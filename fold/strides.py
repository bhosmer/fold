# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Tuple

from .shape import *

#
# Strides - generalized stride representation
#
# Like Shape, a Strides object is a tuple of Dims, with each entry
# representing a linear walk through the array at that dimension.
# So the innermost stride entry is guaranteed to walk the entire
# array element by element. This means that any sequence of array
# positions can be translated directly into Strides metadata - this
# is the basis of our universal use of Views to represent all array
# indexing expressions.
#


@dataclass
class Strides:
    dims: Tuple[Dim, ...]

    def __init__(self, *dims):
        # unlike shapes, lengths of stride dims aren't uniquely
        # determined, so we just accept the passed dims as-is.
        self.dims = tuple(dim(d) for d in dims)

    def __repr__(self) -> str:
        return f"Strides{self.dims}"

    def __str__(self) -> str:
        return f"({', '.join(str(d) for d in self.dims)})"

    def __len__(self):
        return len(self.dims)

    @overload
    def __getitem__(self, i: int) -> Dim:
        ...

    @overload
    def __getitem__(self, i: slice) -> Tuple[Dim, ...]:
        ...

    def __getitem__(self, i: Union[int, slice]) -> Union[Dim, Tuple[Dim, ...]]:
        return self.dims[i]

    # note: here for mypy. Python only needs __len__ and __getitem__
    def __iter__(self) -> Iterator[Dim]:
        for i in range(len(self)):
            yield self[i]

    @property
    def ndim(self) -> int:
        return len(self.dims)

    def unsqueeze(self, n: int):
        n = wrap_dim(n, self.ndim + 1)
        d = self[n - 1] if n > 0 else dim(0)
        dims = (*self[:n], d, *self[n:])
        return Strides(*dims)

    def equal(self, x) -> bool:
        return (
            isinstance(x, Strides)
            and len(self.dims) == len(x.dims)
            and all(d.equal(e) for d, e in zip(self.dims, x.dims))
        )

    def __eq__(self, x) -> bool:
        return (
            isinstance(x, Strides)
            and len(self.dims) == len(x.dims)
            and all(d == e for d, e in zip(self.dims, x.dims))
        )


# note: compression quality of result depends on a) compression quality
# of passed inner, b) compression quality of shape, c) quality of
# fold() implementation for participating Dims
def from_shape_and_inner(shape: Shape, inner: Dim) -> Strides:
    dims: Tuple[Dim, ...] = ()
    for outer in reversed(shape.dims):
        dims = (inner, *dims)
        inner = inner.fold(outer)
        inner = compress_stride_terminal(inner)
    return Strides(*dims)


def contig_strides(s: Shape) -> Strides:
    return from_shape_and_inner(s, Rect(1, s.numel()))


def zero_strides(s: Shape) -> Strides:
    return from_shape_and_inner(s, Rect(0, s.numel()))


#
# extend given stride dims to fit the given shape.
# new stride extents must be even multiples of their
# originals - i.e., this is normally used to take a
# tuple of stride dims as declared and fit it to its
# intended shape.
#
def adjust_stride_dims(stride_dims: Tuple[Dim, ...], shape: Shape) -> Tuple[Dim, ...]:
    if len(stride_dims) != len(shape):
        raise ValueError(f"shape dims {shape} != strides dims {len(stride_dims)}")
    adjusted: Tuple[Dim, ...] = ()
    for n in range(len(stride_dims)):
        stride_dim = stride_dims[n].length_extend(shape[n].sum())
        adjusted += (stride_dim,)
    return adjusted


def calc_cycles(shape, strides, diffs):
    shape_cycle = lcm(shape.cycle(), diffs.cycle())
    shape_orbit_sum = shape[:shape_cycle].sum()
    strides_cycle = strides.cycle()
    if strides_cycle != 0:
        strides_cycle = lcm(strides.cycle(), shape_orbit_sum)
    if strides_cycle > shape_orbit_sum:
        assert strides_cycle % shape_orbit_sum == 0
        shape_cycle = shape.index_of(strides_cycle)
    if strides_cycle != 0:
        assert len(strides) % strides_cycle == 0
        ncycles = len(strides) // strides_cycle
    else:
        ncycles = 0
    return shape_cycle, strides_cycle, ncycles


def adjust_terminal(d: Dim, delta: int) -> Dim:
    if delta == 0:
        return d
    n = len(d)
    if n == 1:
        return d + delta
    if n == 2:
        return d + Seq([0, delta])
    return d + Runs([0, delta], [n - 1, 1])


def adjust_stride_slices(shape: Dim, strides: Dim, diffs: Dim, slice_offs: Dim):
    res = Dim.EMPTY
    shape_offs = shape.offsets()
    for i in range(len(slice_offs) - 1):
        start, end = slice_offs[i], slice_offs[i + 1]
        shape_slice = shape[start:end]
        diffs_slice = diffs[start:end]
        stride_slice = strides[shape_offs[start] : shape_offs[end]]
        res = res.cat(adjust_stride_terminals(shape_slice, stride_slice, diffs_slice))
    return res


def adjust_stride_terminals(shape: Dim, strides: Dim, diffs: Dim) -> Dim:
    assert len(strides) == shape.sum()
    assert len(diffs) == len(shape)

    shape_cycle, strides_cycle, ncycles = calc_cycles(shape, strides, diffs)

    if ncycles > 1:
        shape_orbit = shape[:shape_cycle]
        strides_orbit = strides[:strides_cycle]
        diffs_orbit = diffs[:shape_cycle]
        adj = adjust_stride_terminals(shape_orbit, strides_orbit, diffs_orbit)
        return adj.repeat(ncycles)

    if isinstance(diffs, Chain):
        slice_offs = dim([len(seq) for seq in diffs.seqs]).offsets()
        return adjust_stride_slices(shape, strides, diffs, slice_offs)

    if isinstance(diffs, Runs):
        slice_offs = diffs.reps.offsets()
        return adjust_stride_slices(shape, strides, diffs, slice_offs)

    return concat_dims(
        adjust_terminal(strides[base : base + wid], diff)
        for base, wid, diff in zip(shape.offsets(), shape, diffs)
        if wid > 0
    )


def linearize_stride_dims(shape, strides):
    assert len(strides) == len(shape), f"len({shape}) != len({strides})"
    linearized = ()
    for d in range(len(strides)):
        shape_dim, stride_dim = shape[d], strides[d]
        if d > 0:
            folded = stride_dim.fold(shape_dim)
            diffs = linearized[d - 1].diff(folded)
            if not is_rect(diffs, 0):
                stride_dim = adjust_stride_terminals(shape_dim, stride_dim, diffs)
        linearized += (stride_dim,)
    return linearized


# note: terminal stride is considered a free value
def is_linearized(outer_strides, shape_dim, stride_dim):
    diffs = outer_strides.diff(stride_dim.fold(shape_dim))
    return len(diffs) <= 1 or diffs[:-1].equal(Rect(0, len(diffs) - 1))


# check stride dims for linearization and agreement with shape
def check_stride_dims(shape_dims, stride_dims, fatal=True):
    ndim = len(shape_dims)
    if ndim != len(stride_dims):
        msg = f"len(shape_dims) {ndim} != len(stride_dims) {len(stride_dims)}"
        if fatal:
            raise ValueError(msg)
        print(msg)
        return
    msgs = []
    for d in range(ndim):
        shape_dim, stride_dim = shape_dims[d], stride_dims[d]
        stride_len = len(stride_dim)
        shape_sum = shape_dim.sum()
        if stride_len != shape_sum:
            msgs += [f"dim {d}: stride len {stride_len} != shape sum {shape_sum}"]
        elif d > 0 and not is_linearized(stride_dims[d - 1], shape_dim, stride_dim):
            msgs += [
                f"dim {d}: strides are not linearized:",
                f"\tshape[{d - 1}]: {shape_dims[d - 1]}",
                f"\tstrides[{d - 1}]: {stride_dims[d - 1]}",
                f"\tshape[{d}]: {shape_dim}",
                f"\tstrides[{d}]: {stride_dim}",
            ]
    if len(msgs) > 0:
        msg = "\n".join(msgs)
        if fatal:
            raise ValueError(msg)
        print(msg)


def spread_stride_dims(
    shape_dims: Sequence[Dim], stride_dims: Sequence[Dim], xdim: Dim
) -> Tuple[Dim, ...]:
    if len(stride_dims) == 0:
        return ()
    parts = calc_spread_partitions(shape_dims)
    return tuple(d.spread(xdim, p) for d, p in zip(stride_dims, parts[1:]))


def zero_prefix_expand(d: Dim, xdim: Dim) -> Dim:
    assert len(xdim) == len(d), f"{len(xdim)} != {len(d)}"
    if isinstance(xdim, Rect) and isinstance(d, Rect):
        if d.w == 0:
            return Rect(d.w, d.n * xdim.w)
        return Runs([0, d.w], [xdim.w - 1, 1]).repeat(d.n)

    def expand_elem(n, w):
        if w == 0:
            return Rect(0, n)
        assert n > 0, f"n <= 0 in expand_elem({n}, {w})"
        if n == 1:
            return Rect(w)
        if n == 2:
            return Seq([0, w])
        return Runs([0, w], [n - 1, 1])

    return concat_dims(expand_elem(n, e) for n, e in zip(xdim, d) if n > 0)


def scale_stride_dims(
    shape_dims: Sequence[Dim], stride_dims: Sequence[Dim], xdim: Dim
) -> Tuple[Dim, ...]:
    if len(stride_dims) == 0:
        return ()
    scaled_outer = zero_prefix_expand(stride_dims[0], xdim)
    scaled_inners = spread_stride_dims(shape_dims, stride_dims, xdim)[1:]
    return (scaled_outer, *scaled_inners)


#
# the final value in a stride array is unused. by adjusting it
# post hoc we get better compression on many common stride patterns,
# without obliging the stride calculation algorithms to be smart.
# however, this only catches obvious patterns, where the terminal
# value is a clear outlier in an otherwise-compressed dim. upstream
# adjustment of the terminal will be more effective - see in particular
# View.calc_strides() and work back from there. Ultimately we may need
# to make use of something like the nesting-traveral logic in
# adjust_stride_terminals() to fully propagate regularity from
# the input strides we're traversing.
#
def compress_stride_terminal(sdim: Dim) -> Dim:
    if len(sdim) == 1:
        return sdim if is_rect(sdim) else Rect(sdim[0], 1)
    if len(sdim) == 2:
        return Rect(sdim[0], 2)
    if isinstance(sdim, Runs) and sdim.reps[-1] == 1:
        nv = len(sdim.vals)
        if nv == 1:  # shouldn't happen
            return Rect(sdim.vals[0], 1)
        if nv == 2:
            return Rect(sdim.vals[0], sdim.reps[0] + 1)
        return Runs(sdim.vals[:-1], sdim.reps[:-2].cat(Rect(sdim.reps[-2] + 1)))
    if isinstance(sdim, Chain):
        ns = len(sdim.seqs)
        if ns == 1:  # shouldn't happen
            return compress_stride_terminal(sdim.seqs[0])
        if ns == 2:
            l, r = sdim.seqs
            if isinstance(l, Repeat) and l.seq[:-1].equal(r[:-1]):
                return Repeat(l.seq, l.n + 1)
        return concat_dims(sdim.seqs[:-1] + [compress_stride_terminal(sdim.seqs[-1])])
    # if not is_rect(sdim):
    #     print(f"HEY stride_dim {sdim}")
    return sdim
