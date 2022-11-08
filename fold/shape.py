# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Tuple

from .dim import *

#
# Shape - generalized shape representation
#
# A Shape is just a tuple of Dims with
# - some invariants enforced/provided at construction time
# - promotion of natural Python values at construction time
# - methods for some ops that act purely on shape metadata.
#
# Full design docs elsewhere (eventually), but the key insight
# to generalization is that each entry in the tuple describes
# not a single width but a *sequence* of widths in that dimension.
# indexed by the positions of its outward dimensions. (So each
# dimension is a linear walk through entire array.)
#


def wrap_dim(n: int, ndim: int):
    if n < 0:
        n = max(n + ndim, 0)
    if n < 0 or n >= ndim:
        raise ValueError(f"dimension {n} out of range for ndim {ndim}")
    return n


def ensure_outer_dim(*dims: RawDim) -> Tuple[RawDim, ...]:
    if len(dims) > 0:
        d = dim(dims[0])
        if len(d) > 1:
            return (len(d), d, *dims[1:])
    return dims


def calc_spread_partitions(shape_dims: Sequence[Dim]) -> Tuple[Dim, ...]:
    if len(shape_dims) == 0:
        return ()
    parts: Tuple[Dim, ...] = (Rect(1, len(shape_dims[0])),)
    for d in shape_dims:
        parts += (d.fold(parts[-1]),)
    return parts


def spread_shape_dims(shape_dims: Sequence[Dim], xdim: Dim) -> Tuple[Dim, ...]:
    if len(shape_dims) == 0:
        return ()
    parts = calc_spread_partitions(shape_dims)
    res = tuple(d.spread(xdim, p) for d, p in zip(shape_dims, parts[:-1]))
    return res


def scale_shape_dims(dims: Sequence[Dim], x: Dim) -> Tuple[Dim, ...]:
    if len(dims) == 0:
        return ()
    scaled_outer = dims[0] * x
    spread_inners = spread_shape_dims(dims[1:], x)
    return (scaled_outer, *spread_inners)


def try_length_extend(d: Dim, n: int) -> Optional[Dim]:
    try:
        return length_extend(d, n)
    except ValueError:
        return None


def length_extend(d: Dim, n: int) -> Dim:
    dlen = len(d)
    if n > 0 and n < dlen:
        msg = f"inner dim {d} maps {dlen} cells, outer frame contains {n} position(s) (too few)"
        raise ValueError(msg)
    if dlen == n or dlen == 0:
        return d
    if n % dlen != 0:
        msg = f"inner dim {d} maps {dlen} cells, outer frame contains {n} positions (not an even multiple)"
        raise ValueError(msg)
    return d.repeat(n // dlen)


def canonicalize(dims: Tuple[Dim, ...]) -> Tuple[Dim, ...]:
    # n = 0 if any(len(d) == 0 for d in dims) else 1
    n = 1
    result: Tuple[Dim, ...] = ()
    for d in dims:
        d = length_extend(d, n)
        result = (*result, d)
        n = d.sum()
    return result


@dataclass
class Shape:
    dims: Tuple[Dim, ...]

    def __init__(self, *dims):
        dims = tuple(dim(d) for d in dims)
        if len(dims) > 0 and len(dims[0]) != 1:
            raise ValueError(f"outer dim must have single entry, got {repr(dims[0])}")
        dims = canonicalize(dims)
        self.dims = dims

    def __repr__(self) -> str:
        return f"Shape{self.dims}"

    def __str__(self) -> str:
        desc = lambda d: d.w if isinstance(d, Rect) else d
        return f"({', '.join(str(desc(d)) for d in self.dims)})"

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

    def numel(self) -> int:
        return 1 if len(self.dims) == 0 else self.dims[-1].sum()

    def cycles(self) -> Tuple[int, ...]:
        cs: Tuple[int, ...] = ()
        x = 1
        for d in reversed(self.dims):
            c = d.cycle()
            n = d.offset_of(c)
            x = c * lcm(n, x) // n
            cs = (x, *cs)
        return cs

    def unsqueeze(self, n: int) -> "Shape":
        n = wrap_dim(n, self.ndim + 1)
        return Shape(*self[:n], 1, *self[n:])

    def expand(self, *xdims: RawDim) -> "Shape":
        if len(xdims) < self.ndim:
            raise ValueError(f"number of dims {len(xdims)} < shape ndims {self.ndim}")
        if len(xdims) > self.ndim:
            return self.unsqueeze(0).expand(*xdims)
        shape_dims = self.dims
        for i in range(self.ndim):
            shape_dim = shape_dims[i]
            xdim = dim(xdims[i])
            if is_rect(shape_dim, 1):
                if not is_rect(xdim, -1):
                    xdim = xdim.length_extend(len(shape_dim))
                    shape_dims = (
                        *shape_dims[:i],
                        *scale_shape_dims(shape_dims[i:], xdim),
                    )
            elif not is_rect(xdim, -1):
                xdim_ext = try_length_extend(xdim, len(shape_dim))
                if xdim_ext is None or not xdim_ext.equal(shape_dim):
                    msg = f"expanded size {xdim_ext} must match existing size {shape_dim} at non-singleton dimension {i}"
                    raise ValueError(msg)
        return Shape(*shape_dims)

    def broadcast_to(self, *dims: RawDim) -> "Shape":
        return self.expand(*dims)

    def infer_neg1_dim(self, *dims: RawDim) -> "Shape":
        try:
            neg1 = next(i for i, d in enumerate(dims) if is_rect(dim(d), -1))
        except StopIteration:
            return Shape(*dims)
        pre, post = dims[:neg1], dims[neg1 + 1 :]
        shpre = Shape(*pre)
        shpost = Shape(*ensure_outer_dim(*post))
        # do the traditional inference if rectangular quantities line up
        if self.numel() % (shpre.numel() * shpost.numel()) == 0:
            n = self.numel() // (shpre.numel() * shpost.numel())
            mult = shpost[0][0] if len(shpost) > len(post) else 1
            return Shape(*shpre, n * mult, *shpost[-len(post) :])
        # if our dim in -1 pos fits with new outers, try it
        neg1dim = self.dims[neg1 + len(self) - len(dims)]
        if shpre.numel() == len(neg1dim):
            reshaped = Shape(*shpre, neg1dim, *shpost[-len(post) :])
            if reshaped.numel() == self.numel():
                return reshaped
        msg = f"-1 not supported for in reshape of shape {self} to {dims}"
        raise ValueError(msg)

        # # if new outer shape fits our dim in -1 pos, use it
        # self_neg1 = self.dims[neg1 + len(self) - len(dims)]
        # if pre_shape.numel() == len(self_neg1):
        #     return Shape(*pre_shape, self_neg1, *post_shape[-len(post) :])
        # # otherwise do the traditional inference if possible
        # if self.numel() % (pre_shape.numel() * post_shape.numel()) != 0:
        #     msg = f"-1 not supported for in reshape of shape {self} to {dims}"
        #     raise ValueError(msg)
        # n = self.numel() // (pre_shape.numel() * post_shape.numel())
        # mult = post_shape[0][0] if len(post_shape) > len(post) else 1
        # return Shape(*pre_shape, n * mult, *post_shape[-len(post) :])

    def equal(self, x) -> bool:
        return (
            isinstance(x, Shape)
            and self.ndim == x.ndim
            and all(d.equal(e) for d, e in zip(self.dims, x.dims))
        )

    def __eq__(self, x) -> bool:
        return (
            isinstance(x, Shape)
            and self.ndim == x.ndim
            and all(d == e for d, e in zip(self.dims, x.dims))
        )
