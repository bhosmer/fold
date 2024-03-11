# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Tuple, Callable
import math
import torch

from .strides import *

#
# View
# Array
# Overlay
#

#
# index prep used by View.__getitem__()
#

# the type of post-prep index tuple entries
IndexEntry = Union[int, slice, "Array"]


def isinstance_any(v, types: Sequence[type]):
    return any(isinstance(v, t) for t in types)


def is_advanced_index(ix):
    return isinstance_any(ix, [Array, Overlay, torch.Tensor, Dim, list])


# {} and {<n>} are syntactic sugar for iota indexes
def is_iota_index(ix):
    if isinstance(ix, set) and len(ix) == 1 and isinstance(list(ix)[0], int):
        return True
    return ix == {}


#
# given the entry position, iota index literal, index shape and
# index tuple len, generate the iota index
#
def gen_iota_index(i, ix, shape, ixlen):
    outers = max(0, ixlen - shape.ndim)
    axis = i if ix == {} else list(ix)[0]
    if axis >= outers:
        return iota(*shape, axis=axis - outers)
    return zeros(*shape, dtype=torch.long)


#
# prepare the advanced indexes in a tuple of index entries:
# - broadcast all non-iota arrays with each other
# - generate iota indexes
#
def prep_advanced_indexes(indexes: Tuple[Any, ...]) -> Tuple[Any, ...]:
    ixs = list(indexes)
    advi = [i for i, ix in enumerate(ixs) if is_advanced_index(ix)]
    ioti = [i for i, ix in enumerate(ixs) if is_iota_index(ix)]
    if len(advi) == 0:
        if len(ioti) > 0:
            raise ValueError(f"use of {{}} requires at least one advanced index")
        return ixs
    # TODO temp until Overlay's array API is built
    overlay_shim = lambda ix: ix.clone() if isinstance(ix, Overlay) else array(ix)
    adva = broadcast_arrays(*[overlay_shim(ix) for ix in [ixs[i] for i in advi]])
    for i, a in zip(advi, adva):
        ixs[i] = a
    for i in ioti:
        ixs[i] = gen_iota_index(i, ixs[i], adva[0].shape, len(ixs))
    return tuple(ixs)


# check that a given value has the type of a supported index entry
def check_index_entry_type(d: int, ix: Any):
    if isinstance_any(ix, [int, slice]) or is_advanced_index(ix) or is_iota_index(ix):
        return
    msg = f"expected int, slice, advanced index, {{}} or {{<n>}}, got {repr(ix)} at dim {d}"
    raise ValueError(msg)


#
# prepare an index value (e.g. as passed into __getitem__/__setitem__):
# ensure it's a tuple of index entry values of supported type, padded
# out to the specified width, with advanced indexes broadcast and iota
# literals converted to arrays
#
def prep_indexes(indexes: Any, ndim: int) -> Tuple[IndexEntry, ...]:
    if not isinstance(indexes, tuple):
        indexes = (indexes,)
    for i, index in enumerate(indexes):
        check_index_entry_type(i, index)
    indexes = prep_advanced_indexes(indexes)
    # normalize rank
    if len(indexes) > ndim:
        raise ValueError(f"too many dimensions ({len(indexes)}), ndims = {ndim}")
    if len(indexes) < ndim:
        indexes += (slice(None),) * (ndim - len(indexes))
    return indexes


#
# helpers for View.expand_indexes
#


# wrap negative values and check range
def wrap_index(i: int, width: int, n: int) -> int:
    if i < 0:
        return max(width + i, 0)
    if i >= width:
        raise ValueError(f"dim {n}: index {i} out of range for width {width}")
    return i


def wrap_index_dim(ixdim: Dim, width: Union[Dim, int], n: int) -> int:
    if ixdim.min() < 0:
        return (ixdim + width).floor(0)
    if isinstance(width, int):
        if ixdim.max() > width:
            oor = [(p, i) for p, i in enumerate(ixdim) if i > width]
            msg = f"{len(oor)} index(es) out of range at dim {n} {width=}:\n\t"
            msg += f"\n\t".join(f"{i} at pos {p}" for p, i in oor[:4])
            msg += ", ..." if len(oor) > 4 else ""
            # msg = f"dim {n}: at least one index element out of range for width {width}"
            raise ValueError(msg)
        return ixdim
    if ixdim.max() <= width.min():
        return ixdim
    if not all(i <= w for i, w in zip(ixdim, width)):
        oor = [(p, i, w) for p, (i, w) in enumerate(zip(ixdim, width)) if i > w]
        msg = f"{len(oor)} index(es) out of range at dim {n}:\n\t"
        msg += f"\n\t".join(f"index {i} at pos {p} width {w}" for p, i, w in oor[:4])
        msg += ", ..." if len(oor) > 4 else ""
        # msg = f"dim {n}: at least one index element out of range for width {width}"
        raise ValueError(msg)
    return ixdim


# wrap and validate an index value against the widths of a dimension
def scatter_index(i: int, widths: Dim, n: int) -> Dim:
    if i < 0:
        return (widths + i).floor(0)
    if widths.min() <= i:
        if is_rect(widths):
            wmsg = f"size at this dimension is {widths[0]}"
        else:
            ws = list(set(w for w in widths if w <= i))
            wmsg = f"sizes at this dimension include {ws}"
        msg = f"index {i} out of range for dimension {n} ({wmsg})"
        raise ValueError(msg)
    return Rect(i, len(widths))


#
# return a sequences of slices generated by adjusting the given slice
# and extent to the widths of the given given dim
#
def scatter_slice(slc: slice, extent: int, widths: Dim) -> Tuple[List[slice], Dim]:
    adjusted = False

    def slice_indexes_at(i):
        nonlocal adjusted
        b = bdim[i] if bdim is not None else None
        e = edim[i] if edim is not None else None
        s = sdim[i] if sdim is not None else None
        start, stop, step = slice(b, e, s).indices(widths[i])
        if start != b or stop != e or step != 1:
            adjusted = True
        return start, stop, step

    bdim = dim(slc.start).length_extend(extent) if slc.start is not None else None
    edim = dim(slc.stop).length_extend(extent) if slc.stop is not None else None
    sdim = dim(slc.step).length_extend(extent) if slc.step is not None else None

    indexes = [slice_indexes_at(i) for i in range(len(widths))]
    slices = [slice(*i) for i in indexes]
    if adjusted:
        shape_dim = concat_dims([dim(slice_len(*i)) for i in indexes])
    elif slc.start == None and slc.stop == None and slc.step == None:
        shape_dim = widths.length_extend(extent)
    else:
        shape_dim = edim.diff(bdim)  # type: ignore
    return slices, shape_dim


#
# View
#
# A view defines how a multidimensional Array's elements are assembled
# from a linear datastore.
#


@dataclass
class View:
    shape: Shape
    strides: Strides
    offset: int

    def __init__(
        self, shape: Shape, strides: Optional[Strides] = None, offset: int = 0
    ):
        if strides is None:
            strides = contig_strides(shape)
        elif len(strides) != len(shape):
            raise ValueError(f"len(strides) {len(strides)} != len(shape) {len(shape)}")
        self.shape = shape
        self.strides = strides
        self.offset = offset

    @property
    def ndim(self) -> int:
        return self.shape.ndim

    def numel(self) -> int:
        return self.shape.numel()

    def __getitem__(self, index: Any) -> "View":
        indexes = prep_indexes(index, self.shape.ndim)
        if len(indexes) == 0:
            return self
        result_shape, positions = self.expand_indexes(indexes)
        result_strides, result_offset = self.calc_strides(result_shape, positions)
        return View(result_shape, result_strides, result_offset)

    def compose(self, view: "View") -> "View":
        result_strides, result_offset = self.calc_strides(view.shape, view.addresses())
        return View(view.shape, result_strides, result_offset)

    def expand_indexes(self, indexes: Tuple[IndexEntry, ...]) -> Tuple[Shape, Dim]:
        shape_dims: Tuple[Dim, ...] = ()
        positions = dim(0)
        last_adv_dim = None
        for n, ix in enumerate(indexes):
            inner_widths = self.shape[n][positions]
            stride_offsets = self.shape[n].offsets()
            pos_bases = stride_offsets[positions]
            if isinstance(ix, int):
                ix = scatter_index(ix, inner_widths, n)
                positions = pos_bases + ix
            elif isinstance(ix, slice):
                extent = shape_dims[-1].sum() if len(shape_dims) > 0 else 1
                slices, shape_dim = scatter_slice(ix, extent, inner_widths)
                shape_dims += (shape_dim,)
                slice_dims = [dim(s) + b for b, s in zip(pos_bases, slices)]
                positions = concat_dims(slice_dims)
            elif isinstance_any(ix, [Array, Overlay]):
                ixdim = ix.reshape(-1).todim()
                if last_adv_dim is None:
                    last_adv_dim = n
                    shape_dims = Shape(*shape_dims, *ix.shape).dims
                    pos_list = concat_dims(
                        wrap_index_dim(ixdim, wid, n) + base
                        for base, wid in zip(pos_bases, inner_widths)
                    )
                else:
                    if n - last_adv_dim > 1:
                        raise ValueError("advanced indexes must be adjacent dims")
                    last_adv_dim = n
                    wrapped = wrap_index_dim(ixdim, inner_widths, n)
                    pos_list = pos_bases + wrapped
                positions = dim(pos_list)
        return Shape(*shape_dims), positions

    # convert given positions to strides over *our view of* those positions,
    # conformant to the given shape.
    def calc_strides(self, shape: Shape, positions: Dim) -> Tuple[Strides, int]:
        if shape.numel() != len(positions):
            msg = f"internal error: shape.numel() {shape.numel()} != len(positions) {len(positions)}"
            raise ValueError(msg)
        stride_offsets = self.strides[-1].offsets()
        if isinstance(positions, Affine) and positions.s == 1:
            # shortcut for dense ranges
            inner_strides = self.strides[-1][positions]
        else:
            positions = positions.cat(dim(self.shape.numel()))
            inner_strides = stride_offsets[positions].fwddiff()
            inner_strides = compress_stride_terminal(inner_strides)
        stride_dims = from_shape_and_inner(shape, inner_strides)
        check_stride_dims(shape, stride_dims)
        offset = self.offset + stride_offsets[positions[0]]
        return Strides(*stride_dims), offset

    def addresses(self, axis=-1) -> Dim:
        if self.shape.ndim == 0:
            return dim(self.offset)
        return self.strides.dims[axis].offsets(self.offset)[:-1]

    def is_contiguous(self) -> bool:
        if self.shape.ndim == 0 or self.numel() == 1:
            return True
        return is_singleton(self.strides.dims[-1])

    def is_space(self) -> bool:
        return self.offset == 0 and self.is_contiguous()

    def space(self) -> "View":
        if self.is_space():
            return self
        return View(self.shape)

    def is_injective(self) -> bool:
        addrs = self.addresses()
        return len(set(addrs)) == len(addrs)

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, View)
            and self.shape.equal(other.shape)
            and self.strides.equal(other.strides)
            and self.offset == other.offset
        )

    def unsqueeze(self, n: int):
        shape = self.shape.unsqueeze(n)
        strides = self.strides.unsqueeze(n)
        check_stride_dims(shape, strides)
        return View(shape, strides, self.offset)

    def expand(self, *xdims: RawDim):
        if len(xdims) < self.ndim:
            msg = f"expand: number of dims {len(xdims)} < view ndims {self.ndim}"
            raise ValueError(msg)
        if len(xdims) > self.ndim:
            return self.unsqueeze(0).expand(*xdims)
        shape_dims = self.shape.dims
        stride_dims = self.strides.dims
        changed = False  # just saves work, expand is idempotent
        for i in range(self.ndim):
            shape_dim = shape_dims[i]
            xdim = dim(xdims[i])
            if is_rect(shape_dim, 1):
                if not is_rect(xdim, -1):
                    changed = True
                    xdim = xdim.length_extend(len(shape_dim))
                    stride_dims = (
                        *stride_dims[:i],
                        *scale_stride_dims(shape_dims[i:], stride_dims[i:], xdim),
                    )
                    shape_dims = (
                        *shape_dims[:i],
                        *scale_shape_dims(shape_dims[i:], xdim),
                    )
            elif not is_rect(xdim, -1):
                xdim_ext = try_length_extend(xdim, len(shape_dim))
                if xdim_ext is None or not xdim_ext.equal(shape_dim):
                    msg = f"expand: expanded size {xdim_ext} must match existing size {shape_dim} at non-singleton dimension {i}"
                    raise ValueError(msg)
        if not changed:
            return self  # just saves work, expand is idempotent
        stride_dims = linearize_stride_dims(shape_dims, stride_dims)
        return View(Shape(*shape_dims), Strides(*stride_dims), self.offset)

    def broadcast_to(self, *dims: RawDim) -> "View":
        return self.expand(*dims)

    # TODO replace the try/catch approach with can_broadcast()/can_expand()
    def try_broadcast_to(self, *dims: RawDim) -> Optional["View"]:
        try:
            return self.broadcast_to(*dims)
        except ValueError as ve:
            if "expand:" in str(ve):
                return None
            raise ve

    def reshape(self, *dims: RawDim) -> "View":
        shape = self.shape.infer_neg1_dim(*dims)
        if self.numel() != shape.numel():
            msg = f"shape {shape} is invalid for target of size {self.numel()}"
            raise ValueError(msg)
        inner = self.strides[-1] if len(shape) > 0 else dim(0)
        strides = from_shape_and_inner(shape, inner)
        return View(shape, strides, self.offset)

    # TODO implement with transpose
    def permute(self, *perm: int) -> "View":
        raise ValueError("TODO: for now, use transpose()")

    # generate a shear index for a swapped ragged inner dimension. currently unused
    def _shear_index(self, inner, outer):
        cells = inner.cut(outer)
        hs = [c.max() for c in cells]
        return concat_dims(
            [
                (c.iota() if is_rect(c) else dim([j for j, w in enumerate(c) if w > i]))
                for h, c in zip(hs, cells)
                for i in range(h)
            ]
        )

    # swap adjacent dimensions.
    # x is the outer (leading) dimension of the pair.
    def _swap_adjacent_dims(self, dims: List[Dim], x: int) -> Shape:
        outer, inner = dims[x], dims[x + 1]
        sheared = None
        if is_rect(inner):
            new_outer = Rect(inner.max(), len(outer))
            new_inner = outer.spread(new_outer)
        else:
            cells = inner.cut(outer)
            hs = [c.max() for c in cells]
            ws = [
                len(c) if is_rect(c) else sum(w > i for w in c)
                for h, c in zip(hs, cells)
                for i in range(h)
            ]
            new_outer = concat_dims(Rect(h) for h in hs)
            new_inner = concat_dims(Rect(w) for w in ws)
            if any(len(c) > 1 and c.fwddiff().max() > 0 for c in cells):
                # note: to support shearing transpose, this index needs to be plumbed through.
                # note the extra complication due to use of adjacent swap chains
                # sheared = self._shear_index(inner, outer)
                #
                # note the omission of dimension indexes in the error message, to avoid
                # confusion due to chains of adjacent swaps having moved dims from their
                # original positions. TODO either track dim origin positions or restructure
                # the algorithm
                shear = [c for c in cells if len(c) > 1 and c.fwddiff().max() > 0]
                offs = outer.offsets()[:-1]
                msg = f"transpose: the following shape(s) will shear:"
                msg += "".join(
                    [f"\n\t{list(c)} at position {i}" for c, i in zip(shear, offs)]
                )
                raise ValueError(msg)

        # ragged dimensions inward of the swap need explicit shape transposition
        new_tail = dims[x + 2 :]
        for i in range(x + 2, len(dims)):
            if not is_rect(dims[i]):
                d = Array(dims[i], View(Shape(*dims[:i])))
                t = d.transpose(x, x + 1)
                new_tail[i - x - 2] = t.eval().data

        dims = dims[:x] + [new_outer, new_inner] + new_tail
        return dims

    # transpose a shape by swapping the specified axes.
    # note that we only allow transpositions that avoid shear, see comments
    # on transpose() for details
    #
    def _transpose_shape(self, x: int, y: int) -> Shape:
        shape = self.shape
        if x >= self.ndim:
            raise ValueError(f"invalid transpose dim {x} for ndim {self.ndim}")
        if y >= self.ndim:
            raise ValueError(f"invalid transpose dim {y} for ndim {self.ndim}")
        x, y = (x, y) if x < y else (y, x)
        x, y = [wrap_dim(i, self.ndim) for i in [x, y]]
        dims = list(shape)
        # for simplicity, swap adjacent pairs
        for i in reversed(range(x, y)):
            dims = self._swap_adjacent_dims(dims, i)
        for i in range(x + 1, y):
            dims = self._swap_adjacent_dims(dims, i)
        return Shape(*dims)

    # transpose the view by swapping the specified axes.
    # transposition is defined in the usual way on rectangular arrays,
    # and extends straightforwardly to ragged shapes.
    #
    # We do make one design choice: a transposition that will cause "shearing"
    # will raise an error, even though it would produce a well-formed array.
    # Shearing happens when the alignment of elements shifts during transposition,
    # making it non-invertible (though idempotent).
    #
    # Shearing occurs when ragged dimensions transpose outward, except when
    # the ragged inner dimension is strictly narrowing (i.e. the width of
    # each element is less than or equal to the width of its predecessor)
    # within each segment partitioned by the outer dimension it is being
    # transposed with.
    #
    # Note that this is a view -> view transformation, so the result is
    # simply a restrided view. To produce a contiguous transposed array,
    # call array.transpose(x, y).eval().
    #
    def transpose(self, x: int, y: int) -> "View":
        tshape = self._transpose_shape(x, y)
        # use swapped-dimension indexing into the new shape to restride
        perm = list(range(self.ndim))
        # note: this indexing scheme breaks in the presence of shear
        perm[x], perm[y] = y, x
        index = tuple(iota(*tshape, axis=perm.index(i)) for i in range(len(perm)))
        return self[index]


#
# Array
#
# The array class is essentially a simple association of a linear data store
# with a View. For Array API functionality that doesn't involve data elements,
# Array just delegates to the View API.
#


@dataclass
class Array:
    data: Union[torch.Tensor, Dim]
    view: View

    def __init__(self, data: Union[torch.Tensor, Dim], v: Union[Shape, View]):
        self.data = data
        self.view = v if isinstance(v, View) else View(v)

    def __getitem__(self, index: Any) -> "Array":
        return Array(self.data, self.view[index])

    def __setitem__(self, index: Any, a: Any):
        self.overlay[index](a).eval()

    @property
    def overlay(self):
        @dataclass
        class OverlayBuilder:
            lhs: Array

            def __getitem__(self, index):
                return lambda src: Overlay(self.lhs, index, array(src))

        return OverlayBuilder(self)

    @property
    def shape(self):
        return self.view.shape

    @property
    def strides(self):
        return self.view.strides

    @property
    def offset(self):
        return self.view.offset

    @property
    def ndim(self) -> int:
        return self.view.ndim

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    def numel(self) -> int:
        return self.view.numel()

    def __len__(self) -> int:
        return 0 if self.ndim == 0 else self.shape[0][0]

    def __iter__(self) -> Iterator:
        for i in range(len(self)):
            yield self[i]

    def space(self) -> View:
        return self.view.space()

    def item(self):
        if self.shape.ndim > 0:
            raise ValueError(f"item() called on non-scalar, shape = {self.shape}")
        if isinstance(self.data, torch.Tensor):
            return self.data[self.offset].item()
        return self.data[self.offset]

    def eval(self):
        return self if self.is_initial() else self.clone()

    def contiguous(self):
        return self if self.is_contiguous() else self.clone()

    def clone(self):
        return Array(self.data[self.view.addresses()], View(self.shape))

    def update(self, positions: Dim, values: Union[torch.Tensor, Dim]):
        if not self.view.is_injective():
            raise ValueError("can't set items (target reuses elements)")
        addresses = self.view.addresses()[positions]
        self.data[addresses] = values  # type: ignore

    def is_contiguous(self) -> bool:
        return self.view.is_contiguous()

    def is_initial(self) -> bool:
        return self.view.numel() == len(self.data) and self.view.is_space()

    def tolist(self) -> List:
        if self.ndim == 0:
            if isinstance(self.data, Dim):
                return self.data[self.view.offset]
            return self.data[self.view.offset].item()
        elif self.ndim == 1:
            return self.data[self.view.addresses()].tolist()
        return [r.tolist() for r in self]

    def todim(self) -> Dim:
        if self.ndim != 1:
            raise ValueError(f"self.ndim must be 1, got {self.ndim}")
        if isinstance(self.data, Dim):
            return self.data[self.view.addresses()]
        storage = self.data.storage()
        addrs = self.view.addresses()
        return Seq([storage[addr] for addr in addrs])

    def item_iter(self) -> Iterator:
        addrs = self.view.addresses()
        if isinstance(self.data, torch.Tensor):
            for a in addrs:
                yield self.data.storage()[a]
        else:
            for a in addrs:
                yield self.data[a]

    def __eq__(self, other) -> bool:
        if not (isinstance(other, Array) and self.shape == other.shape):
            return False
        self_data = self.eval().data
        other_data = other.eval().data
        if isinstance(self_data, torch.Tensor) and isinstance(other_data, Dim):
            other_data = torch.tensor(other_data)
        return self_data.equal(other_data)

    def __str__(self) -> str:
        data = self.eval().data

        def has_neg(data):
            if isinstance(data, torch.Tensor) and torch.any(data < 0):
                return True
            if isinstance(data, Dim) and data.min() < 0:
                return True
            return False

        # TODO: until we have Dim.abs(), which may provoke Dims-use-Tensors port
        def abs_temp(data):
            if isinstance(data, torch.Tensor):
                return data.abs()
            if data.min() >= 0:
                return data
            return dim([abs(x) for x in data])

        def places():
            nsign = 1 if has_neg(data) else 0
            x = 0 if self.numel() == 0 else abs_temp(data).max()  # data.abs().max()
            return nsign + (1 if x == 0 else math.ceil(math.log(x, 10)))

        if isinstance(data, torch.Tensor) and data.is_floating_point():
            fmt = f"{{:{5 + places()}.4f}}"
        elif isinstance(data, torch.Tensor) and data.dtype == torch.bool:
            fmt = "{!s:>5}"
        else:
            fmt = f"{{:{places()}}}"

        rank = self.shape.ndim
        if rank == 0:
            return f"{data[0].item()}"

        def pr(a: "Array", ind: int = 1) -> str:
            if a.shape.ndim == 1:
                data = a.eval().data
                items = [fmt.format(x) for x in data.tolist()]
                sep = ", "
            else:
                items = [pr(row, ind + 1) for row in a]
                sep = "," + ("\n" * (rank - ind)) + (" " * ind)
            return "[" + sep.join(items) + "]"

        return pr(self)

    def pointwise_unary(
        self, op: Callable[[torch.Tensor], torch.Tensor], inplace=False
    ) -> "Array":
        if isinstance(self.data, Dim):
            self.data = torch.tensor(self.data)
        if inplace:
            # if we can, really do it inplace
            # TODO note consequences on gradients per AD
            tv = self.maybe_torch_data_view()
            if tv is not None:
                op(tv)
                return self
        x = self.eval().data
        y = op(x)
        if inplace:
            self[:] = array(y)
            return self
        return Array(y, View(self.shape))

    def maybe_torch_data_view(self):
        # return a (torch) view on our torch.Tensor data, shape allowing
        if (
            isinstance(self.data, torch.Tensor)
            and self.is_initial()
            and self.shape.is_rect()
        ):
            pt_shape = eval(str(self.shape))
            return self.data.view(*pt_shape)
        return None

    def unsqueeze(self, dim: int) -> "Array":
        return Array(self.data, self.view.unsqueeze(dim))

    def expand(self, *dims: RawDim) -> "Array":
        return Array(self.data, self.view.expand(*dims))

    def broadcast_to(self, *dims: RawDim) -> "Array":
        return Array(self.data, self.view.broadcast_to(*dims))

    def try_broadcast_to(self, *dims: RawDim) -> Optional["Array"]:
        bc_view = self.view.try_broadcast_to(*dims)
        return None if bc_view is None else Array(self.data, bc_view)

    def reshape(self, *dims: RawDim) -> "Array":
        return Array(self.data, self.view.reshape(*dims))

    def permute(self, *p: int) -> "Array":
        return Array(self.data, self.view.permute(*p))

    # transpose the given axes.
    #
    # Returns an array with our data and a restrided view.
    # To produce a contiguous array, use eval().
    # For more details, see View.transpose()
    #
    def transpose(self, x: int, y: int) -> "Array":
        return Array(self.data, self.view.transpose(x, y))

    #
    # chunk matches PT definition for rectangular dims. for others there
    # are 3 options:
    # 1. don't allow chunking along nonrectangular dims
    # 2. choose chunk sizes based on max extent along selected dim
    # 3. choose chunk sizes at each extent along selected dim
    #
    # all are relatively straightforward. here we do 2, although use cases
    # could be imagined for 3
    #
    def chunk(self, chunks: int, dim: int):
        len = self.shape[dim].max()
        n = (len + chunks - 1) // chunks
        ix = [slice(None)] * self.ndim
        chunks = ()
        for i in range(0, len, n):
            ix[dim] = slice(i, i + n)
            chunks += (self[tuple(ix)],)
        return chunks

    #
    # scatter() and gather() both take an index whose shape conforms to the
    # source array in all but one specified dimension: locations in that
    # dimension are given explicitly by the index, whereas locations in all
    # other dimensions are simply derived from the positions of these index
    # values.
    #
    # When the index is rectangular, this format is a generalization of the
    # ELL sparse index format. That format always uses the index to specify
    # locations in the innermost dimension, whereas scatter/gather allow any
    # (single) dimension's locations to be specified explicitly.
    #
    # When the index is ragged, the format is a generalization of the CSR/CSC
    # sparse index format. Those formats assume 2 dimensions, whereas scatter/
    # gather allow an arbitrary number of dimensions. (TODO: CSF is a
    # multidimensional variant of CSR, does it relate cleanly?)
    #
    # Here scatter() and gather() are thin wrappers over advanced indexing.
    # Advanced indexing expressions return functionalized results (views and
    # overlays), which can be used as-is or evaluated to produce a materialized
    # result. For now, see test_array.py for usage examples.
    #
    def gather(self, dim: int, index: "Array") -> "Array":
        dim = wrap_dim(dim, self.ndim)
        ixs = tuple(index if i == dim else {} for i in range(self.ndim))
        return self[ixs]

    def scatter(self, dim: int, index: "Array", src: "Array") -> "Array":
        dim = wrap_dim(dim, self.ndim)
        ixs = tuple(index if i == dim else {} for i in range(self.ndim))
        return self.overlay[ixs](src)


#
# Overlay
#
# this is pretty crude, but it's enough to establish the connection between
# deferred assignment and "overlay" formats like sparsity, diag, tri etc.
# biggest things missing:
#
# 1. Overlay needs the full Array interface. Options: inheritance would
# minimize redundancy but make the taxonomy more annoying, duck typing
# would keep classes simple at the cost of extra plumbing for Array method
# impls that just delegate to dest. Note also that we currently return
# plain Arrays when __getitem__ hits exclusively src or dest.
#
# 2. chained overlays: src: Union[Array, Overlay]. this is what __setitem__
# should do. needs #1
#
# 3. different pullback shape(s). full explanation elsewhere (eventually)
# but the TLDR is that the src/image pair isn't uniquely determined for
# a particular mapping of source elements to destination locations:
# we can pick any src view/image view pair that composes to put the
# right elements in the right places in dest. this corresponds to having
# free choice of sparse layout - but in practice we choose this carefully,
# to facilitate performance on materialized values (here, src).
# currently we take src view/image view as given, but when manipulating
# the dest/image/src triple (e.g. on __getitem__) we just collapse to a
# linear src view containing the desired elements. (this is analogous to
# switching to a single-vector linear-address COO sparse index format).
# by throwing away the original source view shape (rather than, say,
# producing a subshape) we're so to speak unilaterally changing formats,
# and probably throwing away regularity. A View.intersect() that produces
# a subshape of the view shape is where to start.
#
# 4. Dim.intersect(). this is necessary for efficiency in View.intersect
# and should also help preserve regularity in the result strides,
# regardless of pullback shape.
#


@dataclass
class Overlay:
    dest: Array
    image: View  # src shape, positions in dest *space*, injective
    src: Array

    def __init__(self, dest: Array, index_or_image: Any, src: Array):
        # convert index arg to image
        if isinstance(index_or_image, View):
            image = index_or_image
        else:
            image = dest.space()[index_or_image]
        # currently we only broadcast when numel is unequal, but could tighten to broadcast-always
        if image.numel() != src.numel():
            bcsrc = src.try_broadcast_to(*image.shape)
            if bcsrc is not None:
                src = bcsrc
            else:
                msg = f"source shape {src.shape} cannot be broadcast to image shape {image.shape}"
                raise ValueError(msg)
        if not image.is_injective():
            msg = "image is not injective (index contains duplicate locations)"
            raise ValueError(msg)
        if image.addresses().max() >= dest.numel():
            msg = f"image address {image.addresses().max()} >= target numel {dest.numel()}"
            raise ValueError(msg)
        self.dest = dest
        self.src = src
        self.image = image

    # overlay[x].eval() == overlay.eval()[x]
    # TODO avoid address expansion with Dim.intersect()
    def __getitem__(self, index: Any) -> Union["Overlay", Array]:
        space = self.dest.space()
        new_addrs = space[index].addresses()
        old_src_addrs = self.image.addresses()
        new_addrs_set, old_src_addrs_set = set(new_addrs), set(old_src_addrs)
        # TODO Dim.intersect yikes
        new_src_index = concat_dims(
            Rect(i) for i, a in enumerate(old_src_addrs) if a in new_addrs_set
        )
        new_img_index = concat_dims(
            Rect(i) for i, a in enumerate(new_addrs) if a in old_src_addrs_set
        )
        new_dest = self.dest[index]
        n = len(new_src_index)
        if n == 0:
            return new_dest
        new_src = self.src.reshape(-1)[new_src_index]
        if n == len(new_addrs):
            return new_src.reshape(*new_dest.shape)
        new_img_strides_offset = space.calc_strides(new_src.shape, new_img_index)
        new_image = View(new_src.shape, *new_img_strides_offset)
        return Overlay(new_dest, new_image, new_src)

    def eval(self) -> Array:
        self.dest.update(self.image.addresses(), self.src.eval().data)
        return self.dest

    def clone(self) -> Array:
        over = Overlay(self.dest.clone(), self.image, self.src)
        return over.eval()

    @property
    def shape(self):
        return self.dest.shape

    @property
    def ndim(self) -> int:
        return self.dest.ndim

    def numel(self) -> int:
        return self.dest.numel()

    def __len__(self) -> int:
        return 0 if self.ndim == 0 else self.shape[0][0]

    def __iter__(self) -> Iterator:
        for i in range(len(self)):
            yield self[i]

    def item(self):
        if self.shape.ndim > 0:
            raise ValueError(f"item() called on non-scalar, shape = {self.shape}")
        return self.src.item()  # TODO nope

    def tolist(self) -> List:
        return self.clone().tolist()

    def __str__(self) -> str:
        return str(self.clone())


#
# builders
#


def array(x: Any, dtype=None) -> Array:
    if isinstance_any(x, [Array, Overlay]):
        return x

    if isinstance(x, torch.Tensor):
        data = x if x.ndim == 1 else x.reshape(-1)
        return Array(data, View(Shape(*x.shape)))

    def collect(x: Sequence):
        if len(x) == 0:
            return None, 1, (0,), x
        if all(isinstance(y, bool) for y in x):
            return bool, 1, (len(x),), x
        if all(isinstance(y, int) for y in x):
            return int, 1, (len(x),), x
        if all(isinstance(y, float) for y in x):
            return float, 1, (len(x),), x
        if all(isinstance(y, Sequence) for y in x):
            types, ranks, dims, datas = zip(*[collect(y) for y in x])
            if len(set(types).difference([None])) != 1:
                msg = f"lists must have uniform element type, got types {set(types)}"
                raise ValueError(msg)
            ty = types[0]
            if len(set(ranks)) != 1:
                msg = f"lists must have uniform nesting depth, got depths {set(ranks)}"
                raise ValueError(msg)
            rank = ranks[0]
            n = len(dims)
            outer = simple_dim([d[0] for d in dims])
            inners = tuple(concat_dims(d[i] for d in dims) for i in range(1, rank))
            dims = (n, outer, *inners)
            data = [d for data in datas for d in data]
            return ty, rank + 1, dims, data
        raise ValueError(f"list cannot have elements of mixed type: {x}")

    if not isinstance(x, Sequence):
        _, _, _, vals = collect([x])
        dims = ()
    else:
        _, _, dims, vals = collect(x)

    data = torch.tensor(vals, dtype=dtype)
    shape = Shape(*dims)
    return Array(data, View(shape))


def rand(*dims: RawDim, dtype=None) -> Array:
    shape = Shape(*dims)
    data = torch.rand(shape.numel(), dtype=dtype)
    return Array(data, View(shape))


def fill(*dims: Tuple[RawDim, ...], value, dtype=None) -> Array:
    shape = Shape(*dims)
    data = torch.full((shape.numel(),), value, dtype=dtype)
    return Array(data, View(shape))


def zeros(*dims: RawDim, dtype=None) -> Array:
    return fill(*dims, value=0.0, dtype=dtype)


def ones(*dims: RawDim, dtype=None) -> Array:
    return fill(*dims, value=1.0, dtype=dtype)


def true(*dims: RawDim):
    return fill(*dims, value=True)


def false(*dims: RawDim):
    return fill(*dims, value=False)


def arange(*dims: RawDim, start=0, dtype=torch.long, const=False) -> Array:
    shape = Shape(*dims)
    n = shape.numel()
    if const and dtype == torch.long:
        data = Affine(start, n, 1) if n > 1 else Rect(start)
    else:
        data = torch.arange(start=start, end=start + n, dtype=dtype)
    return Array(data, View(shape))


#
# broadcast tuple of arrays to a common shape, if compatible
# TODO: currently we just let a pairwise broadcast blow up
# if shapes are incompatible - should catch errors here and
# put out a better error
#
# see torch.broadcast_tensors
# https://pytorch.org/docs/stable/generated/torch.broadcast_tensors.html?highlight=broadcast_tensors#torch.broadcast_tensors
#
def broadcast_arrays(*arrays: Array) -> Tuple[Array, ...]:
    if len(arrays) < 2:
        return arrays
    shape = arrays[0].shape
    for a in arrays[1:]:
        dims = tuple(d if not is_singleton(dim(d)) else dim(-1) for d in a.shape)
        dims = (*shape[: -len(dims)], *dims)
        shape = shape.broadcast_to(*dims)
    return tuple(a.broadcast_to(*shape) for a in arrays)


#
# construct an index array for the given axis of the given shape.
# e.g. `iota(3, 2, axis=1) == array([[0, 1], [0, 1], [0, 1]])`
#
# for rectangular shapes, the equivalent result can be built using
# arange and expansion, e.g. `arange(1, 2).expand(3, 2)` produces
# the same result as the expression above. but ragged shapes aren't
# reproducible this way - e.g.
# `iota(3, [3, 2, 1], axis=1) == array([[0, 1, 2], [0, 1], [0]]))`.
#
# notes:
#
# 1. this approach builds an Array that uses a Dim (rather than a
# torch.Tensor) as a backing store. Array's internal support for
# this is a bit hasty and should be hardened (see TODO). Also,
# the longstanding idea of using a torch.Tensor as a backing store
# for at least some Dims may close the loop further (see TODO).
#
# 2. there's a deeper relationship to be made precise between
# virtualized data (as done here) and repeated traversal of a
# simple sequence using noncontiguous strides, a la the
# traditional arange()-and-expand() approach. For reference
# we have a generalized version of the latter in `strided_iota()`
# below.
#
def iota(*shape_dims: RawDim, axis: int) -> Array:
    shape = Shape(*shape_dims)
    axis = wrap_dim(axis, shape.ndim)
    # indexes of all elements in this dimension
    idxs = shape[axis].iota()
    # distance (in index space) between elements in this dimension
    dist = contig_strides(shape)[axis]
    # indexes for every leaf element
    seq = idxs.spread(dist)
    # reshape as full index on original array
    return Array(seq, shape)


#
# see above - rather than virtualizing the value sequence itself,
# here we generalize the expanded-arange approach, meaning the
# result has a simple data sequence (the arange) and noncontiguous
# strides that traverse it repeatedly. This is less efficient than
# a virtualized data sequence, since the compression is shifted to
# the strides, where (given our linear-walk invariant) the
# backedges that implement repetition must be made explicit.
#
def strided_iota(*shape_dims: RawDim, axis: int) -> Array:
    shape = Shape(*shape_dims)
    axis = wrap_dim(axis, shape.ndim)
    # this important subset of shapes would need additional
    # complexity - fall back to iota() for correctness
    if axis < shape.ndim - 1 and shape[axis + 1].min() == 0:
        return iota(*shape_dims, axis=axis)
    raw = (*[0] * axis, 1, *[0] * (shape.ndim - axis - 1))
    stride_dims = tuple(dim(d) for d in raw)
    adjusted = adjust_stride_dims(stride_dims, shape)
    linearized = linearize_stride_dims(shape, adjusted)
    strides = Strides(*linearized)
    extent = max(shape[axis])
    vals = Affine(0, extent, 1) if extent > 1 else Rect(0, extent)
    return Array(torch.tensor(vals), View(shape, strides))
