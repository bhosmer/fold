# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from bisect import bisect
from dataclasses import dataclass
from functools import reduce
import math
import random
from typing import (
    Any,
    Dict,
    Iterator,
    Iterable,
    List,
    Set,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
)

#
# Dim interface and implementations, plus assorted builders and utils.
# Dims are the building blocks used by Shape and Strides representations.
#

#
# typedefs
#

# dim() promotes DimDesc to Dim
DimDesc = Union[
    int,  # Rect(val, 1)
    slice,  # Affine(start, stop, step) (indexes not wrapped)
    range,  # Affine(start, stop, step) (indexes not wrapped)
    List[int],  # Seq(vals)
]

RawDim = Union[DimDesc, "Dim"]


#
# helpers
#


def lcm(a, b):
    return abs(a * b) // math.gcd(a, b)


def flatmap_optional(f, x):
    return [y for y in map(f, x) if y is not None]


def offsets(data: Sequence[int], base: int = 0) -> List[int]:
    return reduce(lambda xs, y: xs + [xs[-1] + y], data, [base])


def fwddiff(data: Sequence[int]) -> List[int]:
    return [data[i + 1] - data[i] for i in range(len(data) - 1)]


def concat_dims(dims: Iterable["Dim"]) -> "Dim":
    return reduce(lambda acc, l: acc.cat(l), dims, Dim.EMPTY)


def seq_to_dim(seq: Sequence[int]) -> "Dim":
    return concat_dims(Rect(w) for w in seq)


# e.g. split([1, 2, 3, 4], [1, 2]) = [[[1], [2, 3], [4]]
def split(seq: Sequence[int], part: Sequence[int]):
    if sum(part) != len(seq):
        msg = f"sum of segments {part} != sequence length {len(seq)}"
        raise ValueError(msg)
    part = part[:-1]  # final segment length now guaranteed
    f = lambda s, n: s[:-1] + [s[-1][:n], s[-1][n:]]
    return reduce(f, part, [seq])


#
# A Dim is an int sequence. Subclasses are specialized to model
# different kinds of regularity. Most regular is Rect, least regular
# is Seq. Others model repetition, sparsity, concatenation etc.
# Dim API is a bunch int sequence ops. Subclasses implement them to
# take advantage of whatever regularity they model.
#
# Code structure here is: method skeletons with common check code
# and trivial case handling are in Dim directly, with nontrivial
# post-check implementations in the subclasses.
#


class Dim(Sequence[int]):
    EMPTY: "Dim"

    # outer extent (length) of this dimension
    def __len__(self) -> int:
        raise NotImplementedError

    # length of dim's kernel pattern
    def cycle(self) -> int:
        raise NotImplementedError

    # number of cycles - convenience for zero check
    def ncycles(self) -> int:
        c = self.cycle()
        return len(self) // c if c > 0 else 0

    # slice at cycle
    def orbit(self) -> "Dim":
        c = self.cycle()
        if c == 0:
            return Dim.EMPTY
        elif c == len(self):
            return self
        else:
            return self.orbit_impl()

    def orbit_impl(self) -> "Dim":
        return self[: self.cycle()]

    # value at self[i]
    @overload
    def __getitem__(self, i: int) -> int: ...

    # slice Dim at self[start:stop]
    @overload
    def __getitem__(self, i: slice) -> "Dim": ...

    # subdimension at given positions
    @overload
    def __getitem__(self, ixs: Sequence[int]) -> "Dim": ...

    # impl of both __getitem__ overloads
    def __getitem__(self, i: Union[int, slice, Sequence[int]]) -> Union[int, "Dim"]:
        raise NotImplementedError

    # reverse __getitem__: self.select(d) == d.__getitem__(self)
    def select(self, d: "Dim") -> "Dim":
        if len(self) == 0:
            return Dim.EMPTY
        return self.select_impl(d)

    def select_impl(self, d: "Dim") -> "Dim":
        return d[[i for i in self]]

    # note: here for mypy. Python only needs __len__ and __getitem__
    def __iter__(self) -> Iterator[int]:
        for i in range(len(self)):
            yield self[i]

    # offset of a given index - itemwise version of offsets()
    def offset_of(self, i: int) -> int:
        if i > len(self):
            raise IndexError(f"{i} > {len(self)}")
        return self.offset_of_impl(i)

    def offset_of_impl(self, i: int) -> int:
        raise NotImplementedError

    # offsets (integral)
    def offsets(self, base: int = 0) -> "Dim":
        raise NotImplementedError

    # forward difference (derivative)
    def fwddiff(self) -> "Dim":
        raise NotImplementedError

    # index of segment containing linear offset x
    # TODO Seq implementations and maybe others currently assume positive values,
    # so are broken for negative strides. Either fix or remove
    def index_of(self, x: int) -> int:
        if x > self.sum():
            raise IndexError(f"{x} > {self.sum()}")
        return self.index_of_impl(x)

    def index_of_impl(self, x: int) -> int:
        raise NotImplementedError

    # catenate self with d
    def cat(self, d: "Dim") -> "Dim":
        if len(self) == 0:
            return d
        if len(d) == 0:
            return self
        return self.cat_impl(d)

    def cat_impl(self, d: "Dim") -> "Dim":
        raise NotImplementedError

    # repeat self n times
    def repeat(self, n: int) -> "Dim":
        if n == 0:
            return Dim.EMPTY
        if n == 1:
            return self
        return self.repeat_impl(n)

    def repeat_impl(self, n: int) -> "Dim":
        return Repeat(self, n)

    # convenience: use repeat to extend dim
    def length_extend(self, n: int) -> "Dim":
        self_len = len(self)
        if n > 0 and n < self_len:
            msg = f"length_extend: target length {n} < self length {self_len}"
            raise ValueError(msg)
        if self_len == n or self_len == 0:
            return self
        if n % self_len != 0:
            msg = f"length_extend: target length {n} not an even multiple of self length {self_len}"
            raise ValueError(msg)
        return self.repeat(n // self_len)

    # spread: itemwise or partitioned repeat
    def spread(self, x: "Dim", p: Optional["Dim"] = None) -> "Dim":
        if p is None:
            if len(x) != len(self):
                raise ValueError(f"spreads len {len(x)} != self len {len(self)}")
        else:
            if len(p) != len(x):
                raise ValueError(f"partition len {len(p)} != spreads len {len(x)}")
            if p.sum() != len(self):
                raise ValueError(f"partition sum {p.sum()} != self len {len(self)}")
        if len(self) == 0:
            return self
        if p is None or is_rect(p, 1):
            if is_rect(x, 0):
                return Dim.EMPTY
            if is_rect(x, 1):
                return self
            return self.spread_itemwise_impl(x)
        else:
            return self.spread_partitioned_impl(x, p)

    def spread_itemwise_impl(self, x: "Dim") -> "Dim":
        if len(self) == 1:
            return self.repeat(x[0])
        if is_singleton(x):
            return self
        return Runs(self, x)

    def spread_partitioned_impl(self, x: "Dim", p: "Dim") -> "Dim":
        segs = [self[p.offset_of(i) : p.offset_of(i + 1)] for i in range(len(p))]
        return concat_dims(s.repeat(n) for s, n in zip(segs, x))

    # alias for shift
    def __add__(self, x: Union[int, "Dim"]) -> "Dim":
        return self.shift(x)

    # alias for diff
    def __sub__(self, x: Union[int, "Dim"]) -> "Dim":
        return self.diff(x)

    # alias for scale(-1)
    def __neg__(self) -> "Dim":
        return self.scale(-1)

    # alias for scale
    def __mul__(self, x: Union[int, float, "Dim"]) -> "Dim":
        return self.scale(x)

    # scale (itemwise mult) by the given int, float or Dim entries
    def scale(self, x: Union[int, "Dim"]) -> "Dim":
        if isinstance(x, int):
            return self.scale_int_impl(x)
        if isinstance(x, float):
            raise ValueError(f"{x=}")
        if len(self) == 0:
            return self
        if len(x) != len(self):
            raise ValueError(f"len(x) {len(x)} != len(self) {len(self)}")
        if x.cycle() == 1:
            return self.scale(x[0])
        return self.scale_dim_impl(x)

    def scale_int_impl(self, x: int) -> "Dim":
        raise NotImplementedError

    def scale_dim_impl(self, x: "Dim") -> "Dim":
        raise NotImplementedError

    # shift (itemwise add) by the given int or Dim entries
    # scale param is a back pocket fuse - self.shift(x, -1) == self.shift(x.scale(-1))
    def shift(self, x: Union[int, "Dim"], scale: int = 1) -> "Dim":
        if isinstance(x, int):
            if x == 0:
                return self
            return self.shift_int_impl(x, scale)
        if len(x) != len(self):
            raise ValueError(f"len(x) {len(x)} != len(self) {len(self)}")
        if len(self) == 0:
            return self
        if x.cycle() == 1:
            return self.shift(x[0], scale)
        return self.shift_dim_impl(x, scale)

    def shift_int_impl(self, x: int, scale: int) -> "Dim":
        raise NotImplementedError

    def shift_dim_impl(self, x: "Dim", scale: int) -> "Dim":
        raise NotImplementedError

    # [0..w) for w in self, flattened
    # overriden by subs
    def iota(self) -> "Dim":
        def makedim(w):
            if w > 2:
                return Affine(0, w, 1)
            if w > 1:
                return Seq([0, 1])
            return Rect(0, w)

        return concat_dims(makedim(w) for w in self)

    # result of itemwise max(x, self[i])
    def floor(self, x: int) -> "Dim":
        if len(self) == 0 or self.min() >= x:
            return self
        return self.floor_impl(x)

    def floor_impl(self, x: int) -> "Dim":
        raise NotImplementedError

    # result of itemwise min(x, self[i])
    def ceil(self, x: int) -> "Dim":
        if len(self) == 0 or self.max() <= x:
            return self
        return self.ceil_impl(x)

    def ceil_impl(self, x: int) -> "Dim":
        raise NotImplementedError

    # itemwise subtraction
    def diff(self, x: "Dim") -> "Dim":
        return self.shift(x, -1)

    # fold (coalesce) our entries as specified by partition
    def fold(self, part: "Dim") -> "Dim":
        if part.sum() != len(self):
            msg = f"fold: partition sum {part.sum()} != sequence length {len(self)}"
            raise ValueError(msg)
        return self.fold_impl(part)

    def fold_impl(self, part: "Dim") -> "Dim":
        raise NotImplementedError

    # cut into sections as specified by partition
    def cut(self, part: "Dim") -> Sequence["Dim"]:
        if part.sum() != len(self):
            msg = f"cut: partition sum {part.sum()} != sequence length {len(self)}"
            raise ValueError(msg)
        poffs = part.offsets()
        return [self[poffs[i] : poffs[i + 1]] for i in range(len(part))]

    # sum of our inner extents (linear extent in scalar units)
    def sum(self) -> int:
        raise NotImplementedError

    # minimum value
    def min(self) -> int:
        if len(self) == 0:
            raise ValueError("min() of an empty Dim")
        return self.min_impl()

    def min_impl(self) -> int:
        raise NotImplementedError

    # maximum value
    def max(self) -> int:
        if len(self) == 0:
            raise ValueError("max() of an empty Dim")
        return self.max_impl()

    def max_impl(self) -> int:
        raise NotImplementedError

    def tolist(self) -> List[int]:
        # careful
        return [w for w in self]

    # equality mod representation - expensive
    def equal(self, d) -> bool:
        if len(self) != len(d):
            return False
        for i in range(len(self)):
            if self[i] != d[i]:
                return False
        return True

    def check_index(self, i: int) -> int:
        if i < 0:
            i += len(self)
            if i < 0:
                raise IndexError(f"{i} + len(self) {len(self)} < 0")
            return i
        if i >= len(self):
            raise IndexError(f"{i} >= len(self) {len(self)}")
        return i

    @overload
    def check_indexes(self, i: "Dim") -> "Dim": ...

    @overload
    def check_indexes(self, i: Sequence[int]) -> Sequence[int]: ...

    def check_indexes(
        self, i: Union[Sequence[int], "Dim"]
    ) -> Union[Sequence[int], "Dim"]:
        if isinstance(i, Dim):
            return i.as_indexes_on(self)
        return [self.check_index(j) for j in i]

    # reverse check_indexes
    def as_indexes_on(self, d: "Dim") -> "Dim":
        raise NotImplementedError

    def check_slice(self, i: slice) -> Tuple[int, int]:
        start, stop, step = i.indices(len(self))
        if step != 1:
            raise ValueError(f"step > 1 not supported ({step})")
        if start > stop:
            raise ValueError(f"start {start} > stop {stop}")
        return start, stop


#
# Seq is the ground (uncompressed) Dim representation, for use when
# no obvious regularity exists - it's a simple wrapper around an actual
# int list.
#
# However note that we store the offsets (cumulative sum) of the sequence,
# rather than the sequence itself. It turns out that lots of what we do
# a lot is more straightforward and natural using offsets than values.
# For simplicity/compactness we don't store values at all - nothing seems
# to require or benefit from the entire value sequence being materialized,
# and point values can be cheaply recovered from offsets.
#


@dataclass
class Seq(Dim):
    offs: List[int]

    def __init__(self, data: Sequence[int], is_offsets=False, base=0):
        if is_offsets and base != 0:
            msg = f"is_offsets True requires base to be 0, got {base}"
            raise ValueError(msg)
        self.offs = list(data) if is_offsets else offsets(data, base)

    def __repr__(self) -> str:
        # print actual offsets only if base != 0, otherwise extents
        if len(self.offs) > 0 and self.offs[0] != 0:
            return f"Seq({self.offs}, is_offsets=True)"
        return f"Seq({[w for w in self]})"

    def __str__(self) -> str:
        # print actual repr only if base != 0, otherwise list literal
        if len(self.offs) > 0 and self.offs[0] != 0:
            return repr(self)
        return f"{[w for w in self]}"

    def __len__(self) -> int:
        return len(self.offs) - 1

    def cycle(self) -> int:
        return len(self)

    @overload
    def __getitem__(self, i: int) -> int: ...

    @overload
    def __getitem__(self, i: slice) -> Dim: ...

    @overload
    def __getitem__(self, i: Sequence[int]) -> Dim: ...

    def __getitem__(self, i: Union[int, slice, Sequence[int]]) -> Union[int, Dim]:
        if isinstance(i, Dim):
            return i.select(self)
        if isinstance(i, int):
            i = self.check_index(i)
            return self.offs[i + 1] - self.offs[i]
        if isinstance(i, Sequence):
            i = self.check_indexes(i)
            return Seq([self[j] for j in i])
        start, stop = self.check_slice(i)
        if start == 0:
            if stop == len(self):
                return self
            return Seq(self.offs[: stop + 1], is_offsets=True)
        base = self.offs[start]
        offs = [self.offs[x] - base for x in range(start, stop + 1)]
        return Seq(offs, is_offsets=True)

    def as_indexes_on(self, d: Dim) -> Dim:
        if len(self) == 0:
            return Dim.EMPTY
        return Seq([d.check_index(i) for i in self])

    def offset_of_impl(self, i: int) -> int:
        return self.offs[i]

    def offsets(self, base: int = 0) -> Dim:
        if base == 0:
            return Seq(self.offs)  # note that is_offsets is False
        return Seq([base + off for off in self.offs])

    def fwddiff(self) -> Dim:
        return Seq(fwddiff(self))

    def index_of_impl(self, x: int) -> int:
        return bisect(self.offs, x) - 1

    def cat_impl(self, d: Dim) -> Dim:
        if isinstance(d, Seq):
            # TODO move this to constructor once Sparse.offs is fixed
            assert len(self) == 0 or self.offs[0] == 0, f"self base {self.offs[0]}"
            assert len(d) == 0 or d.offs[0] == 0, f"d base {d.offs[0]}"
            if len(self) > 1 and self == d:
                return Repeat(self, 2)
            base = self.offs[-1]
            offs = self.offs[:-1] + [x + base for x in d.offs]
            return Seq(offs, is_offsets=True)
        if isinstance(d, Rect) and d.n == 1:
            return Seq(self.offs + [self.offs[-1] + d.w], is_offsets=True)
        return Chain([self, d])

    def scale_int_impl(self, x: int) -> Dim:
        return Seq([w * x for w in self])

    def scale_dim_impl(self, x: Dim) -> Dim:
        return Seq([self[i] * x[i] for i in range(len(self))])

    def shift_int_impl(self, x: int, scale: int) -> Dim:
        return Seq([int(w + x * scale) for w in self])

    def shift_dim_impl(self, x: Dim, scale: int) -> Dim:
        return Seq([self[i] + x[i] * scale for i in range(len(self))])

    def floor_impl(self, x: int) -> Dim:
        return concat_dims(dim(max(self[i], x)) for i in range(len(self)))

    def ceil_impl(self, x: int) -> Dim:
        return concat_dims(dim(min(self[i], x)) for i in range(len(self)))

    def fold_impl(self, part: Dim) -> Dim:
        poffs = part.offsets()
        return concat_dims(
            Rect(self[poffs[i] : poffs[i + 1]].sum()) for i in range(len(part))
        )

    def sum(self) -> int:
        return self.offs[-1]

    def min_impl(self) -> int:
        return min(w for w in self)

    def max_impl(self) -> int:
        return max(w for w in self)


#
# Rect models sequences of a single value.
#


@dataclass
class Rect(Dim):
    w: int  # width
    n: int = 1  # sequence length

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        return f"Rect({self.w}, {self.n})"

    def __str__(self) -> str:
        return f"{self.w}" if self.n == 1 else repr(self)

    def cycle(self) -> int:
        return min(self.n, 1)

    @overload
    def __getitem__(self, i: int) -> int: ...

    @overload
    def __getitem__(self, i: slice) -> Dim: ...

    @overload
    def __getitem__(self, i: Sequence[int]): ...

    def __getitem__(self, i: Union[int, slice, Sequence[int]]) -> Union[int, Dim]:
        if isinstance(i, Dim):
            i = self.check_indexes(i)
            return Rect(self.w, len(i))
            # return i.select(self)
        if isinstance(i, int):
            i = self.check_index(i)
            return self.w
        if isinstance(i, Sequence):
            i = self.check_indexes(i)
            return Rect(self.w, len(i))
        start, stop = self.check_slice(i)
        return Rect(self.w, stop - start)

    def select_impl(self, d: Dim) -> Dim:
        i = d.check_index(self.w)
        return Rect(d[i], len(self))

    def as_indexes_on(self, d: Dim) -> Dim:
        if len(self) == 0:
            return Dim.EMPTY
        return Rect(d.check_index(self.w), len(self))

    def offset_of_impl(self, i: int) -> int:
        return self.w * i

    def offsets(self, base: int = 0) -> Dim:
        if self.n == 0:
            return Rect(base, 1)
        if self.w == 0:
            return Rect(base, self.n + 1)
        return Affine(base, self.n + 1, self.w)

    def fwddiff(self) -> Dim:
        return Rect(0, max(0, self.n - 1))

    def index_of_impl(self, x: int) -> int:
        n = self.sum()
        if x == n:
            return len(self)
        return x // self.w

    def cat_impl(self, d: Dim) -> Dim:
        if isinstance(d, Rect):
            if d.w == self.w:
                return Rect(self.w, self.n + d.n)
            if d.n == 1 and self.n == 1:
                return Seq([self.w, d.w])
                # return Affine(self.b, 2, d.b - self.b)
            return Runs(Seq([self.w, d.w]), Seq([self.n, d.n]))
        if isinstance(d, Affine):
            if d.w == self.w and d.n == 2:
                return Runs(d, Seq([self.n + 1, 1]))
            if self.n == 1 and d.w == self.w + d.s:
                return Affine(self.w, d.n + 1, d.s)
        if isinstance(d, Runs) and len(d) > 0 and d.vals[0] == self.w:
            return Runs(d.vals, dim(d.reps[0] + self.n).cat(d.reps[1:]))
        if isinstance(d, Seq):
            if self.n == 1:
                return Seq([self.w]).cat(d)
            if len(d) == 1 and d[0] == self.w:
                return Rect(self.w, self.n + 1)
        return Chain([self, d])

    def repeat_impl(self, n: int) -> Dim:
        return Rect(self.w, self.n * n)

    def spread_itemwise_impl(self, x: Dim) -> Dim:
        return Rect(self.w, x.sum())

    def spread_partitioned_impl(self, x: Dim, p: Dim) -> Dim:
        return Rect(self.w, p.scale(x).sum())

    def scale_int_impl(self, x: int) -> Dim:
        return Rect(self.w * x, self.n)

    def scale_dim_impl(self, x: Dim) -> Dim:
        return x.scale(self.w)

    def shift_int_impl(self, x: int, scale: int) -> Dim:
        return Rect(int(self.w + x * scale), self.n)

    def shift_dim_impl(self, x: Dim, scale: int) -> Dim:
        return x.scale(scale).shift(self)

    def iota(self) -> Dim:
        if self.n == 0:
            return Dim.EMPTY
        seq = Affine(0, self.w, 1) if self.w > 1 else Rect(0, self.w)
        return Repeat(seq, self.n) if self.n > 1 else seq

    def floor_impl(self, x: int) -> Dim:
        return Rect(max(self.w, x), self.n)

    def ceil_impl(self, x: int) -> Dim:
        return Rect(min(self.w, x), self.n)

    def fold_impl(self, part: Dim) -> Dim:
        if isinstance(part, Rect):
            return Rect(self.w * part.w, part.n)
        if isinstance(part, Affine):
            if self.w == 0:
                return Rect(0, part.n)
            return Affine(self.w * part.w, part.n, self.w * part.s)
        if isinstance(part, Repeat):
            seq = Rect(self.w, self.n // part.n).fold(part.seq)
            return Repeat(seq, part.n)
        if isinstance(part, Chain):
            return concat_dims(Rect(self.w, s.sum()).fold(s) for s in part.seqs)
        return concat_dims(Rect(p * self.w) for p in part)

    def sum(self) -> int:
        return self.w * self.n

    def min_impl(self) -> int:
        return self.w

    def max_impl(self) -> int:
        return self.w


#
# empty dim global constant.
# The use of Rect to model this is a choice - all subclasses can model an empty
# sequence, so this is basically a bet that an empty Rect will compose better
# than an empty instance of any of the others. But it might turn out that we
# get better representation compression by actually having Dim.empty() that
# returns a zero-length sequence in the representation of your choice.
#
Dim.EMPTY = Rect(0, 0)


@dataclass
class Affine(Dim):
    w: int  # initial width
    n: int  # sequence length
    s: int  # slope

    def __init__(self, w: int, n: int, s: int):
        if n <= 1:
            raise ValueError(f"length must be > 1, got {n} (use Rect instead)")
        if s == 0:
            raise ValueError(f"weight must be != 0 (use Rect instead)")
        self.w = w
        self.n = n
        self.s = s

    def __repr__(self) -> str:
        return f"Affine({self.w}, {self.n}, {self.s})"

    def __len__(self) -> int:
        return self.n

    def cycle(self) -> int:
        return self.n if self.s != 0 else min(self.n, 1)

    @overload
    def __getitem__(self, i: int) -> int: ...

    @overload
    def __getitem__(self, i: slice) -> Dim: ...

    @overload
    def __getitem__(self, i: Sequence[int]): ...

    def __getitem__(self, i: Union[int, slice, Sequence[int]]) -> Union[int, Dim]:
        if isinstance(i, Dim):
            return i.select(self)
        if isinstance(i, int):
            i = self.check_index(i)
            return i * self.s + self.w
        if isinstance(i, Sequence):
            i = self.check_indexes(i)
            return Seq([self[j] for j in i])
        start, stop = self.check_slice(i)
        w = self.w + self.s * start
        n = stop - start
        return Affine(w, n, self.s) if n > 1 else Rect(w, n)

    def select_impl(self, d: Dim) -> Dim:
        if self.s == 1:
            s = self.as_indexes_on(d)
            return d[s.min() : s.max() + 1]
        if isinstance(d, Affine) and self.min() >= 0 and self.max() < len(d):
            return Affine(d.w + self.w * d.s, len(self), d.s * self.s)
        return super().select_impl(d)

    def as_indexes_on(self, d: Dim) -> Dim:
        if len(self) == 0:
            return Dim.EMPTY
        if self.min() >= 0 and self.max() < len(d):
            return self
        return Seq([d.check_index(i) for i in self])

    def offset_of_impl(self, i: int) -> int:
        return (self.w * i) + (self.s * i * (i - 1) // 2)

    def offsets(self, base: int = 0) -> Dim:
        return Seq(offsets(self, base))

    def fwddiff(self) -> Dim:
        return Rect(self.s, max(0, self.n - 1))

    def index_of_impl(self, x: int) -> int:
        n = self.sum()
        if x == n:
            return len(self)
        dx = (
            self.s
            - 2 * self.w
            + math.sqrt(
                4 * self.w**2 - 4 * self.w * self.s + 8 * x * self.s + self.s**2
            )
        ) / (2 * self.s)
        return int(dx)

    def cat_impl(self, d: Dim) -> Dim:
        if isinstance(d, Affine):
            if d == self:
                return Repeat(self, 2)
            if d.s == self.s and d.w == self.w + self.s * self.n:
                return Affine(self.w, self.n + d.n, self.s)
            if d.s == 0 and self.s == 0:
                return Runs(Seq([self.w, d.w]), Seq([self.n, d.n]))
        if isinstance(d, Rect) and d.n == 1 and d.w == self.w + self.s * self.n:
            return Affine(self.w, self.n + 1, self.s)
        if isinstance(d, Seq) and self.n == 1:
            return Seq([self.w]) + d
        return Chain([self, d])

    def scale_int_impl(self, x: int) -> Dim:
        if x == 0:
            return Rect(0, self.n)
        return Affine(int(self.w * x), self.n, int(self.s * x))

    def scale_dim_impl(self, x: Union[int, float, Dim]) -> Dim:
        return Seq([self[i] * x[i] for i in range(len(self))])

    def shift_int_impl(self, x: int, scale: int) -> Dim:
        return Affine(int(self.w + x * scale), self.n, self.s)

    def shift_dim_impl(self, x: Dim, scale: int) -> Dim:
        if isinstance(x, Affine):
            w = self.s + x.s * scale
            if w == 0:
                return Rect(self.w + x.w * scale, self.n)
            return Affine(self.w + x.w * scale, self.n, w)
        return x.scale(scale).shift(self)

    def floor_impl(self, x: int) -> Dim:
        if len(self) == 0:
            return Rect(max(self.w, x), 0)
        if self.min() >= x:
            return self
        if self.max() <= x:
            return Rect(x, len(self))
        if self.s > 0:
            n = 1 + (x - self.w) // self.s
            return Chain([Rect(x, n), self[n:]])
        else:
            n = self.n - 1 - (x - self.w) // self.s
            return Chain([self[:-n], Rect(x, n)])

    def ceil_impl(self, x: int) -> Dim:
        if len(self) == 0:
            return Rect(min(self.w, x), 0)
        if self.max() <= x:
            return self
        if self.min() >= x:
            return Rect(x, len(self))
        if self.s > 0:
            n = self.n - 1 - ((x - self.w) // self.s)
            return Chain([self[:-n], Rect(x, n)])
        else:
            n = 1 + (x - self.w) // self.s
            return Chain([Rect(x, n), self[n:]])

    def fold_impl(self, part: Dim) -> Dim:
        offs = [self.offset_of(part.offset_of(i)) for i in range(len(part) + 1)]
        return Seq(offs, is_offsets=True)

    def sum(self) -> int:
        return self.offset_of(self.n)

    def min_impl(self) -> int:
        return self.w + min(0, self.s * (self.n - 1))

    def max_impl(self) -> int:
        return self.w + max(0, self.s * (self.n - 1))


#
# Repeat represents sequences made up of a repeated subsequence
#


@dataclass
class Repeat(Dim):
    seq: Dim
    n: int

    def __init__(self, seq, n):
        if n <= 1:
            raise ValueError(f"number of repetitions must be > 1, got {n}")
        seq = dim(seq)
        if isinstance(seq, Repeat):
            n *= seq.n
            seq = seq.seq
        self.seq = seq
        self.n = n

    def __repr__(self) -> str:
        return f"Repeat({repr(self.seq)}, {self.n})"

    def __str__(self) -> str:
        return f"Repeat({self.seq}, {self.n})"

    def __len__(self) -> int:
        return len(self.seq) * self.n

    def cycle(self) -> int:
        return self.seq.cycle()

    def orbit_impl(self) -> Dim:
        return self.seq.orbit()

    @overload
    def __getitem__(self, i: int) -> int: ...

    @overload
    def __getitem__(self, i: slice) -> Dim: ...

    @overload
    def __getitem__(self, i: Sequence[int]) -> Dim: ...

    def __getitem__(self, i: Union[int, slice, Sequence[int]]) -> Union[int, Dim]:
        if isinstance(i, Dim):
            return i.select(self)
        unit = len(self.seq)
        if isinstance(i, int):
            i = self.check_index(i)
            x = i % unit
            return self.seq[x]
        if isinstance(i, Sequence):
            i = self.check_indexes(i)
            return Seq([self[j] for j in i])
        start, stop = self.check_slice(i)
        if unit == 0:
            return Dim.EMPTY  # avoid degenerates
        # TODO clean this shit up
        x = start // unit
        y = stop // unit
        adv = start % unit
        ret = stop % unit
        if adv == 0 and ret == 0:
            n = y - x
            if n == 0:
                return Dim.EMPTY
            if n == 1:
                return self.seq
            return Repeat(self.seq, n)
        if x == y:
            return self.seq[adv:ret]
        if adv == 0:
            tail = self.seq[:ret]
            nhead = y - x
            if nhead == 0:
                return tail
            if nhead == 1:
                return self.seq.cat(tail)
            return self.seq.repeat(nhead).cat(tail)
        head = self.seq[adv:]
        tail = self.seq[:ret]
        nmid = y - x - 1
        if nmid == 0:
            return head.cat(tail)
        return head.cat(self.seq.repeat(nmid)).cat(tail)

    def select_impl(self, d: Dim) -> Dim:
        return Repeat(self.seq.select(d), self.n)

    def as_indexes_on(self, d: Dim) -> Dim:
        if len(self) == 0:
            return Dim.EMPTY
        return Repeat(self.seq.as_indexes_on(d), self.n)

    def offset_of_impl(self, i: int) -> int:
        cycle = len(self.seq)
        if cycle == 0:
            return 0
        base = self.seq.sum() * (i // cycle)
        off = self.seq.offset_of(i % cycle)
        return base + off

    def offsets(self, base: int = 0) -> Dim:
        initial = Dim.EMPTY.offsets(base)
        if self.n == 0 or len(self.seq) == 0:
            return initial
        offs = self.seq.offsets(base)
        if self.n == 1:
            return offs
        if self.seq.sum() == 0:
            # return Chain([initial, Repeat(self.seq.offsets(base)[1:], self.n)])
            return Chain([Repeat(self.seq.offsets(base)[:-1], self.n), initial])

        for i in range(self.n - 1):
            seq_offs = self.seq.offsets(offs[-1])
            offs = offs.cat(seq_offs[1:])
        return offs

    def fwddiff(self) -> Dim:
        if len(self.seq) == 0 or self.n == 0:
            return self
        diff = self.seq.fwddiff()
        head = (diff.cat(dim(self.seq[0] - self.seq[-1]))).repeat(self.n - 1)
        return head.cat(diff)

    def index_of_impl(self, x: int) -> int:
        n = self.sum()
        if x == n:
            return len(self)
        n = self.seq.sum()
        base = len(self.seq) * (x // n)
        off = self.seq.index_of(x % n)
        return base + off

    def cat_impl(self, d: Dim) -> Dim:
        if self.seq == d:
            return Repeat(self.seq, self.n + 1)
        if isinstance(d, Repeat) and self.seq == d.seq:
            return Repeat(self.seq, self.n + d.n)
        return Chain([self, d])

    def scale_int_impl(self, x: int) -> Dim:
        return Repeat(self.seq.scale(x), self.n)

    def scale_dim_impl(self, x: Dim) -> Dim:
        c = x.cycle()
        if lcm(c, self.cycle()) <= len(self.seq):
            return Repeat(self.seq.scale(x[: len(self.seq)]), self.n)
        return concat_dims(
            self.seq.scale(s) for s in split(x, [len(self.seq)] * self.n)
        )

    def shift_int_impl(self, x: int, scale: int) -> Dim:
        return Repeat(self.seq.shift(x, scale), self.n)

    def shift_dim_impl(self, x: Dim, scale: int) -> Dim:
        if lcm(x.cycle(), self.cycle()) <= len(self.seq):
            return Repeat(self.seq.shift(x[: len(self.seq)], scale), self.n)
        return concat_dims(
            self.seq.shift(s, scale) for s in split(x, [len(self.seq)] * self.n)
        )

    def iota(self) -> Dim:
        return Repeat(self.seq.iota(), self.n)

    def floor_impl(self, x: int) -> Dim:
        return Repeat(self.seq.floor(x), self.n)

    def ceil_impl(self, x: int) -> Dim:
        return Repeat(self.seq.ceil(x), self.n)

    def repeat_impl(self, n: int) -> Dim:
        return Repeat(self.seq, self.n * n)

    def spread_itemwise_impl(self, x: Dim) -> Dim:
        return self.spread_partitioned_impl(x, Rect(1, len(self)))

    def spread_partitioned_impl(self, x: Dim, p: Dim) -> Dim:
        sc, xc, pc = self.cycle(), x.cycle(), p.cycle()
        if sc % p.offset_of(pc) == 0 and sc % p.offset_of(xc) == 0:
            x = x[: len(x) // self.n]
            p = p[: len(p) // self.n]
            return Repeat(self.seq.spread(x, p), self.n)
        return super().spread_partitioned_impl(x, p)

    def fold_impl(self, part: Dim) -> Dim:
        poff = part.offset_of(part.cycle())
        if poff > 0 and self.cycle() % poff == 0:
            part = part[: len(part) // self.n]
            segs = self.seq.fold(part)
            if len(segs) == 1:
                return Rect(segs[0], self.n)
            return Repeat(segs, self.n)
        poffs = part.offsets()
        return concat_dims(
            Rect(self[poffs[i] : poffs[i + 1]].sum()) for i in range(len(part))
        )

    def sum(self) -> int:
        return self.offset_of(len(self))

    def min_impl(self) -> int:
        return self.seq.min()

    def max_impl(self) -> int:
        return self.seq.max()


#
# Sparse represents sequences made up mostly of zeros.
# note: nonzero values and their distribution are stored in Dims,
# so patterns of regularity in them are capturable.
#
# note: currently misuses Seq for self.offs, see note in concat_impl
#
@dataclass
class Sparse(Dim):
    vals: Dim
    offs: Dim

    def __init__(self, vals, offs=None):
        if Sparse.is_descriptor(vals):
            vals, offs = Sparse.convert_descriptor(vals)
        else:
            if offs is None:
                raise ValueError(f"missing offsets dim")
            vals = dim(vals)
            offs = dim(offs)
            if len(vals) != len(offs):
                raise ValueError(f"len(vals) {len(vals)} != len(offs) {len(offs)}")
        self.vals = vals
        self.offs = offs

    def __repr__(self) -> str:
        return f"Sparse({repr(self.vals)}, {repr(self.offs)})"

    def __str__(self) -> str:
        return f"Sparse({self.vals}, {self.offs})"

    def __len__(self) -> int:
        return self.offs.sum()

    def cycle(self) -> int:
        return len(self)

    @overload
    def __getitem__(self, i: int) -> int: ...

    @overload
    def __getitem__(self, i: slice) -> Dim: ...

    @overload
    def __getitem__(self, i: Sequence[int]) -> Dim: ...

    def __getitem__(self, i: Union[int, slice, Sequence[int]]) -> Union[int, Dim]:
        if isinstance(i, Dim):
            return i.select(self)
        if isinstance(i, int):
            i = self.check_index(i)
            x = self.offs.index_of(i)
            return self.vals[x] if x >= 0 and self.offs.offset_of(x) == i else 0
        if isinstance(i, Sequence):
            # TODO probably a cheaper way to do this
            m = {}
            for j, x in enumerate(i):
                w = self[x]
                if w != 0:
                    m[j] = w
            m[len(i)] = -1
            return Sparse(m)
        start, stop = self.check_slice(i)
        if stop == start:
            return Dim.EMPTY  # avoid degenerates
        x = self.offs.index_of(start - 1) + 1
        y = self.offs.index_of(stop - 1) + 1
        offs = [self.offs.offset_of(i) - start for i in range(x, y)] + [stop - start]
        return Sparse(self.vals[x:y], Seq(offs, is_offsets=True))

    def select_impl(self, d: Dim) -> Dim:
        if d[0] == 0:
            return Sparse(self.vals.select(d), self.offs)
        return super().select_impl(d)

    def as_indexes_on(self, d: Dim) -> Dim:
        if len(self) == 0:
            return Dim.EMPTY
        if d.check_index(0) == 0:
            return Sparse(self.vals.as_indexes_on(d), self.offs)
        return Seq([d.check_index(w) for w in self])

    def offset_of_impl(self, i: int) -> int:
        x = self.offs.index_of(i - 1)
        return 0 if x < 0 else self.vals.offset_of(x + 1)

    def offsets(self, base: int = 0) -> Dim:
        # TODO closed form rle
        return Seq(offsets(self, base))

    def fwddiff(self) -> Dim:
        # TODO closed form
        return sparsify(fwddiff(self))

    def index_of_impl(self, x: int) -> int:
        return self.offs.offset_of(self.vals.index_of(x))

    def cat_impl(self, d: Dim) -> Dim:
        if isinstance(d, Sparse):
            vals = self.vals.cat(d.vals)
            # note: these shims are needed because Sparse.offs breaks the
            # invariant that offsets begin with 0, breaking Seq.concat
            # TODO rework Sparse.offs to respect the invariant and guard on it
            # in Seq ctor, not just Seq.concat_impl
            if len(self.vals) == 0:  # lhs is all zeros
                do = [d.offs.offset_of(i) + len(self) for i in range(len(d.offs) + 1)]
                offs = Seq(do, is_offsets=True)
            elif len(d.vals) == 0:  # rhs is all zeros
                so = [self.offs.offset_of(i) for i in range(len(self.offs))]
                offs = Seq(so + [len(self) + len(d)], is_offsets=True)
            else:
                so = [self.offs.offset_of(i) for i in range(len(self.offs))]
                do = [d.offs.offset_of(i) + len(self) for i in range(len(d.offs) + 1)]
                offs = Seq(so + do, is_offsets=True)
            return Sparse(vals, offs)
        return Chain([self, d])

    def scale_int_impl(self, x: int) -> Dim:
        return Sparse(self.vals.scale(x), self.offs)

    def scale_dim_impl(self, x: Dim) -> Dim:
        x = Seq([x[self.offs.offset_of(i)] for i in range(len(self.vals))])
        return Sparse(self.vals.scale(x), self.offs)

    def shift_int_impl(self, x: int, scale: int) -> Dim:
        return Seq([self[i] + x * scale for i in range(len(self))])

    def shift_dim_impl(self, x: Union[int, Dim], scale: int) -> Dim:
        return Seq([self[i] + x[i] * scale for i in range(len(self))])

    def floor_impl(self, x: int) -> Dim:
        if x == 0:
            return Sparse(self.vals.floor(x), self.offs)
        return Seq([n for n in self]).floor(x)

    def ceil_impl(self, x: int) -> Dim:
        if x == 0:
            return Sparse(self.vals.ceil(x), self.offs)
        return Seq([n for n in self]).ceil(x)

    def fold_impl(self, part: Dim) -> Dim:
        poffs = part.offsets()
        return concat_dims(
            Rect(self[poffs[i] : poffs[i + 1]].sum()) for i in range(len(part))
        )

    def sum(self) -> int:
        return self.vals.sum()

    def min_impl(self) -> int:
        return 0 if len(self) != len(self.vals) else self.vals.min()

    def max_impl(self) -> int:
        return 0 if len(self.vals) == 0 else self.vals.max()

    @staticmethod
    def is_descriptor(x) -> bool:
        return (
            isinstance(x, dict)
            and all(isinstance(k, int) for k in x.keys())
            and all(isinstance(v, int) for v in x.values())
        )

    @staticmethod
    def convert_descriptor(x) -> Tuple[Dim, Dim]:
        if not Sparse.is_descriptor(x):
            raise ValueError(f"not a sparse descriptor: {x}")
        offs = sorted(x)
        nzvals = [x[o] for o in offs]
        if nzvals[-1] == -1:
            nzvals = nzvals[:-1]
        else:
            offs = offs + [offs[-1] + 1]
        return Seq(nzvals), Seq(offs, is_offsets=True)


#
# Runs are vectorized Rects, which lets us abstract over values and/or lengths.
# TODO how useful are vectorized Affines?
#
@dataclass
class Runs(Dim):
    vals: Dim
    reps: Dim

    def __init__(self, vals, reps):
        vals = dim(vals)
        reps = dim(reps).length_extend(len(vals))
        if len(vals) != len(reps):
            raise ValueError(f"len(vals) {len(vals)} != len(reps) {len(reps)}")
        if len(vals) <= 1:
            msg = f"run sequence must be > 1 value, got {len(vals)} (use Rect or Dim.EMPTY instead)"
            raise ValueError(msg)
        if is_rect(vals):
            msg = f"vals dim is constant, use Rect({vals[0]}, {reps.sum()}) instead"
            raise ValueError(msg)
        if reps.min() < 0:
            raise ValueError(f"run lengths cannot be negative, got {reps.min()}")
        if is_singleton(reps):
            raise ValueError(f"run lengths are 1, use {vals} instead")
        self.vals = vals
        self.reps = reps

    def __repr__(self) -> str:
        return f"Runs({repr(self.vals)}, {repr(self.reps)})"

    def __str__(self) -> str:
        # length of reps dim is fixed so print just the width for Rect dims
        reps = self.reps.w if isinstance(self.reps, Rect) else self.reps
        return f"Runs({self.vals}, {reps})"

    def __len__(self) -> int:
        return self.reps.sum()

    def cycle(self) -> int:
        return len(self)

    @overload
    def __getitem__(self, i: int) -> int: ...

    @overload
    def __getitem__(self, i: slice) -> Dim: ...

    @overload
    def __getitem__(self, i: Sequence[int]) -> Dim: ...

    def __getitem__(self, i: Union[int, slice, Sequence[int]]) -> Union[int, Dim]:
        if isinstance(i, Dim):
            return i.select(self)
        if isinstance(i, int):
            i = self.check_index(i)
            return self.vals[self.reps.index_of(i)]
        if isinstance(i, Sequence):
            return Seq([self[j] for j in i])

        start, stop = self.check_slice(i)
        if stop == start:
            return Dim.EMPTY  # avoid degenerates

        x = self.reps.index_of(start)
        y = self.reps.index_of(stop - 1) + 1
        n = y - x

        if n == 1:
            return Rect(self.vals[x], stop - start)

        vals = seq_to_dim(self.vals[x:y])
        head = Rect(self.reps.offset_of(x + 1) - start)
        mid = self.reps[x + 1 : y - 1]
        tail = Rect(stop - self.reps.offset_of(y - 1))
        reps = head.cat(mid).cat(tail)

        if is_rect(vals):
            return Rect(vals.w, reps.sum())
        elif is_rect(reps, 1) or is_singleton(reps):
            return vals
        else:
            return Runs(vals, reps)

    def select_impl(self, d: Dim) -> Dim:
        vals = self.vals.select(d)
        if isinstance(vals, Rect):
            return Rect(vals.w, self.reps.sum())
        return Runs(vals, self.reps)

    def as_indexes_on(self, d: Dim) -> Dim:
        return Runs(self.vals.as_indexes_on(d), self.reps)

    def offset_of_impl(self, i: int) -> int:
        seg = self.reps.index_of(i)
        coal = self.vals.scale(self.reps)
        base = coal.offset_of(seg)
        tail = i - self.reps.offset_of(seg)
        off = 0 if tail == 0 else self.vals[seg] * tail
        return base + off

    def offsets(self, base: int = 0) -> Dim:
        if len(self) == 0:
            return Dim.EMPTY.offsets(base)
        if len(self.vals) == 1:
            assert False, f"internal: singleton Runs {self}"
            return Rect(self.vals[0], self.reps[0]).offsets(base)
        segs = [Rect(w, r) for w, r in zip(self.vals, self.reps) if r > 0]
        expanded = Chain(segs) if len(segs) > 1 else segs[0]
        return expanded.offsets(base)

    def fwddiff(self) -> Dim:
        if len(self) == 0:
            return self
        if len(self.vals) == 1:
            return Rect(self.vals[0], self.reps[0]).fwddiff()
        segs = [Rect(w, r) for w, r in zip(self.vals, self.reps)]
        expanded = Chain(segs) if len(segs) > 1 else segs[0]
        return expanded.fwddiff()

    def index_of_impl(self, x: int) -> int:
        coal = self.vals.scale(self.reps)
        i = coal.index_of(x)
        base = self.reps.offset_of(i)
        tail = x - coal.offset_of(i)
        off = 0 if tail == 0 else tail // self.vals[i]
        return base + off

    def cat_impl(self, d: Dim) -> Dim:
        if isinstance(d, Rect):
            if d.w == self.vals[-1]:
                return Runs(self.vals, self.reps[:-1].cat(Rect(self.reps[-1] + d.n)))
            return Runs(self.vals.cat(Rect(d.w)), self.reps.cat(Rect(d.n)))
        if isinstance(d, Runs):
            if self.vals == d.vals and self.reps == d.reps:
                return Repeat(self, 2)
            return Runs(self.vals.cat(d.vals), self.reps.cat(d.reps))
        return Chain([self, d])

    def spread_itemwise_impl(self, x: Dim) -> Dim:
        return Runs(self.vals, x.fold(self.reps))

    def scale_int_impl(self, x: int) -> Dim:
        return Runs(self.vals.scale(x), self.reps)

    def scale_dim_impl(self, x: Dim) -> Dim:
        if isinstance(x, Runs) and x.reps == self.reps:
            vals = self.vals.scale(x.vals)
            if isinstance(vals, Rect):
                return Rect(vals.w, self.reps.sum())
            return Runs(vals, self.reps)
        return concat_dims(
            Rect(v, r).scale(s)
            for v, r, s in zip(self.vals, self.reps, split(x, self.reps))
        )

    def shift_int_impl(self, x: int, scale: int) -> Dim:
        return Runs(self.vals.shift(x, scale), self.reps)

    def shift_dim_impl(self, x: Dim, scale: int) -> Dim:
        if isinstance(x, Runs) and x.reps == self.reps:
            vals = self.vals.shift(x.vals, scale)
            if isinstance(vals, Rect):
                return Rect(vals.w, self.reps.sum())
            return Runs(vals, self.reps)
        return concat_dims(
            Rect(v, r).shift(s, scale)
            for v, r, s in zip(self.vals, self.reps, split(x, self.reps))
        )

    def iota(self) -> Dim:
        return concat_dims(Rect(v, n).iota() for v, n in zip(self.vals, self.reps))

    def floor_impl(self, x: int) -> Dim:
        vals = self.vals.floor(x)
        if isinstance(vals, Rect):
            return Rect(vals.w, self.reps.sum())
        return Runs(vals, self.reps)

    def ceil_impl(self, x: int) -> Dim:
        vals = self.vals.ceil(x)
        if isinstance(vals, Rect):
            return Rect(vals.w, self.reps.sum())
        return Runs(vals, self.reps)

    def fold_impl(self, part: Dim) -> Dim:
        poffs = part.offsets()
        return concat_dims(
            Rect(self[poffs[i] : poffs[i + 1]].sum()) for i in range(len(part))
        )

    def sum(self) -> int:
        return sum(w * r for w, r in zip(self.vals, self.reps))

    def min_impl(self) -> int:
        return min(self.vals)

    def max_impl(self) -> int:
        return max(self.vals)


#
# Chain represents sequences of concatenated Dims.
#
@dataclass
class Chain(Dim):
    seqs: List[Dim]

    def __init__(self, seqs, lens=None, sums=None):
        def flatten_chains(acc, s):
            tail = s.seqs if isinstance(s, Chain) else [dim(s)]
            return acc + tail

        seqs = reduce(flatten_chains, seqs, [])
        if any(len(s) == 0 for s in seqs):
            raise ValueError(f"Chain sequences cannot be empty")
        if len(seqs) <= 1:
            raise ValueError(f"must specify chain of > 1 sequences, got {len(seqs)}")
        if all(isinstance(s, Seq) for s in seqs):
            raise ValueError(f"all seqs {seqs=}, use concat_dims() instead")
        self.seqs = seqs
        # TODO revisit
        self._lens: Dim = Seq(len(s) for s in self.seqs) if lens is None else lens
        self._sums: Dim = Seq(s.sum() for s in self.seqs) if sums is None else sums

    def __repr__(self) -> str:
        return f"Chain([{', '.join(repr(s) for s in self.seqs)}])"

    def __str__(self) -> str:
        return f"Chain([{', '.join(str(s) for s in self.seqs)}])"

    def __len__(self) -> int:
        return self._lens.sum()

    def cycle(self) -> int:
        return len(self)

    @overload
    def __getitem__(self, i: int) -> int: ...

    @overload
    def __getitem__(self, i: slice) -> Dim: ...

    @overload
    def __getitem__(self, i: Sequence[int]) -> Dim: ...

    def __getitem__(self, i: Union[int, slice, Sequence[int]]) -> Union[int, Dim]:
        if isinstance(i, Dim):
            return i.select(self)
        if isinstance(i, int):
            return self._get_int_item(i)
        if isinstance(i, Sequence):
            # return Seq([self[j] for j in i])
            return concat_dims(dim(self[j]) for j in i)
        return self._get_slice(i)

    def _get_int_item(self, i) -> int:
        i = self.check_index(i)
        base = self._lens.index_of(i)
        i -= self._lens.offset_of(base)
        return self.seqs[base][i]

    def _get_slice(self, i: slice):
        start, stop = self.check_slice(i)
        if stop == start:
            return Dim.EMPTY
        x = self._lens.index_of(start)
        y = self._lens.index_of(stop - 1)
        adv = start - self._lens.offset_of(x)
        ret = stop - self._lens.offset_of(y)
        if x == y:
            return self.seqs[x][adv:ret]
        head = self.seqs[x][adv:]
        tail = self.seqs[y][:ret]
        seqs = [head, *self.seqs[x + 1 : y], tail]
        return concat_dims(seqs)

    def select_impl(self, d: Dim) -> Dim:
        return concat_dims([s.select(d) for s in self.seqs])

    def as_indexes_on(self, d: Dim) -> Dim:
        return concat_dims([s.as_indexes_on(d) for s in self.seqs])

    def offset_of_impl(self, i: int) -> int:
        n = len(self)
        if i == n:
            return self.sum()
        base = self._lens.index_of(i)
        off = i - self._lens.offset_of(base)
        return self._sums.offset_of(base) + self.seqs[base].offset_of(off)

    def offsets(self, base: int = 0) -> Dim:
        if len(self.seqs) == 0:
            return Dim.EMPTY.offsets(base)
        seq_offs = []
        for seq in self.seqs[:-1]:
            seq_off = seq.offsets(base)
            seq_offs.append(seq_off[:-1])
            base = seq_off[-1]
        seq_offs.append(self.seqs[-1].offsets(base))
        offs = concat_dims(seq_offs)
        # offs = self.seqs[0].offsets(base)
        # for seq in self.seqs[1:]:
        #     seq_offs = seq.offsets(offs[-1])
        #     offs += seq_offs[1:]
        return offs

    def fwddiff(self) -> Dim:
        if len(self.seqs) == 0 or len(self) == 0:
            return self

        def seg_diff(i):
            seq = self.seqs[i]
            diff = seq.fwddiff()
            if i < len(seqs) - 1:
                diff = diff.cat(dim(seqs[i + 1][0] - seq[-1]))
            return diff

        seqs = [seq for seq in self.seqs if len(seq) > 0]
        diffs = [seg_diff(i) for i in range(len(seqs))]
        return concat_dims(diffs)

    def index_of_impl(self, x: int) -> int:
        n = self.sum()
        if x == n:
            return len(self)
        i = self._sums.index_of(x)
        base = self._lens.offset_of(i)
        off = self.seqs[i].index_of(x - self._sums.offset_of(i))
        return base + off

    def cat_impl(self, d: Dim) -> Dim:
        if isinstance(d, Chain):
            return Chain(
                self.seqs + d.seqs, self._lens.cat(d._lens), self._sums.cat(d._sums)
            )
            # return Chain(self.seqs + d.seqs)
        tl = self.seqs[-1].cat(d)
        return Chain(self.seqs[:-1] + (tl.seqs if isinstance(tl, Chain) else [tl]))

    def spread_itemwise_impl(self, x: Dim) -> Dim:
        xslice = lambda i: x[self._lens.offset_of(i) : self._lens.offset_of(i + 1)]
        seqs = [self.seqs[i].spread(xslice(i)) for i in range(len(self.seqs))]
        return concat_dims(seqs)

    def scale_int_impl(self, x: int) -> Dim:
        return concat_dims([s.scale(x) for s in self.seqs])

    def scale_dim_impl(self, x: Dim) -> Dim:
        xslice = lambda i: x[self._lens.offset_of(i) : self._lens.offset_of(i + 1)]
        seqs = [self.seqs[i].scale(xslice(i)) for i in range(len(self.seqs))]
        return concat_dims(seqs)

    def shift_int_impl(self, x: int, scale: int) -> Dim:
        return concat_dims([s.shift(x, scale) for s in self.seqs])

    def shift_dim_impl(self, x: Dim, scale: int) -> Dim:
        xslice = lambda i: x[self._lens.offset_of(i) : self._lens.offset_of(i + 1)]
        seqs = [self.seqs[i].shift(xslice(i), scale) for i in range(len(self.seqs))]
        return concat_dims(seqs)

    def iota(self) -> Dim:
        return concat_dims(seq.iota() for seq in self.seqs)

    def floor_impl(self, x: int) -> Dim:
        return concat_dims(seq.floor(x) for seq in self.seqs)

    def ceil_impl(self, x: int) -> Dim:
        return concat_dims(seq.ceil(x) for seq in self.seqs)

    def fold_impl(self, part: Dim) -> Dim:
        poffs = part.offsets()
        return concat_dims(
            Rect(self[poffs[i] : poffs[i + 1]].sum()) for i in range(len(part))
        )

    def sum(self) -> int:
        return self._sums.sum()

    def min_impl(self) -> int:
        return min([s.min() for s in self.seqs])

    def max_impl(self) -> int:
        return max([s.max() for s in self.seqs])


#
# builders
#


# number of items in a python slice
# https://stackoverflow.com/questions/36188429/retrieve-length-of-slice-from-slice-object-in-python
def slice_len(start: int, stop: int, step: int) -> int:
    return max(0, (stop - start + (step - int(math.copysign(1, step)))) // step)


#
# promotes RawDim descriptors to Dims, lets Dims pass through
# see RawDim typedef at top of file
#
def dim(x: RawDim) -> Dim:
    # Dim -> Dim
    if isinstance(x, Dim):
        return x

    # int -> Rect with extent 1 (length is length extended in shape construction)
    if isinstance(x, int):
        return Rect(x)

    # slice and range -> Rect or Affine
    if isinstance(x, slice) or isinstance(x, range):
        n = slice_len(x.start, x.stop, x.step) if isinstance(x, slice) else len(x)
        if n == 0:
            return Dim.EMPTY
        if n == 1:
            return Rect(x.start)
        return Affine(x.start, n, x.step)

    # List[int] -> Seq
    if isinstance(x, list) and all(isinstance(i, int) for i in x):
        return Seq(cast(List[int], x))

    raise ValueError(f"can't promote to dim: {repr(x)}")


def try_dim(x: RawDim) -> Optional[Dim]:
    try:
        return dim(x)
    except ValueError as ve:
        if str(ve).startswith("can't promote to dim"):
            return None
        raise ve


# NOTE: no fancy pattern matching, just a uniqueness check. still
# TODO remove? after move to fold
def simple_dim(vals: List[int]) -> Dim:
    if len(vals) == 0:
        return Dim.EMPTY
    if len(set(vals)) == 1:
        return Rect(vals[0], len(vals))
    return Seq(vals)


#
# Dim classifiers
# these are called quite a bit, but is_singleton hides an expensive
# test. should be hoisted into a Dim method and virtualized, absent
# a much more stringent policy of ensuring that dims always use the
# maximally compressed representation for their values (so e.g. a
# Seq instance full of 1s would be malformed)
#


def is_rect(d: Dim, val: int = None) -> bool:
    if isinstance(d, Rect):
        return val is None or d.w == val
    return False


def is_singleton(d):
    if is_rect(d, 1):
        return True
    # TODO virtualize
    return all(w == 1 for w in d)


#
# it's kind of an experiment grab bag from here to bottom, should be cleaned up/culled
#


# convenience - create sparse dim from sequence.
# throws away any information from the incoming dim.
def sparsify(d: Union[Sequence[int], Dim]) -> Dim:
    pairs = [(i, x) for i, x in enumerate(d) if x != 0]
    offs, vals = zip(*pairs) if len(pairs) > 0 else ([], [])
    return Sparse(Seq(vals), Seq(list(offs) + [len(d)], is_offsets=True))


#
# create sparse dim from sequence containing only ones and zeros
# note: this is kind of a red herring. it's a way of encoding sparsity purely
# as a matter of shape (i.e., not requiring any of the Overlay machinery) by
# adding an inner "ragged" dimension whose widths are all 1 or 0. interesting
# to play around with but doesn't meet a reasonable definition of sparse format
# since it requires a shape change. Also not clear how efficient the kernels
# written against this format could be.
#
def sparsify_ones(d: Union[Sequence[int], Dim]) -> Dim:
    def check(x):
        if x > 1 or x < 0:
            raise ValueError(f"expected 0 or 1, got {x}")
        return x

    offs = [i for i, x in enumerate(d) if check(x) != 0]
    return Sparse(Rect(1, len(offs)), Seq(list(offs) + [len(d)], is_offsets=True))


def rle(seq, step=0):
    # TODO rewrite w/o torch
    import torch

    v = torch.Tensor(seq).long()
    ne = torch.full((len(v) + 1,), True)
    ne[1:-1] = v.diff() != step
    vals = v[ne[:-1]]
    lens = torch.arange(len(v) + 1)[ne].diff()
    return Runs(dim(vals.tolist()), dim(lens.tolist()))


def rect(h, w):
    return Rect(w, h)


def tril(n):
    return Affine(1, n, 1) if n > 1 else Rect(1, n)


def triu(n):
    return Affine(n, n, -1) if n > 1 else Rect(n, n)


def ragged(n, low=None, high=None):
    if low is None:
        low = 0
    if high is None:
        high = max(low, n)
    return Seq([random.randint(low, high) for _ in range(n)])


def block(seq, n):
    return seq.spread(Rect(n, len(seq)))


def diag(n, off=0, bsiz=1):
    if bsiz == 1:
        return Affine(off, n, 1)
    n = n // bsiz
    return Runs(Affine(off * bsiz, n, bsiz), Rect(bsiz, n))
