# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import contextlib
import cProfile
import random
from itertools import chain
from pprint import pprint
from pstats import SortKey
from typing import Tuple, Type
from unittest import TestCase, main

from fold import *
from util import get_size


def flatten(seqs):
    return [x for seq in seqs for x in seq]


# note: unittest is weirdly nondeterministic, even with this seed -
# e.g. deleting a comment will change pattern of randomized
# dim construction.
random.seed(0)


#
# Dim class tests
#
# Since Dim classes simulate simple sequences of ints, these tests
# exercise them by comparing the results of Dim API calls to results
# of analogous operations on ordinary int lists. For binary ops we
# randomly draw different Dim subclasses and mix them together.
#
# Approach here is to be reasonably exhaustive, sweeping lots of
# combos of randomly generated sequences. Since Dims nest this
# can take a while.
#
# TODO
#
# 1. sweeps are a little ad hoc - characteristics of some Dim classes
# are harder to control than others which has shaped coverage. Whole
# thing could probably be improved.
#

ctor_count = {}


def record_ctor(ctor):
    ctor_count[ctor] = ctor_count.get(ctor, 0) + 1


# factories for test Dims - each returns randomized instances of
# a particular Dim subclass along with the int list they represent


# we generate segs to hit a target sum while testing partitions -
# this means we're not testing other kinds of Dims as partitions.
#
def new_seq(n=None, sum=None, bounds=None) -> Tuple[Seq, List[int]]:
    record_ctor("Seq")
    if sum is not None:
        wids = []
        while sum > 0:
            x = min(sum, random.randint(0, 32))
            wids.append(x)
            sum -= x
    else:
        if bounds is None:
            bounds = (0, 256)  # TODO: negative
        if n is None:
            n = random.randint(0, 32)
        wids = [random.randint(*bounds) for _ in range(n)]
    segs = Seq(wids)
    return segs, wids


def new_rect(n=None) -> Tuple[Rect, List[int]]:
    record_ctor("Rect")
    if n is None:
        n = random.randint(0, 8)
    w = random.randint(0, 8)
    return Rect(w, n), [w] * n


def new_affine(n=None) -> Tuple[Affine, List[int]]:
    record_ctor("Affine")
    if n is not None and n <= 1:
        raise ValueError(f"n must be > 1, got {n}")
    w = random.randint(1, 5)
    if random.randint(0, 1) == 1:
        w *= -1
    if n is None:
        n = random.randint(2, 8)
    b = max(random.randint(1, 8), -w * n)
    aff = Affine(b, n, w)
    ref = [b + w * i for i in range(n)]
    return aff, ref


def new_runs() -> Tuple[Runs, List[int]]:
    record_ctor("Runs")
    n = random.randint(2, 4)
    wids = [random.randint(1, 5) for _ in range(n)]
    reps = [random.randint(1, 5) for _ in range(n)]
    runs = Runs(Seq(wids), Seq(reps))
    ref = list(chain(*[[w] * r for w, r in zip(wids, reps)]))
    return runs, ref


def new_sparse(n=None) -> Tuple[Sparse, List[int]]:
    record_ctor("Sparse")
    sparsity = random.randint(50, 100)
    if n is None:
        n = random.randint(0, 16)
    ref = [random.randint(0, 4) for _ in range(n)]
    ref = [w if random.randint(0, 100) >= sparsity else 0 for w in ref]
    wids = [w for w in ref if w != 0]
    offs = [i for i in range(len(ref)) if ref[i] != 0] + [len(ref)]
    spa = Sparse(Seq(wids), Seq(offs, is_offsets=True))
    return spa, ref


def new_repeat() -> Tuple[Repeat, List[int]]:
    record_ctor("Repeat")
    with block_ctors(new_repeat, new_chain):
        n = random.randint(2, 5)
        seq, ref = draw_ctor()()
        ref = ref * n
        rep = Repeat(seq, n)
        return rep, ref


def new_chain() -> Tuple[Chain, List[int]]:
    record_ctor("Chain")
    with block_ctors(new_chain, new_repeat):
        n = random.randint(2, 5)
        seqs: List[Dim] = []
        refs: List[List[int]] = []
        while len(seqs) < n:
            s, r = draw_ctor()()
            if len(s) > 0:
                seqs += [s]
                refs += [r]
        ch = Chain(seqs)
        ref = flatten(refs)
        return ch, ref


DIM_CLASSES_TO_CTORS = {
    Seq: new_seq,
    Rect: new_rect,
    Affine: new_affine,
    Runs: new_runs,
    Sparse: new_sparse,
    Repeat: new_repeat,
    Chain: new_chain,
}


def get_our_ctor(dim: Dim):
    return DIM_CLASSES_TO_CTORS[dim.__class__]


DIM_CTORS = DIM_CLASSES_TO_CTORS.values()


DIM_CTORS_TO_CLASSES = {DIM_CLASSES_TO_CTORS[c]: c for c in DIM_CLASSES_TO_CTORS.keys()}


def draw_ctor():
    return DIM_CTORS[random.randint(0, len(DIM_CTORS) - 1)]


# when building composite dims (Reps, Chain) we use this to
# avoid deep recursion when using get_ctor()
@contextlib.contextmanager
def block_ctors(*ctors):
    global DIM_CTORS
    save = DIM_CTORS
    DIM_CTORS = list(set(DIM_CTORS).difference(ctors))
    try:
        yield
    finally:
        DIM_CTORS = save


DEBUG = False


# simple exhaustive testing - draw a bunch of random instances
# of each Dim subclass, stress test each one's properties against
# those of the int list it represents
class TestDim(TestCase):
    def test_all(self):
        self.check_ctor(new_seq)
        self.check_ctor(new_rect)
        self.check_ctor(new_affine)
        self.check_ctor(new_runs)
        self.check_ctor(new_sparse)
        self.check_ctor(new_repeat)
        self.check_ctor(new_chain)

    def check_ctor(self, ctor):
        print(f"{DIM_CTORS_TO_CLASSES[ctor]}")
        for _ in range(64):
            dim, ref = ctor()
            self.check_dim(dim, ref)

    def check_dim(self, dim: Dim, ref: List[int]):
        if DEBUG:
            print(f"check_dim(dim={repr(dim)})")
        # len(dim)
        self.check_len(dim, ref)
        # dim.orbit(), dim.cycle()
        self.check_cycle_and_orbit(dim, ref)
        # dim.sum()
        self.check_sum(dim, ref)
        if len(ref) > 0:
            # dim.min()
            self.check_min(dim, ref)
            # dim.max()
            self.check_max(dim, ref)
        # dim[i]
        self.check_getitem(dim, ref)
        # dim[[i, ...]]
        self.check_getseq(dim, ref)
        # dim.offset_of(i)
        self.check_offset_of(dim, ref)
        # dim.offsets()
        self.check_offsets(dim, ref)
        # dim.fwddiff()
        self.check_fwddiff(dim, ref)
        # dim.index(x)
        self.check_index_of(dim, ref)

        # dim[start:stop]
        self.check_getslice(dim, ref)

        # dim1 + dim2
        for ctor in DIM_CTORS:
            self.check_add(dim, ref, *ctor())  # type: ignore

        # dim.repeat(n)
        self.check_repeat(dim, ref)

        # dim.select(dim2)
        # use only ctors whose output length is trivial to control
        with block_ctors(new_repeat, new_chain, new_runs, new_affine):
            n = max(ref) + 1 if len(ref) > 0 else 0
            d2 = draw_ctor()(n=n)
            self.check_select(dim, ref, *d2)

        # dim.spread(dim2)
        # use only ctors whose output length is trivial to control
        with block_ctors(new_repeat, new_chain, new_runs, new_affine):
            self.check_spread(dim, ref, *draw_ctor()(n=len(dim)))
            self.check_spread(dim, ref, *draw_ctor()(n=len(dim)))
            part_dim, part_ref = new_seq(sum=len(dim))
            reps_dim, reps_ref = new_seq(n=len(part_dim), bounds=(0, 8))
            self.check_spread_part(dim, ref, reps_dim, reps_ref, part_dim, part_ref)

        # dim.scale(i)
        self.check_scale_int(dim, ref)

        # dim.scale(f)
        self.check_scale_float(dim, ref)

        # dim1.scale(dim2)
        # use only ctors whose output length is trivial to control
        with block_ctors(new_repeat, new_chain, new_runs, new_affine):
            self.check_scale_seq(dim, ref, *draw_ctor()(n=len(dim)))

        # dim.shift(i)
        self.check_shift_int(dim, ref)

        # dim.iota()
        self.check_iota(dim, ref)

        # dim.floor(x), dim.ceil(x)
        self.check_floor(dim, ref)
        self.check_ceil(dim, ref)

        # dim1.shift(dim2)
        # use only ctors whose output length is trivial to control
        with block_ctors(new_repeat, new_chain, new_runs, new_affine):
            self.check_shift_seq(dim, ref, *draw_ctor()(n=len(dim)))

        # dim1.fold(dim2)
        with block_ctors(new_repeat, new_chain, new_runs):
            self.check_fold(dim, ref, *new_seq(sum=len(dim)))  # type: ignore

        # dim1.cut(dim2)
        with block_ctors(new_repeat, new_chain, new_runs):
            self.check_cut(dim, ref, *new_seq(sum=len(dim)))  # type: ignore

        # check repr() and str() roundtripping
        self.check_roundtrip(dim)

    #
    # method checks
    #

    def check_len(self, dim: Dim, ref: List[int]):
        if DEBUG:
            print(f"check_len(dim={repr(dim)})")
        try:
            self.assertEqual(len(dim), len(ref))
        except Exception as e:
            print(f"\ncheck_len(dim={repr(dim)}):\n\rException {e}")
            raise e

    # helper
    # we want to check equality mod representation, but we also want
    # to make sure our representations don't degenerate size-wise
    def checkEquiv(self, dim, ref):
        ratio = get_size(dim) / get_size(ref)
        if ratio == 1.0:
            self.assertEqual(dim, ref)
        else:
            if DEBUG:
                print(
                    type(dim),
                    get_size(dim),
                    type(ref),
                    get_size(ref),
                    get_size(dim) / get_size(ref),
                )
            self.assertTrue(dim.equal(ref))
            self.assertTrue(ratio <= 1, f"ratio {ratio} {dim} {ref}")

    def check_cycle_and_orbit(self, dim: Dim, ref: List[int]):
        if DEBUG:
            print(f"check_cycle_and_orbit(dim={repr(dim)})")
        try:
            # cycle
            c = dim.cycle()
            n = dim.ncycles()
            self.assertEqual(ref[:c] * n, ref)
            self.checkEquiv(dim[:c].repeat(n), dim)
            # orbit
            o = dim.orbit()
            self.checkEquiv(o, dim[:c])
            self.assertTrue(o.equal(ref[:c]))
            self.checkEquiv(o.repeat(n), dim)

        except Exception as e:
            print(f"\ncheck_cycle_and_orbit(dim={repr(dim)}):\n\rException {e}")
            raise e

    def check_sum(self, dim: Dim, ref: List[int]):
        if DEBUG:
            print(f"check_sum(dim={repr(dim)})")
        try:
            self.assertEqual(dim.sum(), sum(ref))
        except Exception as e:
            print(f"\ncheck_sum(dim={repr(dim)}):\n\rException {e}")
            raise e

    def check_min(self, dim: Dim, ref: List[int]):
        if DEBUG:
            print(f"check_min(dim={repr(dim)})")
        try:
            self.assertEqual(dim.min(), min(ref))
        except Exception as e:
            print(f"\ncheck_min(dim={repr(dim)}):\n\rException {e}")
            raise e

    def check_max(self, dim: Dim, ref: List[int]):
        if DEBUG:
            print(f"check_max(dim={repr(dim)})")
        try:
            self.assertEqual(dim.max(), max(ref))
        except Exception as e:
            print(f"\ncheck_max(dim={repr(dim)}):\n\rException {e}")
            raise e

    def check_getitem(self, dim: Dim, ref: List[int]):
        if DEBUG:
            print(f"check_getitem(dim={repr(dim)})")
        try:
            dim_items = [dim[i] for i in range(len(dim))]
            ref_items = [ref[i] for i in range(len(ref))]
            self.assertEqual(dim_items, ref_items)
        except Exception as e:
            print(f"\ncheck_getitem(dim={repr(dim)}, {ref}):\n\rException {e}")
            raise e

    def check_getseq(self, dim: Dim, ref: List[int]):
        if DEBUG:
            print(f"check_getseq(dim={repr(dim)})")
        try:
            ixs = [random.randint(0, len(ref) - 1) for _ in range(len(ref) // 2)]
            subdim = dim[ixs]
            subref = [ref[i] for i in ixs]
            self.assertEqual([w for w in subdim], subref)
        except Exception as e:
            print(f"\ncheck_getseq(dim={repr(dim)}):\n\rException {e}")
            raise e

    def check_getslice(self, dim: Dim, ref: List[int]):
        if DEBUG:
            print(f"check_getslice(dim={repr(dim)})")
        try:
            for stop in range(len(dim) + 1):
                for start in range(stop + 1):
                    d = dim[start:stop]
                    r = ref[start:stop]
                    self.assertEqual([w for w in d], [w for w in r])
        except Exception as e:
            print(f"{repr(dim)} {dim}")
            print(f"{[w for w in dim]}[{start}:{stop}] = {[w for w in d]}")
            print(f"{ref}[{start}:{stop}] = {r}")
            print(
                f"\ncheck_getslice(dim={repr(dim)}) start {start} stop {stop}:\n\rException {e}"
            )
            raise e

    def check_select(self, diml: Dim, refl: List[int], dimr: Dim, refr: List[int]):
        if DEBUG:
            print(f"check_select(diml={repr(diml)}, dimr={repr(dimr)})")
        try:
            d = diml.select(dimr)
            r = [refr[i] for i in refl]
            self.check_getitem(d, r)
        except Exception as e:
            print(
                f"\ncheck_select(diml={repr(diml)}, dimr={repr(dimr)}):\n\tException {e}"
            )
            raise e

    def check_offset_of(self, dim: Dim, ref: List[int]):
        if DEBUG:
            print(f"check_offset_of(dim={repr(dim)})")
        try:
            dim_offsets = [dim.offset_of(i) for i in range(len(dim) + 1)]
            ref_offsets = offsets(ref)
            self.assertEqual(dim_offsets, ref_offsets)
        except Exception as e:
            print(f"\ncheck_offset_of(dim={repr(dim)}):\n\rException {e}")
            raise e

    def check_offsets(self, dim: Dim, ref: List[int]):
        if DEBUG:
            print(f"check_offsets(dim={repr(dim)})")
        try:
            base = random.randint(0, 1000)
            dim_offsets = dim.offsets(base)
            ref_offsets = offsets(ref, base)
            self.check_getitem(dim_offsets, ref_offsets)
        except Exception as e:
            print(f"\ncheck_offsets(dim={repr(dim)}):\n\rException {e}")
            raise e

    def check_fwddiff(self, dim: Dim, ref: List[int]):
        if DEBUG:
            print(f"check_fwddiff(dim={repr(dim)})")
        try:
            dim_fwddiff = dim.fwddiff()
            ref_fwddiff = fwddiff(ref)
            self.check_getitem(dim_fwddiff, ref_fwddiff)
        except Exception as e:
            print(f"\ncheck_fwddiff(dim={repr(dim)}):\n\rException {e}")
            raise e

    def check_index_of(self, dim: Dim, ref: List[int]):
        if DEBUG:
            print(f"check_index_of(dim={repr(dim)})")
        try:
            dim_indexes = [dim.index_of(x) for x in range(dim.sum() + 1)]
            ref_offsets = offsets(ref)
            ref_indexes = [bisect(ref_offsets, x) - 1 for x in range(sum(ref) + 1)]
            self.assertEqual(dim_indexes, ref_indexes)
        except Exception as e:
            print(f"\ncheck_index(dim={repr(dim)}):\n\rException {e}")
            raise e

    def check_add(self, diml: Dim, refl: List[int], dimr: Dim, refr: List[int]):
        if DEBUG:
            print(f"check_add(diml={repr(diml)}, dimr={repr(dimr)})")
        try:
            # print(f"\tcheck_add({repr(diml)}, {repr(dimr)})")
            d = diml.cat(dimr)
            r = refl + refr
            self.assertEqual([w for w in d], [w for w in r])
            # self.check_getitem(d, r)
        except Exception as e:
            print(
                f"\ncheck_add(diml={repr(diml)}, refl={refl}, dimr={repr(dimr)}, refr={refr}):\n\tException {e}"
            )
            raise e

    def check_repeat(self, dim: Dim, ref: List[int]):
        if DEBUG:
            print(f"check_repeat(dim={repr(dim)})")
        try:
            d = dim.repeat(2)
            r = ref * 2
            self.assertEqual([w for w in d], [w for w in r])
            # self.check_getitem(d, r)
        except Exception as e:
            print(f"\ncheck_repeat(dim={repr(dim)}):\n\rException {e}")
            raise e

    def check_spread(self, diml: Dim, refl: List[int], dimr: Dim, refr: List[int]):
        if DEBUG:
            print(f"check_spread(diml={repr(diml)}, dimr={repr(dimr)})")
        try:
            d = diml.spread(dimr)
            r = flatten([[x] * y for x, y in zip(refl, refr)])
            self.assertEqual([w for w in d], [w for w in r])
            # self.check_getitem(d, r)
        except Exception as e:
            print(
                f"\ncheck_scale(dim1={repr(diml)}, dim2={repr(dimr)}):\n\rException {e}"
            )
            raise e

    def check_spread_part(
        self,
        dim: Dim,
        ref: List[int],
        reps_dim: Dim,
        reps_ref: List[int],
        part_dim: Dim,
        part_ref: List[int],
    ):
        if DEBUG:
            print(
                f"check_spread_part(dim={repr(dim)}, reps={repr(reps_dim)}, part={repr(part_dim)})"
            )
        try:
            d = dim.spread(reps_dim, part_dim)
            r = flatten([x * y for x, y in zip(split(ref, part_ref), reps_ref)])
            self.assertEqual([w for w in d], [w for w in r])
            # self.check_getitem(d, r)
        except Exception as e:
            print(
                f"\ncheck_spread_part(dim={repr(dim)}, xdim={repr(reps_dim)}, pdim={repr(part_dim)})):\n\rException {e}"
            )
            raise e

    def check_scale_int(self, dim: Dim, ref: List[int]):
        if DEBUG:
            print(f"check_scale_int(dim={repr(dim)})")
        try:
            n = random.randint(2, 4)
            d = dim.scale(n)
            r = [x * n for x in ref]
            self.assertEqual([w for w in d], [w for w in r])
            # self.check_getitem(d, r)
        except Exception as e:
            print(f"\ncheck_scale_int(dim={repr(dim)}):\n\rException {e}")
            raise e

    def check_scale_float(self, dim: Dim, ref: List[int]):
        if DEBUG:
            print(f"check_scale_float(dim={repr(dim)})")
        try:
            n = 1 / random.randint(2, 4)
            d = dim.scale(n)
            r = [int(x * n) for x in ref]
            self.assertEqual([w for w in d], [w for w in r])
            # self.check_getitem(d, r)
        except Exception as e:
            print(f"\ncheck_scale_float(dim={repr(dim)}):\n\rException {e}")
            raise e

    def check_scale_seq(self, diml: Dim, refl: List[int], dimr: Dim, refr: List[int]):
        if DEBUG:
            print(f"check_scale_seq(diml={repr(diml)}, dimr={repr(dimr)})")
        try:
            d = diml.scale(dimr)
            r = [x * y for x, y in zip(refl, refr)]
            self.assertEqual([w for w in d], [w for w in r])
            # self.check_getitem(d, r)
        except Exception as e:
            print(
                f"\ncheck_scale_seq(dim1={repr(diml)}, dim2={repr(dimr)}):\n\rException {e}"
            )
            raise e

    def check_shift_int(self, dim: Dim, ref: List[int]):
        if DEBUG:
            print(f"check_shift_int(dim={repr(dim)})")
        try:
            n = random.randint(2, 4)
            s = random.randint(0, 1) * 2 - 1
            d = dim.shift(n, s)
            r = [x + n * s for x in ref]
            self.assertEqual([w for w in d], [w for w in r])
            # self.check_getitem(d, r)
        except Exception as e:
            print(f"\ncheck_shift_int(dim={repr(dim)}), scale = {s}:\n\rException {e}")
            raise e

    def check_shift_seq(self, diml: Dim, refl: List[int], dimr: Dim, refr: List[int]):
        if DEBUG:
            print(f"check_shift_seq(diml={repr(diml)}, dimr={repr(dimr)})")
        try:
            s = random.randint(0, 1) * 2 - 1
            d = diml.shift(dimr, s)
            r = [x + y * s for x, y in zip(refl, refr)]
            self.assertEqual([w for w in d], [w for w in r])
            # self.check_getitem(d, r)
        except Exception as e:
            print(
                f"\ncheck_shift_seq(dim1={repr(diml)}, dim2={repr(dimr)}), scale = {s}:\n\rException {e}"
            )
            raise e

    def check_iota(self, dim: Dim, ref: List[int]):
        if DEBUG:
            print(f"check_iota(dim={repr(dim)})")
        try:
            d = dim.iota()
            r = [i for w in ref for i in range(w)]
            self.assertEqual([w for w in d], [w for w in r])
        except Exception as e:
            print(f"\ncheck_iota(dim={repr(dim)}):\n\rException {e}")
            raise e

    def check_floor(self, dim: Dim, ref: List[int]):
        if DEBUG:
            print(f"check_floor(dim={repr(dim)})")
        try:
            n = random.randint(-5, 5)
            d = dim.floor(n)
            r = [max(n, x) for x in ref]
            self.assertEqual([w for w in d], [w for w in r])
        except Exception as e:
            print(f"\ncheck_floor(dim={repr(dim)}), n = {n}:\n\rException {e}")
            raise e

    def check_ceil(self, dim: Dim, ref: List[int]):
        if DEBUG:
            print(f"check_ceil(dim={repr(dim)})")
        try:
            n = random.randint(-5, 5)
            d = dim.ceil(n)
            r = [min(n, x) for x in ref]
            self.assertEqual([w for w in d], [w for w in r])
        except Exception as e:
            print(f"\ncheck_ceil(dim={repr(dim)}), n = {n}:\n\rException {e}")
            raise e

    def check_fold(self, diml: Dim, refl: List[int], dimr: Dim, refr: List[int]):
        if DEBUG:
            print(f"check_fold(diml={repr(diml)}, dimr={repr(dimr)})")
        try:
            d = diml.fold(dimr)
            offs1 = offsets(refl)
            offs2 = offsets(refr)
            offs = [offs1[offs2[i]] for i in range(len(refr) + 1)]
            r = [offs[i + 1] - offs[i] for i in range(len(offs) - 1)]
            self.assertEqual([w for w in d], [w for w in r])
            # self.check_getitem(d, r)
        except Exception as e:
            print(
                f"\ncheck_fold(dim1={repr(diml)}, dim2={repr(dimr)}):\n\rException {e}"
            )
            raise e

    def check_cut(self, diml: Dim, refl: List[int], dimr: Dim, refr: List[int]):
        if DEBUG:
            print(f"check_cut(diml={repr(diml)}, dimr={repr(dimr)})")
        try:
            ds = diml.cut(dimr)
            roffs = offsets(refr)
            rs = [refl[roffs[i] : roffs[i + 1]] for i in range(len(roffs) - 1)]
            self.assertEqual([list(d) for d in ds], [r for r in rs])
        except Exception as e:
            print(
                f"\ncheck_fold(dim1={repr(diml)}, dim2={repr(dimr)}):\n\rException {e}"
            )
            raise e

    def check_roundtrip(self, d: Dim):
        # checking expanded sequence (i.e., [w for w in d] rather
        # than straight equality) because we sometimes generate
        # test dims with regularities that dim() compresses out.
        # since dim() won't fail in the other direction (introducing
        # less compressed representations) this should be ok
        if DEBUG:
            print(f"check_roundtrip(dim={repr(d)})")
        try:
            str_rt = dim(eval(str(d)))
            self.assertEqual([w for w in d], [w for w in str_rt])
            repr_rt = eval(repr(d))
            self.assertEqual([w for w in d], [w for w in repr_rt])
        except Exception as e:
            print(f"\ncheck_roundtrip({d}):\n\rException {e}")
            raise e


if __name__ == "__main__":
    try:
        if DEBUG:
            cProfile.run("main()", sort=SortKey.CUMULATIVE)
        else:
            main()
    finally:
        pprint(ctor_count)
