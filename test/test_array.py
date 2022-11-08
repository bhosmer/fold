# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from unittest import TestCase, main

from fold import *
from util import get_size, sample

#
# Array, View, Overlay tests
#
# TODO
#
# 1. lots of things here just print rather than actually checking
# 2. some tests precede stride support and only test the contiguous case
# 3. probably good to split overlay/sparsity into its own test file
# 4. many things are spot checked and should be made more exhaustive
# 5. comments, some stuff here is more subtle than it looks
#


class TestGetIndex(TestCase):
    def check_view(self, base, view, shape, values):
        self.assertTrue(view.data is base.data)
        self.assertTrue(view.shape.equal(Shape(*shape)))
        self.assertEqual(view.tolist(), values)

    def test_slice_ragged_length_extend(self):
        m = arange(8, 8)
        x = m[1:5, :[1, 2]]
        self.check_view(m, x, (4, [1, 2]), [[8], [16, 17], [24], [32, 33]])

    def test_explicit_ragged_length_extend(self):
        m = arange(8, 8)
        x = m[[[1, 2], [5, 7]], :[1, 2]]
        self.assertTrue(x.shape.equal(Shape(2, 2, [1, 2])))
        self.assertEqual(x.tolist(), [[[8], [16, 17]], [[40], [56, 57]]])

    def test_explicit_rect_1(self):
        r = arange(4, 4, 4)
        i = [[0, 1], [2, 3]]
        x = r[i, 1]

        tr = torch.arange(64).reshape(4, 4, 4)
        ti = torch.Tensor([[0, 1], [2, 3]]).long()
        tx = tr[ti, 1]

        # TODO we want the exact check, uncomment when ixmap is done
        # self.assertEqual(x.shape, Shape(*tuple(tx.shape)))
        self.assertTrue(x.shape.equal(Shape(*tuple(tx.shape))))
        self.assertTrue(x.eval().data.equal(tx.reshape(-1)))

    def test_explicit_ragged_1(self):
        r = arange(4, 4, 4)
        x = r[[[1, 0], [3]], 1]
        # TODO we want the exact check, uncomment when ixmap is done
        # self.assertEqual(x.shape, Shape(2, [2, 1], 4))
        self.assertTrue(x.shape.equal(Shape(2, [2, 1], 4)))
        self.assertEqual(
            x.tolist(), [[[20, 21, 22, 23], [4, 5, 6, 7]], [[52, 53, 54, 55]]]
        )

    def test_explicit_ragged_2(self):
        m = arange(8, 8)
        x = m[[[0, 1, 3], [5, 7]], :[1, 2, 1, 2, 1]]
        self.assertTrue(x.shape.equal(Shape(2, [3, 2], [1, 2, 1, 2, 1])))
        self.assertEqual(x.tolist(), [[[0], [8, 9], [24]], [[40, 41], [56]]])

    def test_ragged_slice_stop_only_mixed(self):
        r = arange(4, 4, 4)
        x = r[1:3, 2, :[2, 4]]
        self.assertTrue(x.shape.equal(Shape(2, [2, 4])))
        self.assertTrue(x.data is r.data)
        self.assertEqual(x.tolist(), [[24, 25], [40, 41, 42, 43]])

    def test_triu_shape(self):
        t = arange(8, triu(4).cat(tril(4)))
        self.assertEqual(t.shape, Shape(8, Affine(4, 4, -1).cat(Affine(1, 4, 1))))
        self.assertEqual(t.numel(), 20)

    def test_affine_slice_1(self):
        t = arange(8, triu(4).cat(tril(4)))
        vec = t[:, 0]
        self.assertEqual(vec.shape, Shape(8))
        self.assertEqual(vec.tolist(), [0, 4, 7, 9, 10, 11, 13, 16])

    def test_affine_slice_2(self):
        t = arange(8, triu(4).cat(tril(4)))
        vec = t[:, -1]
        self.assertEqual(vec.shape, Shape(8))
        self.assertEqual(vec.tolist(), [3, 6, 8, 9, 10, 12, 15, 19])

    def test_affine_slice_3(self):
        t = arange(8, triu(4).cat(tril(4)))
        ut = t[:4]
        self.assertEqual(ut.shape[1], t.shape[1][:4])
        self.assertEqual(ut.tolist(), [[0, 1, 2, 3], [4, 5, 6], [7, 8], [9]])

    def test_affine_slice_4(self):
        t = arange(8, triu(4).cat(tril(4)))
        lt = t[4:]
        self.assertEqual(lt.shape[1], t.shape[1][4:])
        self.assertEqual(lt.tolist(), [[10], [11, 12], [13, 14, 15], [16, 17, 18, 19]])

    def test_affine_slice_5(self):
        t = arange(8, triu(4).cat(tril(4)))
        vrt = t[:, 2:]
        self.assertEqual([w for w in vrt.shape[1]], [max(w - 2, 0) for w in t.shape[1]])
        self.assertEqual(vrt.tolist(), [[2, 3], [6], [], [], [], [], [15], [18, 19]])

    # these are new, should dedup w/above

    def test_affine_slice_trailing_zero_rows_2d(self):
        tr = arange(8, Affine(8, 8, -1))
        trcols = tr[:, 4:5]
        self.assertEqual(trcols.shape[0], Rect(8))
        self.assertTrue(trcols.shape[1].equal(Rect(1, 4).cat(Rect(0, 4))))
        self.assertEqual(trcols.tolist(), [[4], [12], [19], [25], [], [], [], []])

    def test_affine_slice_trailing_zero_rows_3d(self):
        tr3 = arange(8, Affine(8, 8, -1), 2)
        tr3cols = tr3[:, 4:5]
        self.assertEqual(tr3cols.shape[0], Rect(8))
        self.assertTrue(tr3cols.shape[1].equal(Rect(1, 4).cat(Rect(0, 4))))
        self.assertTrue(tr3cols.shape[2].equal(Rect(2, 4)))  # NOTE
        self.assertEqual(
            tr3cols.tolist(),
            [[[8, 9]], [[24, 25]], [[38, 39]], [[50, 51]], [], [], [], []],
        )

    def test_affine_mixed_trailing_zero_rows_3d_inner(self):
        tr3 = arange(8, Affine(8, 8, -1), 2)
        tr3col = tr3[:, 4:5, 1]
        self.assertEqual(tr3col.ndim, 2)
        self.assertEqual(tr3col.shape[0], Rect(8))
        self.assertTrue(tr3col.shape[1].equal(Rect(1, 4).cat(Rect(0, 4))))
        self.assertEqual(tr3col.tolist(), [[9], [25], [39], [51], [], [], [], []])

    def test_affine_slice_mid_zero_rows_2d(self):
        tr = arange(8, Affine(4, 4, -1).cat(Affine(1, 4, 1)))
        trcols = tr[:, 2:4]
        self.assertEqual(trcols.shape[0], Rect(8))
        self.assertTrue(trcols.shape[1].equal(Seq([2, 1, 0, 0, 0, 0, 1, 2])))
        self.assertEqual(trcols.tolist(), [[2, 3], [6], [], [], [], [], [15], [18, 19]])

    def test_affine_slice_mid_zero_rows_3d(self):
        tr3 = arange(8, Affine(4, 4, -1).cat(Affine(1, 4, 1)), 2)
        tr3cols = tr3[:, 2:4]
        self.assertEqual(tr3cols.shape[0], Rect(8))
        self.assertTrue(tr3cols.shape[1].equal(Seq([2, 1, 0, 0, 0, 0, 1, 2])))
        self.assertTrue(tr3cols.shape[2].equal(Rect(2, 6)))  # NOTE
        self.assertEqual(
            tr3cols.tolist(),
            [
                [[4, 5], [6, 7]],
                [[12, 13]],
                [],
                [],
                [],
                [],
                [[30, 31]],
                [[36, 37], [38, 39]],
            ],
        )

    def test_affine_mixed_mid_zero_rows_3d_inner(self):
        tr3 = arange(8, Affine(4, 4, -1).cat(Affine(1, 4, 1)), 2)
        tr3cols = tr3[:, 2:4, 1]
        self.assertEqual(tr3cols.ndim, 2)
        self.assertEqual(tr3cols.shape[0], dim(8))
        self.assertTrue(tr3cols.shape[1].equal(dim([2, 1, 0, 0, 0, 0, 1, 2])))
        self.assertEqual(
            tr3cols.tolist(), [[5, 7], [13], [], [], [], [], [31], [37, 39]]
        )

    # TODO placeholders, need to sweep
    def test_rect_slice_1(self):
        t = arange(4, 5, 6)
        cube = t[1:3, 2:4, 3:5]
        self.check_view(
            t, cube, (2, 2, 2), [[[45, 46], [51, 52]], [[75, 76], [81, 82]]]
        )

    def test_rect_slice_2(self):
        t = arange(4, 5, 6)
        r = t[1, 2:4, 3:5]
        self.check_view(t, r, (2, 2), [[45, 46], [51, 52]])

    def test_rect_slice_3(self):
        t = arange(4, 5, 6)
        r = t[1:3, 2, 3:5]
        self.check_view(t, r, (2, 2), [[45, 46], [75, 76]])

    def test_rect_slice_4(self):
        t = arange(4, 5, 6)
        r = t[1:3, 2:4, 3]
        self.check_view(t, r, (2, 2), [[45, 51], [75, 81]])

    def test_rect_slice_5(self):
        t = arange(4, 5, 6)
        x = t[2, 2, 2]
        self.assertEqual(x.shape, Shape())
        self.assertEqual(x.item(), 74)

    def test_py_shape(self):
        py = arange(*Shape(10, tril(10), Chain([tril(w) for w in tril(10)])))
        self.assertEqual(py.ndim, 3)
        self.assertEqual(py.numel(), 220)

    def test_py_slice_1(self):
        py = arange(*Shape(10, tril(10), Chain([tril(w) for w in tril(10)])))
        r0 = py[:, 0]
        self.assertEqual(r0.shape, Shape(10, 1))
        self.assertEqual(
            r0.tolist(), [[0], [1], [4], [10], [20], [35], [56], [84], [120], [165]]
        )

    def test_py_slice_2(self):
        py = arange(*Shape(10, tril(10), Chain([tril(w) for w in tril(10)])))
        rn1 = py[:, -1]
        self.assertEqual(rn1.shape, Shape(10, Affine(1, 10, 1)))
        cn1 = rn1[:, -1]
        self.assertEqual(cn1.tolist(), [0, 3, 9, 19, 34, 55, 83, 119, 164, 219])


class TestViewComposition(TestCase):
    def test_expand_commutativity(self):
        m = arange(4).expand(4, -1)
        v = m[:, 1:3]
        w = m[0, 1:3].expand(4, 2)
        v.data == w.data
        # note: representations may differ currently. in particular,
        # expanded strides are better compressed than calculated view
        # strides, so w will be more compactly represented than v
        v.view.addresses().equal(w.view.addresses())


class TestFromData(TestCase):
    def test_ragged_int_1(self):
        ragged_ints = [[[1, 2], [3, 4, 5]], [[6], [7], [8, 9, 10, 11, 12]]]
        d = array(ragged_ints)
        s = Shape(2, [2, 3], [2, 3, 1, 1, 5])
        t = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        self.assertEqual(d, Array(t, s))

    def test_ragged_int_2(self):
        ragged_ints = [[1, 2, 3], [], [4, 5, 6]]
        d = array(ragged_ints)
        s = Shape(3, [3, 0, 3])
        t = torch.tensor([1, 2, 3, 4, 5, 6])
        self.assertEqual(d, Array(t, s))

    def test_rect_int(self):
        rect_ints = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
        d = array(rect_ints)
        s = Shape(2, 2, 3)
        t = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        self.assertEqual(d, Array(t, s))


class TestPrints(TestCase):
    # just for eyeballing
    def test_prints(self):
        print("---")
        m = arange(8, 8)
        print(m)
        print("---")
        print(m.pointwise_unary(lambda t: t * 1000.0))
        print("---")
        r = arange(4, 4, 4)
        print(r)
        print("---")
        tr = arange(20, triu(10).cat(tril(10)))
        print(tr)
        print("---")
        tr = rand(20, triu(10).cat(tril(10)))
        print(tr.pointwise_unary(lambda t: t * 1000))
        print("---")
        print(tr.pointwise_unary(lambda t: t * -1000))
        print("---")
        py = arange(8, triu(8), concat_dims([triu(w) for w in triu(8)]))
        print(py)
        print("---")
        print(py.pointwise_unary(lambda t: t * -99.9))
        print("---")
        b = rand(4, 4).pointwise_unary(lambda t: t > 0.5)
        print(b)
        print("---")
        m = arange(8, 8)
        print(m[:, : triu(8)])
        print("---")
        m = arange(8, 8)
        print(m[:, : tril(8)])
        print("---")
        m = arange(16, tril(8).cat(triu(8)))
        print(m)
        print("---")
        m = arange(16, 8)
        print(m[:, : tril(8).cat(triu(8))])
        print("---")
        m = rand(8, 8)
        m[:, : triu(8)] = zeros(8, triu(8))  # TODO broadcast
        print(m)
        print("---")
        m = rand(8, 8)
        m[:, Affine(8, 8, -1) :] = zeros(8, Affine(0, 8, 1))
        print(m)
        print("---")
        m = zeros(8, 8)
        m[:, Affine(0, 8, 1) : Affine(1, 8, 1)] = rand(8, 1)
        print(m)
        print("---")
        m = zeros(8, 8)
        m[:, Affine(7, 8, -1) : Affine(8, 8, -1)] = rand(8, 1)
        print(m)
        print("---")
        m = arange(16, 16)
        print(m[:, diag(16) : diag(16, 1)])
        m[:, diag(16) : diag(16, 1)] = zeros(16, 1, dtype=int)
        print(m)
        print("---")
        m = arange(16, 16)
        print(m[:, diag(16, 0, 4) : diag(16, 1, 4)])
        m[:, diag(16, 0, 4) : diag(16, 1, 4)] = zeros(16, 4, dtype=int)
        print(m)
        print("--- banded diagonal ---")
        m = zeros(16, 20, dtype=int)
        m[:, Affine(0, 16, 1) : Affine(5, 16, 1)] = arange(16, 5)
        print(m)

    def test_todo(self):
        # here to make sure they don't blow up
        # TODO make tests, move to where they go
        m = arange(8, 8)
        print(m[:3, [1, 3, 5]])
        print(m[:4, [[1, 3], [5]]])
        #
        m3 = arange(4, 4, 4)
        mx = m3[:2, [0, 1], :]
        mz = m3[:, [[0, 1], [1, 2]], :]  # shaped index list
        mz = m3[:, [[0, 1], [2]], :]  # ragged shaped index list
        t3 = torch.arange(4 * 4 * 4).reshape(4, 4, 4)
        print(m[[0, 2, 4], [1, 3, 5]])


class TestSparseIndexFormatExtractions(TestCase):
    #
    # tests of extractions (__getitem__, gather()) using index
    # shapes corresponding to standard sparse index formats.
    # note: these are not equivalent to sparse tensors - those
    # are the injections (__setitem__, scatter()) - see
    # TestSparseIndexFormatInjections
    #

    def test_ell_extract(self):
        #
        # ell is the standard scatter/gather index format.
        # test gather over rectangular indexes, check against
        # PyTorch gather and advanced indexing equivalent.
        #
        m3 = arange(8, 8, 8)
        i3 = arange(2, 2, 2)
        tm3 = torch.arange(512).reshape(8, 8, 8)
        ti3 = torch.arange(8).reshape(2, 2, 2)

        g = m3.gather(0, i3)
        self.assertEqual(g.tolist(), tm3.gather(0, ti3).tolist())
        self.assertEqual(g.tolist(), m3[i3, {}, {}].tolist())

        g = m3.gather(1, i3)
        self.assertEqual(g.tolist(), tm3.gather(1, ti3).tolist())
        self.assertEqual(g.tolist(), m3[{}, i3, {}].tolist())

        g = m3.gather(2, i3)
        self.assertEqual(g.tolist(), tm3.gather(2, ti3).tolist())
        self.assertEqual(g.tolist(), m3[{}, {}, i3].tolist())

    def test_permuted_ell_extract(self):
        #
        # test shuffling indexes to do "permuted gathers", e.g. use
        # an ell index to designate row rather than column locations
        #
        m = arange(8, 8)
        i = arange(2, 2)
        self.assertEqual(m[i, {0}].tolist(), [[0, 8], [17, 25]])
        # some 3d permutations
        m3 = arange(8, 8, 8)
        i3 = arange(2, 2, 2)
        x = m3[{0}, i3, {1}]
        self.assertEqual(x.tolist(), [[[0, 8], [17, 25]], [[96, 104], [113, 121]]])
        x = m3[{1}, i3, {0}]
        self.assertEqual(x.tolist(), [[[0, 8], [80, 88]], [[33, 41], [113, 121]]])
        x = m3[{1}, {0}, i3]
        self.assertEqual(x.tolist(), [[[0, 1], [66, 67]], [[12, 13], [78, 79]]])

    def test_csr_extract(self):
        #
        # CSR is "ragged gather" with explicit index on the innermost
        # dimension. check against advanced indexing equivalent.
        #
        # note that we're testing with sparse indexes (i.e., some rows
        # are empty), both to simulate real-world sparsity and to test
        # our synthetic iota index generation in such cases
        #

        # 2d csr, enumerated index
        m = arange(4, 4)
        i = array([[3], [], [0, 1, 2], [2, 3]])
        g = m.gather(1, i)
        self.assertEqual(g.tolist(), [[3], [], [8, 9, 10], [14, 15]])
        x = m[{}, i]
        self.assertEqual(x.tolist(), g.tolist())

        # 2d csr, index from sparse dim
        # note: the arange-based indexes aren't realistic, just used for brevity
        m = arange(8, 8)
        i = arange(8, Sparse({1: 2, 4: 3, 5: 1, 7: 2}))
        g = m.gather(1, i)
        self.assertEqual(
            g.tolist(), [[], [8, 9], [], [], [34, 35, 36], [45], [], [62, 63]]
        )
        x = m[{}, i]
        self.assertEqual(x.tolist(), g.tolist())

        # with batch dimension, same sparsity pattern per batch
        m = arange(2, 8, 16)
        i = arange(2, 8, Sparse({1: 2, 4: 3, 5: 1, 7: 2}))
        g = m.gather(2, i)
        self.assertEqual(
            g.tolist(),
            [
                [[], [16, 17], [], [], [66, 67, 68], [85], [], [118, 119]],
                [[], [152, 153], [], [], [202, 203, 204], [221], [], [254, 255]],
            ],
        )
        x = m[{}, {}, i]
        self.assertEqual(x.tolist(), g.tolist())

        # with batch dimension, different sparsity per batch
        m = arange(2, 8, 16)
        i = arange(2, 8, Sparse({1: 2, 4: 3, 5: 1, 7: 2, 10: 5, 13: 2, 16: -1}))
        g = m.gather(2, i)
        self.assertEqual(
            g.tolist(),
            [
                [[], [16, 17], [], [], [66, 67, 68], [85], [], [118, 119]],
                [[], [], [168, 169, 170, 171, 172], [], [], [221, 222], [], []],
            ],
        )
        x = m[{}, {}, i]
        self.assertEqual(x.tolist(), g.tolist())

    def test_csc_extract(self):
        #
        # csc is equivalent to ragged gather with explicit index
        # on the next-innermost dimension. check against advanced
        # indexing equivalent.
        #
        m = arange(4, 4)
        i = array([[3], [], [0, 1, 2], [2, 3]])
        x = m[i, {0}]
        self.assertEqual(x.tolist(), [[12], [], [2, 6, 10], [11, 15]])

        m = arange(8, 8)
        i = arange(8, Sparse({1: 2, 4: 3, 5: 1, 7: 2}))
        x = m[i, {0}]
        self.assertEqual(
            x.tolist(), [[], [1, 9], [], [], [20, 28, 36], [45], [], [55, 63]]
        )

        # with batch dimension, same sparsity pattern per batch
        m = arange(2, 16, 8)
        i = arange(2, 8, Sparse({1: 2, 4: 3, 5: 1, 7: 2}))
        x = m[{}, i, {1}]
        self.assertEqual(
            x.tolist(),
            [
                [[], [1, 9], [], [], [20, 28, 36], [45], [], [55, 63]],
                [[], [193, 201], [], [], [212, 220, 228], [237], [], [247, 255]],
            ],
        )

        # with batch dimension, different sparsity per batch
        m = arange(2, 16, 8)
        i = arange(2, 8, Sparse({1: 2, 4: 3, 5: 1, 7: 2, 10: 5, 13: 2, 16: -1}))
        x = m[{}, i, {1}]
        self.assertEqual(
            x.tolist(),
            [
                [[], [1, 9], [], [], [20, 28, 36], [45], [], [55, 63]],
                [[], [], [194, 202, 210, 218, 226], [], [], [237, 245], [], []],
            ],
        )

    def test_coo_extract(self):
        #
        # advanced indexing expressions using linear vector indexes
        # - equivalent to sparse COO index format.
        #
        m3 = arange(8, 8, 8)
        tm3 = torch.arange(512).reshape(8, 8, 8)

        x = m3[{}, [3, 2, 1], [1, 2, 3]]
        y = m3[[0], [3, 2, 1], [1, 2, 3]]
        self.assertEqual(x.tolist(), y.tolist())
        t = tm3[[0], [3, 2, 1], [1, 2, 3]]
        self.assertEqual(x.tolist(), t.tolist())

        x = m3[[3, 2, 1], {}, [1, 2, 3]]
        y = m3[[3, 2, 1], [0], [1, 2, 3]]
        self.assertEqual(x.tolist(), y.tolist())
        t = tm3[[3, 2, 1], [0], [1, 2, 3]]
        self.assertEqual(x.tolist(), t.tolist())

        x = m3[{}, {}, Affine(7, 8, -1)]
        y = m3[[0], [0], Affine(7, 8, -1)]
        self.assertEqual(x.tolist(), y.tolist())
        t = tm3[[0], [0], [7, 6, 5, 4, 3, 2, 1, 0]]
        self.assertEqual(x.tolist(), t.tolist())

        x = m3[{}, Affine(7, 8, -1), {}]
        y = m3[[0], Affine(7, 8, -1), Affine(0, 8, 1)]
        self.assertEqual(x.tolist(), y.tolist())
        t = tm3[[0], [7, 6, 5, 4, 3, 2, 1, 0], torch.arange(8)]
        self.assertEqual(x.tolist(), t.tolist())

        x = m3[Affine(7, 8, -1), {}, {}]
        y = m3[Affine(7, 8, -1), [0], Affine(0, 8, 1)]
        self.assertEqual(x.tolist(), y.tolist())
        t = tm3[[7, 6, 5, 4, 3, 2, 1, 0], [0], torch.arange(8)]
        self.assertEqual(x.tolist(), t.tolist())


class TestSparseIndexFormatInjections(TestCase):
    #
    # tests of injections (__setitem__, scatter()) using index
    # shapes corresponding to standard sparse index formats.
    #
    # note: these tests use expanded-zero destination arrays,
    # both to evoke sparse tensors and to stress-test overlays
    # in the presence of destination striding.
    #

    def test_ell_inject(self):
        #
        # ell is the standard scatter/gather index format.
        # test scatter over rectangular indexes, check against
        # PyTorch and advanced indexing equivalent.
        #
        m3 = arange().expand(8, 8, 8)
        i3 = arange(2, 2, 2)
        x3 = arange(2, 2, 2, start=100)
        tm3 = torch.tensor(m3.tolist())
        ti3 = torch.tensor(i3.tolist())
        tx3 = torch.tensor(x3.tolist())

        sc = m3.scatter(0, i3, x3)
        self.assertEqual(sc.tolist(), tm3.scatter(0, ti3, tx3).tolist())
        ov = m3.overlay[i3, {}, {}](x3)
        self.assertEqual(ov.tolist(), sc.tolist())

        sc = m3.scatter(1, i3, x3)
        self.assertEqual(sc.tolist(), tm3.scatter(1, ti3, tx3).tolist())
        ov = m3.overlay[{}, i3, {}](x3)
        self.assertEqual(ov.tolist(), sc.tolist())

        sc = m3.scatter(2, i3, x3)
        self.assertEqual(sc.tolist(), tm3.scatter(2, ti3, tx3).tolist())
        ov = m3.overlay[{}, {}, i3](x3)
        self.assertEqual(ov.tolist(), sc.tolist())

    def test_permuted_ell_inject(self):
        #
        # test shuffling indexes to do "permuted scatters", e.g. use
        # an ell index to designate row rather than column locations
        #
        m = arange().expand(4, 2)
        i = arange(2, 2)
        x = arange(i.numel(), start=100)
        ov = m.overlay[i, {0}](x)
        self.assertEqual(ov.tolist(), [[100, 0], [101, 0], [0, 102], [0, 103]])

        # some 3d permutations
        m3 = arange().expand(2, 8, 2)
        i3 = arange(2, 2, 2, const=True)
        x3 = arange(i3.numel(), start=100)

        ov = m3.overlay[{0}, i3, {1}](x3)
        self.assertEqual(
            ov.tolist(),
            [
                [
                    [100, 0],
                    [101, 0],
                    [0, 102],
                    [0, 103],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [104, 0],
                    [105, 0],
                    [0, 106],
                    [0, 107],
                ],
            ],
        )

        ov = m3.overlay[{1}, i3, {0}](x3)
        self.assertEqual(
            ov.tolist(),
            [
                [
                    [100, 0],
                    [101, 0],
                    [0, 0],
                    [0, 0],
                    [0, 104],
                    [0, 105],
                    [0, 0],
                    [0, 0],
                ],
                [
                    [0, 0],
                    [0, 0],
                    [102, 0],
                    [103, 0],
                    [0, 0],
                    [0, 0],
                    [0, 106],
                    [0, 107],
                ],
            ],
        )

        m3 = arange().expand(2, 2, 8)
        ov = m3.overlay[{1}, {0}, i3](x3)
        self.assertEqual(
            ov.tolist(),
            [
                [[100, 101, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 104, 105, 0, 0]],
                [[0, 0, 102, 103, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 106, 107]],
            ],
        )

    def test_csr_inject(self):
        #
        # CSR is "ragged gather" with explicit index on the innermost
        # dimension. check against advanced indexing equivalent.
        #
        # note that we're testing with sparse indexes (i.e., some rows
        # are empty), both to simulate real-world sparsity and to test
        # our synthetic iota index generation in such cases
        #

        # 2d csr, enumerated index
        m = arange().expand(4, 4)
        i = array([[3], [], [0, 1, 2], [2, 3]])
        x = arange(i.numel(), start=100)
        sc = m.scatter(1, i, x)
        self.assertEqual(
            sc.tolist(),
            [[0, 0, 0, 100], [0, 0, 0, 0], [101, 102, 103, 0], [0, 0, 104, 105]],
        )
        ov = m.overlay[{}, i](x)
        self.assertEqual(ov.tolist(), sc.tolist())

        # 2d csr, index from sparse dim
        # note: arange-based indexes aren't realistic distributions, just easy to build
        m = arange().expand(4, 4)
        i = arange(4, Sparse({0: 1, 2: 2, 3: 1}))
        x = arange(i.numel(), start=100)
        sc = m.scatter(1, i, x)
        self.assertEqual(
            sc.tolist(),
            [[100, 0, 0, 0], [0, 0, 0, 0], [0, 101, 102, 0], [0, 0, 0, 103]],
        )
        ov = m.overlay[{}, i](x)
        self.assertEqual(ov.tolist(), sc.tolist())

        #
        # @@@ FROM HERE @@@
        #

        # with batch dimension, same sparsity pattern per batch
        m = arange(2, 8, 16)
        i = arange(2, 8, Sparse({1: 2, 4: 3, 5: 1, 7: 2}))
        g = m.gather(2, i)
        self.assertEqual(
            g.tolist(),
            [
                [[], [16, 17], [], [], [66, 67, 68], [85], [], [118, 119]],
                [[], [152, 153], [], [], [202, 203, 204], [221], [], [254, 255]],
            ],
        )
        x = m[{}, {}, i]
        self.assertEqual(x.tolist(), g.tolist())

        # with batch dimension, different sparsity per batch
        m = arange(2, 8, 16)
        i = arange(2, 8, Sparse({1: 2, 4: 3, 5: 1, 7: 2, 10: 5, 13: 2, 16: -1}))
        g = m.gather(2, i)
        self.assertEqual(
            g.tolist(),
            [
                [[], [16, 17], [], [], [66, 67, 68], [85], [], [118, 119]],
                [[], [], [168, 169, 170, 171, 172], [], [], [221, 222], [], []],
            ],
        )
        x = m[{}, {}, i]
        self.assertEqual(x.tolist(), g.tolist())

    def test_csc_extract(self):
        #
        # csc is equivalent to ragged gather with explicit index
        # on the next-innermost dimension. check against advanced
        # indexing equivalent.
        #
        m = arange(4, 4)
        i = array([[3], [], [0, 1, 2], [2, 3]])
        x = m[i, {0}]
        self.assertEqual(x.tolist(), [[12], [], [2, 6, 10], [11, 15]])

        m = arange(8, 8)
        i = arange(8, Sparse({1: 2, 4: 3, 5: 1, 7: 2}))
        x = m[i, {0}]
        self.assertEqual(
            x.tolist(), [[], [1, 9], [], [], [20, 28, 36], [45], [], [55, 63]]
        )

        # with batch dimension, same sparsity pattern per batch
        m = arange(2, 16, 8)
        i = arange(2, 8, Sparse({1: 2, 4: 3, 5: 1, 7: 2}))
        x = m[{}, i, {1}]
        self.assertEqual(
            x.tolist(),
            [
                [[], [1, 9], [], [], [20, 28, 36], [45], [], [55, 63]],
                [[], [193, 201], [], [], [212, 220, 228], [237], [], [247, 255]],
            ],
        )

        # with batch dimension, different sparsity per batch
        m = arange(2, 16, 8)
        i = arange(2, 8, Sparse({1: 2, 4: 3, 5: 1, 7: 2, 10: 5, 13: 2, 16: -1}))
        x = m[{}, i, {1}]
        self.assertEqual(
            x.tolist(),
            [
                [[], [1, 9], [], [], [20, 28, 36], [45], [], [55, 63]],
                [[], [], [194, 202, 210, 218, 226], [], [], [237, 245], [], []],
            ],
        )

    def test_coo_extract(self):
        #
        # advanced indexing expressions using linear vector indexes
        # - equivalent to sparse COO index format.
        #
        m3 = arange(8, 8, 8)
        tm3 = torch.arange(512).reshape(8, 8, 8)

        x = m3[{}, [3, 2, 1], [1, 2, 3]]
        y = m3[[0], [3, 2, 1], [1, 2, 3]]
        self.assertEqual(x.tolist(), y.tolist())
        t = tm3[[0], [3, 2, 1], [1, 2, 3]]
        self.assertEqual(x.tolist(), t.tolist())

        x = m3[[3, 2, 1], {}, [1, 2, 3]]
        y = m3[[3, 2, 1], [0], [1, 2, 3]]
        self.assertEqual(x.tolist(), y.tolist())
        t = tm3[[3, 2, 1], [0], [1, 2, 3]]
        self.assertEqual(x.tolist(), t.tolist())

        x = m3[{}, {}, Affine(7, 8, -1)]
        y = m3[[0], [0], Affine(7, 8, -1)]
        self.assertEqual(x.tolist(), y.tolist())
        t = tm3[[0], [0], [7, 6, 5, 4, 3, 2, 1, 0]]
        self.assertEqual(x.tolist(), t.tolist())

        x = m3[{}, Affine(7, 8, -1), {}]
        y = m3[[0], Affine(7, 8, -1), Affine(0, 8, 1)]
        self.assertEqual(x.tolist(), y.tolist())
        t = tm3[[0], [7, 6, 5, 4, 3, 2, 1, 0], torch.arange(8)]
        self.assertEqual(x.tolist(), t.tolist())

        x = m3[Affine(7, 8, -1), {}, {}]
        y = m3[Affine(7, 8, -1), [0], Affine(0, 8, 1)]
        self.assertEqual(x.tolist(), y.tolist())
        t = tm3[[7, 6, 5, 4, 3, 2, 1, 0], [0], torch.arange(8)]
        self.assertEqual(x.tolist(), t.tolist())


class TestScatter(TestCase):
    #
    # test scatter and advanced indexing equivalents over some typical shapes
    #
    def test_scatter_rect_inner(self):
        tdest = torch.rand(4, 4, 4)
        tsrc = torch.rand(4, 4, 2)
        tidx = torch.tensor(sample([(4, 2)] * 16)).reshape(4, 4, 2)
        tscat = torch.scatter(tdest, -1, tidx, tsrc)
        dest = array(tdest)
        src = array(tsrc)
        idx = array(tidx)
        scat = dest.scatter(-1, idx, src)
        self.assertEqual(scat.tolist(), tscat.tolist())
        scat2 = dest.overlay[{}, {}, idx](src)
        self.assertEqual(scat.tolist(), scat2.tolist())

    def test_scatter_rect_mid(self):
        tdest = torch.rand(4, 4, 4)
        tsrc = torch.rand(4, 2, 4)
        tidx = torch.tensor(sample([(4, 2)] * 16)).reshape(4, 4, 2).permute(0, 2, 1)
        tscat = torch.scatter(tdest, 1, tidx, tsrc)
        dest = array(tdest)
        src = array(tsrc)
        idx = array(tidx)
        scat = dest.scatter(1, idx, src)
        self.assertEqual(scat.tolist(), tscat.tolist())
        scat2 = dest.overlay[{}, idx, {}](src)
        self.assertEqual(scat.tolist(), scat2.tolist())

    def test_scatter_rect_outer(self):
        tdest = torch.rand(4, 4, 4)
        tsrc = torch.rand(2, 4, 4)
        tidx = torch.tensor(sample([(4, 2)] * 16)).reshape(4, 4, 2).permute(2, 0, 1)
        tscat = torch.scatter(tdest, 0, tidx, tsrc)
        dest = array(tdest)
        src = array(tsrc)
        idx = array(tidx)
        scat = dest.scatter(0, idx, src)
        self.assertEqual(scat.tolist(), tscat.tolist())
        scat2 = dest.overlay[idx, {}, {}](src)
        self.assertEqual(scat.tolist(), scat2.tolist())

    # TODO other ragged tests
    def test_scatter_ragged_mid(self):
        dest = rand(4, [4, 3, 2, 1], 10)
        src = zeros().expand(4, [4, 3, 2, 1], 5)
        idx = array(sample([(10, 5)] * 10)).reshape(4, [4, 3, 2, 1], 5)
        scat = dest.scatter(-1, idx, src)
        scat2 = dest.overlay[{}, {}, idx](src)
        self.assertEqual(scat.tolist(), scat2.tolist())


class TestScalarGetitemContiguous(TestCase):
    def test_scalar_getitem_contiguous(self):
        def check(a):
            for i in range(a.shape[0][0]):
                for j in range(a.shape[1][i]):
                    x = a[i, j]
                    self.assertTrue(x.data is a.data)
                    self.assertEqual(x.shape, Shape())
                    self.assertEqual(x.strides, Strides())
                    self.assertEqual(x.offset, a.shape[1].offset_of(i) + j)
                    y = x.eval()
                    self.assertEqual(y.item(), x.item())
                    self.assertEqual(y.shape, x.shape)
                    self.assertEqual(y.strides, x.strides)
                    self.assertEqual(y.offset, 0)

        check(arange(10, 10))
        check(arange(4, [4, 3, 2, 1]))

    def test_scalar_getitem_contiguous_neg(self):
        def check(a):
            for i in range(1, a.shape[0][0] + 1):
                for j in range(1, a.shape[1][-i] + 1):
                    x = a[-i, -j]
                    self.assertTrue(x.data is a.data)
                    self.assertEqual(x.shape, Shape())
                    self.assertEqual(x.strides, Strides())
                    self.assertEqual(
                        x.offset, a.shape[1].offset_of(len(a.shape[1]) + 1 - i) - j
                    )
                    y = x.eval()
                    self.assertEqual(y.item(), x.item())
                    self.assertEqual(y.shape, x.shape)
                    self.assertEqual(y.strides, x.strides)
                    self.assertEqual(y.offset, 0)

        check(arange(10, 10))
        check(arange(4, [4, 3, 2, 1]))


class TestBuilders(TestCase):
    def test_ctor(self):
        # quick spot check of Array constructor and defaults
        t = torch.rand(100)
        a = array(t)
        self.assertTrue(a.data is t)
        self.assertEqual(a.shape, Shape(100))
        self.assertEqual(a.strides, Strides(Rect(1, 100)))
        self.assertEqual(a.offset, 0)

    def test_arange(self):
        a = arange(10, 10)
        self.assertTrue(a.data.equal(torch.arange(100)))
        self.assertEqual(a.shape, Shape(10, 10))
        self.assertEqual(a.strides, Strides(Rect(10, 10), Rect(1, 100)))
        self.assertEqual(a.offset, 0)

    # TODO others


class TestEdgeMisc(TestCase):
    def test_zero_dim_getitem(self):
        z = arange()
        self.assertEqual(z.__getitem__(()).tolist(), z.item())


class TestSliceMisc(TestCase):
    def test_empty_slice(self):
        m = arange(8, 8)
        x = m[:0]
        self.assertEqual(len(x), 0)

    def test_descending_slice_1(self):
        v = arange(8)[::-1]
        x = array(range(8)[::-1])
        self.assertEqual(v.tolist(), x.tolist())

    def test_descending_slice_2(self):
        m = arange(3, 3)
        self.assertEqual(m[::-1].tolist(), [[6, 7, 8], [3, 4, 5], [0, 1, 2]])
        self.assertEqual(m[:, ::-1].tolist(), [[2, 1, 0], [5, 4, 3], [8, 7, 6]])
        self.assertEqual(m[::-1, ::-1].tolist(), [[8, 7, 6], [5, 4, 3], [2, 1, 0]])

    def test_descending_slice_3(self):
        t = arange(4, Affine(4, 4, -1))  # triu(4)
        self.assertEqual(t[::-1].tolist(), [[9], [7, 8], [4, 5, 6], [0, 1, 2, 3]])
        self.assertEqual(t[:, ::-1].tolist(), [[3, 2, 1, 0], [6, 5, 4], [8, 7], [9]])
        self.assertEqual(t[::-1, ::-1].tolist(), [[9], [8, 7], [6, 5, 4], [3, 2, 1, 0]])

    def test_skip_slice_1(self):
        v = arange(8)[::2]
        x = array(range(8)[::2])
        self.assertEqual(v.tolist(), x.tolist())

    def test_skip_slice_2(self):
        m = arange(4, 4)
        self.assertEqual(m[::2].tolist(), [[0, 1, 2, 3], [8, 9, 10, 11]])
        self.assertEqual(m[:, ::2].tolist(), [[0, 2], [4, 6], [8, 10], [12, 14]])
        self.assertEqual(m[::2, ::2].tolist(), [[0, 2], [8, 10]])

    def test_skip_slice_3(self):
        t = arange(4, Affine(4, 4, -1))  # triu(4)
        self.assertEqual(t[::2].tolist(), [[0, 1, 2, 3], [7, 8]])
        self.assertEqual(t[:, ::2].tolist(), [[0, 2], [4, 6], [7], [9]])
        self.assertEqual(t[::2, ::2].tolist(), [[0, 2], [7]])


class TestOverlayBasics(TestCase):
    #
    # tests:
    # - __getitem__/tolist gives the correct values
    # - slice crossing both source and target gives the correct values
    # - slice in source/dest-only regions gives an array over source/target data
    # - clone gives an array with the correct values
    # - eval updates target, or correctly raises non-injective error
    # - other target updates are observed
    # - other source updates are observed
    #
    # test all combinations of:
    # {contiguous, strided} x {source, target, both}
    # {rect, ragged} (both sides)
    #
    def test_rect_contig_1(self):
        m = arange(3, 3)
        c = arange(3)
        ov = m.overlay[:, 1](c)
        ref = [[0, 0, 2], [3, 1, 5], [6, 2, 8]]
        self.assertEqual(ov.tolist(), ref)
        ovrow = ov[0]
        self.assertEqual(ovrow.tolist(), ref[0])
        ov_destonly = ov[{}, [0, 2]]
        self.assertTrue(ov_destonly.data is m.data)
        ov_srconly = ov[:, 1]
        self.assertTrue(ov_srconly.data is c.data)
        cl = ov.clone()
        self.assertIsInstance(cl, Array)
        self.assertEqual(cl.tolist(), ref)
        self.assertEqual(m, arange(3, 3))
        ov.eval()
        self.assertEqual(m.tolist(), ref)
        m[0, 0] = 999
        self.assertEqual(ov[0, 0].tolist(), 999)
        c[1] = 999
        self.assertEqual(ov[1, 1].tolist(), 999)

    def test_rect_contig_2(self):
        m = arange(3, 3)
        n = arange(2, 2)
        ov = m.overlay[1:, 1:](n)
        ref = [[0, 1, 2], [3, 0, 1], [6, 2, 3]]
        self.assertEqual(ov.tolist(), ref)
        self.assertEqual(ov[0].tolist(), ref[0])
        self.assertTrue(ov[{}, [[0, 1, 2], [0], [0]]].data is m.data)
        self.assertTrue(ov[1:, 1:].data is n.data)
        cl = ov.clone()
        self.assertIsInstance(cl, Array)
        self.assertEqual(cl.tolist(), ref)
        self.assertEqual(m, arange(3, 3))
        ov.eval()
        self.assertEqual(m.tolist(), ref)
        m[0, 0] = 999
        self.assertEqual(ov[0, 0].tolist(), 999)
        n[1, 1] = 999
        self.assertEqual(ov[2, 2].tolist(), 999)

    def test_rect_strided_source_1(self):
        m = arange(3, 3)
        c = arange().expand(3)
        ov = m.overlay[:, 1](c)
        ref = [[0, 0, 2], [3, 0, 5], [6, 0, 8]]
        self.assertEqual(ov.tolist(), ref)
        self.assertEqual(ov[0].tolist(), ref[0])
        self.assertTrue(ov[{}, [0, 2]].data is m.data)
        self.assertTrue(ov[:, 1].data is c.data)
        cl = ov.clone()
        self.assertIsInstance(cl, Array)
        self.assertEqual(cl.tolist(), ref)
        self.assertEqual(m, arange(3, 3))
        ov.eval()
        self.assertEqual(m.tolist(), ref)
        m[0, 0] = 999
        self.assertEqual(ov[0, 0].tolist(), 999)

    def test_rect_strided_source_2(self):
        m = arange(3, 3)
        n = arange().expand(2, 2)
        ov = m.overlay[1:, 1:](n)
        ref = [[0, 1, 2], [3, 0, 0], [6, 0, 0]]
        self.assertEqual(ov.tolist(), ref)
        self.assertEqual(ov[0].tolist(), ref[0])
        self.assertTrue(ov[{}, [[0, 1, 2], [0], [0]]].data is m.data)
        self.assertTrue(ov[1:, 1:].data is n.data)
        cl = ov.clone()
        self.assertIsInstance(cl, Array)
        self.assertEqual(cl.tolist(), ref)
        self.assertEqual(m, arange(3, 3))
        ov.eval()
        self.assertEqual(m.tolist(), ref)
        m[0, 0] = 999
        self.assertEqual(ov[0, 0].tolist(), 999)

    def test_rect_strided_target_1(self):
        m = arange().expand(3, 3)
        c = arange(3)
        ov = m.overlay[:, 1](c)
        ref = [[0, 0, 0], [0, 1, 0], [0, 2, 0]]
        self.assertEqual(ov.tolist(), ref)
        self.assertEqual(ov[0].tolist(), ref[0])
        self.assertTrue(ov[{}, [0, 2]].data is m.data)
        self.assertTrue(ov[:, 1].data is c.data)
        cl = ov.clone()
        self.assertIsInstance(cl, Array)
        self.assertEqual(cl.tolist(), ref)
        self.assertEqual(m, arange().expand(3, 3))
        self.assertRaisesRegex(ValueError, "can't set items", ov.eval)
        c[1] = 999
        self.assertEqual(ov[1, 1].tolist(), 999)

    def test_rect_strided_target_2(self):
        m = arange().expand(3, 3)
        n = arange(2, 2)
        ov = m.overlay[1:, 1:](n)
        ref = [[0, 0, 0], [0, 0, 1], [0, 2, 3]]
        self.assertEqual(ov.tolist(), ref)
        self.assertEqual(ov[0].tolist(), ref[0])
        self.assertTrue(ov[{}, [[0, 1, 2], [0], [0]]].data is m.data)
        self.assertTrue(ov[1:, 1:].data is n.data)
        cl = ov.clone()
        self.assertEqual(cl.tolist(), ref)
        self.assertEqual(m, arange().expand(3, 3))
        self.assertRaisesRegex(ValueError, "can't set items", ov.eval)
        n[1, 1] = 999
        self.assertEqual(ov[2, 2].tolist(), 999)

    def test_rect_strided_target_and_source_1(self):
        m = arange().expand(3, 3)
        c = arange(start=1).expand(3)
        o = m.overlay[:, 1](c)
        ref = [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
        self.assertEqual(o.tolist(), ref)
        self.assertEqual(o[0].tolist(), ref[0])
        self.assertTrue(o[{}, [0, 2]].data is m.data)
        self.assertTrue(o[:, 1].data is c.data)
        cl = o.clone()
        self.assertIsInstance(cl, Array)
        self.assertEqual(cl.tolist(), ref)
        self.assertEqual(m, arange().expand(3, 3))
        self.assertRaisesRegex(ValueError, "can't set items", o.eval)

    def test_rect_strided_target_and_source_2(self):
        m = arange().expand(3, 3)
        n = arange(start=1).expand(2, 2)
        o = m.overlay[1:, 1:](n)
        ref = [[0, 0, 0], [0, 1, 1], [0, 1, 1]]
        self.assertEqual(o.tolist(), ref)
        self.assertEqual(o[0].tolist(), ref[0])
        self.assertTrue(o[{}, [[0, 1, 2], [0], [0]]].data is m.data)
        self.assertTrue(o[1:, 1:].data is n.data)
        cl = o.clone()
        self.assertIsInstance(cl, Array)
        self.assertEqual(cl.tolist(), ref)
        self.assertEqual(m, arange().expand(3, 3))
        self.assertRaisesRegex(ValueError, "can't set items", o.eval)

    def test_ragged_contig_1(self):
        m = arange(3, Affine(3, 3, -1))
        c = arange(2)
        ov = m.overlay[:2, 1](c)
        ref = [[0, 0, 2], [3, 1], [5]]
        self.assertEqual(ov.tolist(), ref)
        self.assertEqual(ov[0].tolist(), ref[0])
        self.assertTrue(ov[{}, [[0, 2], [0], [0]]].data is m.data)
        self.assertTrue(ov[:-1, 1].data is c.data)
        cl = ov.clone()
        self.assertIsInstance(cl, Array)
        self.assertEqual(cl.tolist(), ref)
        self.assertEqual(m, arange(3, Affine(3, 3, -1)))
        ov.eval()
        self.assertEqual(m.tolist(), ref)
        m[0, 0] = 999
        self.assertEqual(ov[0, 0].tolist(), 999)
        c[1] = 999
        self.assertEqual(ov[1, 1].tolist(), 999)

    def test_ragged_contig_2(self):
        m = arange(3, Affine(3, 3, -1))
        n = arange(2, Affine(2, 2, -1))
        ov = m.overlay[:-1, 1:](n)
        ref = [[0, 0, 1], [3, 2], [5]]
        self.assertEqual(ov.tolist(), ref)
        self.assertEqual(ov[0].tolist(), ref[0])
        self.assertTrue(ov[:, 0].data is m.data)
        self.assertTrue(ov[:-1, 1:].data is n.data)
        cl = ov.clone()
        self.assertIsInstance(cl, Array)
        self.assertEqual(cl.tolist(), ref)
        self.assertEqual(m, arange(3, Affine(3, 3, -1)))
        ov.eval()
        self.assertEqual(m.tolist(), ref)
        m[0, 0] = 999
        self.assertEqual(ov[0, 0].tolist(), 999)
        n[1, 0] = 999
        self.assertEqual(ov[1, 1].tolist(), 999)

    def test_ragged_strided_source_1(self):
        m = arange(3, Affine(3, 3, -1))
        c = arange().expand(2)
        ov = m.overlay[:2, 1](c)
        ref = [[0, 0, 2], [3, 0], [5]]
        self.assertEqual(ov.tolist(), ref)
        self.assertEqual(ov[0].tolist(), ref[0])
        self.assertTrue(ov[{}, [[0, 2], [0], [0]]].data is m.data)
        self.assertTrue(ov[:-1, 1].data is c.data)
        cl = ov.clone()
        self.assertIsInstance(cl, Array)
        self.assertEqual(cl.tolist(), ref)
        self.assertEqual(m, arange(3, Affine(3, 3, -1)))
        ov.eval()
        self.assertEqual(m.tolist(), ref)
        m[0, 0] = 999
        self.assertEqual(ov[0, 0].tolist(), 999)

    def test_ragged_strided_source_2(self):
        m = arange(3, Affine(3, 3, -1))
        n = arange().expand(2, Affine(2, 2, -1))
        ov = m.overlay[:-1, 1:](n)
        ref = [[0, 0, 0], [3, 0], [5]]
        self.assertEqual(ov.tolist(), ref)
        self.assertEqual(ov[0].tolist(), ref[0])
        self.assertTrue(ov[:, 0].data is m.data)
        self.assertTrue(ov[:-1, 1:].data is n.data)
        cl = ov.clone()
        self.assertIsInstance(cl, Array)
        self.assertEqual(cl.tolist(), ref)
        self.assertEqual(m, arange(3, Affine(3, 3, -1)))
        ov.eval()
        self.assertEqual(m.tolist(), ref)
        m[0, 0] = 999
        self.assertEqual(ov[0, 0].tolist(), 999)

    def test_ragged_strided_target_1(self):
        m = arange().expand(3, Affine(3, 3, -1))
        c = arange(2, start=1)
        ov = m.overlay[:2, 1](c)
        ref = [[0, 1, 0], [0, 2], [0]]
        self.assertEqual(ov.tolist(), ref)
        self.assertEqual(ov[0].tolist(), ref[0])
        self.assertTrue(ov[{}, [[0, 2], [0], [0]]].data is m.data)
        self.assertTrue(ov[:-1, 1].data is c.data)
        cl = ov.clone()
        self.assertIsInstance(cl, Array)
        self.assertEqual(cl.tolist(), ref)
        self.assertEqual(m, arange().expand(3, Affine(3, 3, -1)))
        self.assertRaisesRegex(ValueError, "can't set items", ov.eval)
        c[1] = 999
        self.assertEqual(ov[1, 1].tolist(), 999)

    def test_ragged_strided_target_2(self):
        m = arange().expand(3, Affine(3, 3, -1))
        n = arange(2, Affine(2, 2, -1), start=1)
        ov = m.overlay[:-1, 1:](n)
        ref = [[0, 1, 2], [0, 3], [0]]
        self.assertEqual(ov.tolist(), ref)
        self.assertEqual(ov[0].tolist(), ref[0])
        self.assertTrue(ov[:, 0].data is m.data)
        self.assertTrue(ov[:-1, 1:].data is n.data)
        cl = ov.clone()
        self.assertIsInstance(cl, Array)
        self.assertEqual(cl.tolist(), ref)
        self.assertEqual(m, arange().expand(3, Affine(3, 3, -1)))
        self.assertRaisesRegex(ValueError, "can't set items", ov.eval)
        n[1, 0] = 999
        self.assertEqual(ov[1, 1].tolist(), 999)

    def test_ragged_strided_target_and_source_1(self):
        m = arange().expand(3, Affine(3, 3, -1))
        c = arange(start=1).expand(2)
        o = m.overlay[:2, 1](c)
        ref = [[0, 1, 0], [0, 1], [0]]
        self.assertEqual(o.tolist(), ref)
        self.assertEqual(o[0].tolist(), ref[0])
        self.assertTrue(o[{}, [[0, 2], [0], [0]]].data is m.data)
        self.assertTrue(o[:-1, 1].data is c.data)
        cl = o.clone()
        self.assertIsInstance(cl, Array)
        self.assertEqual(cl.tolist(), ref)
        self.assertEqual(m, arange().expand(3, Affine(3, 3, -1)))
        self.assertRaisesRegex(ValueError, "can't set items", o.eval)

    def test_ragged_strided_target_and_source_2(self):
        m = arange().expand(3, Affine(3, 3, -1))
        n = arange(start=1).expand(2, Affine(2, 2, -1))
        o = m.overlay[:-1, 1:](n)
        ref = [[0, 1, 1], [0, 1], [0]]
        self.assertEqual(o.tolist(), ref)
        self.assertEqual(o[0].tolist(), ref[0])
        self.assertTrue(o[:, 0].data is m.data)
        self.assertTrue(o[:-1, 1:].data is n.data)
        cl = o.clone()
        self.assertIsInstance(cl, Array)
        self.assertEqual(cl.tolist(), ref)
        self.assertEqual(m, arange().expand(3, Affine(3, 3, -1)))
        self.assertRaisesRegex(ValueError, "can't set items", o.eval)

    def test_simple_1(self):
        m = fill(4, 4, value=99)
        n = arange(2, 2)
        o = m.overlay[1:3, 1:3](n)
        ref = [[99, 99, 99, 99], [99, 0, 1, 99], [99, 2, 3, 99], [99, 99, 99, 99]]
        self.assertEqual(o.tolist(), ref)
        m[1:3, 1:3] = n
        self.assertEqual(m.tolist(), ref)
        self.assertEqual(m, o.clone())
        cl = o.clone()


class TestUnsqueeze(TestCase):
    def test_contiguous_unsqueeze(self):
        m = arange(8, 8)
        self.assertEqual(m.unsqueeze(2), arange(8, 8, 1))
        self.assertEqual(m.unsqueeze(1), arange(8, 1, 8))
        self.assertEqual(m.unsqueeze(0), arange(1, 8, 8))
        self.assertEqual(m.unsqueeze(-1), arange(8, 8, 1))
        self.assertEqual(m.unsqueeze(-2), arange(8, 1, 8))
        self.assertEqual(m.unsqueeze(-3), arange(1, 8, 8))


class TestReshape(TestCase):
    def test_contiguous_reshape_rect(self):
        m = arange(4, 4)
        n = m.reshape(2, 2, 2, 2)
        self.assertEqual(n[1, 1, 1, 1].item(), 15)
        n[1, 1, 1, 1] = array(99, dtype=torch.long)
        self.assertEqual(m[-1, -1].item(), 99)

    def test_contiguous_reshape_tri(self):
        triu = arange(8, Affine(8, 8, -1))
        self.assertEqual(triu[0], arange(8))
        tril = triu.reshape(8, Affine(1, 8, 1))
        triu[-1, -1] = array(99, dtype=torch.long)
        self.assertEqual(triu[-1, -1], tril[-1, -1])

    def test_neg1_reshape_rect(self):
        m = arange(3, 4, 5)
        self.assertEqual(m.reshape(-1).shape, Shape(60))
        self.assertEqual(m.reshape(-1, 3).shape, Shape(20, 3))
        self.assertEqual(m.reshape(-1, 4).shape, Shape(15, 4))
        self.assertEqual(m.reshape(-1, 5).shape, Shape(12, 5))
        self.assertEqual(m.reshape(3, -1).shape, Shape(3, 20))
        self.assertEqual(m.reshape(4, -1).shape, Shape(4, 15))
        self.assertEqual(m.reshape(5, -1).shape, Shape(5, 12))
        self.assertEqual(m.reshape(-1, 4, 5).shape, Shape(3, 4, 5))
        self.assertEqual(m.reshape(3, -1, 5).shape, Shape(3, 4, 5))
        self.assertEqual(m.reshape(3, 4, -1).shape, Shape(3, 4, 5))
        self.assertEqual(m.reshape(-1, 2, 2, 5).shape, Shape(3, 2, 2, 5))
        self.assertEqual(m.reshape(3, -1, 2, 5).shape, Shape(3, 2, 2, 5))
        self.assertEqual(m.reshape(3, 2, -1, 5).shape, Shape(3, 2, 2, 5))
        self.assertEqual(m.reshape(3, 2, 2, -1).shape, Shape(3, 2, 2, 5))

    def test_neg1_reshape_over_ragged(self):
        # spot checks
        m = arange(2, 3, 4)
        self.assertEqual(m.reshape(-1, [1, 2, 3]).shape, Shape(12, [1, 2, 3]))
        self.assertEqual(m.reshape(-1, [0, 1, 2]).shape, Shape(24, [0, 1, 2]))
        self.assertEqual(m.reshape(2, -1, [1, 2, 3]).shape, Shape(2, 6, [1, 2, 3]))
        self.assertEqual(m.reshape(4, -1, [1, 2, 3]).shape, Shape(4, 3, [1, 2, 3]))
        self.assertEqual(m.reshape(2, [1, 2], -1).shape, Shape(2, [1, 2], 8))
        self.assertEqual(
            m.reshape(4, [1, 2, 3, 6], -1).shape, Shape(4, [1, 2, 3, 6], 2)
        )

    def test_neg1_reshape_ragged_3d(self):
        a = arange(3, 4, [4, 5, 3, 2, 6, 1, 3, 7, 5, 3, 4, 1])
        b = a.reshape(12, -1)
        self.assertEqual(b.shape, Shape(12, a.shape[-1]))

    def test_neg1_reshape_ragged_4d(self):
        a = arange(3, 4, [4, 5, 3, 2, 6, 1, 3, 7, 5, 3, 4, 1], 10)
        b = a.reshape(12, -1, 10)
        self.assertEqual(b.shape, Shape(12, a.shape[-2], 10))


class TestIota(TestCase):
    def test_iota_rect(self):
        shape = Shape(3, 4, 5)
        x = iota(*shape, axis=0)
        self.assertEqual(x.shape, shape)
        self.assertEqual(x.reshape(-1).tolist(), [0] * 20 + [1] * 20 + [2] * 20)
        y = iota(*shape, axis=1)
        self.assertEqual(y.shape, shape)
        self.assertEqual(
            y.reshape(-1).tolist(), ([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5) * 3
        )
        z = iota(*shape, axis=2)
        self.assertEqual(z.shape, shape)
        self.assertEqual(z.reshape(-1).tolist(), [i for i in range(5)] * 12)

    def test_iota_affine(self):
        shape = Shape(6, 8, Affine(8, 8, -1))
        x = iota(*shape, axis=0)
        self.assertEqual(x.shape, shape)
        self.assertEqual(
            x.reshape(-1).tolist(), [i for i in range(6) for _ in range(36)]
        )
        y = iota(*shape, axis=1)
        self.assertEqual(y.shape, shape)
        self.assertEqual(
            y.reshape(-1).tolist(),
            [j for _ in range(6) for j in range(8) for _ in range(0, 8 - j)],
        )
        z = iota(*shape, axis=2)
        self.assertEqual(z.shape, shape)
        self.assertEqual(
            z.reshape(-1).tolist(),
            [k for _ in range(6) for j in range(8) for k in range(0, 8 - j)],
        )

    def test_iota_sparse(self):
        shape = Shape(2, 8, Sparse({1: 2, 4: 3, 5: 1, 7: 2, 10: 5, 13: 2, 16: -1}))
        x = iota(*shape, axis=0)
        self.assertEqual(x.shape, shape)
        self.assertEqual(
            x.tolist(),
            [
                [[], [0, 0], [], [], [0, 0, 0], [0], [], [0, 0]],
                [[], [], [1, 1, 1, 1, 1], [], [], [1, 1], [], []],
            ],
        )
        y = iota(*shape, axis=1)
        self.assertEqual(y.shape, shape)
        self.assertEqual(
            y.tolist(),
            [
                [[], [1, 1], [], [], [4, 4, 4], [5], [], [7, 7]],
                [[], [], [2, 2, 2, 2, 2], [], [], [5, 5], [], []],
            ],
        )
        z = iota(*shape, axis=2)
        self.assertEqual(z.shape, shape)
        self.assertEqual(
            z.tolist(),
            [
                [[], [0, 1], [], [], [0, 1, 2], [0], [], [0, 1]],
                [[], [], [0, 1, 2, 3, 4], [], [], [0, 1], [], []],
            ],
        )


class TestExpand(TestCase):
    def test_expand_rect_contiguous_inner(self):
        x = arange(4, 4).unsqueeze(-1).expand(-1, -1, 3)
        self.assertEqual(x.shape, Shape(4, 4, 3))
        vals = [i for i in range(16) for _ in range(3)]
        self.assertEqual(x.reshape(-1).tolist(), vals)
        s = Strides(Rect(4, 4), Rect(1, 16), Repeat(Runs([0, 1], [2, 1]), 16))
        self.assertEqual(x.strides, s)

    def test_expand_rect_contiguous_middle(self):
        x = arange(4, 4).unsqueeze(1).expand(-1, 3, -1)
        self.assertEqual(x.shape, Shape(4, 3, 4))
        vals = [k for i in range(4) for _ in range(3) for k in range(i * 4, i * 4 + 4)]
        self.assertEqual(x.reshape(-1).tolist(), vals)
        s = Strides(
            Rect(4, 4),
            Repeat(Runs([0, 4], [2, 1]), 4),
            Repeat(Chain([Repeat(Runs([1, -3], [3, 1]), 2), Rect(1, 4)]), 4),
        )
        self.assertEqual(x.strides, s)

    def test_expand_rect_contiguous_outer(self):
        x = arange(4, 4).unsqueeze(0).expand(3, -1, -1)
        self.assertEqual(x.shape, Shape(3, 4, 4))
        vals = [j for i in range(3) for j in range(16)]
        self.assertEqual(x.reshape(-1).tolist(), vals)
        s = Strides(
            Rect(0, 3),
            Repeat(Runs([4, -12], [3, 1]), 3),
            Repeat(Runs([1, -15], [15, 1]), 3),
        )
        self.assertEqual(x.strides, s)

    def test_expand_rect_contiguous_multi(self):
        x = arange(5, 7).unsqueeze(1).unsqueeze(-1).expand(4, -1, 6, -1, 8)
        self.assertEqual(x.shape, Shape(4, 5, 6, 7, 8))
        # vals = [j for i in range(3) for j in range(16)]
        # self.assertEqual(x.reshape(-1).tolist(), vals)
        s = Strides(
            Rect(0, 4),
            Repeat(Runs([7, -28], [4, 1]), 4),
            Repeat(Chain([Repeat(Runs([0, 7], [5, 1]), 4), Runs([0, -28], [5, 1])]), 4),
            Repeat(
                Chain(
                    [
                        Repeat(
                            Chain([Repeat(Runs([1, -6], [6, 1]), 5), Rect(1, 7)]), 4
                        ),
                        Repeat(Runs([1, -6], [6, 1]), 5),
                        Runs([1, -34], [6, 1]),
                    ]
                ),
                4,
            ),
            Repeat(
                Chain(
                    [
                        Repeat(
                            Chain(
                                [
                                    Repeat(
                                        Chain(
                                            [
                                                Repeat(Runs([0, 1], [7, 1]), 6),
                                                Runs([0, -6], [7, 1]),
                                            ]
                                        ),
                                        5,
                                    ),
                                    Repeat(Runs([0, 1], [7, 1]), 7),
                                ]
                            ),
                            4,
                        ),
                        Repeat(
                            Chain(
                                [Repeat(Runs([0, 1], [7, 1]), 6), Runs([0, -6], [7, 1])]
                            ),
                            5,
                        ),
                        Repeat(Runs([0, 1], [7, 1]), 6),
                        Runs([0, -34], [7, 1]),
                    ]
                ),
                4,
            ),
        )
        self.assertEqual(x.strides, s)

    def test_expand_tri_contiguous_inner(self):
        x = arange(4, Affine(4, 4, -1)).unsqueeze(-1).expand(-1, -1, 3)
        self.assertEqual(x.shape, Shape(4, Affine(4, 4, -1), 3))
        vals = [i for i in range(10) for _ in range(3)]
        self.assertEqual(x.reshape(-1).tolist(), vals)
        s = Strides(Affine(4, 4, -1), Rect(1, 10), Repeat(Runs([0, 1], [2, 1]), 10))
        self.assertEqual(x.strides, s)

    def test_expand_tri_contiguous_middle(self):
        x = arange(4, Affine(4, 4, -1)).unsqueeze(1).expand(-1, 3, -1)
        self.assertEqual(x.shape, Shape(4, 3, Runs(Affine(4, 4, -1), 3)))
        vals = [
            k
            for i in [(4, 0), (3, 4), (2, 7), (1, 9)]
            for _ in range(3)
            for k in range(i[1], i[1] + i[0])
        ]

        self.assertEqual(x.reshape(-1).tolist(), vals)
        s = Strides(
            Affine(4, 4, -1),
            Runs([0, 4, 0, 3, 0, 2, 0, 1], Repeat([2, 1], 4)),
            Chain(
                [
                    Repeat(Runs([1, -3], [3, 1]), 2),
                    Rect(1, 4),
                    Repeat(Runs([1, -2], [2, 1]), 2),
                    Rect(1, 3),
                    Repeat([1, -1], 2),
                    Runs([1, 0, 1], [2, 2, 1]),
                ]
            ),
        )
        self.assertEqual(x.strides, s)

    def test_expand_tri_contiguous_outer(self):
        x = arange(4, Affine(4, 4, -1)).unsqueeze(0).expand(3, -1, -1)
        self.assertEqual(x.shape, Shape(3, 4, Repeat(Affine(4, 4, -1), 3)))
        vals = [i for _ in range(3) for i in range(10)]
        self.assertEqual(x.reshape(-1).tolist(), vals)
        s = Strides(
            Rect(0, 3),
            Repeat(Chain([Affine(4, 3, -1), -9]), 3),
            Repeat(Runs([1, -9], [9, 1]), 3),
        )
        self.assertEqual(x.strides, s)

    # TODO add permuted


class TestBroadcasting(TestCase):
    @staticmethod
    def try_broadcast_to(a: Array, *dims: RawDim) -> Optional[Array]:
        try:
            return a.broadcast_to(*dims)
        except ValueError as e:
            if str(e).find("must match existing size") > 0:
                return None
            raise e

    def check(self, a, shape, target):
        if isinstance(a, tuple):
            a = arange(*a)
        if isinstance(target, tuple):
            target = Shape(*target)
        bc = self.try_broadcast_to(a, *shape)
        if target is None:
            self.assertIsNone(bc)
        elif isinstance(target, Array):
            self.assertEqual(bc.tolist(), target.tolist())
        else:  # shape
            self.assertEqual(bc.shape, target)

    def test_broadcast_shape_only(self):
        # TODO these are initial random examples... should add data checking
        self.check((3, 1, 5), (3, 4, 5), (3, 4, 5))
        self.check((1, 3, 4, 5), (99, 2, -1, -1, 5), (99, 2, 3, 4, 5))
        self.check((3, [3, 4, 5], 5), (2, 3, -1, 5), (2, 3, [3, 4, 5], 5))
        self.check((3, [3, 2, 1], 5), (2, 3, [3, 2, 1], 5), (2, 3, [3, 2, 1], 5))
        self.check((5, 1), (5, [2, 3, 1, 0, 6]), (5, [2, 3, 1, 0, 6]))
        self.check((3, [1, 2, 3], 5), (3, [3, 2, 1], 5), None)
        self.check((3, 1, [1, 2, 3]), (-1, 3, -1), (3, 3, Runs([1, 2, 3], 3)))

        # scalar
        self.check((), (4,), (4,))
        self.check((), (5, 4), (5, 4))
        self.check((), (2, 5, 4), (2, 5, 4))
        self.check((), (3, [2, 3, 4], 4), (3, [2, 3, 4], 4))
        self.check((), (2, 3, [2, 3, 4], 4), (2, 3, [2, 3, 4], 4))
        self.check((), (2, [1, 2], [1, 2, 3]), (2, [1, 2], [1, 2, 3]))

        # vector
        self.check((4,), (5, 4), (5, 4))
        self.check((4,), (2, 5, 4), (2, 5, 4))
        self.check((4,), (3, [2, 3, 4], 4), (3, [2, 3, 4], 4))
        self.check((4,), (2, 3, [2, 3, 4], 4), (2, 3, [2, 3, 4], 4))
        self.check((4,), (2, 3, [2, 3, 4], 4), (2, 3, [2, 3, 4], 4))
        self.check((4,), (2, [1, 2], [1, 2, 3], 4), (2, [1, 2], [1, 2, 3], 4))

        # column vector
        self.check((3, 1), (3, 4), (3, 4))
        self.check((3, 1), (2, 3, 4), (2, 3, 4))
        self.check((3, 1), (3, [3, 4, 5]), (3, [3, 4, 5]))
        self.check((3, 1), (2, 3, [3, 4, 5]), (2, 3, [3, 4, 5]))
        self.check((3, 1), (2, 2, 3, [3, 4, 5]), (2, 2, 3, [3, 4, 5]))
        self.check((3, 1), (2, [1, 2], 3, [3, 4, 5]), (2, [1, 2], 3, [3, 4, 5]))

        # mid singleton
        self.check((3, 1, 2), (3, 4, 2), (3, 4, 2))
        self.check((3, 1, 2), (2, 3, 4, 2), (2, 3, 4, 2))
        self.check((3, 1, 2), (3, [3, 4, 5], 2), (3, [3, 4, 5], 2))
        self.check((3, 1, 2), (2, 3, [3, 4, 5], 2), (2, 3, [3, 4, 5], 2))
        self.check(
            (3, 1, 2), (2, [1, 2], 3, [3, 4, 5], 2), (2, [1, 2], 3, [3, 4, 5], 2)
        )

        # ragged inner with mid singleton
        self.check((3, 1, [3, 4, 5]), (3, 3, -1), (3, 3, Runs([3, 4, 5], 3)))
        self.check((3, 1, [3, 4, 5]), (3, 3, [3, 4, 5]), None)
        self.check((3, 1, [3, 4, 5]), (2, 3, 3, -1), (2, 3, 3, Runs([3, 4, 5], 3)))

        #
        # with data
        #

    def test_broadcast_shape_and_data(self):
        # scalar
        a = arange()
        self.check(a, (3,), array([0, 0, 0]))
        self.check(a, (3, 4), zeros(3, 4, dtype=int))
        self.check(a, (2, 3, 4), zeros(2, 3, 4, dtype=int))
        self.check(a, (8, Affine(1, 8, 1), 4), zeros(8, Affine(1, 8, 1), 4, dtype=int))
        self.check(
            a, (2, 8, Affine(1, 8, 1), 4), zeros(2, 8, Affine(1, 8, 1), 4, dtype=int)
        )

        # vector
        l = [0, 1, 2, 3]
        a = arange(4)
        self.check(a, (3, 4), array([l] * 3))
        self.check(a, (2, 3, 4), array([[l] * 3] * 2))
        self.check(a, (3, [2, 3, 4], 4), array([[l] * 2, [l] * 3, [l] * 4]))
        self.check(a, (2, 2, [3, 4], 4), array([[[l] * 3, [l] * 4]] * 2))

        # column vector
        smear_rect = lambda l, n: [i * n for i in l]
        smear_ragged = lambda l, ns: [i * n for i, n in zip(l, ns)]
        l = [[0], [1], [2], [3]]
        a = arange(4, 1)
        self.check(a, (4, 4), array(smear_rect(l, 4)))
        self.check(
            a,
            (-1, Affine(1, 4, 1)),
            array(smear_ragged(l, [1, 2, 3, 4])),
        )
        self.check(a, (4, Affine(1, 4, 1)), array(smear_ragged(l, [1, 2, 3, 4])))

        # mid singleton
        # TODO modernize and add data check
        a = arange(3, 1, 2)
        self.check(a, (3, 4, 2), (3, 4, 2))
        self.check(a, (2, 3, 4, 2), (2, 3, 4, 2))
        self.check(a, (3, [3, 4, 5], 2), (3, [3, 4, 5], 2))
        self.check(a, (2, 3, [3, 4, 5], 2), (2, 3, [3, 4, 5], 2))
        self.check(a, (2, [1, 2], 3, [1, 2, 3], 2), (2, [1, 2], 3, [1, 2, 3], 2))

        # ragged inner with mid singleton
        self.check((3, 1, [3, 4, 5]), (3, 3, -1), (3, 3, Runs([3, 4, 5], 3)))
        self.check((3, 1, [3, 4, 5]), (3, 3, [3, 4, 5]), None)
        self.check((3, 1, [3, 4, 5]), (2, 3, 3, -1), (2, 3, 3, Runs([3, 4, 5], 3)))
        self.check(
            (3, 1, [3, 4, 5]),
            (2, 3, [1, 2, 3], -1),
            (2, 3, [1, 2, 3], Runs([3, 4, 5], [1, 2, 3])),
        )

        # multiple ragged, multiple singleton
        self.check(
            (2, 1, [1, 2], 1, [3, 4, 5]),
            (-1, 2, -1, 2, -1),
            (
                2,
                2,
                Runs([1, 2], 2),
                2,
                Chain([Repeat([3], 4), Repeat(Runs([4, 5], 2), 2)]),
            ),
        )

    def test_broadcast_arrays(self):
        arrays = [arange(3, 1, 1), arange(1, 4, 1), arange(1, 1, 5)]
        a, b, c = broadcast_arrays(*arrays)
        self.assertEqual(a, arange(3, 1, 1).expand(3, 4, 5))
        self.assertEqual(b, arange(1, 4, 1).expand(3, 4, 5))
        self.assertEqual(c, arange(1, 1, 5).expand(3, 4, 5))


class TestGetItemBroadcasting(TestCase):
    def test_rect_getitem(self):
        a = arange(8, 8, 8)
        x = array([[[0, 1], [8, 9]], [[64, 65], [72, 73]]])
        b = a[
            arange(2, 1, 1, const=True),
            arange(1, 2, 1, const=True),
            arange(1, 1, 2, const=True),
        ]
        self.assertEqual(b, x)
        c = a[arange(2, 1, 1), arange(2, 1), arange(2)]
        self.assertEqual(c, x)


class TestSetItemBroadcasting(TestCase):
    #
    #  spot checks on __setitem__/overlay with broadcasting and rhs promotion
    #
    def test_setitem_rect_broadcast_2d(self):
        m = arange(8, 8)
        m[:, [3, 4]] = [99, 100]
        self.assertEqual(
            m.tolist(),
            [
                [0, 1, 2, 99, 100, 5, 6, 7],
                [8, 9, 10, 99, 100, 13, 14, 15],
                [16, 17, 18, 99, 100, 21, 22, 23],
                [24, 25, 26, 99, 100, 29, 30, 31],
                [32, 33, 34, 99, 100, 37, 38, 39],
                [40, 41, 42, 99, 100, 45, 46, 47],
                [48, 49, 50, 99, 100, 53, 54, 55],
                [56, 57, 58, 99, 100, 61, 62, 63],
            ],
        )
        ov = arange(8, 8).overlay[:, [3, 4]]([99, 100])
        self.assertEqual(ov.eval().tolist(), m.tolist())

    def test_setitem_tri_broadcast_2d(self):
        m = arange(8, 8)
        m[:, Affine(1, 8, 1) :] = 0
        self.assertEqual(
            m.tolist(),
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [8, 9, 0, 0, 0, 0, 0, 0],
                [16, 17, 18, 0, 0, 0, 0, 0],
                [24, 25, 26, 27, 0, 0, 0, 0],
                [32, 33, 34, 35, 36, 0, 0, 0],
                [40, 41, 42, 43, 44, 45, 0, 0],
                [48, 49, 50, 51, 52, 53, 54, 0],
                [56, 57, 58, 59, 60, 61, 62, 63],
            ],
        )
        ov = arange(8, 8).overlay[:, Affine(1, 8, 1) :](0)
        self.assertEqual(ov.eval().tolist(), m.tolist())

    def test_setitem_tri_dest_2d(self):
        tr = arange(8, Affine(8, 8, -1))
        tr[:, 2:4] = 0
        self.assertEqual(
            tr.tolist(),
            [
                [0, 1, 0, 0, 4, 5, 6, 7],
                [8, 9, 0, 0, 12, 13, 14],
                [15, 16, 0, 0, 19, 20],
                [21, 22, 0, 0, 25],
                [26, 27, 0, 0],
                [30, 31, 0],
                [33, 34],
                [35],
            ],
        )
        ov = arange(8, Affine(8, 8, -1)).overlay[:, 2:4](0)
        self.assertEqual(ov.eval().tolist(), tr.tolist())


class TestPermute(TestCase):
    def check_rect_perm(self, shape, perm):
        a = arange(*shape).permute(*perm)
        t = torch.arange(a.numel()).reshape(*shape).permute(*perm)
        self.assertEqual(a.tolist(), t.tolist())

    def test_2d_rect(self):
        # TODO shape edge cases
        self.check_rect_perm((3, 5), (0, 1))
        self.check_rect_perm((3, 5), (1, 0))

    def test_3d_rect(self):
        self.check_rect_perm((3, 4, 5), (0, 1, 2))
        self.check_rect_perm((3, 4, 5), (0, 2, 1))
        self.check_rect_perm((3, 4, 5), (2, 0, 1))
        self.check_rect_perm((3, 4, 5), (1, 0, 2))
        self.check_rect_perm((3, 4, 5), (1, 2, 0))
        self.check_rect_perm((3, 4, 5), (2, 1, 0))


class TestIndexSlicing(TestCase):
    def test_diag_example(self):
        m = arange(3, 3)
        d = arange(3).broadcast_to(2, -1)
        self.assertEqual(m[tuple(d)], array([0, 4, 8]))


class TestOverlayAsIndex(TestCase):
    # TODO rn this relies on cloning the overlay into an array
    # before it gets too deep into the __getitem__ machinery.
    # still, it tests that the basic laws are in place.
    # once the shim is removed it'll stress Overlay's array
    # API more thoroughly - broadcasting, item iteration etc.
    def test_basics(self):
        m = arange(8, 8)
        ref = m[[0, 1, 0, 1]]
        i1 = ones(4, dtype=torch.long)
        i2 = zeros(2, dtype=torch.long)
        self.assertEqual(m[i1.overlay[::2](i2)], ref)
        self.assertEqual(m[i1].overlay[::2](m[i2]).tolist(), ref.tolist())


#
# TODO test_chunk
#


if __name__ == "__main__":
    main()
