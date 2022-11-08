# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from unittest import TestCase, main
from fold.shape import *

# shape-specific tests - not much here since
# Array/View tests stress Shape stuff extensively,
# but serves to bisect some obvious stuff.


class TestUnsqueeze(TestCase):
    def test_contiguous_unsqueeze(self):
        m = Shape(8, 8)
        self.assertEqual(m.unsqueeze(2), Shape(8, 8, 1))
        self.assertEqual(m.unsqueeze(1), Shape(8, 1, 8))
        self.assertEqual(m.unsqueeze(0), Shape(1, 8, 8))
        self.assertEqual(m.unsqueeze(-1), Shape(8, 8, 1))
        self.assertEqual(m.unsqueeze(-2), Shape(8, 1, 8))
        self.assertEqual(m.unsqueeze(-3), Shape(1, 8, 8))


class TestExpand(TestCase):
    def test_expand_rect_contiguous_inner(self):
        x = Shape(4, 4).unsqueeze(-1).expand(-1, -1, 3)
        self.assertEqual(x, Shape(4, 4, 3))

    def test_expand_rect_contiguous_middle(self):
        x = Shape(4, 4).unsqueeze(1).expand(-1, 3, -1)
        self.assertEqual(x, Shape(4, 3, 4))

    def test_expand_rect_contiguous_outer(self):
        x = Shape(4, 4).unsqueeze(0).expand(3, -1, -1)
        self.assertEqual(x, Shape(3, 4, 4))

    def test_expand_rect_contiguous_multi(self):
        x = Shape(5, 7).unsqueeze(1).unsqueeze(-1).expand(4, -1, 6, -1, 8)
        self.assertEqual(x, Shape(4, 5, 6, 7, 8))

    def test_expand_tri_contiguous_inner(self):
        x = Shape(4, (4, 4, -1)).unsqueeze(-1).expand(-1, -1, 3)
        self.assertEqual(x, Shape(4, (4, 4, -1), 3))

    def test_expand_tri_contiguous_middle(self):
        x = Shape(4, (4, 4, -1)).unsqueeze(1).expand(-1, 3, -1)
        self.assertEqual(x, Shape(4, 3, ((4, 4, -1), (3, 4))))

    def test_expand_tri_contiguous_outer(self):
        x = Shape(4, (4, 4, -1)).unsqueeze(0).expand(3, -1, -1)
        self.assertEqual(x, Shape(3, 4, ((4, 4, -1), 3)))


class TestBroadcast(TestCase):
    def check_broadcast(self, shapes, target):
        shape = Shape(*shapes[0])
        for sh in shapes[1:]:
            shape = shape.broadcast_to(*sh)
        self.assertEqual(shape, Shape(*target))

    def test_rect(self):
        shapes = [(3, 1, 1), (-1, 4, -1), (-1, -1, 5)]
        self.check_broadcast(shapes, (3, 4, 5))
        shapes = [(3, 1, 1), (-1, 4, -1), (-1, -1, [4, 3, 2, 1])]
        self.check_broadcast(shapes, (3, 4, [4, 3, 2, 1]))


if __name__ == "__main__":
    main()
