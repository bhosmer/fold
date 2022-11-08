# fold

A unified data model for multidimensional arrays.

Building on a foundation of **generalized shapes**, the **fold** project aims to factor the evolving multidimensional array repertiore - including advanced indexing, indexed assignments, strided storage, ragged arrays, views, sparsity and other structured storage techniques, scatter/gather, etc. - into a small collection of orthogonal concepts:

* **extraction** vs **injection** along an index: do we want to
  * _extract_ elements of array `a` at the locations in index `i` to form a new array: `b = a[i]`
  * _inject_ the elements of array `b` into array `a` at `i`'s locations: `a[i] = b`

* **materialization** vs **functionalization**: given an operation that produces an observable result, do we want to
  * perform the operation immediately and _materialize_ the result? 
  * defer the operation and provide a _functional_ interface through which the result can be observed piecemeal?

* **explicit** vs **implicit** indexing: given a set of locations we wish to express, should we
  * _explicitly_ provide the locations as values? or
  * give the locations _implicitly_, by placing some associated values at those locations?

To make this factoring concrete, we need the following foundations:

* a data model for **generalized shapes**. Much of the power of multidimensional arrays comes from the reification of shape into metadata, decoupling abstract shape from physical layout and letting us define many important operations as simple metadata transforms. But the conventional model - a tuple of positive integers, one per dimension - is limited to expressing (hyper)rectangles. We need to generalize this model to include representations of the nonrectangular shapes we'll need, while preserving the current representations of rectangular shapes as a subset.

* a data model for **generalized strides** that is expressive enough to represent _any_ traversal of underlying storage to visit the elements of an array of any generalized shape. This universality lets us exploit the derivative/integral relationship between _strides_ and _positions_ (analogous to the relationship between _extents_ and _offsets_ in shapes).

* a model for **encoded sequences**: by capturing regularity in shape and stride patterns, not only do these ensure that the quantity of metadata needed is correlated to entropy rather array size, interpreting per-dimension metadata as an encoded sequence is what lets express the shapes and strides of rectangular arrays in the usual way within our generalized abstraction.

## This repo 

This repo contains a prototype implementation of the **fold** array model, implemented in Python and using PyTorch Tensors as array backing stores (although full integration into PyTorch isn't attempted). 

For a working introduction, check out these [notebooks](https://github.com/bhosmer/fold/tree/main/notebooks), starting with the one on [generalized shapes](https://github.com/bhosmer/fold/blob/main/notebooks/fold_shapes.ipynb).




