
## try file splitting again
- earlier attempt hit a wall bc circular deps forced local imports

## finish dim ctor cutover
- port tests (maybe some minor internal use) to use class ctors
- remove extended literal support from dim()

## Overlay.__getindex__ is the source of lots of bad indexes

## dim method quality pass
- any method that falls back to returning a Seq() needs to be checked, some were written hastily

## get rid of int arg overrides in Dim api
- port callsites to pass the equivalent rect
- if getting length at the callsite is a hardship maybe do length extension in Dim super

## more sparse examples using overlay

## harden Array use of Dim as a backing store
- currently we make do with spot checks where Dim and torch.Tensor APIs differ, should probably wrap in some sort of Storage
- in particular we're papering over the lack of Dim.abs() in the Array prettyprinter
- using torch.Tensor as Dim backing store may present further opportunities for rationalization

## use torch.Tensor as Dim backing store
- would give us access to tensor utilities, be a first step to closing the loop on differentiability
- question remains for non-sequence data (affine coefficients, say)

## strides produced by view indexing are bad, expand strides are good
update to use stuff in expand/iota codepath - current impl produces uncompressed dims more or less all the time

## more examples
1. sparse layout factory functions etc
2. straight adv ind/coo
3. csr/csc
4. also diag and triu/tril
5. nonsparse structured layouts, circulants, lu thing, etc

## finish commenting

## Dim class hierarchy
1. dim.py getting pretty big, should probably be broken up per subclass.
2. audit methods - any old stuff we can cull? there's a lot
3. notes on some of the trickier impls

## permute
non-rectangular shapes: either/both of
a) shapes can be constructud successfully that aren't legit) and
b) legitimate shapes aren't handled properly by current iota-based
indexing approach.

## generalized affine dim?
strides from shapes with affine variation but >1 granularity (so e.g. trianges over a fixed inner dimension) have no natural compression approach. new dim class? 

## Overlay needs Array interface 
and maybe there should be a subclass relationship ugh

## Dim.intersect() 
1. need this for overlap inverse image on getitem
2. View.intersect() with shape-preserving pullback

## test_dim
negative values are untested currently. should be simple to add
mod at least one harness issue (using a generated dim as a partition)

## test_array
1. ragged shape in src
2. ragged shape in dest
3. overlay slicing (intersection)
4. overlay modify underlying
5. overlay splats
6. *any setitem test should have an overlay version*
7. *everything should have permuted/broadcast/other noncontiguous versions*
8. test_setitem for each dim, cycle through point, slice, linear points, shaped points
9. relaxed shape constraint in overlay
10. test expansion stride compression against equivalent iota() - should be the same, but currently expansion strides are lower quality (though correct)
