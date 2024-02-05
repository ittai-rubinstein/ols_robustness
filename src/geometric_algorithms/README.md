# Optimization Bounding Library

A library of algorithms for bounding the following optimization problems (where our goal is to bound the optimal solution from above).

## The Optimization Problems

### Problem 1 (`sum_ips`):
Given a set of vectors `X_i` for `i \in {1, ..., n}`, maximize `\sum_{i \in T} <X_i, v>` over `v` in `S^(d-1)` and `|T| = k`.

Note, that this optimization is equivalent to maximizing the `||\sum_{i\in T} X_i||`.

### Problem 2 (`sum_ips_squared`):
Given a set of vectors `X_i` for `i \in {1, ..., n}`, maximize `\sum_{i \in T} <X_i, v>^2` over `v` in `S^(d-1)` and `|T| = k`.

## Lossy Reductions

Note that problem 2 reduces to problem 1 by defining `Y_i = X_i \otimes X_i`, and maximizing over `V\in S^{d^2-1}` (which we think of as being `v\otimes v` for some `v` in `S^{d-1}`). 
For such a vector `V`, we have the property that `<Y_i, V> = <X_i, v>^2`, and therefore solving problem 1 for this new list of vectors produces a bound on problem 2 for the `X_i`.

We can expand this reduction by replacing the condition `<V, V> = 1` with `<V, Q V> = 1` for a carefully chosen matrix `Q` for which `<v \otimes v, Q v \otimes v> = <v \otimes v, v \otimes v>` for all `v` in `R^d`.
We explore a class of such matrices (that obtain good results when paired with spectral algorithms for moderate set sizes `|T|`) in the file `hypercontractivity_reductions.py`.

Note that both of these reductions increase the effective dimension quadratically, so they are not suited for the coarse-net and ransac algorithms whose runtime may be exponential in this dimension.

## Algorithms

### Efficient Algorithms

Efficient algorithms (total runtime `<= n^3 + d^6`) that give reasonably good results on our regime of interest for Gaussian samples.

#### Triangle Inequality

The algorithm takes as input the Gram matrix, and utilizes the observation that for any set `T` of samples,
the `l2` norm squared of the sum of these samples is given by the following expression which can be relaxed as follows:
`l2` norm squared = `max_T sum_{i, j in T} <X_i, X_j>` <= (relaxation)
                <= `max_{T, S_i} sum_{i in T, j in S_i} <X_i, X_j>`
This relaxed version of the problem is then solved with a greedy algorithm.

#### Spectral Algorithms

Compute an upper bound on the `L2` squared norm of the sum of any `k` rows of the input array
using spectral decomposition.

**Methodology:**
1. Perform eigenvalue decomposition of the outer product matrix if not provided.
2. Calculate bounds on the coefficients in the eigen-basis. These bounds are based on `|<1_T, v_i>|^2 <= max{|sum over k largest entries in v_i|^2, |sum over k most negative entries in v_i|^2}`
3. Compute the upper bound on the `L2` norm squared.

#### LP / SDP Algorithms

An extension of the spectral / triangle inequality algorithms that can take more constraints into account in our relaxation of the problem. Current implementation does not run well in practice.

### `exp(epsilon d)`-Time Algorithms

A suite of algorithms that run in longer, but still manageable time (`exp(\tilde{O}(epsilon d))`), which can provably produce good bounds even on highly non-hypercontractive datasets.

#### RANSAC Algorithms
#### Coarse-Net Algorithms
