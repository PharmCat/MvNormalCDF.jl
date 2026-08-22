# MvNormalCDF

Quasi-Monte-Carlo numerical computation of multivariate normal probabilities.

![Tier 1](https://github.com/PharmCat/MvNormalCDF.jl/workflows/Tier%201/badge.svg)

[![codecov](https://codecov.io/gh/PharmCat/MvNormalCDF.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/PharmCat/MvNormalCDF.jl)


This function uses an algorithm given in the paper
"Numerical Computation of Multivariate Normal Probabilities", in
 J. of Computational and Graphical Stat., 1(1992), pp. 141-149, by
Alan Genz, WSU Math, PO Box 643113, Pullman, WA 99164-3113
Email : alangenz@wsu.edu

The primary references for the numerical integration are
"On a Number-Theoretical Integration Method"
H. Niederreiter, Aequationes Mathematicae, 8(1972), pp. 304-11, and
"Randomization of Number Theoretic Methods for Multiple Integration"
R. Cranley and T.N.L. Patterson, SIAM J Numer Anal, 13(1976), pp. 904-14.

Re-coded in Julia from the MATLAB function qsimvnv(m,r,a,b)
Alan Genz is the author the MATLAB qsimvnv() function.

Alan Genz software website: http://archive.is/jdeRh
Source code to MATLAB qsimvnv() function: http://archive.is/h5L37

Quasi-random sequence made with Richtmyer generator.

![MVNCDFFORMULA](https://raw.githubusercontent.com/PharmCat/MvNormalCDF.jl/main/.github/images/mvnpi.png)

# Installation

```
import Pkg; Pkg.add("MvNormalCDF")
```

# Methods

```
    mvnormcdf(dist::MvNormal, a, b; m::Integer = 1000*size(dist.Σ,1), rng = RandomDevice())
```

Computes the Multivariate Normal probability integral using a quasi-Monte-Carlo
algorithm with m points for multivariate normal distributions ([MvNormal](https://juliastats.org/Distributions.jl/stable/multivariate/#Distributions.MvNormal))

Results will vary slightly from run-to-run due to the quasi-Monte-Carlo
    algorithm.

There is no covariance matrix Σ positive definite check.    

Probability p is output with error estimate e.

> ⚠️ **Always check the error estimate `e`.**
>
> `e` is a stochastic estimate — three standard errors over the 12 randomizations
> of the quasi-random sequence. It measures the spread between those 12 samples
> and is not a bound on the total error.
>
> `e == 0` means all 12 samples agreed exactly. This happens in three very
> different situations:
>
> * **the result is exact** — the variables are independent after reordering
>   (diagonal or block-diagonal `Σ`), or a closed-form shortcut was taken
>   (`a == b`, `a == -Inf` with `b == +Inf`, or the 2-/3-dimensional positive
>   orthant);
> * **`Σ` is singular or ill-conditioned** — the integral collapsed to a lower
>   dimension. `p` is usually still correct, but that is not guaranteed;
> * **underflow / saturation** — the probability is so small (or so close to 1)
>   that every integration point returned the same value. `p` is then reported
>   as exactly `0.0` or `1.0` while the true value is not.
>
> So `e == 0` is not an error signal by itself — it means `e` carries no
> information, and `p` has to be judged some other way: check `rank(Σ)` and
> `cond(Σ)`, and check whether `p` sits at exactly `0.0` or `1.0`.

## Arguments
- `dist::MvNormal`: multivariate normal distributions from Distributions.jl
- `a::AbstractVector`: lower integration limit column vector
- `b::AbstractVector`: upper integration limit column vector
- `m::Integer`:        number of integration points (default 1000*dimension)
- `rng`: random number generator (for quasi-random Richtmyer sequence)


```
    mvnormcdf(μ::AbstractVector{<:Real}, Σ::AbstractMatrix{<:Real}, a::AbstractVector{<:Real}, b::AbstractVector{<:Real}; m::Integer = 1000*size(Σ,1), rng = RandomDevice())
```

Computes the Non-central Multivariate Normal probability integral with covariance matrix Σ and mean vector μ [x,...].

## Arguments
- `μ::AbstractVector`: vector of means
- `Σ::AbstractMatrix`: positive-definite covariance matrix of MVN distribution
- `a::AbstractVector`: lower integration limit column vector
- `b::AbstractVector`: upper integration limit column vector
- `m::Integer`:        number of integration points (default 1000*dimension)
- `rng`: random number generator

## Example
```julia
Σ = [4 3 2 1; 3 5 -1 1; 2 -1 4 2; 1 1 2 5]
μ = zeros(4)
a = [-Inf; -Inf; -Inf; -Inf]
b = [1; 2; 3; 4]
m = 5000
(p,e) = mvnormcdf(μ, Σ, a, b; m=m)
#(0.605219554009911, 0.0015718064928452481)
```

```
     mvnormcdf(Σ::AbstractMatrix, a::AbstractVector, b::AbstractVector; m::Integer = 1000*size(Σ,1), rng = RandomDevice())
```

Computes the Multivariate Normal probability integral with covariance matrix Σ, mean [0,...].

## Example
```julia
Σ = [4 2; 2 3]
μ = [1; 2]
a = [-Inf; -Inf]
b = [2; 2]
(p,e) = mvnormcdf(Σ, a-μ, b-μ)
#(0.4306346895870772, 0.00015776288569406053)
```

## Not exported

```
    MvNormalCDF.qsimvnv!(Σ::AbstractMatrix{T}, a::AbstractVector{T}, b::AbstractVector{T}, m::Integer, rng) where T
```

Re-coded in Julia from the MATLAB function qsimvnv(m,r,a,b); mutate Σ, a, b. Computes the Multivariate Normal probability integral with covariance matrix Σ, mean [0,...].

# Reference
- Genz, A. (1992). Numerical computation of multivariate normal probabilities. Journal of Computational and Graphical Statistics, 1, 141--150
- Genz, A. (1993). Comparison of methods for the computation of multivariate normal probabilities. Computing Science and Statistics, 25, 400--405

# P.S.

Idea was taken from this [PR](https://github.com/JuliaStats/StatsFuns.jl/pull/114) to StatsFuns.jl.

See discourse discussion [here](https://discourse.julialang.org/t/mvn-cdf-have-it-coded-need-help-getting-integrating-into-distributions-jl/38631).

Thanks to @blackeneth.

# P.P.S
>
> QSIMVNV(m,r,a,b) and _chlrdr(r,a,b)
>
> Copyright (C) 2013, Alan Genz,  All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided the following conditions are met:
>   1. Redistributions of source code must retain the above copyright
>      notice, this list of conditions and the following disclaimer.
>   2. Redistributions in binary form must reproduce the above copyright
>      notice, this list of conditions and the following disclaimer in
>      the documentation and/or other materials provided with the
>      distribution.
>   3. The contributor name(s) may not be used to endorse or promote
>      products derived from this software without specific prior
>      written permission.
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
> "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
> LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
> FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
> COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
> INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
> BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
> OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
> ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
> TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF USE
> OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
>
