# MvNormalCDF
# Copyright © 2019-2021 Vladimir Arnautov aka PharmCat <mail@pharmcat.net>, Andrew Gough

#=
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
% QSIMVNV(m,r,a,b) and _chlrdr(r,a,b)
%
% Copyright (C) 2013, Alan Genz,  All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided the following conditions are met:
%   1. Redistributions of source code must retain the above copyright
%      notice, this list of conditions and the following disclaimer.
%   2. Redistributions in binary form must reproduce the above copyright
%      notice, this list of conditions and the following disclaimer in
%      the documentation and/or other materials provided with the
%      distribution.
%   3. The contributor name(s) may not be used to endorse or promote
%      products derived from this software without specific prior
%      written permission.
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
% "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
% LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
% FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
% COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
% INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
% BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
% OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
% ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
% TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF USE
% OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%
=#

copy_oftype(A::AbstractArray{T}, ::Type{T}) where {T} = copy(A)
copy_oftype(A::AbstractArray{T,N}, ::Type{S}) where {T,N,S} = convert(AbstractArray{S,N}, A)

"""
    mvnormcdf(dist::MvNormal, a, b; m::Integer = 1000*size(dist.Σ,1), rng = RandomDevice())

Computes the Multivariate Normal probability integral using a quasi-Monte-Carlo
algorithm with m points for multivariate normal distributions ([MvNormal](https://juliastats.org/Distributions.jl/stable/multivariate/#Distributions.MvNormal))

Probability p is output with error estimate e.

# Arguments
- `dist::MvNormal`: multivariate normal distributions from Distributions.jl
- `a::AbstractVector`: lower integration limit column vector
- `b::AbstractVector`: upper integration limit column vector
- `m::Integer`:        number of integration points (default 1000*dimension)
- `rng`: random number generator

# Reference
- Genz, A. (1992). Numerical computation of multivariate normal probabilities. Journal of Computational and Graphical Statistics, 1, 141--150
- Genz, A. (1993). Comparison of methods for the computation of multivariate normal probabilities. Computing Science and Statistics, 25, 400--405
"""
function mvnormcdf(dist::MvNormal, a, b; m::Integer = 1000*size(dist.Σ,1), rng = RandomDevice())
    mvnormcdf(dist.μ, dist.Σ, a, b, m = m, rng = rng)
end

"""
    mvnormcdf(μ::AbstractVector, Σ::AbstractMatrix, a::AbstractVector, b::AbstractVector; m::Integer = 1000*size(Σ,1), rng = RandomDevice())

Computes the Multivariate Normal probability integral using a quasi-Monte-Carlo
algorithm with m points for positive definite covariance matrix Σ, mean [0,...], with lower
integration limit vector a and upper integration limit vector b.
```math
\\Phi_k(\\mathbf{a},\\mathbf{b},\\mathbf{\\Sigma} ) = \\frac{1}{\\sqrt{\\left | \\mathbf{\\Sigma}  \\right |{(2\\pi )^k}}}\\int_{a_1}^{b_1}\\int_{a_2}^{b_2}\\begin{align*}
 &...\\end{align*} \\int_{a_k}^{b_k}e^{^{-\\frac{1}{2}}\\mathbf{x}^t{\\mathbf{\\Sigma }}^{-1}\\boldsymbol{\\mathbf{x}}}dx_k...dx_1
```
Probability p is output with error estimate e.
# Arguments
- `μ::AbstractVector`: vector of means
- `Σ::AbstractMatrix`: positive-definite covariance matrix of MVN distribution
- `a::AbstractVector`: lower integration limit column vector
- `b::AbstractVector`: upper integration limit column vector
- `m::Integer`:        number of integration points (default 1000*dimension)
- `rng`: random number generator

# Example
```julia
Σ = [4 3 2 1; 3 5 -1 1; 2 -1 4 2; 1 1 2 5]
μ = zeros(4)
a = [-Inf; -Inf; -Inf; -Inf]
b = [1; 2; 3; 4]
m = 5000
(p,e) = mvnormcdf(μ, Σ, a, b; m=m)
#(0.605219554009911, 0.0015718064928452481)
```
Results will vary slightly from run-to-run due to the quasi-Monte-Carlo
algorithm.
"""
function mvnormcdf(Σ::AbstractMatrix, a::AbstractVector, b::AbstractVector; m::Integer = 1000*size(Σ,1), rng = RandomDevice())
    mvnormcdf(Zeros{promote_type(eltype(Σ), eltype(a), eltype(b), Float64)}(size(Σ, 1)), Σ, a, b, m = m, rng = rng)
end

"""
     mvnormcdf(Σ::AbstractMatrix, a::AbstractVector, b::AbstractVector; m::Integer = 1000*size(Σ,1), rng = RandomDevice())

Non-central MVN distributions (with non-zero mean) can use this function by adjusting
the integration limits. Subtract the mean vector, μ, from each
integration vector.
# Example
```julia
Σ = [4 2; 2 3]
μ = [1; 2]
a = [-Inf; -Inf]
b = [2; 2]
(p,e) = mvnormcdf(Σ, a-μ, b-μ)
#(0.4306346895870772, 0.00015776288569406053)
```
"""
function mvnormcdf(μ::AbstractVector{T1}, Σ::AbstractMatrix{T2}, a::AbstractVector, b::AbstractVector; m::Integer = 1000*size(Σ,1), rng = RandomDevice()) where T1 where T2
    T = promote_type(T1, T2, Float64)
    # check for proper dimensions
    n  = size(Σ, 1)
    nc = size(Σ, 2) 	# assume square Cov matrix nxn
    # check dimension > 1
    n >= 2   || throw(ErrorException("dimension of Σ must be 2 or greater. Σ dimension: $(size(Σ))"))
    n == nc  || throw(DimensionMismatch("Σ matrix must be square. Σ dimension: $(size(Σ))"))
    # check dimensions of lower vector, upper vector, and cov matrix match
    (n == length(a) == length(b)) || throw(DimensionMismatch("iconsistent argument dimensions. Sizes: Σ $(size(Σ))  a $(size(a))  b $(size(b))"))

    # check that lower integration limit a <= upper integration limit b for all elements
    all( x -> x[1] <= x[2], zip(a, b)) || throw(ArgumentError("lower integration limit a must be <= upper integration limit b"))
    # check that Σ is positive definate; if not, print warning
    # isposdef(Σ) || @warn "covariance matrix Σ fails positive definite check"
    # check if Σ, a, or b contains NaNs
    if any(isnan, Σ) || any(isnan, a) || any(isnan, b)
        p = NaN
        e = NaN
        return (p, e)
    end
    # check if a==b
    if a == b
        p = zero(T)
        e = zero(T)
        return (p, e)
    end
    # check if a = -Inf & b = +Inf
    if all(x -> x == -Inf, a) && all(x -> x == Inf, b)
        p = one(T)
        e = zero(T)
        return (p, e)
    end
    ##################################################################
    #
    # Special cases: positive Orthant probabilities for 2- and
    # 3-dimesional Σ have exact solutions. Integration range [0,∞]
    #
    ##################################################################
    if all(iszero, a) && all(x -> x == Inf, b) && n <= 3
        #Σstd = sqrt.(diag(Σ))
        Σstd  = Vector{Float64}(undef, n)
        @inbounds for i in 1:n
            Σstd[i] = sqrt(Σ[i, i])
        end
        Rcorr = cov2cor(Σ, Σstd)
        if n == 2
            p = 1/4 + asin(Rcorr[1, 2]) / (2π)
            e = eps()
        elseif n == 3
            p = 1/8 + (asin(Rcorr[1, 2]) + asin(Rcorr[2, 3]) + asin(Rcorr[1, 3])) / (4π)
            e = eps()
        end
        return (p, e)
    end
    #
    at = copy_oftype(a, T)
    bt = copy_oftype(b, T)
    at .-= μ
    bt .-= μ
    qsimvnv!(copy_oftype(Σ, T), at, bt, m, rng)
end

"""
Re-coded in Julia from the MATLAB function qsimvnv(m,r,a,b)
Alan Genz is the author the MATLAB qsimvnv() function.

! Mutate a, b. Return new Σ
"""
function qsimvnv!(Σ::AbstractMatrix{T}, a::AbstractVector, b::AbstractVector, m::Integer, rng) where T
    #T = promote_type(T1, T2)
    ##################################################################
    #
    # get lower cholesky matrix and (potentially) re-ordered integration vectors
    #
    ##################################################################
    (ch, as, bs) = _chlrdr!(Σ, a, b) # ch =lower cholesky; as=lower vec; bs=upper vec
    ##################################################################
    #
    # quasi-Monte Carlo integration of MVN integral
    #
    ##################################################################
    ### setup initial values
    ai = as[1]
    bi = bs[1]
    ct = ch[1, 1]
    #  unitnorm = Normal() # unit normal distribution
    # rng = RandomDevice()
    # if ai is -infinity, explicity set c=0
    # implicitly, the algorith classifies anythign > 9 std. deviations as infinity
    if ai > -9ct
        if ai < 9ct
            c1 = cdf(ZDIST, ai / ct)
        else
            c1 = one(T)
        end
    else
        c1 = zero(T)
    end
    # if bi is +infinity, explicity set d=0
    if bi > -9ct
        if bi < 9ct
            d1 = cdf(ZDIST, bi / ct)
        else
            d1 = one(T)
        end
    else
        d1 = zero(T)
    end
    n   = size(Σ, 1) 	# assume square Cov matrix nxn
    cxi = c1			# initial cxi; genz uses ci but it conflicts with Lin. Alg. ci variable
    dci = d1 - cxi		# initial dcxi
    p   = zero(T)       # probablity = 0
    e   = 0.0			# error = 0
    # Richtmyer generators
    ps  = sqrt.(primes(Int(floor(5 * n * log(n + 1) / 4)))) # Richtmyer generators
    q   = ps[1:n - 1, 1]
    ns  = 12
    nv  = Int(max(floor(m / ns), 1))
    ##
    Jnv    = ones(1, nv)
    cfill  = fill(cxi, nv) 	            # evaulate at nv quasirandom points row vec
    dpfill = fill(dci, nv)
    y      = zeros(T, nv, n - 1)			# n-1 (cols), nv (rows), preset to zero # change row-col for col-operation
    #=
    Randomization loop for ns samples
    j is the number of samples to integrate over,
    but each with a vector nv in length
    i is the number of dimensions, or integrals to comptue
    =#
    c  = zeros(T, length(cfill))
    dc = zeros(T, length(dpfill))
    pv = zeros(T, length(dpfill))
    d  = zeros(T, length(Jnv))
    tv = zeros(T, nv)

    for j in 1:ns					# loop for ns samples
        copyto!(c, cfill)
        copyto!(dc, dpfill)
        copyto!(pv, dpfill)

        @inbounds for i in 2:n                # n - dimentions
            xr = rand(rng)
            @inbounds for cnt = 1:length(c)
                x = abs(2 * mod(cnt * q[i - 1] + xr, 1) - 1)
                y[cnt, i - 1] = quantile(ZDIST, c[cnt] + x * dc[cnt])
            end

            s = mul!(tv, view(y, :, 1:i - 1), view(ch, i, 1:i - 1))
            ct = ch[i, i]                                       # ch is cholesky matrix
            copyto!(c, Jnv)										# preset to 1 (>9 sd, +∞)
            copyto!(d, Jnv)										# preset to 1 (>9 sd, +∞)
            asi = as[i]
            bsi = bs[i]

            @inbounds for cnt in 1:length(c)
                aicnt = asi - s[cnt]
                bicnt = bsi - s[cnt]
                if isless(aicnt, -9ct)
                    c[cnt] = 0.0
                elseif isless(abs(aicnt), 9ct)
                    c[cnt] = cdf(ZDIST, aicnt / ct)
                end
                if isless(bicnt, -9ct)
                    d[cnt] = 0.0
                elseif isless(abs(bicnt), 9ct)
                    d[cnt] = cdf(ZDIST, bicnt / ct)
                end
            end
            @. dc = d - c
            @. pv = pv * dc
        end # for i

        dm = (mean(pv) - p) / j
        p += dm
        e = (j - 2) * e / j + dm * dm
    end # for j
    e = 3 * sqrt(e) 	# error estimate is 3 times standard error with ns samples
    return (p, e)  	# return probability value and error estimate
end # function qsimvnv

"""
Computes permuted lower Cholesky factor c for R which may be singular,
  also permuting integration limit vectors a and b.

! Mutate Σ, a, b

# Arguments
	Σ		matrix			Matrix for which to compute lower Cholesky matrix,
                            this is a covariance matrix
	a		vector			column vector for the lower integration limit
							algorithm may permutate this vector to improve integration
							accuracy
	b		vector			column vector for the upper integration limit
							algorithm may pertmutate this vector to improve integration
							accuracy
Output
			tuple		An a tuple with 3 returned arrays:
							1 - lower Cholesky root of r
							2 - lower integration limit (perhaps permutated)
							3 - upper integration limit (perhaps permutated)
# Examples
r = [1 0.25 0.2; 0.25 1 0.333333333; 0.2 0.333333333 1]
a = [-1; -4; -2]
b = [1; 4; 2]
(c, ap, bp) = _chlrdr(r,a,b)
result:
Lower cholesky root:
c = [ 1.00  0.0000  0.0000,
      0.20  0.9798  0.0000,
      0.25  0.2892  0.9241 ]
Permutated upper input vector:
ap = [-1, -2, -4]
Permutated lower input vector:
bp = [1, 2, 4]
"""
#############
function _chlrdr!(Σ::AbstractMatrix{T}, a::AbstractVector, b::AbstractVector) where T
    
    # define constants
    ep = 1e-10 # singularity tolerance
    ϵ  = eps()
    # unit normal distribution
    #unitnorm = Normal()
    n   = size(Σ, 1) # covariance matrix n x n square
    ckk = zero(T)
    dem = zero(T)
    am  = zero(T)
    bm  = zero(T)
    im  = zero(T)
    #c   = copyto!(Matrix{T}(undef, n, n), Σ)
    c   = Σ
    ap  = a
    bp  = b
    d   = Vector{T}(undef, n)
    @inbounds for i in 1:n
        d[i] = sqrt(c[i, i])
    end
    @inbounds for i in 1:n
        if d[i] > 0
            c[:, i] /= d[i]
            c[i, :] /= d[i]
            ap[i]   /= d[i]     # ap n x 1 vector
            bp[i]   /= d[i]     # bp n x 1 vector
        end
    end
    y = zeros(T, n) # n x 1 zero vector to start
    @inbounds for k in 1:n
        im  = k
        ckk = zero(T)
        dem = one(T)
        s   = zero(T)
        @inbounds for i in k:n
            if c[i, i] > ϵ  # machine error
                cii = sqrt(max(c[i, i], 0))
                if i > 1  && k > 1
                    #=
                    s = ch(i,1:i-1)*y(1:i-1);
                    seems this it multiplication of row-vector on column-vector
                    so this is dot product of two vectors in Julia,
                    not element whise multiplication
                    =#
                    #s=(c[i,1:(k-1)] .* y[1:(k-1)])[1]
                    #test
                    s = dot(view(c, i, 1:(k - 1)), view(y, 1:(k - 1)))
                    #s = dot(c[i, 1:(k-1)], y[1:(k-1)])
                end
                ai=(ap[i] - s) / cii
                bi=(bp[i] - s) / cii
                de = cdf(ZDIST, bi) - cdf(ZDIST, ai)
                if de <= dem
                    ckk = cii
                    dem = de
                    am  = ai
                    bm  = bi
                    im  = i
                end
            end # if c[i,i]> ϵ
        end # for i=
        if im > k
            c[im, im] = c[k, k]
            ap[im] , ap[k] = ap[k] , ap[im]
            bp[im] , bp[k] = bp[k] , bp[im]
            if k > 1
                c[im, 1:(k - 1)], c[k, 1:(k - 1)] = c[k, 1:(k - 1)], c[im, 1:(k - 1)]
            end
            if im < n
                c[(im + 1):n, im], c[(im + 1):n, k] = c[(im + 1):n, k], c[(im + 1):n, im]
            end
            if k <= (n - 1) && im <= n
                c[(k + 1):(im - 1), k], c[im, (k + 1):(im - 1)] = transpose(c[im, (k + 1):(im - 1)]), transpose(c[(k + 1):(im - 1), k])
            end
        end # if im>k
        if k < n
            c[k:k, (k + 1):n] .= zero(T)
        end
        if ckk > k * ep
            c[k, k] = ckk
            for i in k + 1:n
                c[i, k] /= ckk
                #c[i:i, (k+1):i] -= c[i,k]*transpose(c[(k+1):i,k])
                axpy!(-c[i, k], view(c, k + 1:i, k), view(c, i:i, k + 1:i))
            end
            if abs(dem) > ep
                y[k] = (exp(-am^2 / 2) - exp(-bm^2 / 2)) / (sqrt2π * dem)
            else
                if am < -10
                    y[k] = bm
                elseif bm > 10
                    y[k] = am
                else
                    y[k] = (am + bm)/2
                end
            end # if abs
        else
            c[k:n, k] .= zero(T)
            y[k] = zero(T)
        end # if ckk>ep*k
    end # for k=
    return (c, ap, bp)
end
