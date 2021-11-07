module MvNormalCDF
    using Distributions, Primes, Random, LinearAlgebra
    export MvNormal, cdf, mvnormcdf
    const sqrt2π = sqrt(2π)
    include("functions.jl")
end # module
