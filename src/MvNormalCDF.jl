# MvNormalCDF
# Copyright © 2019-2021 Vladimir Arnautov aka PharmCat <mail@pharmcat.net>, Andrew Gough

module MvNormalCDF
    using Distributions, Primes, Random, LinearAlgebra

    export MvNormal, cdf, mvnormcdf

    const sqrt2π = sqrt(2π)

    include("functions.jl")
end # module
