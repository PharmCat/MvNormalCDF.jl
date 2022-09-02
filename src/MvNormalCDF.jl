# MvNormalCDF
# Copyright © 2019-2021 Vladimir Arnautov aka PharmCat <mail@pharmcat.net>, Andrew Gough

module MvNormalCDF
    using Distributions, Primes, Random, LinearAlgebra, StatsBase, FillArrays

    export MvNormal, mvnormcdf

    const sqrt2π = sqrt(2π)
    const ZDIST  = Normal()

    include("functions.jl")
end # module
