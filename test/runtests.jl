# MvNormalCDF
# Copyright © 2019-2021 Vladimir Arnautov aka PharmCat <mail@pharmcat.net>, Andrew Gough
using MvNormalCDF
using Test, Distributions, StableRNGs, ForwardDiff

td = Array{Any}(undef,(14,9))
#  1-cov mtx 2-a 3-b 4-m 5-p 6-ptol 7-e 8-etol

# from MATLAB documenation 4 dim
td[1,1] = [4 3 2 1;3 5 -1 1;2 -1 4 2;1 1 2 5]  # Σ cov Matrix
td[1,2] = [-Inf; -Inf; -Inf; -Inf]             # a lower integration limit
td[1,3] = [1; 2; 3; 4 ]                        # b upper integration limit
td[1,4] = 5000
td[1,5] = 0.605653                            	# expected p value
td[1,6] = 0.001557374                          	# ± p tolerance
td[1,7] = 0.001394971                           # expected e (error) value
td[1,8] = 0.0009277058                        	# ± e tolerance

# 3 dim
td[2,1] =[1  3/5  1/3; 3/5 1  11/15; 1/3 11/15 1]
td[2,2] = [-Inf;-Inf;-Inf]
td[2,3] = [1;4;2]
td[2,4] = 3000
td[2,5] = 0.827985
td[2,6] = 3.529068e-5
td[2,7] = 2.8608229e-5
td[2,8] = 3.72604931e-5

# 3 dim
td[3,1] = [1 0.25 0.2; 0.25 1 0.333333333; 0.2 0.333333333 1]
td[3,2] = [-1;-4;-2]
td[3,3] = [1;4;2]
td[3,4] = 4000
td[3,5] = 0.65368
td[3,6] = 0.000002089699
td[3,7] = 0.000001799225
td[3,8] = 0.000001161755

# Genz book eq. 1.5 p. 4-5 & p. 63
# Genz gives wrong answer on p. 4-5
td[4,1] = [1/3 3/5 1/3; 3/5 1.0 11/15; 1/3 11/15 1.0]
td[4,2] = [-Inf; -Inf; -Inf]
td[4,3] = [1; 4; 2]
td[4,4] = 4000
td[4,5] = 0.943174
td[4,6] = 0.00006724871
td[4,7] = 0.0000509768
td[4,8] = 0.00003337435

# Genz p. 63 uses different r matrix
td[5,1] = [1 0 0; 3/5 1 0; 1/3 11/15 1]
td[5,2] = [-Inf; -Inf; -Inf]
td[5,3] = [1; 4; 2]
td[5,4] = 4000
td[5,5] = 0.827985
td[5,6] = 0.00001024322
td[5,7] = 0.000008933135
td[5,8] = 0.000005194712

# singular example
# problem reduces to univariate problem with
# p = cdf.(Normal(),1) = 0.84134476068543
td[6,1] = [1 1 1; 1 1 1; 1 1 1]
td[6,2] = [-Inf, -Inf, -Inf]
td[6,3] = [1, 1, 1]
td[6,4] = 3000
td[6,5] = 0.841345
td[6,6] = 4.440892E-16
td[6,7] = 0.0
td[6,8] = 2.220446E-16

# 5 dim example
td[7,1] = [1 1 1 1 1;
		 1 2 2 2 2;
		 1 2 3 3 3;
		 1 2 3 4 4;
		 1 2 3 4 5]
td[7,2] = [-1,-2,-3,-4,-5]
td[7,3] = [2,3,4,5,6]
td[7,4] = 6000
td[7,5] = 0.761243
td[7,6] = 0.0004715559
td[7,7] = 0.0003493157
td[7,8] = 0.0002319606

# Genz used wrong integration limits when computing
# above. see p. 63
td[8,1] = td[7,1]
td[8,2] = sort(td[7,2])
td[8,3] = 1 .- td[8,2]
td[8,4] = 6000
td[8,5] = 0.474128
td[8,6] = 0.000037335
td[8,7] = 0.00003169846
td[8,8] = 0.0000178166

# positive orthant probability of above
td[9,1] = td[7,1]
td[9,2] = [0,0,0,0,0]
td[9,3] = td[8,3]
td[9,4] = 6000
td[9,5] = 0.113537
td[9,6] = 0.0001740044
td[9,7] = 0.0001584365
td[9,8] = 0.0001053655

# test 7 Cov matrix, but now -Inf lower limit
td[10,1] = td[7,1]
td[10,2] = [-Inf,-Inf,-Inf,-Inf,-Inf]
td[10,3] = td[8,3]
td[10,4] = 6000
td[10,5] = 0.810315
td[10,6] = 0.00003136998
td[10,7] = 0.0000287006
td[10,8] = 0.00001567681

# eight dimensional test
td[11,1] = [1 1 1 1 1 1 1 1;
		 	1 2 2 2 2 2 2 2;
		 	1 2 3 3 3 3 3 3;
		 	1 2 3 4 4 4 4 4;
		 	1 2 3 4 5 5 5 5;
		 	1 2 3 4 5 6 6 6;
		 	1 2 3 4 5 6 7 7;
		 	1 2 3 4 5 6 7 8]
td[11,2] =  -1*[1,2,3,4,5,6,7,8]
td[11,3] = [2,3,4,5,6,7,8,9]
td[11,4] = 9000
td[11,5] = 0.759474
td[11,6] = 0.0006248131
td[11,7] = 0.0005161476
td[11,8] = 0.0003407714

# orthant probability of above
td[12,1] = td[11,1]
td[12,2] = [0,0,0,0,0,0,0,0]
td[12,3] = [Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf]
td[12,4] = 9000
td[12,5] = 0.19638
td[12,6] = 0.0002654318
td[12,7] = 0.0002384606
td[12,8] = 0.0001693536

# with dim with lower limit -Inf (exact result)
td[13,1] = td[11,1]
td[13,2] = -Inf*[1,1,1,1,1,1,1,1]
td[13,3] = td[12,3]
td[13,4] = 9000
td[13,5] = 1.0
td[13,6] = 2.220446E-16
td[13,7] = 0.0
td[13,8] = 2.220446E-16

# 25 dimensions
td[14,1] = [59.227 2.601 3.38 8.303 -0.334 11.029 10.908 0.739 4.703 7.075 8.049 1.403 9.838 5.46 11.949 2.272 7.234 15.215 -9.091 12.265 3.01 -3.199 10.608 8.464 -8.685;
 2.601 77.213 0.882 2.99 -2.536 -4.55 -3.874 -3.607 6.023 3.129 15.7 -7.271 -10.655 8.456 15.387 -5.764 6.617 -6.331 -2.244 -0.925 -7.516 13.836 9.243 -0.84 -3.781;
 3.38 0.882 79.72 -1.465 3.179 -1.799 9.842 9.165 -1.54 -8.03 0.778 9.053 -2.598 -8.844 15.857 13.613 -1.878 8.18 11.806 -0.242 4.711 2.258 9.554 1.184 -8.047;
 8.303 2.99 -1.465 65.041 3.803 2.13 -0.936 -5.996 2.719 -4.648 4.611 4.486 13.38 -0.376 0.179 14.654 -7.089 -1.194 9.357 5.12 4.943 -0.475 4.764 -8.56 2.337;
 -0.334 -2.536 3.179 3.803 64.128 4.85 -14.767 -10.044 12.437 5.065 8.191 0.391 0.068 9.754 -0.062 0.429 9.265 5.502 4.227 0.559 0.811 3.169 4.558 -1.878 -3.885;
 11.029 -4.55 -1.799 2.13 4.85 64.912 12.989 2.675 1.227 8.205 3.3 3.545 6.225 11.936 -2.956 6.188 -2.206 1.184 10.546 2.492 12.035 -15.789 4.296 1.086 7.225;
 10.908 -3.874 9.842 -0.936 -14.767 12.989 78.424 2.03 5.595 4.921 1.573 -7.355 -9.425 -4.024 4.912 -12.05 -2.034 -2.435 -10.355 0.985 8.23 2.806 6.254 -1.494 5.531;
 0.739 -3.607 9.165 -5.996 -10.044 2.675 2.03 93.184 8.84 3.843 -14.968 16.386 -0.223 4.398 -4.786 1.731 4.025 0.479 3.12 -15.591 12.614 -8.279 -3.582 6.597 -1.915;
 4.703 6.023 -1.54 2.719 12.437 1.227 5.595 8.84 73.981 4.019 -6.404 5.869 -4.305 5.936 2.016 5.987 10.051 -0.705 13.229 -1.715 7.102 12.89 10.967 5.262 15.954;
 7.075 3.129 -8.03 -4.648 5.065 8.205 4.921 3.843 4.019 68.048 7.346 7.412 9.956 6.743 2.547 0.177 0.844 2.147 -4.072 11.832 -3.55 -0.096 -1.96 -1.381 -3.249;
 8.049 15.7 0.778 4.611 8.191 3.3 1.573 -14.968 -6.404 7.346 66.708 -6.22 -8.952 0.647 3.039 -12.078 7.618 10.398 -5.055 1.577 -12.77 11.477 8.272 2.071 7.728;
 1.403 -7.271 9.053 4.486 0.391 3.545 -7.355 16.386 5.869 7.412 -6.22 55.025 0.273 -12.049 -2.71 11.531 12.604 1.294 2.791 -5.698 -2.231 15.025 12.229 -5.876 -3.374;
 9.838 -10.655 -2.598 13.38 0.068 6.225 -9.425 -0.223 -4.305 9.956 -8.952 0.273 82.184 3.466 -9.297 -4.347 12.586 4.372 13.705 -6.795 -5.818 -6.78 5.11 10.099 5.05;
 5.46 8.456 -8.844 -0.376 9.754 11.936 -4.024 4.398 5.936 6.743 0.647 -12.049 3.466 69.352 10.855 7.282 6.615 4.58 0.306 6.482 14.589 5.081 -9.141 -4.657 10.763;
 11.949 15.387 15.857 0.179 -0.062 -2.956 4.912 -4.786 2.016 2.547 3.039 -2.71 -9.297 10.855 87.988 2.504 4.226 10.461 9.703 -3.112 -13.348 0.944 -2.824 -4.498 10.551;
 2.272 -5.764 13.613 14.654 0.429 6.188 -12.05 1.731 5.987 0.177 -12.078 11.531 -4.347 7.282 2.504 63.611 3.024 2.35 -5.191 -6.101 -6.324 -0.483 9.899 5.768 2.382;
 7.234 6.617 -1.878 -7.089 9.265 -2.206 -2.034 4.025 10.051 0.844 7.618 12.604 12.586 6.615 4.226 3.024 78.3 1.54 -6.868 2.613 6.006 5.49 9.06 -4.229 -4.395;
 15.215 -6.331 8.18 -1.194 5.502 1.184 -2.435 0.479 -0.705 2.147 10.398 1.294 4.372 4.58 10.461 2.35 1.54 61.499 11.083 15.428 1.771 2.517 5.181 13.476 6.829;
 -9.091 -2.244 11.806 9.357 4.227 10.546 -10.355 3.12 13.229 -4.072 -5.055 2.791 13.705 0.306 9.703 -5.191 -6.868 11.083 76.538 4.287 6.564 -2.49 15.558 11.202 -16.964;
 12.265 -0.925 -0.242 5.12 0.559 2.492 0.985 -15.591 -1.715 11.832 1.577 -5.698 -6.795 6.482 -3.112 -6.101 2.613 15.428 4.287 59.057 -5.619 4.374 8.934 -6.203 14.687;
 3.01 -7.516 4.711 4.943 0.811 12.035 8.23 12.614 7.102 -3.55 -12.77 -2.231 -5.818 14.589 -13.348 -6.324 6.006 1.771 6.564 -5.619 69.833 -1.233 -7.922 -7.027 -2.315;
 -3.199 13.836 2.258 -0.475 3.169 -15.789 2.806 -8.279 12.89 -0.096 11.477 15.025 -6.78 5.081 0.944 -0.483 5.49 2.517 -2.49 4.374 -1.233 65.458 5.786 4.613 2.223;
 10.608 9.243 9.554 4.764 4.558 4.296 6.254 -3.582 10.967 -1.96 8.272 12.229 5.11 -9.141 -2.824 9.899 9.06 5.181 15.558 8.934 -7.922 5.786 69.574 3.527 -2.802;
 8.464 -0.84 1.184 -8.56 -1.878 1.086 -1.494 6.597 5.262 -1.381 2.071 -5.876 10.099 -4.657 -4.498 5.768 -4.229 13.476 11.202 -6.203 -7.027 4.613 3.527 65.98 3.495;
 -8.685 -3.781 -8.047 2.337 -3.885 7.225 5.531 -1.915 15.954 -3.249 7.728 -3.374 5.05 10.763 10.551 2.382 -4.395 6.829 -16.964 14.687 -2.315 2.223 -2.802 3.495 58.327]
 td[14,2] = vec(-Inf*fill(1,(25,1)))
 td[14,3] = [6.0; 9.0; Inf; Inf; Inf; Inf; Inf; Inf; Inf; Inf; Inf; Inf; Inf; Inf; Inf; Inf; Inf; Inf; Inf; Inf; Inf; Inf; Inf; Inf; Inf]
 td[14,4] = 35000
 td[14,5] = 0.665342
 td[14,6] = 0.000001975944
 td[14,7] = 0.00000181746
 td[14,8] = 0.000001083068


 td[1,9] = 0.6054064156442294
 td[2,9] = 0.8279766144831429
 td[3,9] = 0.6536793504453967
 td[4,9] = 0.9431576518489296
 td[5,9] = 0.8279829100356028
 td[6,9] = 0.8413447460685427
 td[7,9] = 0.7612423324911164
 td[8,9] = 0.4741170481094123
 td[9,9] = 0.11351703757272677
 td[10,9] = 0.8103089978617527
 td[11,9] = 0.7593693258920418
 td[12,9] = 0.19640599118630003
 td[13,9] = 1.0
 td[14,9] = 0.6653426686040154

 @testset "MvNormalCDF test" begin
	for i in 1:14
	    r = td[i,1]
	    a = td[i,2]
	    b = td[i,3]
	    m = td[i,4]
	    pexpected = td[i,5]
	    eexpected = td[i,7]
	    ptol = td[i,6]
	    etol = td[i,8]

	    (p,e) = MvNormalCDF.mvnormcdf(r, a, b; m=m, rng = StableRNG(1234))
		v = p
		p = round(p, digits=6)

	    @test p ≈ pexpected atol=ptol
	    @test e ≈ eexpected atol=etol

		@test v ≈ td[i, 9] atol=1e-10
	end

	#just test not validation
	@test_nowarn MvNormalCDF.mvnormcdf(td[3,1], [0, 0, 0], [Inf, Inf, Inf]; m=m, rng = StableRNG(1234))

    # test warning on singular Σ
    r = td[6,1]
    a = td[6,2]
    b = td[6,3]
    m = td[6,4]

    #@test_logs (:warn,"covariance matrix Σ fails positive definite check") A.qsimvnv(r,a,b;m=m)

    # Σ dimension 1 throws error
    r = Array{Float64}(undef,(1,1))
    r[1,1] = 5
    @test_throws ErrorException MvNormalCDF.mvnormcdf(MvNormal(r),a,b)

    # a < b throws error
    r = td[3,1]
    a = [-Inf, 3, 0]
    b = [0, 0, 1]
    @test_throws ArgumentError MvNormalCDF.mvnormcdf(MvNormal(r),a,b)

    # non-square Σ throws error
    r = [1 2; 3 4; 5 6]
    a = [-Inf, -Inf, -Inf]
    b = [0, 0, 0]
    @test_throws DimensionMismatch MvNormalCDF.mvnormcdf(MvNormal(r),a,b)

	# mismatched dimensions throws errors
	r = [1 2; 3 4; 5 6]
    a = [-Inf, -Inf, -Inf]
	b = [0, 0]

	@test_throws DimensionMismatch MvNormalCDF.mvnormcdf(MvNormal(r),a,b)

 end

@testset "ForwardDiff test" begin
  μ = [1., 2., 3.] 
  Σ = [1 0.25 0.2; 0.25 1 0.333333333; 0.2 0.333333333 1]
  ag = [-1; -4; -2]
  bg = [1; 4; 2]
  gf(x) = MvNormalCDF.mvnormcdf(x, Σ, ag, bg)[1]
  @test_nowarn ForwardDiff.gradient(gf, [1, 2, 3])

  μ = [1., 2., 3.] 
  Σ = [1 0.25 0.2; 0.25 1 0.333333333; 0.2 0.333333333 1]
  ag = [-1; -4; -2]
  bg = [1; 4; 2]
  gf2(x) = MvNormalCDF.mvnormcdf(μ, reshape(x, 3, 3), ag, bg)[1]
  @test_nowarn ForwardDiff.gradient(gf2, [1, 0.25, 0.2, 0.25, 1, 0.333333333, 0.2, 0.333333333, 1])
end


#=
using BenchmarkTools
mvn = MvNormal(td[14,1])
a = td[14,2]
b = td[14,3]
MvNormalCDF.mvnormcdf(mvn,a,b)
@benchmark  MvNormalCDF.mvnormcdf($mvn, $a, $b)

BenchmarkTools.Trial: 262 samples with 1 evaluation.
 Range (min … max):  18.809 ms …  29.450 ms  ┊ GC (min … max): 0.00% … 35.47%
 Time  (median):     19.086 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   19.140 ms ± 676.836 μs  ┊ GC (mean ± σ):  0.21% ±  2.19%

               ▁  ▁▂ ▁▃▃ ▂▂▂▄▁▂██▃▃▃▁▂▃ ▄  ▄   ▁
  ▃▁▃▁▆▁▁▃█▃▄▃▄█▆▇██▆███▆██████████████▇██▇█▇▇▇█▆█▄▆▆▄▄▃▃▄▁▃▃▃ ▄
  18.8 ms         Histogram: frequency by time         19.4 ms <

 Memory estimate: 892.28 KiB, allocs estimate: 2334.
 =#


#=
using BenchmarkTools
Σ = [4 3 2 1; 3 5 -1 1; 2 -1 4 2; 1 1 2 5]
μ = zeros(4)
a = [-Inf; -Inf; -Inf; -Inf]
b = [1; 2; 3; 4]
m = 5000
mvn = MvNormal(Σ)
@benchmark  MvNormalCDF.mvnormcdf($mvn, $a, $b)

BenchmarkTools.Trial: 7939 samples with 1 evaluation.
 Range (min … max):  602.700 μs …  18.335 ms  ┊ GC (min … max): 0.00% … 96.37%
 Time  (median):     621.000 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   627.835 μs ± 199.151 μs  ┊ GC (mean ± σ):  0.35% ±  1.08%

              █▇▁
  ▂▁▁▂▁▂▂▂▂▃▅▇███▆▄▄▄▄▅▄▄▄▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂ ▃
  603 μs           Histogram: frequency by time          682 μs <

 Memory estimate: 38.27 KiB, allocs estimate: 98.

julia> @time MvNormalCDF.mvnormcdf(mvn, a, b)
  0.000663 seconds (98 allocations: 38.266 KiB)
(0.6055881147694031, 0.0009341283457742374)
@time MvNormalCDF.mvnormcdf(Σ, a, b)
  0.000761 seconds (98 allocations: 38.266 KiB)
(0.6064777984403792, 0.0012501174586413868)
julia> @time MvNormalCDF.mvnormcdf(μ, Σ, a, b)
  0.000668 seconds (98 allocations: 38.266 KiB)
(0.6057553050182675, 0.0019361170915866777)

=#
#=
r = td[10,1]
a = td[10,2]
b = td[10,3]
m = td[10,4]


julia> @benchmark  MvNormalCDF.mvnormcdf($r, $a, $b; m=$m, rng = StableRNG(1234))
BenchmarkTools.Trial: 1525 samples with 1 evaluation.
 Range (min … max):  3.083 ms …  10.807 ms  ┊ GC (min … max): 0.00% … 69.49%
 Time  (median):     3.180 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   3.268 ms ± 311.169 μs  ┊ GC (mean ± σ):  0.15% ±  1.78%

   ▃█▅▁
  ▆████▇▇▇▆▄▄▃▃▃▃▃▃▂▃▃▂▃▂▃▂▂▂▂▂▂▁▂▂▂▂▂▂▂▂▂▁▂▂▂▂▂▁▁▂▂▂▂▂▂▁▁▂▂▂ ▃
  3.08 ms         Histogram: frequency by time        4.38 ms <

 Memory estimate: 52.70 KiB, allocs estimate: 63.

=#