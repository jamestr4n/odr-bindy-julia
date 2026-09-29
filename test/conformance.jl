# =============================================================================
# Conformance checks, one per extension point (docs/DESIGN.md §6).
#
# Each takes an implementation and checks it against its contract, so it works
# for the package's own components and for a user's:
#
#     include("test/conformance.jl")
#     @testset "my library" begin test_library(MyLibrary(...), 3) end
# =============================================================================

using ODRBINDy
using ODRBINDy: cost
using Test, LinearAlgebra, SparseArrays, Random

"""
    test_library(lib, D; X, h = 1e-6, rtol = 1e-6)

Shapes and names, and `dtheta` against central differences of `theta`.
"""
function test_library(lib::AbstractLibrary, D::Int;
                      X = randn(MersenneTwister(0), 25, D), h = 1e-6, rtol = 1e-6)
    M = nterms(lib)
    Th = theta(lib, X)
    dTh = dtheta(lib, X)
    @test size(Th) == (size(X, 1), M)
    @test size(dTh) == (size(X, 1), M, D)
    @test length(term_names(lib)) == M
    @test length(state_names(lib, D)) == D
    for e in 1:D
        Xp = copy(X); Xp[:, e] .+= h
        Xm = copy(X); Xm[:, e] .-= h
        fd = (theta(lib, Xp) .- theta(lib, Xm)) ./ (2h)
        @test isapprox(dTh[:, :, e], fd; rtol = rtol, atol = rtol)
    end
end

"""
    test_discretisation(disc, t; f = sin, df = cos, atol)

Sizes and types, and the defining property of the operators: for a smooth
function they reproduce the model with no error, `DMat*f ≈ IMat*f'`.
"""
function test_discretisation(disc::AbstractDiscretisation, t::AbstractVector;
                             f = sin, df = cos, atol = 1e-6)
    IMat, DMat = operators(disc, t)
    @test IMat isa SparseMatrixCSC{Float64}
    @test DMat isa SparseMatrixCSC{Float64}
    @test size(IMat) == size(DMat)
    @test size(IMat, 2) == length(t)
    @test length(collocation_times(disc, t)) == size(IMat, 1)
    @test maximum(abs, DMat * f.(t) .- IMat * df.(t)) < atol
end

"""
    test_optimiser(opt)

Solves a small least-squares problem with a known optimum and a non-zero
residual there, from a start that needs several steps. Then checks that it
reports `converged = false` when stopped after one step.

    r(z) = [z1^2 - 1, z1^2 - 3, 2(z2 - 0.5), sin(z2 - 0.5)]

has its minimum at `z = (√2, 0.5)` with cost 1.
"""
function test_optimiser(opt::AbstractOptimiser)
    fr(z) = [z[1]^2 - 1, z[1]^2 - 3, 2 * (z[2] - 0.5), sin(z[2] - 0.5)]
    fJ(z) = sparse([2z[1] 0.0; 2z[1] 0.0; 0.0 2.0; 0.0 cos(z[2] - 0.5)])
    z0 = [3.0, -2.0]

    res = optimise(opt, fr, fJ, z0; maxiter = 200)
    @test res isa LMResult
    @test res.converged
    @test res.z ≈ [sqrt(2), 0.5] atol = 1e-5
    @test res.r ≈ fr(res.z)
    @test res.cost ≈ cost(res.r)
    @test res.cost ≈ 1 atol = 1e-8

    capped = optimise(opt, fr, fJ, z0; maxiter = 1)
    @test !capped.converged
end

"""
    test_selector(sel, prob, true_mask; opts)

The selector recovers `true_mask` on `prob`, and returns the lowest `-log(E)`
in its history.
"""
function test_selector(sel::AbstractModelSelector, prob::ODRProblem,
                       true_mask::AbstractMatrix{Bool};
                       opts = ODROptions(verbose = 0, rng = MersenneTwister(1)),
                       optimiser = BuiltinLM())
    res = odr_bindy(prob, opts; optimiser = optimiser, selector = sel)
    @test res isa ODRResult
    @test res.mask == true_mask
    @test all(any(res.mask; dims = 1))
    @test res.nlevidence == minimum(h.nlevidence for h in res.history)
    @test res.nlevidence == res.fit.nlevidence
    return res
end
