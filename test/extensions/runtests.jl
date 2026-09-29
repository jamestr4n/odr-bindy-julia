# =============================================================================
# Tests for the components that live in package extensions.
#
#     julia --project=test/extensions test/extensions/runtests.jl
#
# The environment in this folder adds DataDrivenDiffEq, NonlinearSolve and
# Symbolics to ODRBINDy (from ../..). The first run precompiles them, which can
# take many minutes; loading DataDrivenDiffEq takes a minute or two after that.
# =============================================================================

using ODRBINDy
using NonlinearSolve
using DataDrivenDiffEq, Symbolics

include(joinpath(@__DIR__, "..", "conformance.jl"))
include(joinpath(@__DIR__, "..", "problems.jl"))

quiet(seed = 1) = ODROptions(verbose = 0, rng = MersenneTwister(seed))

@testset "ODRBINDy extensions" begin

@testset "NonlinearSolveOptimiser" begin
    @test Base.get_extension(ODRBINDy, :ODRBINDyNonlinearSolveExt) !== nothing

    for alg in (TrustRegion(), LevenbergMarquardt())
        test_optimiser(NonlinearSolveOptimiser(alg))
    end

    # a full-library ODR fit lands where BuiltinLM does
    Xd, t, sx, _ = vanderpol_data()
    prob = ODRProblem(Xd, t, PolynomialLibrary(2, 3);
                      sigma_x = sx, sigma_y = 1e-2, sigma_p = 1e2)
    mask = trues(10, 2)
    xi0 = bootstrap_ridge(prob, mask, prob.Xdata; rng = MersenneTwister(3))
    ref = fit_model(prob, mask, quiet(); optimiser = BuiltinLM(), xi0 = xi0)
    for alg in (TrustRegion(), LevenbergMarquardt())
        f = fit_model(prob, mask, quiet(); optimiser = NonlinearSolveOptimiser(alg),
                      xi0 = xi0)
        @test f.converged
        @test f.Xi ≈ ref.Xi atol = 1e-4
        @test f.nlevidence ≈ ref.nlevidence atol = 1e-3
    end

    # and selection through it recovers the model
    Xd, t, sx, true_osc = oscillator_data()
    osc = ODRProblem(Xd, t, PolynomialLibrary(2, 1);
                     sigma_x = sx, sigma_y = 1e-2, sigma_p = 1e2)
    test_selector(GreedyBackward(), osc, true_osc;
                  optimiser = NonlinearSolveOptimiser(TrustRegion()))
end

@testset "BasisLibrary" begin
    @test Base.get_extension(ODRBINDy, :ODRBINDyDataDrivenDiffEqExt) !== nothing

    @variables x y
    lib = BasisLibrary(Basis(polynomial_basis([x, y], 3), [x, y]))
    test_library(lib, 2)
    @test state_names(lib, 2) == ["x", "y"]

    # the same functions as PolynomialLibrary, in another order
    poly = PolynomialLibrary(2, 3)
    X = randn(MersenneTwister(5), 20, 2)
    Tb, Tp = theta(lib, X), theta(poly, X)
    perm = [findfirst(j -> Tb[:, i] ≈ Tp[:, j], 1:nterms(poly)) for i in 1:nterms(lib)]
    @test sort(perm) == 1:nterms(poly)
    @test dtheta(lib, X) ≈ dtheta(poly, X)[:, perm, :]

    # parameters are held at their defaults, and dtheta stays exact
    @variables w = 0.7
    trig = BasisLibrary(Basis([x, y, sin(w * x), cos(w * y), x * sin(y)], [x, y];
                              parameters = [w]))
    test_library(trig, 2)
    @test theta(trig, [1.0 2.0])[3] ≈ sin(0.7)

    # fits select the same model as with PolynomialLibrary
    Xd, t, sx, true_osc = oscillator_data()
    lin = BasisLibrary(Basis(Num[1, x, y], [x, y]))
    osc = ODRProblem(Xd, t, lin; sigma_x = sx, sigma_y = 1e-2, sigma_p = 1e2)
    res = odr_bindy(osc, quiet())
    oscp = ODRProblem(Xd, t, PolynomialLibrary(2, 1);
                      sigma_x = sx, sigma_y = 1e-2, sigma_p = 1e2)
    resp = odr_bindy(oscp, quiet())
    @test res.mask == resp.mask == true_osc
    @test res.nlevidence ≈ resp.nlevidence rtol = 1e-8

    # what the library interface cannot express yet is refused
    @variables t_
    @test_throws ArgumentError BasisLibrary(Basis([x, t_ * x], [x]; iv = t_))
end

end # testset
