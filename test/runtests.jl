# =============================================================================
# Tests for the pluggable components (Aim 2b).
#
#     julia --project=. test/runtests.jl      (or `] test`)
#
# Components that need other packages (BasisLibrary, NonlinearSolveOptimiser)
# are tested in test/extensions/runtests.jl. Bit-identical results for the
# default components are checked by test/refactor_fixture.jl. The frozen public
# API is checked by test/api.jl, included at the end.
# =============================================================================

include("conformance.jl")
include("problems.jl")

quiet(seed = 1) = ODROptions(verbose = 0, rng = MersenneTwister(seed))

@testset "ODRBINDy components" begin

@testset "libraries" begin
    @testset "PolynomialLibrary" begin
        test_library(PolynomialLibrary(3, 2), 3)
    end

    @testset "FourierLibrary" begin
        lib = FourierLibrary(2, 2; varnames = ["a", "b"])
        test_library(lib, 2)
        @test term_names(lib) == ["sin(a)", "cos(a)", "sin(b)", "cos(b)",
                                  "sin(2*a)", "cos(2*a)", "sin(2*b)", "cos(2*b)"]
        @test state_names(lib, 2) == ["a", "b"]
        X = [0.3 -1.2]
        @test theta(lib, X) ≈ [sin(0.3) cos(0.3) sin(-1.2) cos(-1.2) sin(0.6) cos(0.6) sin(-2.4) cos(-2.4)]
    end

    @testset "CustomLibrary" begin
        lib = CustomLibrary(2,
            [x -> 1.0, x -> x[1] * x[2], x -> exp(-x[1]^2), x -> tanh(x[2])],
            [x -> [0.0, 0.0], x -> [x[2], x[1]], x -> [-2x[1] * exp(-x[1]^2), 0.0],
             x -> [0.0, 1 - tanh(x[2])^2]];
            names = ["1", "x*y", "exp(-x^2)", "tanh(y)"], varnames = ["x", "y"])
        test_library(lib, 2)
        @test state_names(lib, 2) == ["x", "y"]
        @test_throws ArgumentError CustomLibrary(2, [x -> 1.0], [])
        # a custom copy of a polynomial library gives the same Theta
        poly = PolynomialLibrary(2, 1)
        same = CustomLibrary(2, [x -> 1.0, x -> x[1], x -> x[2]],
                             [x -> [0.0, 0.0], x -> [1.0, 0.0], x -> [0.0, 1.0]])
        X = randn(MersenneTwister(2), 10, 2)
        @test theta(same, X) == theta(poly, X)
        @test dtheta(same, X) == dtheta(poly, X)
    end

    @testset "CombinedLibrary" begin
        a = PolynomialLibrary(2, 2; varnames = ["x", "y"])
        b = FourierLibrary(2, 1; varnames = ["x", "y"])
        lib = CombinedLibrary(a, b)
        test_library(lib, 2)
        X = randn(MersenneTwister(3), 10, 2)
        @test nterms(lib) == nterms(a) + nterms(b)
        @test theta(lib, X) == hcat(theta(a, X), theta(b, X))
        @test term_names(lib) == vcat(term_names(a), term_names(b))
        @test state_names(lib, 2) == ["x", "y"]
    end

    @testset "BasisLibrary needs its extension" begin
        # the constructor from a Basis is only defined by the extension
        @test_throws MethodError BasisLibrary(nothing)
    end
end

@testset "discretisations" begin
    t = collect(range(0, 10; length = 400))
    tu = uneven_times(400, 10.0, 0.4, MersenneTwister(4))

    @testset "fornberg_weights" begin
        for n in (2, 4, 6, 8)
            s = collect(-n ÷ 2:n ÷ 2) .* 1.0
            @test fornberg_weights(0.0, s, 1)[:, 2] ≈ central_fd_coefficients(n) atol = 1e-13
        end
        # second derivative on uneven nodes, exact for a quadratic
        x = [0.0, 0.3, 1.1, 1.5]
        w = fornberg_weights(0.7, x, 2)
        @test sum(w[:, 3] .* x .^ 2) ≈ 2
        @test sum(w[:, 1] .* x .^ 3) ≈ 0.7^3
    end

    @testset "FiniteDifference, uniform" begin
        test_discretisation(FiniteDifference(6), t; atol = 1e-8)
        # identical to the matrices the original API builds
        @test operators(FiniteDifference(6), t) ==
              finite_difference_matrices(400, 6, t[2] - t[1])
    end

    @testset "FiniteDifference, uneven" begin
        test_discretisation(FiniteDifference(6), tu; atol = 1e-8)
        test_discretisation(FiniteDifference(2), tu; atol = 1e-2)
        @test collocation_times(FiniteDifference(6), tu) == tu[4:(end - 3)]
        # sixth order: halving the spacing shrinks the error ~64x
        err(N) = (tt = uneven_times(N, 10.0, 0.4, MersenneTwister(N));
                  (I, D) = operators(FiniteDifference(6), tt);
                  maximum(abs, D * sin.(tt) .- I * cos.(tt)))
        @test err(100) / err(200) > 30
        @test_throws ArgumentError operators(FiniteDifference(6), reverse(t))
    end

    @testset "WeakForm" begin
        for tt in (t, tu), disc in (WeakForm(8), WeakForm(5; stride = 3, degree = 2))
            test_discretisation(disc, tt; atol = 1e-8)
            IMat, _ = operators(disc, tt)
            @test all(sum(IMat; dims = 2) .≈ 1)          # each phi integrates to 1
            @test all(>=(0), nonzeros(IMat))
        end
        IMat, _ = operators(WeakForm(8; stride = 4), t)
        @test size(IMat, 1) == (400 - 17) ÷ 4 + 1
        @test collocation_times(WeakForm(8; stride = 4), t) == t[9:4:(9 + 4 * (size(IMat, 1) - 1))]
        @test_throws ArgumentError WeakForm(0)
        @test_throws ArgumentError operators(WeakForm(300), t)
    end

    @testset "derivative_matrix" begin
        Dt = derivative_matrix(tu, 6)
        @test size(Dt) == (400, 400)
        @test maximum(abs, Dt * sin.(tu) .- cos.(tu)) < 1e-6   # one-sided ends too
    end
end

@testset "optimisers" begin
    @testset "BuiltinLM" begin
        test_optimiser(BuiltinLM())
        test_optimiser(BuiltinLM(damping = :levenberg, accel = true))
    end

    @testset "gauss_newton_decrement" begin
        J = sparse([1.0 0.0; 0.0 2.0; 1.0 1.0])
        r = [0.5, -1.0, 0.25]
        g = J' * r
        @test gauss_newton_decrement(r, J) ≈ dot(g, (J' * J) \ g) / 2
        @test gauss_newton_decrement(zeros(3), J) == 0
    end

    @testset "NonlinearSolveOptimiser needs its extension" begin
        opt = NonlinearSolveOptimiser(:any_algorithm; xtol = 1e-9)
        @test opt.xtol == 1e-9
        @test_throws MethodError optimise(opt, identity, identity, [1.0]; maxiter = 1)
    end
end

@testset "selectors" begin
    Xd, t, sx, true_osc = oscillator_data()
    osc = ODRProblem(Xd, t, PolynomialLibrary(2, 1; varnames = ["x", "y"]);
                     sigma_x = sx, sigma_y = 1e-2, sigma_p = 1e2)

    @testset "GreedyBackward" begin
        test_selector(GreedyBackward(), osc, true_osc)
    end

    @testset "Exhaustive" begin
        res = test_selector(Exhaustive(), osc, true_osc)
        @test length(res.history) == n_models(3, 2) == 49
        @test length(unique(h.mask for h in res.history)) == 49
        # greedy found the evidence-optimal model
        g = odr_bindy(osc, quiet(); selector = GreedyBackward())
        @test g.mask == res.mask
        @test n_models(10, 3) == 1023^3
        @test_throws ArgumentError odr_bindy(osc, quiet(); selector = Exhaustive(max_models = 10))
    end

    @testset "BeamSearch" begin
        test_selector(BeamSearch(3), osc, true_osc)
        @test_throws ArgumentError BeamSearch(0)
    end

    Xd, t, sx, true_vdp = vanderpol_data()
    vdp = ODRProblem(Xd, t, PolynomialLibrary(2, 3; varnames = ["x", "y"]);
                     sigma_x = sx, sigma_y = 1e-2, sigma_p = 1e2)

    @testset "BeamSearch(1) is GreedyBackward" begin
        for kw in ((;), (trial_xi_init = :regress, refine_after_removal = false,
                         stop_after_rises = 3, trial_maxiter = 60))
            g = odr_bindy(vdp, quiet(); selector = GreedyBackward(; kw...))
            b = odr_bindy(vdp, quiet(); selector = BeamSearch(1; kw...))
            @test b.mask == g.mask
            @test b.Xi == g.Xi
            @test b.X == g.X
            @test b.nlevidence == g.nlevidence
            @test b.history == g.history
        end
    end

    @testset "BeamSearch(3) on Van der Pol" begin
        test_selector(BeamSearch(3), vdp, true_vdp)
    end
end

@testset "end to end" begin
    @testset "WeakForm recovers Van der Pol" begin
        Xd, t, sx, true_vdp = vanderpol_data()
        prob = ODRProblem(Xd, t, PolynomialLibrary(2, 3); discretisation = WeakForm(8),
                          sigma_x = sx, sigma_y = 1e-2, sigma_p = 1e2)
        @test odr_bindy(prob, quiet()).mask == true_vdp
    end

    @testset "uneven sampling recovers Van der Pol" begin
        tu = uneven_times(400, 7.98, 0.4, MersenneTwister(5))
        Xd, _, sx, true_vdp = vanderpol_data(t = tu)
        for disc in (FiniteDifference(6), WeakForm(8))
            prob = ODRProblem(Xd, tu, PolynomialLibrary(2, 3); discretisation = disc,
                              sigma_x = sx, sigma_y = 1e-2, sigma_p = 1e2)
            @test odr_bindy(prob, quiet()).mask == true_vdp
        end
    end

    @testset "Fourier terms recover a pendulum" begin
        Xd, t, sx, true_pend, _ = pendulum_data()
        lib = CombinedLibrary(PolynomialLibrary(2, 1; varnames = ["θ", "ω"]),
                              FourierLibrary(2, 1; varnames = ["θ", "ω"]))
        prob = ODRProblem(Xd, t, lib; sigma_x = sx, sigma_y = 1e-2, sigma_p = 1e2)
        res = odr_bindy(prob, quiet())
        @test res.mask == true_pend
        @test res.Xi[4, 2] ≈ -2 rtol = 0.05                 # the sin θ coefficient
        io = IOBuffer()
        print_model(res, lib; io = io)
        @test occursin("dω/dt", String(take!(io)))
    end
end

include("api.jl")

end # testset
