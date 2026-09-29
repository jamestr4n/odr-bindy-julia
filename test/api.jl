# =============================================================================
# The public API, frozen for v1.0 (docs/API.md).
#
# The export list is pinned, so adding, removing or renaming an exported name
# fails here and has to be a deliberate change, made in docs/API.md too.
# Included by test/runtests.jl.
# =============================================================================

const EXPORTED_API = Set([
    # libraries
    :AbstractLibrary, :PolynomialLibrary, :FourierLibrary, :CustomLibrary,
    :CombinedLibrary, :BasisLibrary, :nterms, :theta, :dtheta, :term_names,
    # discretisations
    :AbstractDiscretisation, :FiniteDifference, :WeakForm, :operators,
    :central_fd_coefficients, :finite_difference_matrices, :collocation_times,
    :fornberg_weights, :derivative_matrix,
    # problem setup
    :ODRProblem, :ODRHyperParameters, :ODROptions,
    # optimisers
    :AbstractOptimiser, :BuiltinLM, :NonlinearSolveOptimiser, :optimise,
    :levenberg_marquardt, :LMResult, :gauss_newton_decrement,
    # fitting one model
    :bootstrap_ridge, :reduced_hessian, :neg_log_evidence,
    :ODRFit, :fit_model, :fit_model_multistart,
    # model selection and the algorithm
    :AbstractModelSelector, :GreedyBackward, :BeamSearch, :Exhaustive,
    :select_model, :n_models,
    :ODRResult, :odr_bindy, :print_model, :state_names,
])

@testset "public API" begin
    @testset "export list is frozen" begin
        exported = Set(n for n in names(ODRBINDy)
                       if n !== :ODRBINDy && Base.isexported(ODRBINDy, n))
        @test isempty(setdiff(exported, EXPORTED_API))    # nothing new
        @test isempty(setdiff(EXPORTED_API, exported))    # nothing gone
        @test all(n -> isdefined(ODRBINDy, n), EXPORTED_API)
    end

    @testset "residual and jacobian are public, not exported" begin
        for n in (:residual, :jacobian)
            @test isdefined(ODRBINDy, n)
            @test !Base.isexported(ODRBINDy, n)
            VERSION >= v"1.11" && @test Base.ispublic(ODRBINDy, n)
        end
    end

    @testset "deprecated ODROptions keywords" begin
        # Each old keyword warns, and still configures the default components
        # exactly as the new ones do.
        old = @test_deprecated ODROptions(
            lm_damping = :levenberg, lm_accel = true,
            ftol = 1e-9, xtol = 1e-13, gtol = 1e-11,
            lm_maxiter = 60, lm_maxiter_refine = 3000, warm_start = false,
            trial_xi_init = :regress, refine_after_removal = false,
            stop_after_rises = 3)
        @test ODRBINDy.default_optimiser(old) ==
              BuiltinLM(damping = :levenberg, accel = true,
                        ftol = 1e-9, xtol = 1e-13, gtol = 1e-11)
        @test ODRBINDy.default_selector(old) ==
              GreedyBackward(trial_maxiter = 60, refine_maxiter = 3000,
                             warm_start = false, trial_xi_init = :regress,
                             refine_after_removal = false, stop_after_rises = 3)

        # Without them, the defaults are the components' own defaults.
        new = ODROptions()
        @test ODRBINDy.default_optimiser(new) == BuiltinLM()
        @test ODRBINDy.default_selector(new) == GreedyBackward()
        @test_throws MethodError ODROptions(not_an_option = 1)
    end
end
