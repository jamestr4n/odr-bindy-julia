# =============================================================================
# NonlinearSolveOptimiser: any NonlinearSolve.jl least-squares algorithm as an
# ODR-BINDy optimiser. The type lives in the core (src/lm.jl); this extension
# adds its `optimise` method.
# =============================================================================

module ODRBINDyNonlinearSolveExt

using ODRBINDy
using ODRBINDy: cost, gauss_newton_decrement
using LinearAlgebra: norm
using NonlinearSolve: NonlinearSolve, NonlinearFunction, NonlinearLeastSquaresProblem,
                      AbsNormSafeBestTerminationMode
using NonlinearSolve.NonlinearSolveBase: get_u, not_terminated

function ODRBINDy.optimise(opt::NonlinearSolveOptimiser, fr, fJ, z0::AbstractVector;
                           maxiter::Int)
    z0 = Vector{Float64}(z0)
    f = NonlinearFunction{false}((z, _) -> fr(z);
                                 jac = (z, _) -> fJ(z),
                                 resid_prototype = fr(z0),
                                 jac_prototype = fJ(z0))
    prob = NonlinearLeastSquaresProblem(f, z0)

    # NonlinearSolve's own stopping rule for least squares looks at the residual
    # norm, which an ODR residual never brings near zero, or waits for
    # `stalled_steps` steps shorter than `xtol`. Some algorithms (its
    # LevenbergMarquardt) rarely take steps that short, and would run to
    # `maxiter` every time. So the solver is stepped here, and stopped as soon
    # as the Gauss-Newton decrement test passes: the same test decides
    # `converged`, whatever the algorithm (see the NonlinearSolveOptimiser
    # docstring).
    tc = AbsNormSafeBestTerminationMode(Base.Fix2(norm, 2);
                                        max_stalled_steps = opt.stalled_steps)
    cache = NonlinearSolve.init(prob, opt.alg; maxiters = maxiter, abstol = opt.xtol,
                                termination_condition = tc, opt.kwargs...)

    z, r = z0, fr(z0)
    converged = _converged(opt, fJ, z, r)
    iterations = 0
    while !converged && not_terminated(cache)
        NonlinearSolve.step!(cache)
        iterations += 1
        znew = get_u(cache)
        znew == z && continue                     # a rejected step: nothing to test
        z = Vector{Float64}(znew)
        r = fr(z)
        converged = _converged(opt, fJ, z, r)
    end
    # a stop by NonlinearSolve's own rule may have rolled back to its best point
    zfinal = Vector{Float64}(get_u(cache))
    if zfinal != z
        z, r = zfinal, fr(zfinal)
        converged = _converged(opt, fJ, z, r)
    end
    return LMResult(z, r, cost(r), iterations, converged)
end

"One more Gauss-Newton step could lower the cost by at most `ftol` of it."
function _converged(opt::NonlinearSolveOptimiser, fJ, z, r)
    C = cost(r)
    return isfinite(C) && gauss_newton_decrement(r, fJ(z)) <= opt.ftol * max(C, eps())
end

end # module
