# =============================================================================
# Fit one sparsity pattern.
#
# Replaces ODR_BINDy_Regression.m and ODR_BINDy_Regression_MultiStart.m.
#
#   1. bootstrap ridge regression for an initial Xi          (initialguess.jl)
#   2. pack z = [vec(X0); xi0]                               (problem.jl)
#   3. Levenberg-Marquardt on the ODR residual               (lm.jl)
#   4. unpack, scatter xi back into the M x D matrix
#   5. score by Bayesian evidence                            (evidence.jl)
#
# A non-converged fit is reported with `nlevidence = Inf`, which the greedy
# search reads as "do not remove that term" (paper section 2.3, point 3).
# =============================================================================

"""
    ODRFit

One fitted model.

- `Xi`         `M x D` coefficients, zero outside `mask`
- `X`          `Nx x D` denoised states
- `nlevidence` negative log evidence, `Inf` if the optimisation did not converge
- `cost`       `L(X, Xi)` at the optimum
- `mask`       the sparsity pattern that was fitted
- `converged`, `iterations` from the optimiser
"""
struct ODRFit{T<:Real}
    Xi::Matrix{T}
    X::Matrix{T}
    nlevidence::T
    cost::T
    mask::BitMatrix
    converged::Bool
    iterations::Int
end

"""
    fit_model(prob, mask, opts; X0, xi0, maxiter, rng) -> ODRFit

Jointly optimise the states and the active coefficients of `mask`, then score
the result.

`X0` is the starting guess for the states — the raw data for a cold start, or
the previous model's denoised states for a warm start (paper section 2.3, point 2:
the single biggest speedup in the algorithm). `xi0` skips the bootstrap when a
coefficient guess is already available.
"""
function fit_model(prob::ODRProblem{L,T}, mask::AbstractMatrix{Bool},
                   opts::ODROptions = ODROptions();
                   X0::AbstractMatrix = prob.Xdata,
                   xi0::Union{Nothing,AbstractVector} = nothing,
                   maxiter::Int = opts.lm_maxiter_refine,
                   rng::AbstractRNG = opts.rng) where {L,T}
    size(mask) == (prob.M, prob.D) ||
        throw(DimensionMismatch("mask must be $(prob.M) x $(prob.D)"))
    size(X0) == (prob.Nx, prob.D) ||
        throw(DimensionMismatch("X0 must be $(prob.Nx) x $(prob.D)"))
    any(mask) || throw(ArgumentError("mask must keep at least one term"))

    fr = z -> residual(prob, mask, z)
    fJ = z -> jacobian(prob, mask, z)

    xistart = xi0 === nothing ?
        bootstrap_ridge(prob, mask, X0; nsamples = opts.bootstrap_samples,
                        bragging = opts.bragging, rng = rng) :
        collect(xi0)

    z0 = vcat(vec(Matrix{T}(X0)), Vector{T}(xistart))
    res = levenberg_marquardt(fr, fJ, z0; maxiter = maxiter,
                              ftol = opts.ftol, xtol = opts.xtol, gtol = opts.gtol,
                              damping = opts.lm_damping, accel = opts.lm_accel)

    X, xi = unpack(prob, res.z)
    Xi = zeros(T, prob.M, prob.D)
    Xi[mask] .= xi                                  # inverse of Xi[mask]

    res.converged || return ODRFit(Xi, Matrix(X), T(Inf), T(res.cost),
                                   BitMatrix(mask), false, res.iterations)

    nle, _ = neg_log_evidence(prob, mask, fJ(res.z), res.r)
    return ODRFit(Xi, Matrix(X), nle, T(res.cost), BitMatrix(mask),
                  true, res.iterations)
end

"""
    fit_model_multistart(prob, mask, opts; X0, nstarts, maxiter) -> ODRFit

Run `fit_model` from several independent bootstrap starting points and keep the
best-scoring result.

The loss of eq. (6) is not convex, and each bootstrap resample lands in a
slightly different place, so this is cheap multi-start global optimisation. Used
for the initial full-library fit, where there is no previous model to warm-start
from.
"""
function fit_model_multistart(prob::ODRProblem{L,T}, mask::AbstractMatrix{Bool},
                              opts::ODROptions = ODROptions();
                              X0::AbstractMatrix = prob.Xdata,
                              nstarts::Int = opts.n_multistart,
                              maxiter::Int = opts.lm_maxiter_refine) where {L,T}
    best = nothing
    for s in 1:max(nstarts, 1)
        f = fit_model(prob, mask, opts; X0 = X0, maxiter = maxiter)
        if opts.verbose >= 2
            @printf("    start %2d/%2d: -log(E) = %s  (%d LM steps%s)\n",
                    s, max(nstarts, 1), _fmt_evidence(f.nlevidence), f.iterations,
                    f.converged ? "" : ", did not converge")
        end
        if best === nothing || _better(f, best)
            best = f
        end
    end
    return best::ODRFit{T}
end

"Prefer higher evidence; fall back on the loss when no start scored finitely."
_better(a::ODRFit, b::ODRFit) =
    isfinite(a.nlevidence) || isfinite(b.nlevidence) ?
        a.nlevidence < b.nlevidence : a.cost < b.cost

_fmt_evidence(v::Real) = isfinite(v) ? @sprintf("%.6g", v) : "Inf"
