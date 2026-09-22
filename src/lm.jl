# =============================================================================
# Levenberg-Marquardt for sparse nonlinear least squares.
#
# Replaces MATLAB's `lsqnonlin` with 'trust-region-reflective'. Kept in-package
# so that ODRBINDy has no dependency outside the standard library; the only
# contract the rest of the code relies on is `LMResult`, so swapping in
# NonlinearSolve.jl or LeastSquaresOptim.jl later touches nothing else.
#
# Minimise C(z) = 1/2 ||r(z)||^2 by repeatedly solving the damped normal
# equations
#
#     ( J'J + lambda * diag(J'J) ) delta = -J' r
#
# lambda -> 0   : Gauss-Newton, fast near the solution
# lambda -> inf : gradient descent with a vanishing step, safe far from it
#
# lambda is adapted by the Nielsen / Madsen gain-ratio rule: compare the cost
# reduction actually achieved against the one the linearised model predicted,
#
#     rho = (C - Cnew) / (0.5 * delta' * (lambda * D * delta - g))
#
# and shrink lambda smoothly when the model is trustworthy (rho near 1), grow it
# geometrically while steps keep failing. A fixed shrink/grow factor makes the
# damping oscillate on badly scaled problems -- and this one is badly scaled,
# since 1/sigma_y is large and 1/sigma_x is not.
#
# NOTE: the normal equations are factorised with a *sparse* Cholesky, which in
# the standard library (SuiteSparse/CHOLMOD) supports `Float64` only. That is
# the working precision throughout the package.
# =============================================================================

"""
    LMResult

Outcome of [`levenberg_marquardt`](@ref).

- `z`          the final iterate
- `r`          the residual there
- `cost`       `1/2 ||r||^2`
- `iterations` LM steps taken (accepted *and* rejected)
- `converged`  whether a tolerance was met before `maxiter`

`converged == false` is deliberately *informative* rather than an error: paper
§2.3 notes that a trial model missing a necessary term frequently fails to
converge, and the greedy search uses that as a reason to keep the term.
"""
struct LMResult{T<:Real}
    z::Vector{T}
    r::Vector{T}
    cost::T
    iterations::Int
    converged::Bool
end

"""
    levenberg_marquardt(fr, fJ, z0; kwargs...) -> LMResult

Minimise `1/2 ||fr(z)||^2` starting from `z0`, with `fJ(z)` the (sparse)
Jacobian of `fr`.

| keyword | meaning |
|---|---|
| `maxiter` | iteration cap |
| `ftol` | stop on a relative cost decrease below this |
| `xtol` | stop on a relative step below this |
| `gtol` | stop on `norm(J'r, Inf)` below this |
| `lambda0`, `lambda_min`, `lambda_max` | damping schedule |

Exhausting `lambda_max` without an improving step returns `converged = false`.
"""
function levenberg_marquardt(fr, fJ, z0::AbstractVector{T};
                             maxiter::Int = 1000,
                             ftol::Real = 5e-8,
                             xtol::Real = 1e-12,
                             gtol::Real = 1e-10,
                             lambda0::Real = 1e-3,
                             lambda_min::Real = 1e-12,
                             lambda_max::Real = 1e12) where {T<:Real}
    z = Vector{T}(z0)
    r = fr(z)
    C = cost(r)

    J = fJ(z)
    g = J' * r
    H = J' * J
    dH = _damping_diagonal(H)

    lambda = T(lambda0)
    nu = T(2)                                      # failure growth, doubles on
    iter = 0                                       # each consecutive failure
    converged = false

    while iter < maxiter
        iter += 1

        if norm(g, Inf) <= gtol
            converged = true
            break
        end

        delta = _lm_step(H, dH, g, lambda)
        if delta === nothing                       # not positive definite
            lambda *= nu
            nu *= 2
            lambda > lambda_max && break
            continue
        end

        znew = z .+ delta
        rnew = fr(znew)
        Cnew = cost(rnew)

        # Gain ratio: achieved reduction over the reduction the damped linear
        # model promised. Positive means the step is worth taking.
        pred = dot(delta, lambda .* dH .* delta .- g) / 2
        rho = pred > 0 ? (C - Cnew) / pred : T(-1)

        if isfinite(Cnew) && rho > 0
            snorm = norm(delta)
            dC = C - Cnew
            z, r = znew, rnew
            Cold, C = C, Cnew
            lambda = max(lambda * max(T(1) / 3, 1 - (2rho - 1)^3), T(lambda_min))
            nu = T(2)

            if dC <= ftol * max(Cold, eps(T)) || snorm <= xtol * (xtol + norm(z))
                converged = true
                break
            end

            J = fJ(z)                              # only recomputed on success
            g = J' * r
            H = J' * J
            dH = _damping_diagonal(H)
        else
            lambda *= nu
            nu *= 2
            lambda > lambda_max && break
        end
    end

    return LMResult(z, r, C, iter, converged)
end

"Diagonal of `J'J`, floored so that the damping term never vanishes."
function _damping_diagonal(H::AbstractMatrix{T}) where {T}
    d = Vector{T}(diag(H))
    floorval = sqrt(eps(T))
    @inbounds for k in eachindex(d)
        d[k] = max(d[k], floorval)
    end
    return d
end

"One damped Gauss-Newton step, or `nothing` if the damped system is not SPD."
function _lm_step(H::AbstractMatrix, dH::AbstractVector, g::AbstractVector, lambda::Real)
    try
        F = cholesky(Symmetric(H + lambda * Diagonal(dH)))
        return -(F \ g)
    catch err
        err isa InterruptException && rethrow()
        return nothing
    end
end
