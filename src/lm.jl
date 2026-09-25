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
| `damping` | `:marquardt` damps with `diag(J'J)`; `:levenberg` damps with the identity |
| `accel` | add geodesic acceleration (below) |
| `accel_h`, `accel_alpha` | its finite-difference step, and the largest accepted `2|a|/|v|` |

Exhausting `lambda_max` without an improving step returns `converged = false`.

`:levenberg` exists because Marquardt scaling suits this problem badly when
`sigma_y` is small. The diagonal of `J'J` for a state variable is dominated by
the model term (~ `1/sigma_y^2`), yet the directions the trajectory actually has
to move along -- towards solutions of the ODE -- have curvature of only
~ `1/sigma_x^2`. Scaled damping then smothers exactly those directions, and the
iteration crawls. With `:levenberg`, `lambda0` and `lambda_max` are taken
relative to `max(diag(J'J))`; `lambda_min` stays absolute.

`accel = true` adds geodesic acceleration (Transtrum & Sethna 2012,
arXiv:1201.5885; as in GSL's `lmaccel`). The Gauss-Newton step `v` is followed
by a second-order correction `a` for the curvature of the residual along `v`,

    r_vv ~ (2/h) * ( (r(z + h v) - r(z)) / h - J v )
    a    = -(J'J + lambda D) \\ (J' r_vv)          (same factorisation)
    step = v + a / 2

rejected as unreliable when `2|a| / |v| > accel_alpha`. It costs one residual
evaluation and one extra back-substitution per step. Here the curvature is large
where the trajectory has to bend towards solutions of the ODE, and plain
Gauss-Newton only creeps there (gain ratio ~0.6 for hundreds of steps); with
acceleration, fits at `sigma_y = 1e-3` converge in about a tenth of the steps.
"""
function levenberg_marquardt(fr, fJ, z0::AbstractVector{T};
                             maxiter::Int = 1000,
                             ftol::Real = 5e-8,
                             xtol::Real = 1e-12,
                             gtol::Real = 1e-10,
                             lambda0::Real = 1e-3,
                             lambda_min::Real = 1e-12,
                             lambda_max::Real = 1e12,
                             damping::Symbol = :marquardt,
                             accel::Bool = false,
                             accel_h::Real = 0.1,
                             accel_alpha::Real = 0.75) where {T<:Real}
    damping in (:marquardt, :levenberg) ||
        throw(ArgumentError("damping must be :marquardt or :levenberg"))
    z = Vector{T}(z0)
    r = fr(z)
    C = cost(r)

    J = fJ(z)
    g = J' * r
    H = J' * J
    marquardt = damping === :marquardt
    dH = marquardt ? _damping_diagonal(H) : ones(T, length(z))

    scale = marquardt ? one(T) : maximum(diag(H))
    lambda = T(lambda0) * scale
    lambda_max = lambda_max * scale
    nu = T(2)                                      # failure growth, doubles on
    iter = 0                                       # each consecutive failure
    converged = false
    solver = _StepSolver()                         # keeps the symbolic factorisation

    while iter < maxiter
        iter += 1

        if norm(g, Inf) <= gtol
            converged = true
            break
        end

        v = _lm_step!(solver, H, dH, g, lambda)
        if v === nothing                           # not positive definite
            lambda *= nu
            nu *= 2
            lambda > lambda_max && break
            continue
        end

        delta = v
        if accel
            rvv = (2 / accel_h) .* ((fr(z .+ accel_h .* v) .- r) ./ accel_h .- J * v)
            a = -(solver.F \ (J' * rvv))
            if !all(isfinite, a) || 2 * norm(a) > accel_alpha * norm(v)
                lambda *= nu                       # curvature too large to trust:
                nu *= 2                            # shorten the step instead
                lambda > lambda_max && break
                continue
            end
            delta = v .+ a ./ 2
        end

        znew = z .+ delta
        rnew = fr(znew)
        Cnew = cost(rnew)

        # Gain ratio: achieved reduction over the reduction the damped linear
        # model promised for the Gauss-Newton step. Positive means the step is
        # worth taking.
        pred = dot(v, lambda .* dH .* v .- g) / 2
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
            marquardt && (dH = _damping_diagonal(H))
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

"""
Solves the damped normal equations, step after step.

The sparsity pattern of `J'J` does not change during a fit, so the symbolic
analysis of the sparse Cholesky (the fill-reducing ordering, most of the cost)
is done once and only the numeric factorisation is repeated. If the pattern
does change -- a Jacobian entry that happens to be exactly zero is not stored --
the analysis is simply redone.
"""
mutable struct _StepSolver
    F::Any
    colptr::Vector{Int}
    rowval::Vector{Int}
end
_StepSolver() = _StepSolver(nothing, Int[], Int[])

_factorise!(S::_StepSolver, A::AbstractMatrix) = (S.F = cholesky(Symmetric(A)); S)

function _factorise!(S::_StepSolver, A::SparseMatrixCSC)
    if S.F !== nothing && A.colptr == S.colptr && A.rowval == S.rowval
        cholesky!(S.F, Symmetric(A))
    else
        S.F = nothing                              # stays unset if analysis throws
        S.F = cholesky(Symmetric(A))
        S.colptr, S.rowval = copy(A.colptr), copy(A.rowval)
    end
    return S
end

"One damped Gauss-Newton step, or `nothing` if the damped system is not SPD."
function _lm_step!(S::_StepSolver, H::AbstractMatrix, dH::AbstractVector,
                   g::AbstractVector, lambda::Real)
    try
        _factorise!(S, H + lambda * Diagonal(dH))
        return -(S.F \ g)
    catch err
        err isa InterruptException && rethrow()
        return nothing
    end
end
