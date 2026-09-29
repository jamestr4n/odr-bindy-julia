# =============================================================================
# Problem definition, hyperparameters and solver options
# =============================================================================

"""
    ODRHyperParameters(sigma_x, sigma_y, sigma_p)

Noise and prior standard deviations, each allowed to vary pointwise.

- `sigma_x :: Nx x D`  measurement noise on the states
- `sigma_y :: Neq x D` model error on the collocated residual (truncation error
  of the discretisation, plus any stochastic forcing)
- `sigma_p :: M x D`   prior standard deviation on the coefficients

These are *not* tuned: each one is the inverse square root of the weight that
its term carries in the loss, fixed by what you believe about the data.
"""
struct ODRHyperParameters{T<:Real}
    sigma_x::Matrix{T}
    sigma_y::Matrix{T}
    sigma_p::Matrix{T}
end

"""
    ODRHyperParameters(; sigma_x, sigma_y, sigma_p, Nx, Neq, M, D)

Scalar-broadcasting convenience constructor.
"""
function ODRHyperParameters(; sigma_x::Real, sigma_y::Real, sigma_p::Real,
                              Nx::Int, Neq::Int, M::Int, D::Int)
    T = promote_type(typeof(float(sigma_x)), typeof(float(sigma_y)), typeof(float(sigma_p)))
    return ODRHyperParameters{T}(fill(T(sigma_x), Nx, D),
                                 fill(T(sigma_y), Neq, D),
                                 fill(T(sigma_p), M, D))
end

"""
    ODROptions(; n_multistart = 8, bootstrap_samples = 100, bragging = true,
                 verbose = 1, rng = Random.default_rng())

Settings shared by every optimiser and selector.

| option | meaning |
|---|---|
| `n_multistart` | independent bootstrap starts for the initial full-library fit |
| `bootstrap_samples` | resamples used by the ridge-regression initial guess |
| `bragging` | median (`true`) rather than mean (`false`) over bootstrap samples |
| `verbose` | 0 silent, 1 per removal, 2 per trial |
| `rng` | random number generator for the bootstrap |

# Deprecated keywords

The solver and search settings have moved into the components that use them.
The old keywords still work, with a deprecation warning, and are removed in
v1.0. Each one configures the *default* optimiser or selector, so it is ignored
when `odr_bindy` is given an `optimiser` or `selector` explicitly.

| deprecated keyword | use instead |
|---|---|
| `lm_damping`, `lm_accel`, `ftol`, `xtol`, `gtol` | `optimiser = BuiltinLM(damping, accel, ftol, xtol, gtol)` |
| `lm_maxiter` | `selector = GreedyBackward(trial_maxiter)` |
| `lm_maxiter_refine` | `selector = GreedyBackward(refine_maxiter)` |
| `warm_start`, `trial_xi_init`, `refine_after_removal`, `stop_after_rises` | `selector = GreedyBackward(...)`, same names |
"""
mutable struct ODROptions
    n_multistart::Int
    bootstrap_samples::Int
    bragging::Bool
    # deprecated: read only by `default_optimiser` and `default_selector`
    lm_maxiter::Int
    lm_maxiter_refine::Int
    lm_damping::Symbol
    lm_accel::Bool
    ftol::Float64
    xtol::Float64
    gtol::Float64
    warm_start::Bool
    trial_xi_init::Symbol
    refine_after_removal::Bool
    stop_after_rises::Int
    verbose::Int
    rng::AbstractRNG
end

function ODROptions(; n_multistart::Int = 8, bootstrap_samples::Int = 100,
                    bragging::Bool = true, verbose::Int = 1,
                    rng::AbstractRNG = Random.default_rng(),
                    lm_maxiter = nothing, lm_maxiter_refine = nothing,
                    lm_damping = nothing, lm_accel = nothing,
                    ftol = nothing, xtol = nothing, gtol = nothing,
                    warm_start = nothing, trial_xi_init = nothing,
                    refine_after_removal = nothing, stop_after_rises = nothing)
    lm = "optimiser = BuiltinLM"
    gb = "selector = GreedyBackward"
    return ODROptions(
        n_multistart, bootstrap_samples, bragging,
        _moved_option(:lm_maxiter, lm_maxiter, 100, "$gb(trial_maxiter = ...)"),
        _moved_option(:lm_maxiter_refine, lm_maxiter_refine, 1000,
                      "$gb(refine_maxiter = ...)"),
        _moved_option(:lm_damping, lm_damping, :marquardt, "$lm(damping = ...)"),
        _moved_option(:lm_accel, lm_accel, false, "$lm(accel = ...)"),
        _moved_option(:ftol, ftol, 5e-8, "$lm(ftol = ...)"),
        _moved_option(:xtol, xtol, 1e-12, "$lm(xtol = ...)"),
        _moved_option(:gtol, gtol, 1e-10, "$lm(gtol = ...)"),
        _moved_option(:warm_start, warm_start, true, "$gb(warm_start = ...)"),
        _moved_option(:trial_xi_init, trial_xi_init, :previous,
                      "$gb(trial_xi_init = ...)"),
        _moved_option(:refine_after_removal, refine_after_removal, true,
                      "$gb(refine_after_removal = ...)"),
        _moved_option(:stop_after_rises, stop_after_rises, 2,
                      "$gb(stop_after_rises = ...)"),
        verbose, rng)
end

"`value` if the deprecated keyword `name` was passed (with a warning), else `default`."
function _moved_option(name::Symbol, value, default, home::String)
    value === nothing && return default
    Base.depwarn("`ODROptions($name = ...)` is deprecated and will be removed in " *
                 "v1.0; pass `$home` to `odr_bindy` instead.", :ODROptions)
    return value
end

"""
    ODRProblem(Xdata, lib, IMat, DMat, hyper)

Everything the optimiser needs. `Xdata` is the raw noisy series (`Nx x D`).
"""
struct ODRProblem{L<:AbstractLibrary,T<:Real}
    Xdata::Matrix{T}
    lib::L
    IMat::SparseMatrixCSC{T,Int}
    DMat::SparseMatrixCSC{T,Int}
    hyper::ODRHyperParameters{T}
    Nx::Int
    Neq::Int
    D::Int
    M::Int
end

function ODRProblem(Xdata::AbstractMatrix{T}, lib::AbstractLibrary,
                    IMat::AbstractMatrix, DMat::AbstractMatrix,
                    hyper::ODRHyperParameters) where {T<:Real}
    Nx, D = size(Xdata)
    Neq = size(IMat, 1)
    M = nterms(lib)

    size(DMat) == size(IMat) || throw(DimensionMismatch("IMat and DMat must match"))
    size(IMat, 2) == Nx || throw(DimensionMismatch("operators must have $Nx columns"))
    size(hyper.sigma_x) == (Nx, D) || throw(DimensionMismatch("sigma_x must be $Nx x $D"))
    size(hyper.sigma_y) == (Neq, D) || throw(DimensionMismatch("sigma_y must be $Neq x $D"))
    size(hyper.sigma_p) == (M, D) || throw(DimensionMismatch("sigma_p must be $M x $D"))

    return ODRProblem{typeof(lib),T}(Matrix{T}(Xdata), lib,
                                     sparse(T.(IMat)), sparse(T.(DMat)),
                                     hyper, Nx, Neq, D, M)
end

"""
    ODRProblem(Xdata, t, lib; discretisation = FiniteDifference(6),
               sigma_x, sigma_y, sigma_p)

Build the problem from the raw series `Xdata` (`Nx x D`) sampled at times `t`.
The operators come from `discretisation` (an [`AbstractDiscretisation`](@ref)),
so `Neq` never has to be worked out by hand. Each `sigma` can be

- a number, used everywhere;
- a length-`D` vector, one value per state (e.g. per-state noise levels);
- a full matrix: `Nx x D` for `sigma_x`, `Neq x D` for `sigma_y`, `M x D`
  for `sigma_p`.

See [`ODRHyperParameters`](@ref) for what each one means.
"""
function ODRProblem(Xdata::AbstractMatrix, t::AbstractVector, lib::AbstractLibrary;
                    discretisation::AbstractDiscretisation = FiniteDifference(6),
                    sigma_x, sigma_y, sigma_p)
    Nx, D = size(Xdata)
    length(t) == Nx ||
        throw(DimensionMismatch("t must have $Nx entries, one per row of Xdata"))
    IMat, DMat = operators(discretisation, t)
    Neq, M = size(IMat, 1), nterms(lib)
    hyper = ODRHyperParameters(_sigma_matrix(sigma_x, Nx, D, "sigma_x"),
                               _sigma_matrix(sigma_y, Neq, D, "sigma_y"),
                               _sigma_matrix(sigma_p, M, D, "sigma_p"))
    return ODRProblem(Xdata, lib, IMat, DMat, hyper)
end

"Broadcast a scalar, per-state vector or full matrix `s` to an `n x D` matrix."
_sigma_matrix(s::Real, n::Int, D::Int, name) = fill(Float64(s), n, D)

function _sigma_matrix(s::AbstractVector, n::Int, D::Int, name)
    length(s) == D ||
        throw(DimensionMismatch("$name as a vector needs one entry per state ($D)"))
    return repeat(Float64.(permutedims(s)), n, 1)
end

function _sigma_matrix(s::AbstractMatrix, n::Int, D::Int, name)
    size(s) == (n, D) || throw(DimensionMismatch("$name must be $n x $D"))
    return Matrix{Float64}(s)
end

"Column ranges of the flattened parameter vector belonging to each state dimension."
function param_ranges(mask::AbstractMatrix{Bool})
    D = size(mask, 2)
    rngs = Vector{UnitRange{Int}}(undef, D)
    offset = 0
    for d in 1:D
        k = count(view(mask, :, d))
        rngs[d] = (offset + 1):(offset + k)
        offset += k
    end
    return rngs
end

"Split the flat unknown vector `z` into the state matrix `X` and coefficients `xi`."
function unpack(prob::ODRProblem, z::AbstractVector)
    nx = prob.Nx * prob.D
    X = reshape(view(z, 1:nx), prob.Nx, prob.D)
    xi = view(z, (nx + 1):length(z))
    return X, xi
end
