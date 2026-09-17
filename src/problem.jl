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
    ODROptions(; kwargs...)

| option | meaning |
|---|---|
| `n_multistart` | independent bootstrap starts for the initial full-library fit |
| `bootstrap_samples` | resamples used by the ridge-regression initial guess |
| `bragging` | median (`true`) rather than mean (`false`) over bootstrap samples |
| `lm_maxiter` | LM iteration cap during greedy trials (deliberately tight: a trial that will not converge is evidence the term is needed) |
| `lm_maxiter_refine` | iteration cap for the full fit and post-removal refinement |
| `warm_start` | seed each trial with the previous model's denoised `X` |
| `stop_after_rises` | stop after this many consecutive drops in evidence |
| `verbose` | 0 silent, 1 per removal, 2 per trial |
"""
Base.@kwdef mutable struct ODROptions
    n_multistart::Int = 8
    bootstrap_samples::Int = 100
    bragging::Bool = true
    lm_maxiter::Int = 100
    lm_maxiter_refine::Int = 1000
    ftol::Float64 = 5e-8
    xtol::Float64 = 1e-12
    gtol::Float64 = 1e-10
    warm_start::Bool = true
    refine_after_removal::Bool = true
    stop_after_rises::Int = 2
    verbose::Int = 1
    rng::AbstractRNG = Random.default_rng()
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
