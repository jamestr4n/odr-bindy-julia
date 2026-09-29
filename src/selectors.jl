# =============================================================================
# Alternative model selectors (the default, GreedyBackward, is in greedy.jl).
#
#   Exhaustive   every sparsity pattern: ground truth for small problems
#   BeamSearch   greedy elimination that keeps the `width` best models per
#                level; `width = 1` is exactly GreedyBackward
# =============================================================================

const _History{T} = Vector{NamedTuple{(:nterms, :nlevidence, :mask),Tuple{Int,T,BitMatrix}}}

_entry(f::ODRFit) = (nterms = count(f.mask), nlevidence = f.nlevidence,
                     mask = BitMatrix(f.mask))

"Every state keeps at least one term."
_admissible(mask::AbstractMatrix{Bool}) = all(any(mask; dims = 1))

"Fit the full library from several bootstrap starts, warning if it failed."
function _full_fit(prob::ODRProblem, optimiser, opts::ODROptions, maxiter::Int)
    mask = trues(prob.M, prob.D)
    opts.verbose >= 1 && @printf("full library: %d terms\n", count(mask))
    full = fit_model_multistart(prob, mask, opts; optimiser = optimiser,
                                maxiter = maxiter)
    opts.verbose >= 1 && @printf("  Np = %2d   -log(E) = %s\n",
                                 count(mask), _fmt_evidence(full.nlevidence))
    full.converged || @warn "ODRBINDy: the initial full-library fit did not " *
        "converge. Raise the selector's iteration cap, or raise `sigma_y` " *
        "towards the discretisation's actual truncation error."
    return full
end

# -----------------------------------------------------------------------------
# Exhaustive search
# -----------------------------------------------------------------------------

"""
    Exhaustive(; maxiter = 1000, max_models = 10_000, warm_start = true)

Fit every sparsity pattern and keep the one with the lowest `-log(E)`.

There are `(2^M - 1)^D` patterns (each state needs at least one of its `M`
terms), and the states cannot be searched separately because they share the
denoised `X`. That is 3,969 fits for `D = 2, M = 6`, and ~10⁹ for Lorenz with
quadratics, so this is for small problems only; it refuses to start beyond
`max_models`. Its use is as ground truth: does greedy find the model with the
best evidence?

The full library is fitted first, from `opts.n_multistart` bootstrap starts.
With `warm_start`, every other pattern starts from its denoised states, with
coefficients from a bootstrap ridge regression on them; otherwise from the raw
data. Every fit gets `maxiter` iterations. `history` holds every model, in the
order they were fitted.

```julia
res = odr_bindy(prob; selector = Exhaustive())
```
"""
struct Exhaustive <: AbstractModelSelector
    maxiter::Int
    max_models::Int
    warm_start::Bool
end

function Exhaustive(; maxiter::Int = 1000, max_models::Int = 10_000,
                    warm_start::Bool = true)
    maxiter >= 1 || throw(ArgumentError("maxiter must be >= 1"))
    return Exhaustive(maxiter, max_models, warm_start)
end

"""
    n_models(M, D) -> BigInt

The number of sparsity patterns of an `M x D` coefficient matrix in which every
state keeps at least one term: `(2^M - 1)^D`.
"""
n_models(M::Int, D::Int) = (big(2)^M - 1)^D

function select_model(sel::Exhaustive, prob::ODRProblem{L,T},
                      optimiser::AbstractOptimiser, opts::ODROptions) where {L,T}
    M, D = prob.M, prob.D
    total = n_models(M, D)
    total <= sel.max_models || throw(ArgumentError(
        "Exhaustive would fit $total models (M = $M terms, D = $D states), more " *
        "than max_models = $(sel.max_models). Use GreedyBackward or BeamSearch, " *
        "or raise max_models."))

    full = _full_fit(prob, optimiser, opts, sel.maxiter)
    X0 = sel.warm_start ? full.X : prob.Xdata
    opts.verbose >= 1 && @printf("fitting all %d models\n", total)

    history = _History{T}()
    best = full
    codes = Iterators.product(ntuple(_ -> 1:(2^M - 1), D)...)
    for (count_done, code) in enumerate(codes)
        mask = falses(M, D)
        for d in 1:D, n in 1:M
            mask[n, d] = isodd(code[d] >> (n - 1))
        end
        f = all(mask) ? full :
            fit_model(prob, mask, opts; optimiser = optimiser, X0 = X0,
                      maxiter = sel.maxiter)
        push!(history, _entry(f))
        f.nlevidence < best.nlevidence && (best = f)

        if opts.verbose >= 2
            @printf("    %5d/%d  Np = %2d   -log(E) = %s\n", count_done, total,
                    count(mask), _fmt_evidence(f.nlevidence))
        end
    end

    opts.verbose >= 1 && @printf("selected %d terms, -log(E) = %s\n",
                                 count(best.mask), _fmt_evidence(best.nlevidence))
    return ODRResult(best.Xi, best.X, BitMatrix(best.mask), best.nlevidence,
                     history, best)
end

# -----------------------------------------------------------------------------
# Beam search
# -----------------------------------------------------------------------------

"""
    BeamSearch(width = 3; stop_after_rises = 2, warm_start = true,
               trial_xi_init = :previous, refine_after_removal = true,
               trial_maxiter = 100, refine_maxiter = 1000)

Backward elimination that keeps the `width` best models at each level instead
of one. Each level tries removing every active term from every model in the
beam (a pattern reached from two parents is fitted once, from the first), and
the `width` best trials by `-log(E)` form the next beam. It stops after
`stop_after_rises` consecutive levels whose best model fails to beat the
previous level's best, and returns the best model seen.

It costs about `width` times as much as greedy, and can recover from an early
greedy mistake that looked good for one step. The options mean the same as in
[`GreedyBackward`](@ref); each trial warm-starts from its own parent, and every
model in the beam is refined. With `width = 1` it makes exactly the same fits,
in the same order, as `GreedyBackward` with the same options, and returns the
same result.

```julia
res = odr_bindy(prob; selector = BeamSearch(3))
```
"""
struct BeamSearch <: AbstractModelSelector
    width::Int
    stop_after_rises::Int
    warm_start::Bool
    trial_xi_init::Symbol
    refine_after_removal::Bool
    trial_maxiter::Int
    refine_maxiter::Int
end

function BeamSearch(width::Int = 3; stop_after_rises::Int = 2, warm_start::Bool = true,
                    trial_xi_init::Symbol = :previous,
                    refine_after_removal::Bool = true,
                    trial_maxiter::Int = 100, refine_maxiter::Int = 1000)
    width >= 1 || throw(ArgumentError("width must be >= 1"))
    trial_xi_init in (:previous, :regress) ||
        throw(ArgumentError("trial_xi_init must be :previous or :regress"))
    stop_after_rises >= 1 || throw(ArgumentError("stop_after_rises must be >= 1"))
    (trial_maxiter >= 1 && refine_maxiter >= 1) ||
        throw(ArgumentError("iteration caps must be >= 1"))
    return BeamSearch(width, stop_after_rises, warm_start, trial_xi_init,
                      refine_after_removal, trial_maxiter, refine_maxiter)
end

function select_model(sel::BeamSearch, prob::ODRProblem{L,T},
                      optimiser::AbstractOptimiser, opts::ODROptions) where {L,T}
    full = _full_fit(prob, optimiser, opts, sel.refine_maxiter)
    beam = [full]
    best = full
    history = _History{T}()
    push!(history, _entry(full))

    rises = 0
    while rises < sel.stop_after_rises
        trials = ODRFit{T}[]
        seen = Set{BitMatrix}()
        for parent in beam, i in findall(vec(parent.mask))
            mask = copy(parent.mask)
            mask[i] = false
            (_admissible(mask) && !(mask in seen)) || continue
            push!(seen, mask)

            X0 = sel.warm_start ? parent.X : prob.Xdata
            xi0 = sel.warm_start && sel.trial_xi_init === :previous ?
                parent.Xi[mask] : nothing
            f = fit_model(prob, mask, opts; optimiser = optimiser, X0 = X0,
                          xi0 = xi0, maxiter = sel.trial_maxiter)
            push!(trials, f)
            opts.verbose >= 2 && @printf("    Np = %2d  trial -> -log(E) = %s\n",
                                         count(mask), _fmt_evidence(f.nlevidence))
        end

        # Keep the finite trials, best first; a stable sort keeps ties in the
        # order they were fitted, as greedy does.
        ok = sort!(filter(f -> isfinite(f.nlevidence), trials);
                   by = f -> f.nlevidence, alg = MergeSort)
        isempty(ok) && break
        next = ok[1:min(sel.width, length(ok))]

        if sel.refine_after_removal
            next = map(next) do f
                refined = fit_model(prob, f.mask, opts; optimiser = optimiser,
                                    X0 = f.X, xi0 = f.Xi[f.mask],
                                    maxiter = sel.refine_maxiter)
                isfinite(refined.nlevidence) ? refined : f
            end
            sort!(next; by = f -> f.nlevidence, alg = MergeSort)
        end

        rises = next[1].nlevidence >= beam[1].nlevidence ? rises + 1 : 0
        beam = next
        push!(history, _entry(beam[1]))
        beam[1].nlevidence < best.nlevidence && (best = beam[1])

        if opts.verbose >= 1
            @printf("  Np = %2d   -log(E) = %-12s  (beam of %d)%s\n",
                    count(beam[1].mask), _fmt_evidence(beam[1].nlevidence),
                    length(beam),
                    rises > 0 ? "   (evidence rose, $rises/$(sel.stop_after_rises))" : "")
        end
    end

    opts.verbose >= 1 && @printf("selected %d terms, -log(E) = %s\n",
                                 count(best.mask), _fmt_evidence(best.nlevidence))
    return ODRResult(best.Xi, best.X, BitMatrix(best.mask), best.nlevidence,
                     history, best)
end
