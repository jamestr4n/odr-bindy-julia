# =============================================================================
# Model selection, and greedy term elimination by Bayesian evidence
# (paper section 2.3).
#
# Replaces ODR_BINDy_Greedy.m and poolDataLIST.m.
#
# Greedy backward elimination, the default `AbstractModelSelector`: start from
# the full library. Try deleting each active term in turn, keep the deletion
# that scores best, repeat. Stop after `stop_after_rises` consecutive failures
# to improve, and return the best model ever seen rather than the last.
#
# Cost is O(M^2 D^2 / 2) fits instead of the 2^(MD) of exhaustive search.
# =============================================================================

"""
    ODRResult

What [`odr_bindy`](@ref) returns.

- `Xi`         `M x D` discovered coefficients
- `X`          `Nx x D` denoised states
- `mask`       the selected sparsity pattern
- `nlevidence` its negative log evidence
- `history`    the models along the search path, as `(nterms, nlevidence, mask)`,
               for plotting the evidence path. For greedy, one per accepted
               removal; the removed term is where consecutive masks differ.
- `fit`        the underlying [`ODRFit`](@ref)
"""
struct ODRResult{T<:Real}
    Xi::Matrix{T}
    X::Matrix{T}
    mask::BitMatrix
    nlevidence::T
    history::Vector{NamedTuple{(:nterms, :nlevidence, :mask),Tuple{Int,T,BitMatrix}}}
    fit::ODRFit{T}
end

"""
    AbstractModelSelector

Decides which sparsity patterns are fitted, and which one is returned. A
subtype must provide

    select_model(sel, prob, optimiser, opts) -> ODRResult

building its fits with [`fit_model`](@ref) and
[`fit_model_multistart`](@ref), passing `optimiser` on. The selector owns the
search: which masks to try and in what order, warm starts, the iteration caps,
the stopping rule. Every state must keep at least one term, and the result
should be the lowest `-log(E)` among the models evaluated, with the path in
`history`.
"""
abstract type AbstractModelSelector end

"""
    select_model(sel, prob, optimiser, opts) -> ODRResult

Run model selector `sel` on `prob`. See [`AbstractModelSelector`](@ref).
"""
function select_model end

"""
    GreedyBackward(; stop_after_rises = 2, warm_start = true,
                     trial_xi_init = :previous, refine_after_removal = true,
                     trial_maxiter = 100, refine_maxiter = 1000)

Greedy backward elimination (paper section 2.3), the default selector.

| option | meaning |
|---|---|
| `stop_after_rises` | stop after this many consecutive removals that fail to lower `-log(E)` |
| `warm_start` | seed each trial with the previous model's denoised `X` |
| `trial_xi_init` | trial coefficients: `:previous` reuses the previous model's; `:regress` re-fits them by bootstrap ridge on the warm-start `X`, as MATLAB does |
| `refine_after_removal` | refit the chosen trial with `refine_maxiter` |
| `trial_maxiter` | iteration cap for each trial (deliberately tight: a trial that will not converge is evidence the term is needed) |
| `refine_maxiter` | iteration cap for the full-library fit and the refits |
"""
struct GreedyBackward <: AbstractModelSelector
    stop_after_rises::Int
    warm_start::Bool
    trial_xi_init::Symbol
    refine_after_removal::Bool
    trial_maxiter::Int
    refine_maxiter::Int
end

function GreedyBackward(; stop_after_rises::Int = 2, warm_start::Bool = true,
                        trial_xi_init::Symbol = :previous,
                        refine_after_removal::Bool = true,
                        trial_maxiter::Int = 100, refine_maxiter::Int = 1000)
    trial_xi_init in (:previous, :regress) ||
        throw(ArgumentError("trial_xi_init must be :previous or :regress"))
    stop_after_rises >= 1 || throw(ArgumentError("stop_after_rises must be >= 1"))
    (trial_maxiter >= 1 && refine_maxiter >= 1) ||
        throw(ArgumentError("iteration caps must be >= 1"))
    return GreedyBackward(stop_after_rises, warm_start, trial_xi_init,
                          refine_after_removal, trial_maxiter, refine_maxiter)
end

"The selector described by the search fields of `opts`."
default_selector(opts::ODROptions) =
    GreedyBackward(stop_after_rises = opts.stop_after_rises,
                   warm_start = opts.warm_start,
                   trial_xi_init = opts.trial_xi_init,
                   refine_after_removal = opts.refine_after_removal,
                   trial_maxiter = opts.lm_maxiter,
                   refine_maxiter = opts.lm_maxiter_refine)

"""
    odr_bindy(prob, opts = ODROptions(); optimiser, selector) -> ODRResult

Discover a sparse model for `prob` by orthogonal distance regression and
Bayesian model selection.

There is no sparsity threshold: terms are dropped while the evidence

    -log(E) = L + sum(log sigma_p) + logdet(H_red)/2

keeps improving. `optimiser` (an [`AbstractOptimiser`](@ref)) fits each
candidate model and `selector` (an [`AbstractModelSelector`](@ref)) runs the
search. By default they are [`BuiltinLM`](@ref) and [`GreedyBackward`](@ref),
configured from the fields of `opts`; when passed explicitly, those fields of
`opts` are ignored.
"""
odr_bindy(prob::ODRProblem, opts::ODROptions = ODROptions();
          optimiser::AbstractOptimiser = default_optimiser(opts),
          selector::AbstractModelSelector = default_selector(opts)) =
    select_model(selector, prob, optimiser, opts)

function select_model(sel::GreedyBackward, prob::ODRProblem{L,T},
                      optimiser::AbstractOptimiser, opts::ODROptions) where {L,T}
    mask = trues(prob.M, prob.D)
    names = term_names(prob.lib)

    opts.verbose >= 1 && @printf("full library: %d terms\n", count(mask))
    current = fit_model_multistart(prob, mask, opts; optimiser = optimiser,
                                   maxiter = sel.refine_maxiter)
    opts.verbose >= 1 && @printf("  Np = %2d   -log(E) = %s\n",
                                 count(mask), _fmt_evidence(current.nlevidence))

    # Every trial warm-starts from this fit, so if it did not converge the whole
    # search is built on sand -- each trial, capped at the much tighter
    # `trial_maxiter`, will not converge either, and no term can be removed.
    current.converged || @warn "ODRBINDy: the initial full-library fit did not " *
        "converge; every trial will score -log(E) = Inf and no term can be " *
        "removed. Raise `refine_maxiter` (`lm_maxiter_refine` in ODROptions), " *
        "or raise `sigma_y` towards the " *
        "discretisation's actual truncation error -- too small a `sigma_y` " *
        "turns the soft model constraint into a nearly hard one, which is the " *
        "stiffness the method exists to avoid."

    best, best_mask = current, copy(mask)
    history = NamedTuple{(:nterms, :nlevidence, :mask),Tuple{Int,T,BitMatrix}}[]
    push!(history, (nterms = count(mask), nlevidence = current.nlevidence,
                    mask = BitMatrix(mask)))

    rises = 0
    while count(mask) > prob.D && rises < sel.stop_after_rises
        cand_fit, cand_mask = nothing, nothing

        for i in findall(vec(mask))
            trial = copy(mask)
            trial[i] = false
            all(any(trial; dims = 1)) || continue   # every state keeps a term

            # Paper section 2.3 point 2: the preceding, larger model's (X, Xi) is a
            # reliable initial guess, so a trial converges in a handful of steps
            # -- and a trial missing a necessary term conspicuously does not.
            # `:regress` (the MATLAB behaviour) re-runs the bootstrap ridge on the
            # warm-start X instead, so the surviving terms can absorb the removed
            # one at once -- which matters when terms are nearly collinear.
            X0 = sel.warm_start ? current.X : prob.Xdata
            xi0 = sel.warm_start && sel.trial_xi_init === :previous ?
                current.Xi[trial] : nothing
            f = fit_model(prob, trial, opts; optimiser = optimiser, X0 = X0,
                          xi0 = xi0, maxiter = sel.trial_maxiter)

            if opts.verbose >= 2
                @printf("    drop %-8s from %-6s -> -log(E) = %s\n",
                        names[_row(i, prob.M)], _dstate(prob, _col(i, prob.M)),
                        _fmt_evidence(f.nlevidence))
            end
            if cand_fit === nothing || f.nlevidence < cand_fit.nlevidence
                cand_fit, cand_mask = f, trial
            end
        end

        # Every remaining term is either structurally required or its removal
        # broke the optimisation: nothing left to delete.
        (cand_fit === nothing || !isfinite(cand_fit.nlevidence)) && break

        if sel.refine_after_removal
            refined = fit_model(prob, cand_mask, opts; optimiser = optimiser,
                                X0 = cand_fit.X, xi0 = cand_fit.Xi[cand_mask],
                                maxiter = sel.refine_maxiter)
            isfinite(refined.nlevidence) && (cand_fit = refined)
        end

        removed = findfirst(vec(mask) .& .!vec(cand_mask))
        rises = cand_fit.nlevidence >= current.nlevidence ? rises + 1 : 0
        current, mask = cand_fit, cand_mask
        push!(history, (nterms = count(mask), nlevidence = current.nlevidence,
                        mask = BitMatrix(mask)))

        if opts.verbose >= 1
            @printf("  Np = %2d   -log(E) = %-12s  removed %s from %s%s\n",
                    count(mask), _fmt_evidence(current.nlevidence),
                    names[_row(removed, prob.M)], _dstate(prob, _col(removed, prob.M)),
                    rises > 0 ? "   (evidence rose, $rises/$(sel.stop_after_rises))" : "")
        end

        if current.nlevidence < best.nlevidence
            best, best_mask = current, copy(mask)
        end
    end

    opts.verbose >= 1 && @printf("selected %d terms, -log(E) = %s\n",
                                 count(best_mask), _fmt_evidence(best.nlevidence))

    return ODRResult(best.Xi, best.X, BitMatrix(best_mask), best.nlevidence,
                     history, best)
end

# Linear index i of an M x D matrix -> (row, column).
_row(i::Integer, M::Int) = mod1(Int(i), M)
_col(i::Integer, M::Int) = div(Int(i) - 1, M) + 1
_dstate(prob::ODRProblem, d::Int) = "d" * state_names(prob.lib, prob.D)[d] * "/dt"

"""
    state_names(lib, D) -> Vector{String}

Names of the `D` state variables, taken from the library's degree-one terms
when they look like bare variable names, and `x1, x2, ...` otherwise.
"""
function state_names(lib::AbstractLibrary, D::Int)
    fallback = ["x$i" for i in 1:D]
    names = term_names(lib)
    length(names) >= D + 1 || return fallback
    cand = names[2:(D + 1)]
    ok = all(n -> !occursin('*', n) && !occursin('^', n) && n != "1", cand)
    return ok ? cand : fallback
end

"""
    print_model(res, lib; io, digits)

Print the discovered equations, one line per state. Replaces `poolDataLIST.m`.
"""
print_model(res::ODRResult, lib::AbstractLibrary; kwargs...) =
    print_model(res.Xi, lib; kwargs...)

function print_model(Xi::AbstractMatrix, lib::AbstractLibrary;
                     io::IO = stdout, digits::Int = 4)
    M, D = size(Xi)
    M == nterms(lib) || throw(DimensionMismatch("Xi must have $(nterms(lib)) rows"))
    names = term_names(lib)
    states = state_names(lib, D)
    for d in 1:D
        print(io, "d", states[d], "/dt = ")
        first = true
        for n in 1:M
            c = Xi[n, d]
            iszero(c) && continue
            if first
                print(io, c < 0 ? "-" : "")
                first = false
            else
                print(io, c < 0 ? " - " : " + ")
            end
            mag = _round_str(abs(c), digits)
            print(io, names[n] == "1" ? mag : mag * " * " * names[n])
        end
        first && print(io, "0")
        println(io)
    end
    return nothing
end

_round_str(v::Real, digits::Int) = rstrip(rstrip(string(round(float(v); digits = digits)), '0'), '.')
