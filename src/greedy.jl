# =============================================================================
# Greedy term elimination by Bayesian evidence (paper section 2.3).
#
# Replaces ODR_BINDy_Greedy.m and poolDataLIST.m.
#
# Start from the full library. Try deleting each active term in turn, keep the
# deletion that scores best, repeat. Stop after `stop_after_rises` consecutive
# failures to improve, and return the best model ever seen rather than the last.
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
- `history`    one entry per accepted removal, for plotting the evidence path
- `fit`        the underlying [`ODRFit`](@ref)
"""
struct ODRResult{T<:Real}
    Xi::Matrix{T}
    X::Matrix{T}
    mask::BitMatrix
    nlevidence::T
    history::Vector{NamedTuple{(:nterms, :nlevidence, :removed),Tuple{Int,T,Int}}}
    fit::ODRFit{T}
end

"""
    odr_bindy(prob, opts = ODROptions()) -> ODRResult

Discover a sparse model for `prob` by orthogonal distance regression and
Bayesian model selection.

There is no sparsity threshold: terms are dropped while the evidence

    -log(E) = L + sum(log sigma_p) + logdet(H_red)/2

keeps improving, and the search stops once it has stopped improving
`opts.stop_after_rises` times in a row.
"""
function odr_bindy(prob::ODRProblem{L,T}, opts::ODROptions = ODROptions()) where {L,T}
    mask = trues(prob.M, prob.D)
    names = term_names(prob.lib)

    opts.verbose >= 1 && @printf("full library: %d terms\n", count(mask))
    current = fit_model_multistart(prob, mask, opts;
                                   maxiter = opts.lm_maxiter_refine)
    opts.verbose >= 1 && @printf("  Np = %2d   -log(E) = %s\n",
                                 count(mask), _fmt_evidence(current.nlevidence))

    # Every trial warm-starts from this fit, so if it did not converge the whole
    # search is built on sand -- each trial, capped at the much tighter
    # `lm_maxiter`, will not converge either, and no term can be removed.
    current.converged || @warn "ODRBINDy: the initial full-library fit did not " *
        "converge; every trial will score -log(E) = Inf and no term can be " *
        "removed. Raise `lm_maxiter_refine`, or raise `sigma_y` towards the " *
        "discretisation's actual truncation error -- too small a `sigma_y` " *
        "turns the soft model constraint into a nearly hard one, which is the " *
        "stiffness the method exists to avoid."

    best, best_mask = current, copy(mask)
    history = NamedTuple{(:nterms, :nlevidence, :removed),Tuple{Int,T,Int}}[]
    push!(history, (nterms = count(mask), nlevidence = current.nlevidence, removed = 0))

    rises = 0
    while count(mask) > prob.D && rises < opts.stop_after_rises
        cand_fit, cand_mask = nothing, nothing

        for i in findall(vec(mask))
            trial = copy(mask)
            trial[i] = false
            all(any(trial; dims = 1)) || continue   # every state keeps a term

            # Paper section 2.3 point 2: the preceding, larger model's (X, Xi) is a
            # reliable initial guess, so a trial converges in a handful of steps
            # -- and a trial missing a necessary term conspicuously does not.
            X0 = opts.warm_start ? current.X : prob.Xdata
            xi0 = opts.warm_start ? current.Xi[trial] : nothing
            f = fit_model(prob, trial, opts; X0 = X0, xi0 = xi0,
                          maxiter = opts.lm_maxiter)

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

        if opts.refine_after_removal
            refined = fit_model(prob, cand_mask, opts; X0 = cand_fit.X,
                                xi0 = cand_fit.Xi[cand_mask],
                                maxiter = opts.lm_maxiter_refine)
            isfinite(refined.nlevidence) && (cand_fit = refined)
        end

        removed = findfirst(vec(mask) .& .!vec(cand_mask))
        rises = cand_fit.nlevidence >= current.nlevidence ? rises + 1 : 0
        current, mask = cand_fit, cand_mask
        push!(history, (nterms = count(mask), nlevidence = current.nlevidence,
                        removed = removed))

        if opts.verbose >= 1
            @printf("  Np = %2d   -log(E) = %-12s  removed %s from %s%s\n",
                    count(mask), _fmt_evidence(current.nlevidence),
                    names[_row(removed, prob.M)], _dstate(prob, _col(removed, prob.M)),
                    rises > 0 ? "   (evidence rose, $rises/$(opts.stop_after_rises))" : "")
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
