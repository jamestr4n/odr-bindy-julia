# =============================================================================
# Initial guess for Xi by bootstrapped, prior-regularised linear regression.
#
# The nonlinear ODR problem is not convex, so the starting point matters. We
# get one cheaply by pretending the states are noise-free and solving the
# ordinary SINDy linear problem with the Gaussian prior acting as a ridge:
#
#     xi_d = ( Theta' W Theta + diag(1/sigma_p^2) )^{-1} Theta' W xdot_d
#     W    = diag(1 / sigma_y[:,d]^2)
#
# Resampling the rows with replacement gives (a) a more robust estimate via
# the median across samples ("bragging"), and (b) a spread of distinct starting
# points for multi-start global optimisation.
# =============================================================================

"""
    bootstrap_ridge(prob, mask, X; nsamples, bragging, rng) -> Vector

Initial coefficient guess for the sparsity pattern `mask`, regressed on the
states `X` (either the raw data, or the current denoised estimate).
"""
function bootstrap_ridge(prob::ODRProblem{L,T}, mask::AbstractMatrix{Bool},
                         X::AbstractMatrix;
                         nsamples::Int = 100,
                         bragging::Bool = true,
                         rng::AbstractRNG = Random.default_rng()) where {L,T}
    ThI = prob.IMat * theta(prob.lib, X)          # Neq x M
    Xdot = prob.DMat * X                          # Neq x D
    idx = active_indices(mask)

    xi = T[]
    for d in 1:prob.D
        isempty(idx[d]) && continue
        Td = ThI[:, idx[d]]
        y = Xdot[:, d]
        w = one(T) ./ view(prob.hyper.sigma_y, :, d) .^ 2
        prior = one(T) ./ prob.hyper.sigma_p[idx[d], d] .^ 2

        coeffs = Matrix{T}(undef, length(idx[d]), nsamples)
        for s in 1:nsamples
            rows = rand(rng, 1:prob.Neq, prob.Neq)   # resample with replacement
            Ts = Td[rows, :]
            ws = w[rows]
            Amat = Ts' * (Ts .* ws) + Diagonal(prior)
            bvec = Ts' * (y[rows] .* ws)
            coeffs[:, s] = Amat \ bvec
        end
        est = bragging ? vec(median(coeffs; dims = 2)) : vec(mean(coeffs; dims = 2))
        append!(xi, est)
    end
    return xi
end
