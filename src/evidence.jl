# =============================================================================
# Bayesian evidence by the Laplace approximation (paper section A.2.1).
#
# Replaces d2Jdp2_GaussNewtonEst.m and d2Jdp2.m.
#
# The evidence needs d^2 L / dXi^2, but X is not free: for each Xi it sits at
# the minimiser X*(Xi). Paper eq. (A6) carries dX*/dXi through every term. With
# the Gauss-Newton approximation H ~ J'J and the implicit function theorem at
# the ODR optimum (dL/dX = 0, paper eq. (A1)),
#
#     dX*/dXi = -Hxx^{-1} Hxp
#
# so eq. (A6) collapses to the Schur complement of J'J:
#
#     H_red = Hpp - Hxp' Hxx^{-1} Hxp
#
# Hxx contains diag(1/sigma_x^2), hence is positive definite unconditionally;
# so is its Schur complement. The MATLAB retry loop on negative eigenvalues has
# no counterpart here.
#
# The log(2*pi) terms of the Laplace formula cancel exactly, leaving
#
#     -log(E) = L(Xi*, X*) + sum(log sigma_p over active) + logdet(H_red)/2
#
# Neither d2Theta nor d3Theta appears: they live only in the terms Gauss-Newton
# drops, which is why `libraries.jl` needs half of what the MATLAB libraries do.
# =============================================================================

"""
    reduced_hessian(prob, J) -> Symmetric

Gauss-Newton Hessian of the loss with respect to the *active coefficients only*,
after eliminating the states: the Schur complement of `J'J`.

`J` is the residual Jacobian at the optimum, as returned by [`jacobian`](@ref).
"""
function reduced_hessian(prob::ODRProblem{L,T}, J::AbstractMatrix) where {L,T}
    nx = prob.Nx * prob.D
    Np = size(J, 2) - nx
    Np > 0 || throw(ArgumentError("no active coefficients to score"))

    H = J' * J
    Hxx = H[1:nx, 1:nx]
    Hxp = Matrix(H[1:nx, (nx + 1):end])            # nx x Np, few columns
    Hpp = Matrix(H[(nx + 1):end, (nx + 1):end])

    F = cholesky(Symmetric(Hxx))                   # sparse, banded: cheap
    S = Hpp - Hxp' * (F \ Hxp)                     # solve, never form an inverse
    return Symmetric((S .+ S') ./ 2)               # symmetrise away round-off
end

"""
    neg_log_evidence(prob, mask, J, r) -> (nle, H_red)

Negative log Bayesian evidence of the model `mask`, evaluated at an optimum
with residual `r` and Jacobian `J`:

    -log(E) = L + sum(log sigma_p over active terms) + logdet(H_red)/2

the three terms reading as *fit*, *number of terms*, and the *Occam factor*.
Lower is better. Returns `(Inf, nothing)` if the Hessian cannot be factorised.
"""
function neg_log_evidence(prob::ODRProblem{L,T}, mask::AbstractMatrix{Bool},
                          J::AbstractMatrix, r::AbstractVector) where {L,T}
    local S, ch
    try
        S = reduced_hessian(prob, J)
        ch = cholesky(S)
    catch err
        err isa InterruptException && rethrow()
        return T(Inf), nothing
    end
    nle = cost(r) + sum(log, prob.hyper.sigma_p[mask]) + logdet(ch) / 2
    return T(nle), S
end
