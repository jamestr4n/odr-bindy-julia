# =============================================================================
# The ODR residual and its analytic sparse Jacobian.
#
# Loss:   L(X, Xi) = 1/2 * ||r||^2
#
#         r = [ vec(eta ./ sigma_y) ;  vec(zeta ./ sigma_x) ;  xi ./ sigma_p ]
#
#         eta  = DMat*X - (IMat*Theta(X))*Xi      (Neq x D)   model error
#         zeta = X - Xdata                        (Nx  x D)   data error
#
# Unknown vector: z = [vec(X) ; xi], with xi the masked entries of Xi taken in
# column-major order (so dimension 1's active terms first, then dimension 2's).
#
# Jacobian block structure (rows = residual blocks, cols = [X, xi]):
#
#         J = [ A  B ]        A = d(eta/sigma_y)/dX     B = d(eta/sigma_y)/dxi
#             [ C  0 ]        C = diag(1 ./ sigma_x)
#             [ 0  P ]        P = diag(1 ./ sigma_p)
#
# with, for output dimension d and input dimension e,
#
#     A[d,e] = diag(1/sigma_y[:,d]) * ( delta_de * DMat - IMat * diag(g_de) )
#     g_de   = (dTheta[:, active_d, e]) * xi_d            (an Nx-vector)
#
# and B block-diagonal, B[d,d] = -diag(1/sigma_y[:,d]) * (IMat*Theta)[:, active_d].
# =============================================================================

"Active library-row indices for each state dimension."
active_indices(mask::AbstractMatrix{Bool}) =
    [findall(view(mask, :, d)) for d in 1:size(mask, 2)]

"""
    residual(prob, mask, z) -> Vector

Stacked residual whose half-squared-norm is the ODR loss.
"""
function residual(prob::ODRProblem{L,T}, mask::AbstractMatrix{Bool},
                  z::AbstractVector) where {L,T}
    X, xi = unpack(prob, z)
    idx = active_indices(mask)
    rngs = param_ranges(mask)
    ThI = prob.IMat * theta(prob.lib, X)          # Neq x M
    Xdot = prob.DMat * X                          # Neq x D

    eta = Matrix{T}(undef, prob.Neq, prob.D)
    @inbounds for d in 1:prob.D
        pred = isempty(idx[d]) ? zeros(T, prob.Neq) : ThI[:, idx[d]] * xi[rngs[d]]
        eta[:, d] .= (Xdot[:, d] .- pred) ./ view(prob.hyper.sigma_y, :, d)
    end
    zeta = (X .- prob.Xdata) ./ prob.hyper.sigma_x

    return vcat(vec(eta), vec(zeta), xi ./ prob.hyper.sigma_p[mask])
end

"""
    jacobian(prob, mask, z) -> SparseMatrixCSC

Exact Jacobian of [`residual`](@ref). Sparse and banded, because `IMat` and
`DMat` are: this is what makes the Newton step affordable.
"""
function jacobian(prob::ODRProblem{L,T}, mask::AbstractMatrix{Bool},
                  z::AbstractVector) where {L,T}
    X, xi = unpack(prob, z)
    idx = active_indices(mask)
    rngs = param_ranges(mask)
    Nx, Neq, D = prob.Nx, prob.Neq, prob.D
    Np = length(xi)

    ThI = prob.IMat * theta(prob.lib, X)          # Neq x M
    dTh = dtheta(prob.lib, X)                     # Nx x M x D

    # --- A : d(eta)/dX -------------------------------------------------------
    Ablocks = Matrix{SparseMatrixCSC{T,Int}}(undef, D, D)
    for d in 1:D
        Wd = Diagonal(one(T) ./ view(prob.hyper.sigma_y, :, d))
        for e in 1:D
            # g[i] = sum_n dTheta[i, n, e] * xi_d[n]   over active n
            g = isempty(idx[d]) ? zeros(T, Nx) : dTh[:, idx[d], e] * xi[rngs[d]]
            blk = prob.IMat * Diagonal(g)
            blk = (d == e) ? (prob.DMat - blk) : -blk
            Ablocks[d, e] = sparse(Wd * blk)
        end
    end
    A = reduce(vcat, [reduce(hcat, Ablocks[d, :]) for d in 1:D])

    # --- B : d(eta)/dxi, block diagonal over state dimensions ----------------
    Bd = [sparse(Diagonal(one(T) ./ view(prob.hyper.sigma_y, :, d)) *
                 (-ThI[:, idx[d]])) for d in 1:D]
    B = blockdiag(Bd...)

    # --- C and P -------------------------------------------------------------
    C = spdiagm(0 => vec(one(T) ./ prob.hyper.sigma_x))
    P = spdiagm(0 => one(T) ./ prob.hyper.sigma_p[mask])

    nx = Nx * D
    return [A                B;
            C                spzeros(T, nx, Np);
            spzeros(T, Np, nx) P]
end

"L(X, Xi) = 1/2 ||r||^2"
cost(r::AbstractVector) = 0.5 * sum(abs2, r)
