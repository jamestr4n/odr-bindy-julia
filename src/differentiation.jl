# =============================================================================
# Discretisation operators L_I (IMat) and L_dt (DMat)
#
# Both map the Nx sampled states onto Neq collocation points:
#
#     eta = DMat * X  -  (IMat * Theta(X)) * Xi
#
# For central finite differences of accuracy order `n`, Neq = Nx - n: the n/2
# points at each end are dropped because the stencil runs off the data.
#
# Replacing this file's output with weak-form / integral operators is all that
# is needed to swap discretisation -- nothing downstream knows the difference.
# =============================================================================

"""
    central_fd_coefficients(n) -> Vector

First-derivative central finite-difference weights on the stencil `-n/2:n/2`
for a unit grid spacing, accurate to order `n` (`n` even).

Obtained by solving the Vandermonde system `V c = e_1`, where
`V[i+1, k] = s_k^i` for `i = 0..n` and `s` the stencil. Row `i` enforces
exactness on the monomial `t^i`; the right-hand side `e_1` asks for the
first derivative.
"""
function central_fd_coefficients(n::Int)
    (n >= 2 && iseven(n)) || throw(ArgumentError("n must be a positive even integer"))
    s = collect(-n ÷ 2:n ÷ 2)
    V = [float(sk)^i for i in 0:n, sk in s]
    rhs = zeros(n + 1)
    rhs[2] = 1.0                       # d/dt of t^1
    c = V \ rhs
    c[abs.(c) .< 1e4 * eps()] .= 0.0
    return c
end

"""
    finite_difference_matrices(Nx, n, dt) -> (IMat, DMat)

Sparse banded operators of size `Neq x Nx` with `Neq = Nx - n`.
`IMat` selects the stencil centre, `DMat` applies the derivative stencil.
`n` is the order of accuracy (even); `dt` the sampling interval.
"""
function finite_difference_matrices(Nx::Int, n::Int, dt::Real)
    (n >= 2 && iseven(n)) || throw(ArgumentError("n must be a positive even integer"))
    Nx > n || throw(ArgumentError("need Nx > n data points"))
    cD = central_fd_coefficients(n) ./ dt
    cI = zeros(n + 1)
    cI[n ÷ 2 + 1] = 1.0
    Neq = Nx - n
    return _banded(cI, Neq, Nx), _banded(cD, Neq, Nx)
end

function _banded(c::Vector{Float64}, Neq::Int, Nx::Int)
    rows, cols, vals = Int[], Int[], Float64[]
    for r in 1:Neq, k in eachindex(c)
        c[k] == 0 && continue
        push!(rows, r)
        push!(cols, r + k - 1)
        push!(vals, c[k])
    end
    return sparse(rows, cols, vals, Neq, Nx)
end

"""
    collocation_times(t, n) -> Vector

The subset of `t` that the operators from `finite_difference_matrices` map onto.
Useful for plotting and for building a time-varying `sigma_y`.
"""
collocation_times(t::AbstractVector, n::Int) = t[(n ÷ 2 + 1):(length(t) - n ÷ 2)]
