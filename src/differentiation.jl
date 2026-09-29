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

# -----------------------------------------------------------------------------
# Pluggable discretisations
# -----------------------------------------------------------------------------

"""
    AbstractDiscretisation

How `dX/dt = Theta(X) Xi` becomes equations at discrete points. A subtype must
provide

- `operators(disc, t) -> (IMat, DMat)`: both `Neq x Nx`, sparse, `Float64`
- `collocation_times(disc, t) -> Vector`: where each of the `Neq` equations lives

where `t` holds the `Nx` sample times. The operators must be linear and must not
depend on `X`, since [`jacobian`](@ref) differentiates
`eta = DMat*X - (IMat*Theta(X))*Xi` using them directly. Keep them sparse and
banded: the sparse Cholesky in every LM step relies on it.
"""
abstract type AbstractDiscretisation end

"""
    operators(disc, t) -> (IMat, DMat)

The discretisation operators of `disc` for sample times `t`.
See [`AbstractDiscretisation`](@ref).
"""
function operators end

"""
    FiniteDifference(order)

Central finite differences accurate to `order` (even). The `order ÷ 2` samples
at each end have no full stencil, so `Neq = Nx - order`.

Evenly spaced samples use the fixed stencil of
[`finite_difference_matrices`](@ref). Unevenly spaced ones get a separate
stencil for every row, from [`fornberg_weights`](@ref) on that row's
`order + 1` sample times, so still accurate to `order` in the local spacing:

```julia
t = sort(10 .* rand(500))                       # uneven sampling
prob = ODRProblem(Xdata, t, lib; discretisation = FiniteDifference(6), ...)
```
"""
struct FiniteDifference <: AbstractDiscretisation
    order::Int
    function FiniteDifference(order::Int)
        (order >= 2 && iseven(order)) ||
            throw(ArgumentError("order must be a positive even integer"))
        return new(order)
    end
end

function operators(disc::FiniteDifference, t::AbstractVector)
    dt = _uniform_step(t)
    dt === nothing || return finite_difference_matrices(length(t), disc.order, dt)
    return _uneven_fd_matrices(Float64.(t), disc.order)
end

collocation_times(disc::FiniteDifference, t::AbstractVector) =
    collocation_times(t, disc.order)

"""
The spacing of `t` if it is uniform, and `nothing` otherwise. The tolerance
allows for times read back from a file with a few digits; a relative error of
1e-4 in `dt` is far below the noise the method is built for.
"""
function _uniform_step(t::AbstractVector)
    length(t) >= 2 || throw(ArgumentError("need at least two sample times"))
    all(>(0), diff(t)) || throw(ArgumentError("sample times must be increasing"))
    dt = (t[end] - t[1]) / (length(t) - 1)
    return maximum(abs, diff(t) .- dt) <= 1e-4 * dt ? float(dt) : nothing
end

function _uneven_fd_matrices(t::Vector{Float64}, n::Int)
    Nx = length(t)
    Nx > n || throw(ArgumentError("need Nx > n data points"))
    Neq = Nx - n
    rows, cols, vals = Int[], Int[], Float64[]
    for r in 1:Neq
        stencil = r:(r + n)
        w = fornberg_weights(t[r + n ÷ 2], t[stencil], 1)[:, 2]
        for (k, i) in enumerate(stencil)
            push!(rows, r)
            push!(cols, i)
            push!(vals, w[k])
        end
    end
    IMat = sparse(1:Neq, (1:Neq) .+ n ÷ 2, ones(Neq), Neq, Nx)
    return IMat, sparse(rows, cols, vals, Neq, Nx)
end

"""
    fornberg_weights(z, x, m) -> Matrix

Finite-difference weights on the arbitrary nodes `x` for every derivative
order `0..m` at the point `z`: `W[j, k + 1]` is the weight of `f(x[j])` in the
approximation of the `k`-th derivative. B. Fornberg, *Generation of finite
difference formulas on arbitrarily spaced grids*, Math. Comp. 51 (1988).
"""
function fornberg_weights(z::Real, x::AbstractVector{<:Real}, m::Int)
    n = length(x) - 1
    n >= m || throw(ArgumentError("need at least $(m + 1) nodes"))
    C = zeros(n + 1, m + 1)
    C[1, 1] = 1.0
    c1 = 1.0
    c4 = x[1] - z
    for i in 1:n
        mn = min(i, m)
        c2 = 1.0
        c5 = c4
        c4 = x[i + 1] - z
        for j in 0:(i - 1)
            c3 = x[i + 1] - x[j + 1]
            c2 *= c3
            if j == i - 1
                for k in mn:-1:1
                    C[i + 1, k + 1] = c1 * (k * C[i, k] - c5 * C[i, k + 1]) / c2
                end
                C[i + 1, 1] = -c1 * c5 * C[i, 1] / c2
            end
            for k in mn:-1:1
                C[j + 1, k + 1] = (c4 * C[j + 1, k + 1] - k * C[j + 1, k]) / c3
            end
            C[j + 1, 1] = c4 * C[j + 1, 1] / c3
        end
        c1 = c2
    end
    return C
end

# -----------------------------------------------------------------------------
# Weak form
# -----------------------------------------------------------------------------

"""
    WeakForm(radius; stride = 1, degree = 4, order = 6)

The weak (integral) form of the model. Instead of asking `dx/dt = f(x)` at each
point, it asks that both sides agree when averaged against a set of test
functions `phi_k`, which vanish at both ends of their support:

    ∫ phi_k(t) dx/dt dt  =  ∫ phi_k(t) f(x(t)) dt

Each `phi_k` covers `2*radius + 1` consecutive samples (so it works on uneven
sampling too), is the polynomial bump `((t - a)(b - t))^degree` on that span,
and is normalised to integrate to one. `IMat` holds its trapezoidal quadrature
weights, `IMat[k, i] = w_i * phi_k(t_i)`.

The derivative is moved off the noisy data and onto `phi_k` by *summation* by
parts, the discrete form of integrating by parts: `DMat = IMat * Dt`, with `Dt`
the full-length finite-difference derivative of accuracy `order`. Each row of
`DMat` is then a smooth discrete `-(w phi_k)'`, with about a tenth of the noise
gain of a finite-difference stencil, and `DMat*x = IMat*dx/dt` is exact to the
finite-difference order on any sampling. (Using `phi_k'` directly, as in
WSINDy, adds a quadrature error of ~3e-4 relative at `radius = 8` that does not
shrink as the sampling gets finer, and is only second order on uneven grids.)

A new test function starts every `stride` samples, so
`Neq = (Nx - 2*radius - 1) ÷ stride + 1`.

Because each `phi_k` integrates to one, `eta` is a local average of the
pointwise model error and `sigma_y` keeps roughly its finite-difference
meaning. Neighbouring rows overlap when `stride < 2*radius + 1`, which makes
their errors correlated; the diagonal `sigma_y` ignores that (docs/DESIGN.md
§8, question 1).

```julia
prob = ODRProblem(Xdata, t, lib; discretisation = WeakForm(8), ...)
```
"""
struct WeakForm <: AbstractDiscretisation
    radius::Int
    stride::Int
    degree::Int
    order::Int
    function WeakForm(radius::Int; stride::Int = 1, degree::Int = 4, order::Int = 6)
        radius >= 1 || throw(ArgumentError("radius must be >= 1"))
        stride >= 1 || throw(ArgumentError("stride must be >= 1"))
        degree >= 1 || throw(ArgumentError("degree must be >= 1"))
        (order >= 2 && iseven(order)) ||
            throw(ArgumentError("order must be a positive even integer"))
        return new(radius, stride, degree, order)
    end
end

"First sample of each test function's support."
function _weak_starts(disc::WeakForm, Nx::Int)
    width = 2 * disc.radius + 1
    Nx >= width || throw(ArgumentError("need at least $width samples for this WeakForm"))
    return 1:disc.stride:(Nx - width + 1)
end

function operators(disc::WeakForm, t::AbstractVector)
    _uniform_step(t)                                 # checks t is increasing
    t = Float64.(t)
    Nx = length(t)
    Nx > disc.order || throw(ArgumentError("need more than $(disc.order) samples"))
    starts = _weak_starts(disc, Nx)
    rows, cols, vals = Int[], Int[], Float64[]
    for (k, s) in enumerate(starts)
        a, b = t[s], t[s + 2 * disc.radius]
        idx = (s + 1):(s + 2 * disc.radius - 1)     # phi is zero at both ends
        w = [(t[i + 1] - t[i - 1]) / 2 for i in idx] # trapezoidal weights
        phi = [((t[i] - a) * (b - t[i]))^disc.degree for i in idx]
        wphi = w .* phi
        append!(rows, fill(k, length(idx)))
        append!(cols, idx)
        append!(vals, wphi ./ sum(wphi))            # so that ∫ phi = 1
    end
    IMat = sparse(rows, cols, vals, length(starts), Nx)
    return IMat, IMat * derivative_matrix(t, disc.order)
end

"""
    derivative_matrix(t, order) -> SparseMatrixCSC

`Nx x Nx` first-derivative matrix on the samples `t`, accurate to `order`:
each row uses the `order + 1` samples centred on it where possible, and the
nearest `order + 1` at the ends ([`fornberg_weights`](@ref)).
"""
function derivative_matrix(t::AbstractVector, order::Int)
    Nx = length(t)
    Nx > order || throw(ArgumentError("need more than $order samples"))
    rows, cols, vals = Int[], Int[], Float64[]
    for i in 1:Nx
        stencil = clamp(i - order ÷ 2, 1, Nx - order) .+ (0:order)
        append!(rows, fill(i, order + 1))
        append!(cols, stencil)
        append!(vals, fornberg_weights(t[i], t[stencil], 1)[:, 2])
    end
    return sparse(rows, cols, vals, Nx, Nx)
end

"The centre sample of each test function."
collocation_times(disc::WeakForm, t::AbstractVector) =
    t[_weak_starts(disc, length(t)) .+ disc.radius]
