# =============================================================================
# Candidate function libraries (the "Basis" of SINDy)
#
# A library must supply, for data X (Nx x D):
#   theta(lib, X)   -> Nx x M      : Theta[i,n] = phi_n(x_i)
#   dtheta(lib, X)  -> Nx x M x D  : dTheta[i,n,e] = d phi_n / d x_e  at x_i
#
# NOTE: because the evidence uses the Gauss-Newton approximation, no second or
# third derivatives are needed (unlike the MATLAB version, which carries
# ddTheta_fun and dddTheta_fun_f).
# =============================================================================

abstract type AbstractLibrary end

"""
    nterms(lib) -> M

Number of candidate functions in the library.
"""
function nterms end

"""
    theta(lib, X) -> Matrix (Nx x M)

Evaluate every candidate function at every row of `X`.
"""
function theta end

"""
    dtheta(lib, X) -> Array (Nx x M x D)

`dtheta(lib,X)[i,n,e]` is the partial derivative of candidate `n` with respect
to state `e`, evaluated at row `i` of `X`.
"""
function dtheta end

"""
    term_names(lib) -> Vector{String}
"""
function term_names end

# -----------------------------------------------------------------------------
# Polynomial library
# -----------------------------------------------------------------------------

"""
    PolynomialLibrary(D, order; varnames)

All monomials in `D` state variables of total degree `<= order`, ordered
graded-lexicographically, i.e. for `D=3, order=2`:

    1, x1, x2, x3, x1^2, x1*x2, x1*x3, x2^2, x2*x3, x3^2

This matches the column ordering of SINDy's `poolData`, so ground-truth
coefficient matrices from the MATLAB examples transfer directly.
"""
struct PolynomialLibrary <: AbstractLibrary
    D::Int
    order::Int
    powers::Vector{Vector{Int}}
    names::Vector{String}
end

function PolynomialLibrary(D::Int, order::Int;
                           varnames::Vector{String} = ["x$i" for i in 1:D])
    D >= 1 || throw(ArgumentError("D must be >= 1"))
    order >= 0 || throw(ArgumentError("order must be >= 0"))
    powers = Vector{Vector{Int}}()
    push!(powers, zeros(Int, D))                 # the constant term
    for deg in 1:order, combo in _multisets(D, deg)
        p = zeros(Int, D)
        for c in combo
            p[c] += 1
        end
        push!(powers, p)
    end
    names = [_monomial_name(p, varnames) for p in powers]
    return PolynomialLibrary(D, order, powers, names)
end

"Non-decreasing index tuples of length `deg` drawn from 1:D (graded-lex order)."
function _multisets(D::Int, deg::Int, start::Int = 1)
    deg == 0 && return [Int[]]
    out = Vector{Vector{Int}}()
    for c in start:D, rest in _multisets(D, deg - 1, c)
        push!(out, vcat(c, rest))
    end
    return out
end

function _monomial_name(p::Vector{Int}, varnames::Vector{String})
    all(iszero, p) && return "1"
    parts = String[]
    for (e, k) in enumerate(p)
        k == 0 && continue
        push!(parts, k == 1 ? varnames[e] : "$(varnames[e])^$k")
    end
    return join(parts, "*")
end

nterms(lib::PolynomialLibrary) = length(lib.powers)
term_names(lib::PolynomialLibrary) = lib.names

function theta(lib::PolynomialLibrary, X::AbstractMatrix{T}) where {T}
    Nx = size(X, 1)
    size(X, 2) == lib.D || throw(DimensionMismatch("X must have $(lib.D) columns"))
    M = nterms(lib)
    Th = ones(T, Nx, M)
    @inbounds for n in 1:M
        p = lib.powers[n]
        for e in 1:lib.D
            p[e] == 0 && continue
            for i in 1:Nx
                Th[i, n] *= X[i, e]^p[e]
            end
        end
    end
    return Th
end

function dtheta(lib::PolynomialLibrary, X::AbstractMatrix{T}) where {T}
    Nx = size(X, 1)
    size(X, 2) == lib.D || throw(DimensionMismatch("X must have $(lib.D) columns"))
    M = nterms(lib)
    dTh = zeros(T, Nx, M, lib.D)
    col = Vector{T}(undef, Nx)
    @inbounds for n in 1:M
        p = lib.powers[n]
        for e in 1:lib.D
            p[e] == 0 && continue          # derivative of a term not containing x_e
            fill!(col, T(p[e]))            # bring down the exponent
            for f in 1:lib.D
                q = (f == e) ? p[f] - 1 : p[f]
                q == 0 && continue
                for i in 1:Nx
                    col[i] *= X[i, f]^q
                end
            end
            dTh[:, n, e] .= col
        end
    end
    return dTh
end

"Check that `X` has the `D` columns a library was built for."
_check_columns(X::AbstractMatrix, D::Int) =
    size(X, 2) == D || throw(DimensionMismatch("X must have $D columns"))

# -----------------------------------------------------------------------------
# Fourier library
# -----------------------------------------------------------------------------

"""
    FourierLibrary(D, nfreq; varnames)

`sin(k*x_e)` and `cos(k*x_e)` for every state `x_e` and every frequency
`k = 1..nfreq`, ordered by frequency, then state:

    sin(x), cos(x), sin(y), cos(y), sin(2*x), cos(2*x), ...

There is no constant term, so it combines with a [`PolynomialLibrary`](@ref)
without duplicating one; see [`CombinedLibrary`](@ref).

```julia
lib = CombinedLibrary(PolynomialLibrary(2, 1; varnames = ["θ", "ω"]),
                      FourierLibrary(2, 1; varnames = ["θ", "ω"]))
# 1, θ, ω, sin(θ), cos(θ), sin(ω), cos(ω)
```
"""
struct FourierLibrary <: AbstractLibrary
    D::Int
    nfreq::Int
    varnames::Vector{String}
    names::Vector{String}
end

function FourierLibrary(D::Int, nfreq::Int;
                        varnames::Vector{String} = ["x$i" for i in 1:D])
    D >= 1 || throw(ArgumentError("D must be >= 1"))
    nfreq >= 1 || throw(ArgumentError("nfreq must be >= 1"))
    length(varnames) == D || throw(ArgumentError("need $D varnames"))
    names = String[]
    for k in 1:nfreq, e in 1:D
        arg = k == 1 ? varnames[e] : "$k*$(varnames[e])"
        push!(names, "sin($arg)", "cos($arg)")
    end
    return FourierLibrary(D, nfreq, varnames, names)
end

nterms(lib::FourierLibrary) = 2 * lib.D * lib.nfreq
term_names(lib::FourierLibrary) = lib.names
state_names(lib::FourierLibrary, D::Int) = lib.varnames

function theta(lib::FourierLibrary, X::AbstractMatrix{T}) where {T}
    _check_columns(X, lib.D)
    Th = Matrix{T}(undef, size(X, 1), nterms(lib))
    n = 0
    @inbounds for k in 1:lib.nfreq, e in 1:lib.D
        for i in axes(X, 1)
            s, c = sincos(k * X[i, e])
            Th[i, n + 1] = s
            Th[i, n + 2] = c
        end
        n += 2
    end
    return Th
end

function dtheta(lib::FourierLibrary, X::AbstractMatrix{T}) where {T}
    _check_columns(X, lib.D)
    dTh = zeros(T, size(X, 1), nterms(lib), lib.D)
    n = 0
    @inbounds for k in 1:lib.nfreq, e in 1:lib.D
        for i in axes(X, 1)                      # each term depends on x_e only
            s, c = sincos(k * X[i, e])
            dTh[i, n + 1, e] = k * c
            dTh[i, n + 2, e] = -k * s
        end
        n += 2
    end
    return dTh
end

# -----------------------------------------------------------------------------
# User-supplied functions
# -----------------------------------------------------------------------------

"""
    CustomLibrary(D, fs, grads; names, varnames)

A library of arbitrary functions of the `D` states. `fs[n](x)` evaluates term
`n` at one state vector `x` (length `D`), and `grads[n](x)` returns its
gradient, a length-`D` vector. The gradients must be exact (derived by hand, or
with an AD package): an approximate one gives a wrong Jacobian, which breaks
both the optimiser and the evidence.

```julia
lib = CustomLibrary(2,
    [x -> x[1], x -> x[2], x -> exp(-x[1]^2)],
    [x -> [1.0, 0.0], x -> [0.0, 1.0], x -> [-2x[1] * exp(-x[1]^2), 0.0]];
    names = ["x", "y", "exp(-x^2)"], varnames = ["x", "y"])
```
"""
struct CustomLibrary{F<:Tuple,G<:Tuple} <: AbstractLibrary
    D::Int
    fs::F
    grads::G
    names::Vector{String}
    varnames::Vector{String}
end

function CustomLibrary(D::Int, fs, grads;
                       names::Vector{String} = ["f$n" for n in eachindex(fs)],
                       varnames::Vector{String} = ["x$i" for i in 1:D])
    D >= 1 || throw(ArgumentError("D must be >= 1"))
    M = length(fs)
    M >= 1 || throw(ArgumentError("need at least one function"))
    length(grads) == M || throw(ArgumentError("need one gradient per function"))
    length(names) == M || throw(ArgumentError("need one name per function"))
    length(varnames) == D || throw(ArgumentError("need $D varnames"))
    return CustomLibrary(D, Tuple(fs), Tuple(grads), names, varnames)
end

nterms(lib::CustomLibrary) = length(lib.fs)
term_names(lib::CustomLibrary) = lib.names
state_names(lib::CustomLibrary, D::Int) = lib.varnames

function theta(lib::CustomLibrary, X::AbstractMatrix{T}) where {T}
    _check_columns(X, lib.D)
    Th = Matrix{T}(undef, size(X, 1), nterms(lib))
    x = Vector{T}(undef, lib.D)
    for i in axes(X, 1)
        x .= view(X, i, :)
        for (n, f) in enumerate(lib.fs)
            Th[i, n] = f(x)
        end
    end
    return Th
end

function dtheta(lib::CustomLibrary, X::AbstractMatrix{T}) where {T}
    _check_columns(X, lib.D)
    dTh = Array{T}(undef, size(X, 1), nterms(lib), lib.D)
    x = Vector{T}(undef, lib.D)
    for i in axes(X, 1)
        x .= view(X, i, :)
        for (n, g) in enumerate(lib.grads)
            gx = g(x)
            length(gx) == lib.D ||
                throw(DimensionMismatch("gradient $n must have $(lib.D) entries"))
            dTh[i, n, :] .= gx
        end
    end
    return dTh
end

# -----------------------------------------------------------------------------
# Several libraries side by side
# -----------------------------------------------------------------------------

"""
    CombinedLibrary(libs...)

The columns of each library in `libs`, side by side, e.g. polynomials and
sines. The state names come from the first library. Nothing removes a term that
appears in two of them, so combine libraries that do not overlap (a
[`FourierLibrary`](@ref) has no constant term for this reason).
"""
struct CombinedLibrary{L<:Tuple} <: AbstractLibrary
    libs::L
end

function CombinedLibrary(libs::AbstractLibrary...)
    isempty(libs) && throw(ArgumentError("need at least one library"))
    return CombinedLibrary(libs)
end

nterms(lib::CombinedLibrary) = sum(nterms, lib.libs)
term_names(lib::CombinedLibrary) = reduce(vcat, [term_names(l) for l in lib.libs])
state_names(lib::CombinedLibrary, D::Int) = state_names(first(lib.libs), D)
theta(lib::CombinedLibrary, X::AbstractMatrix) = reduce(hcat, [theta(l, X) for l in lib.libs])
dtheta(lib::CombinedLibrary, X::AbstractMatrix) =
    cat((dtheta(l, X) for l in lib.libs)...; dims = 2)

# -----------------------------------------------------------------------------
# DataDrivenDiffEq basis
# -----------------------------------------------------------------------------

"""
    BasisLibrary(basis)

A `DataDrivenDiffEq.Basis` used as the library. The constructor lives in a
package extension, so it is available after `using DataDrivenDiffEq`:

```julia
using ODRBINDy, DataDrivenDiffEq, Symbolics
@variables x y z
lib = BasisLibrary(Basis(polynomial_basis([x, y, z], 2), [x, y, z]))
```

`theta` evaluates the basis and `dtheta` its compiled symbolic Jacobian.
Parameters are held at their default values (ODR-BINDy does not fit them), and
the basis must be autonomous, without controls or implicit variables: the
library interface has no time argument yet (docs/DESIGN.md §8, question 2).
Note that `polynomial_basis` orders its terms differently from
[`PolynomialLibrary`](@ref).
"""
struct BasisLibrary{B,J,P} <: AbstractLibrary
    basis::B                 # called as basis(X', p, t) -> M x Nx
    jac::J                   # called as jac(x, p, t)    -> M x D
    p::P                     # parameter values, fixed
    D::Int
    names::Vector{String}
    varnames::Vector{String}
end

nterms(lib::BasisLibrary) = length(lib.names)
term_names(lib::BasisLibrary) = lib.names
state_names(lib::BasisLibrary, D::Int) = lib.varnames

function theta(lib::BasisLibrary, X::AbstractMatrix{T}) where {T}
    _check_columns(X, lib.D)
    Th = lib.basis(permutedims(X), lib.p, zeros(T, size(X, 1)))
    return Matrix{T}(permutedims(Th))
end

function dtheta(lib::BasisLibrary, X::AbstractMatrix{T}) where {T}
    _check_columns(X, lib.D)
    dTh = Array{T}(undef, size(X, 1), nterms(lib), lib.D)
    for i in axes(X, 1)
        dTh[i, :, :] .= lib.jac(X[i, :], lib.p, zero(T))
    end
    return dTh
end
