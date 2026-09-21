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
