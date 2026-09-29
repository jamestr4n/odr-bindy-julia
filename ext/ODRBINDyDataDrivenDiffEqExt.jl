# =============================================================================
# BasisLibrary(basis): a DataDrivenDiffEq `Basis` as an ODR-BINDy library.
#
# The type and its theta/dtheta live in the core (src/libraries.jl): they only
# call the basis and its compiled Jacobian. This extension adds the one method
# that needs DataDrivenDiffEq itself, the constructor. See docs/DESIGN.md §9.1.
# =============================================================================

module ODRBINDyDataDrivenDiffEqExt

using ODRBINDy
using DataDrivenDiffEq: DataDrivenDiffEq, Basis

const DDE = DataDrivenDiffEq

function ODRBINDy.BasisLibrary(basis::Basis)
    DDE.is_implicit(basis) &&
        throw(ArgumentError("BasisLibrary does not support implicit variables"))
    DDE.is_controlled(basis) &&
        throw(ArgumentError("BasisLibrary does not support control inputs"))

    eqs = DDE.equations(basis)
    iv = DDE.value(DDE.get_iv(basis))
    if any(eq -> any(v -> isequal(v, iv), DDE.get_variables(eq.rhs)), eqs)
        throw(ArgumentError("BasisLibrary needs an autonomous basis: a term " *
            "depends on the independent variable $iv, and the library interface " *
            "has no time argument yet (docs/DESIGN.md §8, question 2)"))
    end

    states = DDE.states(basis)
    p = Float64.(DDE.get_parameter_values(basis))
    jac = DDE.jacobian(basis)                    # compiled once, M x D per point
    names = [string(eq.rhs) for eq in eqs]
    # `@variables x(t)` prints as "x(t)"; the state name is the "x"
    varnames = [replace(string(s), r"\(.*\)$" => "") for s in states]
    return BasisLibrary(basis, jac, p, length(states), names, varnames)
end

end # module
