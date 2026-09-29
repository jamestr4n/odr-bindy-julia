"""
    ODRBINDy

Orthogonal Distance Regression based Bayesian Identification of Nonlinear
Dynamics: model discovery from noisy data by jointly denoising the trajectory
and selecting terms through maximising Bayesian evidence.

Julia port of [ODR-BINDy](https://github.com/llfung/ODR-BINDy) by L. Fung,
the code for <https://doi.org/10.1145/3831701>. No dependencies outside the
standard library. `Float64` is the working precision (the sparse Cholesky in
SuiteSparse supports nothing else).

Typical use:

```julia
lib  = PolynomialLibrary(3, 2; varnames = ["x", "y", "z"])
prob = ODRProblem(Xdata, t, lib; discretisation = FiniteDifference(6),
                  sigma_x = 1.6, sigma_y = 1e-2, sigma_p = 1e2)
res  = odr_bindy(prob)
print_model(res, lib)
```

Four parts are pluggable, each through an abstract type: the library
(`AbstractLibrary`), the discretisation (`AbstractDiscretisation`), the
optimiser (`AbstractOptimiser`, passed as `odr_bindy(prob; optimiser)`) and the
model selector (`AbstractModelSelector`, passed as `odr_bindy(prob; selector)`).
See `docs/DESIGN.md`, and `docs/COMPONENTS.md` for the ones provided.

`BasisLibrary` and `NonlinearSolveOptimiser` need DataDrivenDiffEq and
NonlinearSolve respectively; their methods load as package extensions, so the
core stays standard-library only.

See `docs/WALKTHROUGH.md` for how each file implements the paper, and
`docs/PORTING.md` for the correspondence with the MATLAB implementation.
"""
module ODRBINDy

using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Printf

include("libraries.jl")        # Theta and dTheta
include("differentiation.jl")  # L_I and L_dt
include("problem.jl")          # ODRProblem, hyperparameters, packing
include("residual.jl")         # residual and analytic sparse Jacobian
include("initialguess.jl")     # bootstrapped ridge regression
include("lm.jl")               # Levenberg-Marquardt
include("evidence.jl")         # Laplace evidence via the Schur complement
include("regression.jl")       # fit one sparsity pattern
include("greedy.jl")           # greedy elimination, odr_bindy, print_model
include("selectors.jl")        # exhaustive and beam search

# libraries
export AbstractLibrary, PolynomialLibrary, nterms, theta, dtheta, term_names
export FourierLibrary, CustomLibrary, CombinedLibrary, BasisLibrary
# discretisation
export AbstractDiscretisation, FiniteDifference, WeakForm, operators
export central_fd_coefficients, finite_difference_matrices, collocation_times
export fornberg_weights, derivative_matrix
# problem setup
export ODRProblem, ODRHyperParameters, ODROptions
# pieces, exported so they can be used or replaced individually
export bootstrap_ridge
export AbstractOptimiser, BuiltinLM, NonlinearSolveOptimiser, optimise
export levenberg_marquardt, LMResult, gauss_newton_decrement
export reduced_hessian, neg_log_evidence
export ODRFit, fit_model, fit_model_multistart
# model selection and the algorithm
export AbstractModelSelector, GreedyBackward, BeamSearch, Exhaustive, select_model
export n_models
export ODRResult, odr_bindy, print_model, state_names

# Public but not exported: DataDrivenDiffEq and Symbolics export a `jacobian`
# too, and an unqualified call is ambiguous when both are loaded. Use
# `ODRBINDy.residual` and `ODRBINDy.jacobian`. (`public` needs Julia 1.11.)
@static if VERSION >= v"1.11"
    eval(Meta.parse("public residual, jacobian"))
end

# Point at the missing package when an extension's method is not loaded.
function __init__()
    Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, kwargs
        if exc.f === optimise && !isempty(argtypes) &&
           argtypes[1] <: NonlinearSolveOptimiser &&
           Base.get_extension(@__MODULE__, :ODRBINDyNonlinearSolveExt) === nothing
            print(io, "\nNonlinearSolveOptimiser needs NonlinearSolve.jl: " *
                      "run `using NonlinearSolve` first.")
        elseif exc.f === BasisLibrary &&
               Base.get_extension(@__MODULE__, :ODRBINDyDataDrivenDiffEqExt) === nothing
            print(io, "\nBasisLibrary(basis) needs DataDrivenDiffEq.jl: " *
                      "run `using DataDrivenDiffEq` first.")
        end
    end
end

end # module
