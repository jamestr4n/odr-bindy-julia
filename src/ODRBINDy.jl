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
lib   = PolynomialLibrary(3, 2; varnames = ["x", "y", "z"])
IMat, DMat = finite_difference_matrices(size(Xdata, 1), 6, dt)
hyper = ODRHyperParameters(sigma_x = 1.6, sigma_y = 1e-4, sigma_p = 1e2,
                           Nx = size(Xdata, 1), Neq = size(IMat, 1),
                           M = nterms(lib), D = 3)
res   = odr_bindy(ODRProblem(Xdata, lib, IMat, DMat, hyper))
print_model(res, lib)
```

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

# libraries
export AbstractLibrary, PolynomialLibrary, nterms, theta, dtheta, term_names
# discretisation
export central_fd_coefficients, finite_difference_matrices, collocation_times
# problem setup
export ODRProblem, ODRHyperParameters, ODROptions
# pieces, exported so they can be used or replaced individually
export residual, jacobian, bootstrap_ridge
export levenberg_marquardt, LMResult
export reduced_hessian, neg_log_evidence
export ODRFit, fit_model, fit_model_multistart
# the algorithm
export ODRResult, odr_bindy, print_model, state_names

end # module
