# Public API

This is the API that v1.0 will keep stable. Everything listed here follows
semantic versioning from v1.0 on. Its names, signatures and the documented
fields of the result types change only in a major release. Anything that is
not listed, such as `ODRBINDy.cost`, `ODRBINDy.unpack` or
`ODRBINDy.default_optimiser`, is internal and may change in any release.

Before v1.0, a breaking change bumps the minor version (0.1 to 0.2).

`test/api.jl` pins the export list. Adding, removing or renaming an exported
name fails that test, so any change must also be made in this file.

The interfaces for new components are in [DESIGN.md](DESIGN.md) §3, and
examples of each component are in [COMPONENTS.md](COMPONENTS.md).

## Exported

### Running the method

| name | what it is |
|---|---|
| `odr_bindy(prob, opts = ODROptions(); optimiser, selector) -> ODRResult` | the entry point |
| `ODRProblem(Xdata, t, lib; discretisation, sigma_x, sigma_y, sigma_p)` | the problem, built from raw data |
| `ODRProblem(Xdata, lib, IMat, DMat, hyper)` | the problem, built from ready-made operators |
| `ODRHyperParameters(sigma_x, sigma_y, sigma_p)` | noise and prior scales (also a keyword form taking `Nx, Neq, M, D`) |
| `ODROptions(; n_multistart, bootstrap_samples, bragging, verbose, rng)` | settings shared by every component |
| `ODRResult` | fields `Xi`, `X`, `mask`, `nlevidence`, `history` (entries `(nterms, nlevidence, mask)`), `fit` |
| `print_model(res, lib; io, digits)` | prints the discovered equations |
| `state_names(lib, D)` | names of the states, used by `print_model` |

### Libraries

| name | what it is |
|---|---|
| `AbstractLibrary` | interface: `nterms`, `theta`, `dtheta`, `term_names` |
| `nterms(lib)`, `theta(lib, X)`, `dtheta(lib, X)`, `term_names(lib)` | the interface functions |
| `PolynomialLibrary(D, order; varnames)` | the default |
| `FourierLibrary(D, nfreq; varnames)` | `sin` and `cos` of each state |
| `CustomLibrary(D, fs, grads; names, varnames)` | user-supplied functions and gradients |
| `CombinedLibrary(libs...)` | the columns of several libraries, side by side |
| `BasisLibrary(basis)` | a DataDrivenDiffEq `Basis` (needs `using DataDrivenDiffEq`) |

### Discretisations

| name | what it is |
|---|---|
| `AbstractDiscretisation` | interface: `operators` |
| `operators(disc, t) -> (IMat, DMat)` | the interface function |
| `FiniteDifference(order)` | the default; handles uneven samples with Fornberg weights |
| `WeakForm(radius; stride, degree, order)` | weak form, using summation by parts |
| `finite_difference_matrices(Nx, n, dt)` | uniform-step operators, for the second `ODRProblem` constructor |
| `central_fd_coefficients(n)`, `collocation_times(t, n)` | helpers for uniform steps |
| `fornberg_weights(z, x, m)`, `derivative_matrix(t, order)` | helpers for uneven steps |

### Optimisers

| name | what it is |
|---|---|
| `AbstractOptimiser` | interface: `optimise` |
| `optimise(opt, fr, fJ, z0; maxiter) -> LMResult` | the interface function |
| `BuiltinLM(; damping, accel, ftol, xtol, gtol, ...)` | the default |
| `NonlinearSolveOptimiser(alg; ftol, xtol, stalled_steps, kwargs...)` | any NonlinearSolve least-squares algorithm (needs `using NonlinearSolve`) |
| `levenberg_marquardt(fr, fJ, z0; kwargs...) -> LMResult` | the solver behind `BuiltinLM` |
| `LMResult` | fields `z`, `r`, `cost`, `iterations`, `converged` |
| `gauss_newton_decrement(r, J)` | convergence test that works for any algorithm |

### Model selectors

| name | what it is |
|---|---|
| `AbstractModelSelector` | interface: `select_model` |
| `select_model(sel, prob, optimiser, opts) -> ODRResult` | the interface function |
| `GreedyBackward(; stop_after_rises, warm_start, trial_xi_init, refine_after_removal, trial_maxiter, refine_maxiter)` | the default |
| `BeamSearch(width = 3; ...)` | keeps the `width` best models at each step |
| `Exhaustive(; maxiter, max_models, warm_start)` | every sparsity pattern (small libraries only) |
| `n_models(M, D)` | how many patterns `Exhaustive` would fit |

### Building blocks for new selectors

| name | what it is |
|---|---|
| `fit_model(prob, mask, opts; optimiser, X0, xi0, maxiter, rng) -> ODRFit` | fit and score one sparsity pattern |
| `fit_model_multistart(prob, mask, opts; optimiser, X0, nstarts, maxiter) -> ODRFit` | the best of several starts |
| `ODRFit` | fields `Xi`, `X`, `nlevidence`, `cost`, `mask`, `converged`, `iterations` |
| `bootstrap_ridge(prob, mask, X; nsamples, bragging, rng)` | the initial coefficient guess |
| `reduced_hessian(prob, J)`, `neg_log_evidence(prob, mask, J, r)` | the evidence |

## Public, not exported

| name | what it is |
|---|---|
| `ODRBINDy.residual(prob, mask, z)` | the ODR residual vector |
| `ODRBINDy.jacobian(prob, mask, z)` | its exact sparse Jacobian |

These were exported before v0.2. DataDrivenDiffEq and Symbolics also export a
`jacobian`, and an unqualified call is ambiguous when both packages are loaded,
so these two now need the `ODRBINDy.` prefix or
`using ODRBINDy: residual, jacobian`. On Julia 1.11 and later they are marked
`public`.

## Deprecated, removed in v1.0

The solver and search keywords of `ODROptions` still work, but they print a
deprecation warning. These warnings show under `] test`, or when Julia runs
with `--depwarn=yes`.

| old | new |
|---|---|
| `ODROptions(lm_damping, lm_accel, ftol, xtol, gtol)` | `odr_bindy(prob, opts; optimiser = BuiltinLM(damping, accel, ftol, xtol, gtol))` |
| `ODROptions(lm_maxiter)` | `selector = GreedyBackward(trial_maxiter)` |
| `ODROptions(lm_maxiter_refine)` | `selector = GreedyBackward(refine_maxiter)` |
| `ODROptions(warm_start, trial_xi_init, refine_after_removal, stop_after_rises)` | `selector = GreedyBackward(...)`, same names |

## Planned additions

Each of these can be added without breaking anything above, so none of them
blocks v1.0:

- `solve(prob, ODRBINDyAlg(...))` through a CommonSolve extension (DESIGN §9.3).
- Time-dependent libraries (DESIGN §8, question 2). If `theta(lib, X, t)` is
  added, it falls back to `theta(lib, X)`, so existing libraries keep working.
- Spectral / Chebyshev discretisation, and a LeastSquaresOptim.jl optimiser.
