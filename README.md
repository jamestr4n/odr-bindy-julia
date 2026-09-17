# ODRBINDy.jl

Julia port of [llfung/ODR-BINDy](https://github.com/llfung/ODR-BINDy)
(Fung, *Overcoming error-in-variable problem in data-driven model discovery by
orthogonal distance regression*, arXiv:2507.23426).

This project fits sparse ODE models from noisy time-series data by jointly
optimising:

- the denoised latent states `X`, and
- the model coefficients `Xi`

while penalising model complexity through Bayesian evidence.

## Quick start

From the repository root:

```bash
cd odr-bindy-julia
julia --project=. examples/check_derivatives.jl   # run this first after edits
julia --project=. examples/lorenz.jl
```

No external dependencies are required; the project uses only Julia stdlib
packages listed in [Project.toml](Project.toml).

## What the code does

The objective is a penalised residual:

```
L(X, Xi) = ||Xhat - X||_F^2 / (2 sigma_x^2)
         + ||Ldt*X - LI*Theta(X)*Xi||_F^2 / (2 sigma_dt^2)
         + ||Xi||_F^2 / (2 sigma_p^2)
```

where:

- `Xhat` is the observed noisy trajectory,
- `X` is the denoised trajectory inferred jointly with the model,
- `Xi` are the sparse ODE coefficients,
- `Theta(X)` is the library of candidate basis terms,
- `sigma_x` is the measurement-noise scale,
- `sigma_p` is the prior scale on coefficients.

After fitting, the code scores candidate models by Bayesian evidence and greedily
removes weak terms while the evidence improves.

The key difference from modified-SINDy / DySMHO / SIDDs is the soft model
constraint: the fitted trajectory is allowed to drift slightly off the ODE. This
lets one optimisation assimilate data over longer windows than a strict ODE-fit
would allow, and it is the main reason the method is useful in the
error-in-variables regime.

## Repository layout

| File | Purpose |
|---|---|
| `src/libraries.jl` | `AbstractLibrary` interface and `PolynomialLibrary` |
| `src/differentiation.jl` | finite-difference operators `IMat` / `DMat` |
| `src/problem.jl` | `ODRProblem`, hyperparameters, options, mask utilities |
| `src/residual.jl` | residual vector and analytic sparse Jacobian |
| `src/initialguess.jl` | bootstrapped ridge regression starting point |
| `src/lm.jl` | Levenberg–Marquardt on sparse normal equations |
| `src/evidence.jl` | Gauss–Newton Laplace evidence and Schur complement |
| `src/regression.jl` | single-pattern fit and multi-start regression |
| `src/greedy.jl` | greedy term elimination, `odr_bindy`, `print_model` |
| `examples/check_derivatives.jl` | numerical checks for Jacobian and Hessian correctness |
| `examples/lorenz.jl` | Lorenz/Van der Pol demo workflow |

## Correspondence with the MATLAB package

| MATLAB | Julia implementation |
|---|---|
| `Jsq_xp.m` | `residual`, `jacobian` |
| `Jsq_x.m`, `dJdx.m` | not needed — the Gauss–Newton Hessian is used directly |
| `d2Jdp2_GaussNewtonEst.m` | `reduced_hessian` |
| `d2Jdp2.m`, `ddTheta`, `dddTheta` | not needed |
| `EnsembleLinRegress_Wprior.m` | `bootstrap_ridge` |
| `ODR_BINDy_Regression.m` | `fit_model` |
| `ODR_BINDy_Regression_MultiStart.m` | `fit_model_multistart` |
| `ODR_BINDy_Greedy.m` | `odr_bindy` |
| `FD.m` | `finite_difference_matrices` |
| `Polynomial3D2O*.m` | `PolynomialLibrary` |
| `lsqnonlin` | `levenberg_marquardt` |

## Why the Hessian collapses in the profiled problem

At the ODR optimum, the Gauss–Newton Hessian over `z = [vec(X); xi]` can be
blocked and reduced using the implicit function theorem:

```
H_red = Hpp - Hxp' * Hxx^{-1} * Hxp
```

Here `Hxx` contains `diag(1 / sigma_x^2)`, so it is positive definite
unconditionally. This means the Jacobian has full column rank, `J'J` is positive
definite, and the profiled reduced Hessian is well-behaved as well. The MATLAB
workaround `while any(eig(Hessian_p) < 0)` has no direct counterpart here.

The log-evidence simplifies because the `log(2*pi)` terms cancel exactly:

```
-log(evidence) = L(Xi*, X*) + sum(log sigma_p over active terms)
                            + 0.5 * logdet(H_red)
```

This is the Occam-style penalty that favours sparse, parsimonious models.

## Hyperparameters

These are not arbitrary tuning knobs; they are statements of belief about the
noise and prior uncertainty.

- `sigma_x`: measurement noise standard deviation. Estimate this from the data or
  from the sensor.
- `sigma_y`: discretisation truncation error plus stochastic forcing. A value of
  `1e-4` works well for order-6 finite differences at `dt = 0.01`.
  If the noise level varies over the trajectory, pass a full `Neq x D` matrix.
- `sigma_p`: prior standard deviation on coefficients. A large value like `1e2`
  is weakly informative, but it still matters through the evidence term
  `sum(log sigma_p)`.

## Example workflow

The main example is a self-contained Lorenz/Van der Pol demo in
`examples/lorenz.jl`.

- Start with the Van der Pol case; it is cheaper and gives a quicker sanity check.
- Move to the Lorenz example for the more demanding full system identification.
- Run `examples/check_derivatives.jl` after changing the residual or library
  definition; the analytic Jacobian must be verified before trusting results.

## Roadmap

1. `Basis` interop: wrap `DataDrivenDiffEq.Basis` behind `AbstractLibrary`.
2. Swappable optimiser backends (`NonlinearSolve.jl`, `LeastSquaresOptim.jl`).
3. Swappable discretisation (`AbstractDiscretisation` for weak-form / spectral / uneven-sampling variants).
4. Swappable model selector with exhaustive search and beam search options.
5. Parallel trial fits inside the greedy sweep.
6. Performance improvements: cached `theta` evaluations and reused sparse Jacobian structure.
7. Proper tests and CI, replacing the ad hoc derivative checks with a formal test suite.

## Notes

This package is intentionally designed to stay close to the MATLAB reference
implementation while being idiomatic Julia. The current focus is correctness and
algorithmic transparency rather than maximal industrial performance.
