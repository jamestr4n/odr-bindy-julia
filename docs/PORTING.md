# Porting notes

How this package corresponds to the reference MATLAB implementation, and where
the two deliberately differ. For the theory and a file-by-file explanation, see
[WALKTHROUGH.md](WALKTHROUGH.md).

Julia port of [llfung/ODR-BINDy](https://github.com/llfung/ODR-BINDy)
(Fung, *Overcoming error-in-variable problem in data-driven model discovery by
orthogonal distance regression*, arXiv:2507.23426).

## Running it

```bash
cd ODRBINDy
julia --project=. examples/check_derivatives.jl   # do this first
julia --project=. examples/lorenz.jl
```

No external dependencies — everything is Julia stdlib.

## What the code does

Minimise, over **both** the denoised states `X` and the coefficients `Xi`,

```
L(X, Xi) = ||Xhat - X||_F^2 / (2 sigma_x^2)                <- data loss
         + ||Ldt*X - LI*Theta(X)*Xi||_F^2 / (2 sigma_dt^2)  <- model loss (soft constraint)
         + ||Xi||_F^2 / (2 sigma_p^2)                       <- Gaussian prior
```

then score the model by Bayesian evidence and greedily delete terms while the
evidence improves. The soft model constraint is the key difference from
modified-SINDy / DySMHO / SIDDs: the trajectory is allowed to drift slightly
off the ODE, which is what lets a single optimisation assimilate data longer
than a Lyapunov time.

## File map

| file | contents |
|---|---|
| `libraries.jl` | `AbstractLibrary` interface + `PolynomialLibrary` |
| `differentiation.jl` | `IMat` / `DMat` central finite-difference operators |
| `problem.jl` | `ODRProblem`, `ODRHyperParameters`, `ODROptions`, mask helpers |
| `residual.jl` | residual vector and analytic sparse Jacobian |
| `initialguess.jl` | bootstrapped ridge regression starting point |
| `lm.jl` | Levenberg–Marquardt on sparse normal equations |
| `evidence.jl` | Gauss–Newton Laplace evidence (Schur complement) |
| `regression.jl` | fit one sparsity pattern; multi-start |
| `greedy.jl` | greedy term elimination, `odr_bindy`, `print_model` |

## Correspondence with the MATLAB package

| MATLAB | here |
|---|---|
| `Jsq_xp.m` | `residual`, `jacobian` |
| `Jsq_x.m`, `dJdx.m` | not needed — Gauss–Newton Hessian only |
| `d2Jdp2_GaussNewtonEst.m` | `reduced_hessian` (10 lines, see below) |
| `d2Jdp2.m`, `ddTheta`, `dddTheta` | not needed |
| `EnsembleLinRegress_Wprior.m` | `bootstrap_ridge` |
| `ODR_BINDy_Regression.m` | `fit_model` |
| `ODR_BINDy_Regression_MultiStart.m` | `fit_model_multistart` |
| `ODR_BINDy_Greedy.m` | `odr_bindy` |
| `FD.m` | `finite_difference_matrices` |
| `Polynomial3D2O*.m` | `PolynomialLibrary` |
| `lsqnonlin` | `levenberg_marquardt` |

### Why the Hessian collapsed

Blocking the Gauss–Newton Hessian `J'J` over `z = [vec(X); xi]` and using
`dX*/dXi = -Hxx^{-1} Hxp` (implicit function theorem at the ODR optimum) turns
paper eq. (A6) into a Schur complement:

```
H_red = Hpp - Hxp' * Hxx^{-1} * Hxp
```

`Hxx` contains `diag(1/sigma_x^2)`, so it is positive definite unconditionally;
hence `J` has full column rank, `J'J` is positive definite, and `H_red` is too.
The MATLAB `while any(eig(Hessian_p) < 0)` retry loop has no counterpart here.

The `log(2*pi)` terms in the log-evidence cancel exactly, leaving

```
-log(evidence) = L(Xi*, X*) + sum(log sigma_p over active terms)
                            + 0.5 * logdet(H_red)
```

## Hyperparameters

Not tuned — each is a stated belief:

- `sigma_x`: measurement noise std. Estimate from the data or from the sensor.
- `sigma_y`: discretisation truncation error, plus any stochastic forcing, plus
  slack. Keep it comfortably *above* the truncation error, not equal to it:
  `1e-2` for order-6 FD at `dt = 0.01` on Lorenz, where the measured truncation
  error is ~2e-4 RMS. Shrinking `sigma_y` towards that figure turns the soft
  model constraint back into a nearly hard one and the optimisation goes stiff,
  which is the failure mode paper eq. (6) exists to avoid; in practice the
  greedy search then stalls with every trial hitting its iteration cap. If the
  truncation error varies over the trajectory, pass a full `Neq x D` matrix (the
  MATLAB `NonlinearOscillator.m` uses a decaying `sigma_y` this way).
- `sigma_p`: prior std on coefficients. Large (`1e2`) = weakly informative.
  It does real work in the evidence, via the `sum(log sigma_p)` Occam term.

## Roadmap (project aim 2)

1. **`Basis` interop.** Wrap `DataDrivenDiffEq.Basis` behind `AbstractLibrary`:
   `theta` is `basis(X, p, t)`, `dtheta` is `jacobian(basis)`. Everything else
   is untouched, since Gauss–Newton needs no higher derivatives.
2. **Swappable optimiser.** Make the solver a type parameter
   (`ODRSolver{<:AbstractNLLSAlg}`) with backends for `NonlinearSolve.jl`
   (`TrustRegion`, `LevenbergMarquardt`) and `LeastSquaresOptim.jl`. The
   contract is only `LMResult`.
3. **Swappable discretisation.** `AbstractDiscretisation` returning
   `(IMat, DMat)`; add weak-form / spectral / uneven-sampling variants.
4. **Swappable model selector.** `AbstractSelector` over the greedy loop; add
   exhaustive search for small libraries, and a beam search.
5. **Parallelism.** The trial fits inside one greedy sweep are independent —
   `@threads` over them (the MATLAB `parfor` version).
6. **Performance.** `theta` is evaluated twice per LM step; cache it. Reuse the
   sparse pattern of `J` across iterations instead of rebuilding it.
7. **Tests + CI.** Turn `check_derivatives.jl` into a proper test set, add a
   noise-free recovery test and a `Project.toml` `[targets]` entry.
