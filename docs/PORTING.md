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

This is *not* numerically identical to `d2Jdp2_GaussNewtonEst.m`. The MATLAB
version computes `dX*/dXi` from the full Hessian in `X` (its `LHS` includes a
residual-weighted second-derivative term, and its `RHS` a residual-weighted
`dTheta` term) and only then forms the Gauss–Newton outer product. Here both
are Gauss–Newton, so the log-evidence differs from MATLAB's by a term that
scales with the model residual. Paper fig. 11 suggests the choice of Hessian
barely changes success rates, but per-step evidences will not match exactly.

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
  which is the failure mode paper eq. (6) exists to avoid. The paper itself
  uses `1e-3`, and at that value the default solver settings stall (see
  *Differences from the MATLAB implementation* below): use
  `ODROptions(lm_damping = :levenberg, lm_maxiter_refine = 16000)`. If the
  truncation error varies over the trajectory, pass a full `Neq x D` matrix (the
  MATLAB `NonlinearOscillator.m` uses a decaying `sigma_y` this way).
- `sigma_p`: prior std on coefficients. Large (`1e2`) = weakly informative.
  It does real work in the evidence, via the `sum(log sigma_p)` Occam term.

## Differences from the MATLAB implementation

Found while reproducing paper fig. 4 (`benchmarks/lorenz_paper.jl`). Each is a
deliberate or accidental difference in behaviour, not in the model.

| # | MATLAB | this package | effect |
|---|---|---|---|
| 1 | Noise % = `sigma_x / std(X_clean(:))`, one scalar pooled over all states | `examples/lorenz.jl` uses each state's own std | The example's "20%" is ~12–14% by the paper's definition. Benchmarks use the paper's. |
| 2 | `lsqnonlin`, trust-region-reflective, no variable scaling | Levenberg–Marquardt with Marquardt scaling `diag(J'J)` by default | At `sigma_y = 1e-3` Marquardt scaling over-damps the directions that move the trajectory along the ODE (curvature ~`1/sigma_x^2`, while `diag(J'J)` is ~`1/sigma_y^2`), so nearly every greedy trial hits its 100-step cap and no term can be removed. Fixed by the opt-in `lm_damping = :levenberg`. |
| 3 | Initial fit: 4 to 16 attempts, iteration cap doubling from 1000; later refits inherit the doubled cap (≥16000) | `n_multistart` attempts, fixed `lm_maxiter_refine = 1000` | At `sigma_y = 1e-3` the 30-term fit needs ~2000 steps, so the default cap fails every run. Pass `lm_maxiter_refine = 16000`. |
| 4 | Trial coefficients re-regressed on the warm-start `X` (`LinUseDenoise`) | Previous model's coefficients, one zeroed | Opt-in `trial_xi_init = :regress` gives the MATLAB behaviour. Not yet shown to change success rates. |
| 5 | Evidence: `dX/dXi` from the full `X`-Hessian | Pure Gauss–Newton Schur complement | Per-step `-log(E)` differs slightly; see *Why the Hessian collapsed*. |
| 6 | After a removal, keeps the lower of the refit and the trial; retries failed refits (`MaxFailedRun`) | Takes the refit whenever it is finite; no retry | Can change which model is kept at a step. |
| 7 | Rises in `-log(E)` counted from the 3rd removal on | Counted from the 1st | Can stop the search earlier. |
| 8 | Final model chosen among the reduced models only | The full library is also a candidate | Only matters if no removal ever improves the evidence. |
| 9 | Bootstrap: 1000 resamples for full fits and refits, 100 for trials | 100 throughout | Starting points differ; random streams differ anyway. |

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
