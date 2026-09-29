## 1. Goal

ODR-BINDy has four parts that could reasonably be done another way:

| extension point | question it answers | default today |
|---|---|---|
| **Library** | which candidate terms `Theta(X)` can appear in the model? | `PolynomialLibrary` |
| **Discretisation** | how is "`dX/dt = Theta(X) Xi`" turned into equations at discrete points? | central finite differences |
| **Optimiser** | how is the loss `L(X, Xi)` minimised for one sparsity pattern? | built-in Levenberg–Marquardt |
| **Model selector** | which sparsity patterns are tried, and which one is kept? | greedy backward elimination |

The aim is that each can be replaced without touching the other three or the
method itself. Three constraints hold throughout:

1. **The maths does not change.** The residual, its Jacobian and the evidence
   (`residual.jl`, `evidence.jl`) are the method, not extension points.
2. **The core stays stdlib-only.** Components that need heavy packages
   (DataDrivenDiffEq, NonlinearSolve) live in package extensions (Aim 2c).
3. **The defaults give identical results.** With default components and a fixed
   seed, the refactored package must select the same terms, with the same
   coefficients and `-log(E)`, as the current code. A fixture is saved before
   the refactor starts (§7).

## 2. Where each component plugs in

```
 Xdata, t
    │
    ├── Library ──────────► Theta(X), dTheta/dX   called inside every residual/Jacobian evaluation
    ├── Discretisation ───► (IMat, DMat)          built once, when the problem is constructed
    ▼
 ODRProblem
    │
    ▼
 Model selector ──(mask, X0, xi0, maxiter)──► fit_model ──► Optimiser ──► LMResult
      ▲                                           │
      └───────── ODRFit (Xi, X, -log E) ◄─────────┘   (evidence scored inside fit_model)
```

| component | used in | how often |
|---|---|---|
| Library | `residual`, `jacobian`, `bootstrap_ridge`, `print_model` | every LM step (hot path) |
| Discretisation | `ODRProblem` constructor | once |
| Optimiser | one call in `fit_model` | once per fit: ~`M²D²/2` fits for greedy |
| Model selector | `odr_bindy` | once |

Each point touches the rest of the code in only one or two places, which is
why this split is cheap to make: the refactor is mostly moving code behind a
type, not rewriting it.

## 3. The four contracts

Shapes: `Nx` samples, `D` states, `M` library terms, `Neq` equations
(collocation points). Everything is `Float64`, because the sparse Cholesky in
SuiteSparse supports nothing else.

### 3.1 Library — `AbstractLibrary` (exists)

**Must provide**

| method | returns |
|---|---|
| `nterms(lib)` | `M` |
| `theta(lib, X)` | `Nx x M` matrix, `Theta[i, n] = phi_n(X[i, :])` |
| `dtheta(lib, X)` | `Nx x M x D` array, `dTheta[i, n, e] = d phi_n / d x_e` at `X[i, :]` |
| `term_names(lib)` | `Vector{String}` of length `M` |

Optional: `state_names(lib, D)`. The fallback guesses the names from the
degree-one terms, which only works for polynomial-like libraries.

**Rules**

- It must work at *any* `X`, not only the data: the optimiser evaluates it at
  the denoised states, which move on every step.
- `dtheta` must be exact (analytic or automatic differentiation). An
  approximate `dtheta` gives a wrong Jacobian, which breaks both LM and the
  evidence. Only first derivatives are needed, because the evidence uses
  Gauss–Newton (see WALKTHROUGH §2.7).
- It is on the hot path, so it should allocate little. Track E caches `theta`
  within each LM step.

**Alternatives (2b)** *(all done; examples in [COMPONENTS.md](COMPONENTS.md))*

- `BasisLibrary(basis)` wrapping `DataDrivenDiffEq.Basis`. `theta` comes from
  evaluating the basis on the data, `dtheta` from its symbolic Jacobian (§9.1).
  The struct and its `theta`/`dtheta` are in the core, since they only call
  the stored basis and Jacobian. Only the constructor needs DataDrivenDiffEq,
  and it lives in `ext/ODRBINDyDataDrivenDiffEqExt.jl`. It refuses bases that
  depend on time or have controls or implicit variables (§8, question 2).
- `FourierLibrary(D, nfreq)`, and `CustomLibrary(D, fs, grads)` where the user
  supplies each function and its gradient.
- `CombinedLibrary(libs...)`, which puts the columns side by side, e.g.
  polynomials plus sines.

Libraries that know their state names define `state_names(lib, D)`, so the
fallback guess is only used for `PolynomialLibrary`.

### 3.2 Discretisation — `AbstractDiscretisation` (defined)

**Must provide**

| method | returns |
|---|---|
| `operators(disc, t)` | `(IMat, DMat)`, both `SparseMatrixCSC{Float64}`, size `Neq x Nx` |
| `collocation_times(disc, t)` | `Vector` of length `Neq`: where each equation "lives" (for plotting and for a time-varying `sigma_y`) |

`t` is the full vector of sample times, not `dt`, so uneven sampling fits the
same signature. The model residual is then, as now,

    eta = DMat * X  -  (IMat * Theta(X)) * Xi        (Neq x D)

**Rules**

- The operators must be **linear and independent of `X`**. `jacobian` relies on
  both: it differentiates `eta` using `DMat` and `IMat` directly. Any scheme
  that fits this form works with no other code changes.
- They should be **sparse and banded**. The whole method is affordable because
  `J'J` is sparse (WALKTHROUGH §2.6). Dense operators still work, but each LM
  step then costs `O((Nx D)^3)`.
- `sigma_y` means "standard deviation of `eta`", so its sensible scale depends
  on the discretisation (see §8, question 1).

**Default:** `FiniteDifference(order)`, which wraps
`finite_difference_matrices` for evenly spaced `t`.

**Alternatives (2b)**

| scheme | `IMat[k, i]` | `DMat[k, i]` | status |
|---|---|---|---|
| uneven finite differences | selects the centre point | Fornberg weights for that row's stencil | **done**, as the same `FiniteDifference` type. Evenly spaced `t` (to 1e-4 relative) still takes the old code path, bit-identical. |
| weak form (`WeakForm(radius)`) | `w_i * phi_k(t_i)` | `(IMat * Dt)[k, i]` | **done**; see below |
| spectral / Chebyshev | identity | Chebyshev differentiation matrix | not done: dense, so only for small `Nx` |

The weak form (also candidate F.1) fits the contract with no special handling.
`phi_k` is the bump `((t - a)(b - t))^degree` on `2*radius + 1` consecutive
samples, normalised to integrate to one, and `w_i` are trapezoidal weights.

The textbook weak form (WSINDy) sets `DMat[k, i] = -w_i phi_k'(t_i)`, i.e.
it integrates by parts and then applies quadrature. That was the plan, but it
was measured to be inconsistent. `DMat*x - IMat*x'` does not vanish for smooth
`x`: at `radius = 8`, `degree = 4` it is ~3e-4 relative, independent of the
sampling rate, because the error depends on the number of samples per test
function. On uneven samples it is worse, since the trapezoidal rule is only
second order there. For Lorenz (`|x'| ~ 100`) that bias is as large as
`sigma_y`.

The implementation therefore uses *summation* by parts:
`DMat = IMat * Dt`, with `Dt` the full-length finite-difference derivative
(`derivative_matrix(t, order)`). Each row of `DMat` is still a smooth discrete
`-(w phi_k)'`, with the same noise gain as the textbook version (row norm 3.7,
against 43 for an order-6 stencil). But `DMat*x = IMat*x'` now holds to the
finite-difference order on any sampling: 7e-9 at `N = 100`, falling to 4e-14
at `N = 800`, even and uneven alike.

### 3.3 Optimiser — `AbstractOptimiser` (defined)

**Must provide**

```julia
optimise(opt, fr, fJ, z0; maxiter) -> LMResult
```

- `fr(z)` returns the residual vector, and `fJ(z)` its exact sparse Jacobian.
  Backends must **use the supplied Jacobian**, not automatic or
  finite-difference derivatives: `z` has `Nx*D + Np` entries (~3000 for Lorenz)
  and the analytic Jacobian is already verified.
- It returns `LMResult(z, r, cost, iterations, converged)`, with `r = fr(z)`
  and `cost = ||r||² / 2`. The name is kept for now, although the type is no
  longer LM-specific.

**Rules**

- **`maxiter` is passed on every call, not stored in the optimiser**, because
  the selector sets it: greedy uses a tight cap for trials and a loose one for
  refits.
- **`converged` must be honest.** It is `true` only if a tolerance was met, and
  `false` when the iteration cap or damping limit was hit. This matters: the
  greedy search treats "this trial did not converge within the cap" as evidence
  that the removed term is needed (paper §2.3). A backend that reports success
  on hitting the cap would silently change which models are selected.
- Solver-specific settings (damping, acceleration, tolerances) live in the
  optimiser struct, because their meaning differs between backends.

**Default:** `BuiltinLM(; damping = :marquardt, accel = false, ftol, xtol, gtol, lambda0, ...)`,
which wraps `levenberg_marquardt`. It is deliberately not called
`LevenbergMarquardt`, because NonlinearSolve and LeastSquaresOptim both export
that name, and `using` both packages would then give a name clash.

**Alternatives (2b), in extensions**

- `NonlinearSolveOptimiser(alg)` for any NonlinearSolve least-squares
  algorithm, e.g. `TrustRegion()` (closest to MATLAB's `lsqnonlin`). **Done.**
  The struct is in the core and its `optimise` method in
  `ext/ODRBINDyNonlinearSolveExt.jl`. `sol.stats.nsteps` gives `iterations`.
  Mapping the return code to `converged`, as planned, turned out to be unsafe.
  For a least-squares problem, NonlinearSolve's stopping rules look at the
  residual norm, which cannot reach zero here, or at a run of short steps, and
  a trust-region step that is rejected counts as a step of length zero. A run
  of rejections can therefore end in `StalledSuccess` before convergence.
  Instead, `converged` comes from one test that works for any algorithm:
  `gauss_newton_decrement(r, J) <= ftol * cost` (default `ftol = 1e-10`),
  meaning one more Gauss–Newton step could not lower the cost by more than a
  relative `ftol`. The extension also uses this test to decide when to stop,
  advancing the solver one step at a time (`init`/`step!`). Left to its own
  stopping rule, NonlinearSolve's `LevenbergMarquardt` almost never takes the
  run of short steps it waits for, so every greedy trial ran to its cap: one
  Van der Pol selection took ~45 min. With the test it takes 17 steps per full
  fit, and `TrustRegion` takes 4, against 13 for `BuiltinLM`.
- `LSOOptimiser(...)` for LeastSquaresOptim.jl. Not done: not installed, and
  NonlinearSolve can already call it through its own extension.

A caution for comparing backends: selection depends on how many steps a trial
needs relative to the cap (PORTING difference 2), so two correct optimisers can
select different models. This is worth measuring, not a bug to hide.

### 3.4 Model selector — `AbstractModelSelector` (defined)

**Must provide**

```julia
select_model(sel, prob, optimiser, opts) -> ODRResult
```

It builds fits with the existing `fit_model(prob, mask, opts; optimiser, X0, xi0, maxiter)`
and `fit_model_multistart`. (`optimiser` is a keyword, so existing calls
without one keep working.) The selector owns everything about the search:
which masks to try and in what order, warm starts, the two iteration caps, the
stopping rule and which model to return.

**Rules**

- Every state keeps at least one term.
- It returns the lowest `-log(E)` among the models it evaluated, and records
  the path in `history`.
- `history` becomes `(nterms, nlevidence, mask)` instead of
  `(nterms, nlevidence, removed)`, since "removed" only makes sense for greedy.
  For greedy the removed term is the difference between consecutive masks.
- If trials run in parallel (Track E), they must not share a mutable RNG.
  `trial_xi_init = :regress` draws bootstrap samples, so each thread needs its
  own stream.

**Default:** `GreedyBackward(; stop_after_rises = 2, warm_start = true,
trial_xi_init = :previous, refine_after_removal = true, trial_maxiter = 100,
refine_maxiter = 1000)`. This is today's `odr_bindy` loop, moved.

**Alternatives (2b)** *(both done, in `src/selectors.jl`)*

- `Exhaustive()`: every mask. There are `(2^M - 1)^D` of them (`n_models`),
  because the states are coupled through the shared denoised `X`, so the search
  cannot be split per state. Lorenz with quadratics has ~10⁹, so this is only
  for toy problems (e.g. `D = 2, M = 6` has 3,969), and it refuses to start
  beyond `max_models`. Its value is as ground truth: does greedy find the
  evidence-optimal model? On question 3 below, it uses a middle ground: every
  mask warm-starts from the full-library fit's denoised `X`, with ridge
  coefficients, and gets the generous `maxiter`, so a non-convergence is not a
  tight-cap artefact.
- `BeamSearch(width)`: keep the `width` best masks at each level. It costs about
  `width` times as much as greedy. `width = 1` reproduces greedy exactly (same
  fits, same order, same RNG draws), and a test checks this bit for bit.

## 4. Splitting `ODROptions`

`ODROptions` currently mixes settings for all four parts. Each field moves to
the component that uses it:

| field | new home |
|---|---|
| `lm_damping`, `lm_accel`, `ftol`, `xtol`, `gtol` | `BuiltinLM` |
| `lm_maxiter` | `GreedyBackward.trial_maxiter` |
| `lm_maxiter_refine` | `GreedyBackward.refine_maxiter` |
| `warm_start`, `trial_xi_init`, `refine_after_removal`, `stop_after_rises` | `GreedyBackward` |
| `n_multistart`, `bootstrap_samples`, `bragging`, `verbose`, `rng` | stay in `ODROptions` (initial guess and general settings) |

Until v1.0, the old keywords keep working with a deprecation warning that names
the new home. This matters for checking the refactor, because
`benchmarks/lorenz_paper.jl` can then be rerun unchanged and compared.

**Status (v0.2):** done. Passing any of the moved keywords to `ODROptions` gives a
deprecation warning that names the new home (`Base.depwarn`, so it shows under
`] test` or `--depwarn=yes`). The fields still configure the *default*
optimiser and selector (`default_optimiser(opts)`, `default_selector(opts)`)
and are ignored when `odr_bindy` is given an `optimiser` or `selector`. They
are removed in v1.0. The examples, benchmarks and fixture now use the
components directly. The frozen API is listed in [API.md](API.md).

## 5. User-facing API

Today:

```julia
IMat, DMat = finite_difference_matrices(N, 6, dt)
hyper = ODRHyperParameters(sigma_x = sx, sigma_y = 1e-3, sigma_p = 1e2,
                           Nx = N, Neq = size(IMat, 1), M = nterms(lib), D = 3)
prob  = ODRProblem(Xdata, lib, IMat, DMat, hyper)
res   = odr_bindy(prob, ODROptions(lm_damping = :levenberg, lm_accel = true,
                                   lm_maxiter_refine = 16_000))
```

Proposed:

```julia
prob = ODRProblem(Xdata, t, lib; discretisation = FiniteDifference(6),
                  sigma_x = sx, sigma_y = 1e-3, sigma_p = 1e2)
res  = odr_bindy(prob; optimiser = BuiltinLM(damping = :levenberg, accel = true),
                       selector  = GreedyBackward(refine_maxiter = 16_000))
```

- Each `sigma` can be a scalar, a length-`D` vector (per state) or a full
  matrix. The constructor broadcasts it *after* building the operators, so the
  user no longer has to compute `Neq`. This also fixes the "which `Neq` does a
  weak form have?" problem for free.
- The problem also stores `t` and the discretisation, so `collocation_times`
  and plotting work without extra arguments. *(Not done yet: the problem stores
  only the operators it was built from.)*
- `odr_bindy(prob; optimiser, selector, opts)` just calls
  `select_model(selector, prob, optimiser, opts)`.
- **The current constructor stays**: `ODRProblem(Xdata, lib, IMat, DMat, hyper)`
  still works. It lets anyone try a new discretisation without writing a type.

Whether this should be spelled `solve(prob, ODRBINDyAlg(...))` for SciML
consistency is §8, question 5.

## 6. A conformance test for each extension point

Each extension point gets a generic check in `test/` that any implementation,
including a user's own, can run. This is also the Track C item "one test per
pluggable component".

| check | what it verifies |
|---|---|
| `test_library(lib, D)` | shapes; `length(term_names) == M`; `dtheta` against central differences of `theta` (from `check_derivatives.jl` §1) |
| `test_discretisation(disc, t)` | sizes match; `IMat*f` reproduces `f` and `DMat*f` reproduces `f'` on a smooth function, to the expected order (`check_derivatives.jl` §2) |
| `test_optimiser(opt)` | solves a small nonlinear least-squares problem with a known answer; returns `converged = false` when capped at `maxiter = 2` |
| `test_selector(sel)` | recovers the exact Lorenz model on noise-free data; the returned model has the lowest `-log(E)` in `history` |

## 7. Refactor order

1. **Save a fixture first.** Record the selected mask, `Xi` and the `-log(E)`
   path from `examples/lorenz.jl` at a fixed seed, and write them to
   `test/data/`. *(Done: `test/refactor_fixture.jl`, which also covers a 2D
   system with every non-default option switched on.)*
2. Add the three new abstract types, each with its default wrapping the
   existing code (`FiniteDifference`, `BuiltinLM`, `GreedyBackward`). Split
   `ODROptions` (§4). Check against the fixture: the result must be identical,
   not just close. *(Done, including the deprecations in §4. Fixture
   identical.)*
3. Add the new `ODRProblem` constructor (§5). The fixture must still match.
   *(Done.)*
4. Add alternatives one at a time (2b), each with its conformance test.
   *(Done: the checks are in `test/conformance.jl`, and are run by
   `test/runtests.jl` and `test/extensions/runtests.jl`. The fixture still
   matches.)*
5. Move heavy-dependency alternatives into package extensions (2c).
   *(Done: `BasisLibrary` and `NonlinearSolveOptimiser`. `using ODRBINDy`
   loads no package outside the standard library.)*
6. Freeze the public API (2c). *(Done: [API.md](API.md), pinned by
   `test/api.jl`.)*

## 8. Open questions for Lloyd

1. **`sigma_y` for other discretisations.** With the weak form, `eta` is an
   integral over a test function's support, so its natural scale (and a
   sensible `sigma_y`) differs from finite differences. Also, overlapping test
   functions make the rows of `eta` correlated, so a diagonal `sigma_y` is only
   an approximation. Is that acceptable? A full covariance would mean
   whitening `IMat` and `DMat`, which destroys their sparsity.
2. **Time- and parameter-dependent libraries.** A DataDrivenDiffEq `Basis` is
   `f(u, p, t)`, but `theta(lib, X)` has no `t`. Should the contract become
   `theta(lib, X, t)`, with `t` treated as noise-free? And should `dtheta`
   stay with respect to `X` only? The `Basis` prototype (§9.1) shows that a
   basis containing `t` would be evaluated wrongly without this.
3. **Non-convergence as a signal.** Greedy reads "the trial did not converge
   within the cap" as "keep this term". Exhaustive and beam search fit some
   models from cold starts, where non-convergence means something weaker.
   Should they refit from other starting points before scoring a model `Inf`?
4. **The prior and the evidence are fixed here.** Is there a reason to make the
   prior (currently Gaussian, `sigma_p`) a fifth extension point, or should it
   stay out of scope?
5. **SciML conventions.** `solve(prob, alg)` via CommonSolve.jl would make the
   package feel native next to DataDrivenDiffEq. CommonSolve is tiny but is not
   in the standard library. Should it be a core dependency, or only defined in
   the DataDrivenDiffEq extension? **Proposed answer (§9.3):** define it in a
   CommonSolve package extension, so the core stays stdlib-only.

## 9. DataDrivenDiffEq.jl: findings

From reading the source of DataDrivenDiffEq 1.15.5 (and DataDrivenSparse, which
plugs into it) and running a prototype wrapper. The prototype lives outside the
repo; the wrapper itself is reproduced in §9.1.

### 9.1 `Basis` works as a library, unchanged

A `Basis` is called as `basis(X, p, t)` with `X` as `D x N` (columns are
samples), `p` the parameter values and `t` a time vector. The package's
`jacobian(basis)` compiles the symbolic Jacobian with respect to the states,
called as `J(u, p, t)` for one point, giving `M x D`. (Called on a matrix it
concatenates the per-point results into `M x (D*N)`, so a loop over points is
simpler.) The whole wrapper is:

```julia
struct BasisLibrary{B,J} <: AbstractLibrary
    basis::B
    jac::J                  # DataDrivenDiffEq.jacobian(basis), compiled once
    p::Vector{Float64}      # get_parameter_values(basis): fixed, not fitted
    D::Int                  # length(states(basis))
    names::Vector{String}   # [string(eq.rhs) for eq in basis]
end
nterms(l::BasisLibrary) = length(l.basis)
theta(l::BasisLibrary, X) = permutedims(l.basis(permutedims(X), l.p, zeros(size(X, 1))))
function dtheta(l::BasisLibrary, X)
    dTh = zeros(size(X, 1), nterms(l), l.D)
    for i in axes(X, 1)
        dTh[i, :, :] .= l.jac(X[i, :], l.p, 0.0)
    end
    return dTh
end
```

Prototype results:

| check | result |
|---|---|
| `polynomial_basis([x, y, z], 2)` against `PolynomialLibrary(3, 2)` | same 10 terms, identical `theta` and `dtheta`, but a different column order: `1, x, x^2, y, x*y, y^2, z, x*z, y*z, z^2` |
| trig basis with a parameter, `[x, y, sin(w x), cos(w y), x sin(y)]` | `dtheta` matches central differences to 1.7e-10 |
| `examples/lorenz.jl` data, full `odr_bindy`, both libraries | same 7 terms, identical `-log(E)` (794.564414), coefficients agree to 5.9e-10 |
| cost per call at `N = 1000` | `theta` 0.16 ms (vs 0.06), `dtheta` 3.0 ms (vs 0.07) |
| total fit time | 398 s vs 372 s (+7%): the library is not the bottleneck |

What the real extension must add:

- **`state_names`.** The fallback guesses names from the terms and printed
  `dx1/dt` rather than `dx/dt`, because `Basis` orders terms differently.
  Take them from `states(basis)` instead.
- **Time.** The prototype passes `t = 0`, which is only right for autonomous
  bases. A basis containing `t` needs the real sample times, so `theta` needs
  `t` (§8, question 2). This is now a concrete requirement, not a hypothetical.
- **Parameters** are held at their default values. ODR-BINDy does not fit them.
- **Column order.** MATLAB ground-truth `Xi` matrices use `PolynomialLibrary`'s
  order, so comparisons against MATLAB should keep using `PolynomialLibrary`.

### 9.2 `solve` is built around regression on estimated derivatives

`solve(prob, basis, alg)` runs `CommonSolve.init`, which:

1. evaluates `basis(prob)` and takes precomputed derivatives `DX` as the
   target (`get_fit_targets`),
2. optionally denoises and normalises that data, and splits it into
   train/test batches (`DataDrivenCommonOptions`),
3. hands the result to `CommonSolve.solve!(::InternalDataDrivenProblem{Alg})`,
   which each algorithm package defines. It must return a
   `DataDrivenSolution`, whose result objects implement the StatsAPI accessors
   (`coef`, `rss`, `dof`, `nobs`, `loglikelihood`, ...).

ODR-BINDy needs none of steps 1–2. It replaces estimated derivatives with its
own discretisation and denoises `X` jointly with the fit. An adapter is still
possible: define `solve!` for an `ODRBINDyAlg`, ignore the preprocessed
batches, and read the raw `problem.X` (`D x N`), `problem.t` and `basis` from
the internal problem. Then rebuild the answer as a `Basis` with the public
`DataDrivenDiffEq.__construct_basis`, as DataDrivenSparse does. Two cautions:

- `ContinuousDataDrivenProblem(X, t, kernel)` with a collocation *kernel*
  replaces `X` by a smoothed version before any algorithm sees it. ODR-BINDy's
  `sigma_x` then describes the wrong noise. The default (linear interpolation)
  leaves `X` untouched. The adapter must document this and should warn when it
  can tell.
- `DataDrivenSolution` computes its own `rss` against the estimated `DX`,
  which is not ODR-BINDy's loss. `-log(E)` should be reported through the
  result object instead.

### 9.3 Recommendations

1. **`BasisLibrary` in a DataDrivenDiffEq extension (2b, 2c).** It is proven
   to work. It is the cheapest piece of Lloyd's suggestion and should come
   first.
2. **`CommonSolve.solve` in an extension, not in the core.** CommonSolve has no
   dependencies (one 209-line file). Every SciML package re-exports its
   `solve`, and DataDrivenDiffEq does too. Defining
   `CommonSolve.solve(prob::ODRProblem, alg::ODRBINDyAlg)` in
   `ext/ODRBINDyCommonSolveExt.jl` means `solve(prob, alg)` works whenever the
   user has any SciML package loaded, and `using ODRBINDy` stays stdlib-only.
   `ODRBINDyAlg` is a plain struct in the core, bundling the optimiser,
   selector and options. `odr_bindy` stays as the core entry point. This
   answers §8, question 5.
3. **A `solve(::DataDrivenProblem, basis, ODRBINDyAlg())` adapter is optional,
   later.** It lets DataDrivenDiffEq users switch from STLSQ with one line, and
   it would help the SINDy baseline in Track D. But it inherits the two
   cautions above, so it comes after the native API.
4. **Stop exporting `jacobian` (and probably `residual`).** DataDrivenDiffEq,
   Symbolics and others export a `jacobian` too. With both packages loaded, an
   unqualified `jacobian` call fails with an ambiguity error (hit while writing
   the prototype). They stay reachable as `ODRBINDy.jacobian`. *(Done in
   v0.2: both are unexported, and are marked `public` on Julia 1.11+.)*
