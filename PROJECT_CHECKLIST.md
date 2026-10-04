# ODRBINDy.jl project checklist

Big-picture plan for the project, organised around Lloyd's two aims.
`[x]` = done.

---

## Aim 1: Port ODR-BINDy from MATLAB to Julia

- [x] Port the core pipeline: libraries, discretisation, LM solver, evidence, greedy search, regression
- [x] Lorenz63 example recovers the exact 7-term model at 20% noise (single seed)
- [x] Derivative-check script
- [x] README, WALKTHROUGH and PORTING docs
- [x] Core depends only on the standard library
- [ ] Walk Lloyd through the port. He asked to guide the approach before you code ODR-BINDy, so get his sign-off on anything that differs from the MATLAB version
- [ ] **Numerical equivalence with MATLAB**: run both versions on identical data (share a CSV) and compare
  - [ ] selected terms at every greedy step
  - [ ] coefficients (max absolute difference)
  - [ ] log-evidence at every greedy step
  - [ ] record the tolerance reached, e.g. "agrees to 1e-8"
- [x] **Reproduce the paper's headline result** for at least Lorenz63: same noise definition, sample count and number of trials. Lorenz63, T = 10, 64 trials per level: 100% / 100% / 97% at 10 / 20 / 30% noise against the paper's 100 / 100 / 90, all within the 95% CI (`benchmarks/lorenz_paper.jl`, `benchmarks/results/lorenz_paper_summary.csv`)
- [ ] Record any discrepancies and their causes in `docs/PORTING.md`

## Aim 2: Interface design and a full package

### 2a. Design (do this before refactoring)

- [x] Write `docs/DESIGN.md`: the four extension points, what each component must provide and return, and how they plug into `odr_bindy`. Review it with Lloyd.
- [x] Read DataDrivenDiffEq.jl's `Basis` and problem/solve API. Lloyd suggested reusing it, and matching SciML conventions (e.g. CommonSolve `solve(prob, alg)`) makes the package feel native. Findings and a working `Basis` wrapper prototype: `docs/DESIGN.md` §9
- [x] Define the abstract types (defaults `FiniteDifference`, `BuiltinLM`, `GreedyBackward`; results bit-identical, checked by `test/refactor_fixture.jl`):
  - [x] `AbstractLibrary`
  - [x] `AbstractDiscretisation` (returns `(IMat, DMat)`)
  - [x] `AbstractOptimiser` (contract: returns an `LMResult`)
  - [x] `AbstractModelSelector` (wraps the greedy loop)

### 2b. Pluggable components (at least one alternative for each)

| Component | Default (have) | Alternatives |
|---|---|---|
| Library | `PolynomialLibrary` ✓ | [x] DataDrivenDiffEq `Basis` wrapper (Lloyd's suggestion): `BasisLibrary`, in an extension · [x] Fourier / custom-function library: `FourierLibrary`, `CustomLibrary`, plus `CombinedLibrary` |
| Discretisation | Finite difference ✓ | [x] Weak form: `WeakForm`, using summation by parts (DESIGN §3.2) · [ ] spectral / Chebyshev · [x] uneven sampling: `FiniteDifference` with Fornberg weights |
| Optimiser | Built-in LM ✓ | [x] NonlinearSolve.jl backend (TrustRegion / LM): `NonlinearSolveOptimiser`, in an extension · [ ] LeastSquaresOptim.jl |
| Model selector | Greedy backward ✓ | [x] Exhaustive search (small libraries): `Exhaustive` · [x] beam search: `BeamSearch` (`BeamSearch(1)` is bit-identical to greedy) |

- [x] Each alternative has a test and a short docs example (`test/runtests.jl`, `test/extensions/runtests.jl`, `docs/COMPONENTS.md`)
- [x] One example script that swaps every component, to show the interface works end to end (`examples/swap_components.jl`: a pendulum from uneven samples, recovered exactly)

### 2c. Keep the core lightweight

- [x] Load heavy optional dependencies (DataDrivenDiffEq, NonlinearSolve) as **package extensions** (`[weakdeps]` + `[extensions]`, `ext/` folder, Julia 1.9+), so that `using ODRBINDy` stays stdlib-only. `ext/ODRBINDyDataDrivenDiffEqExt.jl` (`BasisLibrary`) and `ext/ODRBINDyNonlinearSolveExt.jl` (`NonlinearSolveOptimiser`); `using ODRBINDy` loads no package outside the standard library
- [x] Freeze the public API for v1.0: listed in `docs/API.md` and pinned by `test/api.jl`. `residual` and `jacobian` are no longer exported (they clashed with DataDrivenDiffEq), and the moved `ODROptions` keywords give deprecation warnings until v1.0. Version bumped to 0.2.0

## Track C: Software quality

- [x] Rename `gitignore` to `.gitignore` (without the dot it has no effect)
- [ ] `test/runtests.jl` with `@testset`s:
  - [ ] derivative checks (from `examples/check_derivatives.jl`)
  - [ ] exact recovery on noise-free data
  - [ ] Lorenz recovery at a fixed seed
  - [ ] MATLAB-equivalence regression test (small fixture saved in `test/data/`)
  - [x] one test per pluggable component (Aim 2): conformance checks in `test/conformance.jl`
  - [ ] type stability (`@inferred`) on hot functions
- [x] Add `[extras]` / `[targets]` for tests in `Project.toml`
- [ ] GitHub Actions CI: Julia 1.9 (LTS/min) and latest, on Linux, Windows and macOS
- [ ] Coverage via Codecov, and README badges (CI, coverage, docs)
- [ ] Static checks: Aqua.jl (ambiguities, stale deps), optionally JET.jl
- [ ] Docstrings for every exported function
- [ ] Documenter.jl site with doctests, deployed to GitHub Pages
- [ ] CompatHelper and TagBot workflows
- [ ] CONTRIBUTING.md and issue/PR templates
- [ ] Agree the package name and registration with Lloyd (it's his method)
- [ ] Register in the Julia General registry

## Track D: Benchmarks and metrics

- [ ] Use the **paper's definition of noise %** everywhere, and write it in the docs
- [ ] Benchmark systems:
  - [x] Lorenz63
  - [ ] Rössler
  - [ ] one or two more (Van der Pol, Duffing, Lorenz96)
- [ ] Noise sweep from 0% to 30%+, with **50–100 seeds per level**
- [ ] Metrics:
  - [ ] exact-model recovery rate, with a 95% confidence interval (Wilson)
  - [ ] TPR / FPR on which terms are selected
  - [ ] mean relative coefficient error
  - [ ] denoising RMS error before → after, averaged over seeds (the 1.62 → 0.16 figure is from a single seed)
  - [ ] runtime per fit
- [ ] Baselines on the same data and seeds:
  - [ ] SINDy with STLSQ (DataDrivenDiffEq.jl), with its threshold tuned fairly
  - [ ] weak / ensemble SINDy (PySINDy via PythonCall.jl, if time allows)
- [ ] Reproducible scripts in `benchmarks/` (fixed seeds) that write CSV results and plots
- [ ] Headline plot: recovery rate vs noise, ODR-BINDy against the baselines (goes in the README)

## Track E: Performance

- [ ] **Save a baseline benchmark before any optimisation** (BenchmarkTools: median time, memory, allocations). Every speedup is measured against it.
- [ ] Profile a Lorenz run to find the hot spots
- [ ] Cache `theta` within each LM step (currently evaluated twice), then re-benchmark
- [ ] Reuse the sparsity pattern of `J` across iterations, then re-benchmark
- [ ] `@threads` over the independent trial fits in each greedy sweep
- [ ] Measure thread scaling on 1 / 2 / 4 / 8 threads
- [ ] Measure how runtime grows with sample count N and library size
- [ ] Runtime against SINDy on the same data, and quantify the speed/accuracy trade-off

## Stretch F: Novel contribution

- [ ] **Decide by the project midpoint, with Lloyd**, whether to attempt one
- [ ] Candidates (check each against the paper so you don't reinvent something already in it):
  - weak-form discretisation inside ODR-BINDy (falls naturally out of Aim 2)
  - unevenly sampled or partially observed data
  - coefficient uncertainty from the Laplace approximation (the reduced Hessian is already computed, so this gives posterior covariance and credible intervals cheaply)
  - better model selectors (beam or exhaustive search against greedy)
- [ ] Show a measurable gain over base ODR-BINDy (noise tolerance, samples needed, runtime or recovery rate)
- [ ] Write it up: technical report, arXiv preprint or workshop paper

## Stretch G: Real-world data

- [ ] Choose one dataset with Lloyd (experimental pendulum, chemical kinetics, predator–prey, ...)
- [ ] Run the full pipeline on it
- [ ] Document how `sigma_x`, `sigma_y` and `sigma_p` were chosen

## Wrap-up

- [ ] Tag v1.0 with release notes
