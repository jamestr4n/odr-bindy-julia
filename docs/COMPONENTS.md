# Components

ODR-BINDy has four pluggable parts. This page lists what the package provides
for each and shows a short example of every alternative. The contracts a new
component must meet are in [DESIGN.md](DESIGN.md) §3. The checks in
[`test/conformance.jl`](../test/conformance.jl) test any implementation
against its contract, including your own.

| part | default | alternatives |
|---|---|---|
| library | `PolynomialLibrary` | `FourierLibrary`, `CustomLibrary`, `CombinedLibrary`, `BasisLibrary`¹ |
| discretisation | `FiniteDifference` (evenly spaced samples) | `FiniteDifference` on uneven samples, `WeakForm` |
| optimiser | `BuiltinLM` | `NonlinearSolveOptimiser`² |
| model selector | `GreedyBackward` | `BeamSearch`, `Exhaustive` |

¹ needs `using DataDrivenDiffEq`. ² needs `using NonlinearSolve`. Both load
as package extensions, so `using ODRBINDy` on its own still depends only on
the standard library.

Every example below assumes data `Xdata` (`Nx x D`) sampled at times `t`, and
per-state noise levels `sx`:

```julia
using ODRBINDy
prob = ODRProblem(Xdata, t, lib; discretisation = FiniteDifference(6),
                  sigma_x = sx, sigma_y = 1e-2, sigma_p = 1e2)
res  = odr_bindy(prob; optimiser = BuiltinLM(), selector = GreedyBackward())
print_model(res, lib)
```

[`examples/swap_components.jl`](../examples/swap_components.jl) replaces all
four at once.

## Libraries

**`FourierLibrary(D, nfreq)`**: `sin(k x_e)` and `cos(k x_e)` for
`k = 1..nfreq`. It has no constant term, so it can be combined with
polynomials without duplicating one.

```julia
lib = FourierLibrary(2, 2; varnames = ["θ", "ω"])
# sin(θ), cos(θ), sin(ω), cos(ω), sin(2*θ), cos(2*θ), sin(2*ω), cos(2*ω)
```

**`CombinedLibrary(libs...)`**: the columns of several libraries side by
side. For a pendulum, where `sin θ` is not a polynomial:

```julia
lib = CombinedLibrary(PolynomialLibrary(2, 1; varnames = ["θ", "ω"]),
                      FourierLibrary(2, 1; varnames = ["θ", "ω"]))
# 1, θ, ω, sin(θ), cos(θ), sin(ω), cos(ω)
```

**`CustomLibrary(D, fs, grads; names, varnames)`**: any functions of the
state, each with its exact gradient.

```julia
lib = CustomLibrary(2,
    [x -> x[1], x -> x[2], x -> x[1] / (1 + x[1]^2)],
    [x -> [1.0, 0.0], x -> [0.0, 1.0], x -> [(1 - x[1]^2) / (1 + x[1]^2)^2, 0.0]];
    names = ["x", "y", "x/(1+x^2)"], varnames = ["x", "y"])
```

**`BasisLibrary(basis)`**: a DataDrivenDiffEq `Basis`, evaluated through
its compiled function and symbolic Jacobian. Parameters stay at their default
values. The basis must be autonomous (no time dependence), with no controls or
implicit variables.

```julia
using ODRBINDy, DataDrivenDiffEq, Symbolics
@variables x y z
lib = BasisLibrary(Basis(polynomial_basis([x, y, z], 2), [x, y, z]))
```

`polynomial_basis` orders its terms differently from `PolynomialLibrary`, so
keep using `PolynomialLibrary` when comparing coefficient matrices with the
MATLAB code.

## Discretisations

**`FiniteDifference(order)` on uneven samples.** Pass any increasing `t`.
Evenly spaced samples use the fixed central stencil, exactly as before.
Otherwise each row gets its own stencil from `fornberg_weights`, which is
still accurate to `order`.

```julia
t = sort(rand(500)) .* 10
prob = ODRProblem(Xdata, t, lib; discretisation = FiniteDifference(6), ...)
```

**`WeakForm(radius; stride = 1, degree = 4, order = 6)`**: asks the model to
hold on average against polynomial bump test functions spanning
`2*radius + 1` samples, instead of pointwise. Summation by parts moves the
derivative onto the test function. Each equation then weights the noisy data
smoothly, with about a tenth of the noise gain of a finite-difference stencil,
while staying exact to the finite-difference `order` on any sampling. Each
test function integrates to one, so `sigma_y` keeps roughly its
finite-difference meaning.

```julia
prob = ODRProblem(Xdata, t, lib; discretisation = WeakForm(8), ...)
```

Two things to bear in mind. First, the weak form constrains the trajectory
only through local averages, so it denoises `X` less than finite differences
do. On the pendulum in `examples/swap_components.jl` it still finds the exact
model, with coefficients within 5%, but it cuts the state error by about a
quarter instead of most of it. Second, `-log(E)` is not comparable between
discretisations: they give different numbers of equations with different
correlations. Compare models only within one discretisation.

## Optimisers

**`NonlinearSolveOptimiser(alg)`**: any least-squares algorithm from
NonlinearSolve.jl. It is given the exact sparse Jacobian. It reports
`converged` from a Gauss–Newton decrement test rather than from NonlinearSolve's
return code (see its docstring for why).

```julia
using ODRBINDy, NonlinearSolve
res = odr_bindy(prob; optimiser = NonlinearSolveOptimiser(TrustRegion()))
res = odr_bindy(prob; optimiser = NonlinearSolveOptimiser(LevenbergMarquardt()))
```

## Model selectors

**`BeamSearch(width)`**: backward elimination that keeps the `width` best
models at each level instead of one, at about `width` times the cost of
greedy. `BeamSearch(1)` makes exactly the fits greedy makes and returns the
same result (tested).

```julia
res = odr_bindy(prob; selector = BeamSearch(3))
```

**`Exhaustive()`**: fits all `(2^M - 1)^D` sparsity patterns (see
`n_models`) and keeps the best, so it is for small problems only. It refuses to
start beyond `max_models` (10,000 by default). Use it to check whether greedy
finds the evidence-optimal model.

```julia
res = odr_bindy(prob; selector = Exhaustive())
res.history      # every model it fitted, with its -log(E)
```

## Tests

```
julia --project=. test/runtests.jl                                 # core components
julia --project=test/extensions test/extensions/runtests.jl        # BasisLibrary, NonlinearSolve
julia --project=. test/refactor_fixture.jl check                   # defaults unchanged
```
