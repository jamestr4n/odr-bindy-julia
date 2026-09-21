## Note

The reference implementation is
[ODR-BINDy](https://github.com/llfung/ODR-BINDy) in MATLAB, which is more mature
and is what the published results were produced with. Go check it out!

# Orthogonal Distance Regression based Bayesian Identification of Nonlinear Dynamics (ODRBINDy.jl)

Model discovery from noisy data for nonlinear dynamical systems, by jointly
denoising the trajectory and selecting model terms through maximising Bayesian
evidence.

Heavily inspired by [SINDy](https://github.com/dynamicslab/pysindy), but far
more noise robust than most SINDy variants and more akin to 4D-Var data
assimilation. There is no sparsity threshold to tune.

Julia port of [ODR-BINDy](https://github.com/llfung/ODR-BINDy) by L. Fung, the
code for [this article](https://doi.org/10.1145/3831701).

## Using the code

### Getting started

Run one of the examples in Julia at the top level folder and start exploring!

```
julia --project=. examples/lorenz.jl
```

### Using it on your own data

```julia
using ODRBINDy

lib = PolynomialLibrary(3, 2; varnames = ["x", "y", "z"])   # candidate terms
IMat, DMat = finite_difference_matrices(size(X, 1), 6, dt)  # discretisation
hyper = ODRHyperParameters(sigma_x = 0.2 * std(vec(X)),     # noise beliefs
                           sigma_y = 1e-4, sigma_p = 1e2,
                           Nx = size(X, 1), Neq = size(IMat, 1),
                           M = nterms(lib), D = 3)

res = odr_bindy(ODRProblem(X, lib, IMat, DMat, hyper))
print_model(res, lib)
```

From Lorenz63 data at 20% noise, this recovers:

```
dx/dt = -10.0027 * x + 9.9984 * y
dy/dt = 27.9931 * x - 0.9987 * y - 0.9999 * x*z
dz/dt = -2.6661 * z + 1.0001 * x*y
```

### Installation

Requires `Julia 1.9+`, with no dependencies outside the standard library. Not
yet registered, so install from this repository:

```julia
julia> ]
pkg> add https://github.com/YOUR-USERNAME/odr-bindy-julia
```

## Documentation

[This paper](https://doi.org/10.1145/3831701) introduces the theoretical
background. You may also want to check out
[the previous work](https://royalsocietypublishing.org/doi/full/10.1098/rspa.2024.0200),
which detailed the advantage of selecting a model by maximising Bayesian
evidence.

For this port, [`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md) explains how each
source file implements the theory, and
[`docs/PORTING.md`](docs/PORTING.md) documents the correspondence with the
MATLAB implementation.

Documentation contributions are welcome. Get in touch!

## Future Work

The success-rate benchmarks from the paper have not been reproduced in Julia
yet, and there is no test suite or CI. Also planned: `DataDrivenDiffEq.jl`
interoperability so a `Basis` can be passed directly, weak-form discretisation,
swappable optimisers, and a parallelised greedy search.

Collaborators and contributions are welcome. Get in touch!

## Relevant packages

If your data is too large for `ODR-BINDy` but not too noisy, try
[`B-SINDy`](https://github.com/llfung/B-SINDy) — the same Bayesian model
selection, but a much faster linear-regression-based technique. There is also
[a MATLAB app](https://github.com/llfung/ODR-BINDy-MATLABApp) for 2D and 3D
systems, which runs without a MATLAB license.

## Note on dependency and license

The algorithm, the method and the reference implementation are the work of
L. Fung. This repository is a reimplementation in Julia and does not modify the
original.

The MATLAB implementation reproduces some utility files (`SparseGalerkin.m`,
`poolData.m`, `poolDataList.m`) from
[SINDy](https://doi.org/10.1073/pnas.1517384113) by Brunton, Proctor & Kutz
(2016, PNAS) under the permission of the original authors. No code from those
files is reproduced here; the equivalent functionality was written independently
for this port. Please refer to `LICENSE` for the full attribution.
