# =============================================================================
# Lorenz 63 at 20% noise -- the headline example of the paper.
#
#     julia --project=. examples/lorenz.jl
#
# Data is generated here with a self-contained RK4 integrator so the example
# has no dependencies. Substepping keeps the integration error well below the
# noise we are about to add.
# =============================================================================

using ODRBINDy
using LinearAlgebra, Random, Statistics, Printf

# --- ground truth ------------------------------------------------------------

const SIGMA, RHO, BETA = 10.0, 28.0, 8 / 3

function lorenz(x)
    return [SIGMA * (x[2] - x[1]),
            x[1] * (RHO - x[3]) - x[2],
            x[1] * x[2] - BETA * x[3]]
end

"Classical RK4, `nsub` substeps per output sample. Returns an `N x D` matrix."
function integrate(f, x0, dt, N; nsub = 10)
    h = dt / nsub
    X = Matrix{Float64}(undef, N, length(x0))
    x = copy(x0)
    X[1, :] = x
    for i in 2:N, _ in 1:nsub
        k1 = f(x); k2 = f(x .+ h / 2 .* k1)
        k3 = f(x .+ h / 2 .* k2); k4 = f(x .+ h .* k3)
        x = x .+ (h / 6) .* (k1 .+ 2k2 .+ 2k3 .+ k4)
        X[i, :] = x
    end
    return X
end

# --- data --------------------------------------------------------------------

rng = MersenneTwister(20240717)

N, dt, D = 500, 0.01, 3
noise_level = 0.20                                   # fraction of each state's std

Xtrue = integrate(lorenz, [-8.0, 7.0, 27.0], dt, N)
sx = noise_level .* vec(std(Xtrue; dims = 1))        # per-state noise std
Xdata = Xtrue .+ randn(rng, N, D) .* sx'

@printf("Lorenz 63: N = %d, dt = %.3f, noise = %.0f%% (sigma = [%.2f %.2f %.2f])\n\n",
        N, dt, 100noise_level, sx...)

# --- problem -----------------------------------------------------------------

fd_order = 6
IMat, DMat = finite_difference_matrices(N, fd_order, dt)
Neq = size(IMat, 1)

lib = PolynomialLibrary(D, 2; varnames = ["x", "y", "z"])
M = nterms(lib)

# sigma_x: what we believe the measurement noise to be (here, what it is).
#
# sigma_y: how far off the discretised equation we are willing to sit. The
#          order-6 stencil's actual truncation error on this trajectory is
#          ~2e-4 RMS (1.8e-3 at worst), so 1e-2 leaves an order of magnitude of
#          slack. Keep it *soft*: driving sigma_y down towards the truncation
#          error turns the model term into a near-hard constraint and the
#          optimisation becomes stiff -- which is the failure mode the soft
#          constraint of paper eq. (6) exists to avoid.
#
# sigma_p: weakly informative prior on the coefficients.
hyper = ODRHyperParameters(repeat(sx', N, 1), fill(1e-2, Neq, D), fill(1e2, M, D))

prob = ODRProblem(Xdata, lib, IMat, DMat, hyper)
opts = ODROptions(n_multistart = 4, verbose = 1, rng = rng)

t0 = time()
res = odr_bindy(prob, opts)
@printf("\nelapsed: %.1f s\n\n", time() - t0)

# --- results -----------------------------------------------------------------

println("discovered:")
print_model(res, lib)

Xi_true = zeros(M, D)                                # 1 x y z x^2 xy xz y^2 yz z^2
Xi_true[2, 1], Xi_true[3, 1] = -SIGMA, SIGMA         # dx/dt = -10x + 10y
Xi_true[2, 2], Xi_true[3, 2] = RHO, -1.0             # dy/dt =  28x - y - xz
Xi_true[7, 2] = -1.0
Xi_true[4, 3], Xi_true[6, 3] = -BETA, 1.0            # dz/dt = -8/3 z + xy

println("\ntruth:")
print_model(Xi_true, lib)

correct = res.mask == (Xi_true .!= 0)
@printf("\nsupport recovered exactly: %s\n", correct ? "yes" : "NO")
if correct
    @printf("max coefficient error:     %.3g\n", maximum(abs, res.Xi .- Xi_true))
end
@printf("state RMSE:  data %.4f -> denoised %.4f\n",
        sqrt(mean(abs2, Xdata .- Xtrue)), sqrt(mean(abs2, res.X .- Xtrue)))
