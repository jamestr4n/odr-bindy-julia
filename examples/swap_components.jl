# =============================================================================
# Every component swapped: a damped pendulum, sampled unevenly, identified
# with none of the defaults.
#
#     library         PolynomialLibrary      -> polynomials + FourierLibrary
#     discretisation  FiniteDifference       -> WeakForm (on uneven samples)
#     optimiser       BuiltinLM              -> NonlinearSolve's TrustRegion
#     selector        GreedyBackward         -> BeamSearch(3)
#
# Needs NonlinearSolve, so run it in an environment that has it, e.g.
#
#     julia --project=test/extensions examples/swap_components.jl
#
# The true model is  θ' = ω,  ω' = -2 sin θ - 0.2 ω.  A polynomial library can
# only approximate sin θ (the pendulum swings to ±2.5 rad, far from sin θ ≈ θ),
# so the Fourier terms are what make the exact model findable.
# =============================================================================

using ODRBINDy
using NonlinearSolve
using Random, Statistics, Printf

"RK4 between the given sample times, `nsub` substeps each."
function integrate(f, x0, t; nsub = 10)
    X = Matrix{Float64}(undef, length(t), length(x0))
    x = collect(float.(x0))
    X[1, :] = x
    for i in 2:length(t)
        h = (t[i] - t[i - 1]) / nsub
        for _ in 1:nsub
            k1 = f(x); k2 = f(x .+ h / 2 .* k1)
            k3 = f(x .+ h / 2 .* k2); k4 = f(x .+ h .* k3)
            x = x .+ (h / 6) .* (k1 .+ 2k2 .+ 2k3 .+ k4)
        end
        X[i, :] = x
    end
    return X
end

# --- data: 300 samples on [0, 15], each moved by up to ±40% of the spacing ----

rng = MersenneTwister(2026)
N, T = 300, 15.0
t = collect(range(0, T; length = N))
t[2:(end - 1)] .+= 0.4 * (T / (N - 1)) .* (2 .* rand(rng, N - 2) .- 1)

pendulum(x) = [x[2], -2sin(x[1]) - 0.2x[2]]
Xtrue = integrate(pendulum, [2.5, 0.0], t)
sx = 0.05 .* vec(std(Xtrue; dims = 1))                 # 5% noise
Xdata = Xtrue .+ randn(rng, N, 2) .* sx'

@printf("pendulum: N = %d uneven samples (spacing %.3f to %.3f), 5%% noise\n\n",
        N, extrema(diff(t))...)

# --- the four components ------------------------------------------------------

vars = ["θ", "ω"]
library = CombinedLibrary(PolynomialLibrary(2, 1; varnames = vars),  # 1, θ, ω
                          FourierLibrary(2, 1; varnames = vars))     # sin, cos of each
discretisation = WeakForm(8)
optimiser = NonlinearSolveOptimiser(TrustRegion())
selector = BeamSearch(3)

prob = ODRProblem(Xdata, t, library; discretisation = discretisation,
                  sigma_x = sx, sigma_y = 1e-2, sigma_p = 1e2)

t0 = time()
res = odr_bindy(prob, ODROptions(rng = rng); optimiser = optimiser, selector = selector)
@printf("\nelapsed: %.1f s\n\n", time() - t0)

# --- results ------------------------------------------------------------------

println("discovered:")
print_model(res, library)

Xi_true = zeros(nterms(library), 2)       # 1 θ ω sin θ cos θ sin ω cos ω
Xi_true[3, 1] = 1.0                       # θ' = ω
Xi_true[3, 2], Xi_true[4, 2] = -0.2, -2.0 # ω' = -0.2 ω - 2 sin θ
println("\ntruth:")
print_model(Xi_true, library)

correct = res.mask == (Xi_true .!= 0)
@printf("\nsupport recovered exactly: %s\n", correct ? "yes" : "NO")
correct && @printf("max coefficient error:     %.3g\n", maximum(abs, res.Xi .- Xi_true))
@printf("state RMSE:  data %.4f -> denoised %.4f\n",
        sqrt(mean(abs2, Xdata .- Xtrue)), sqrt(mean(abs2, res.X .- Xtrue)))
