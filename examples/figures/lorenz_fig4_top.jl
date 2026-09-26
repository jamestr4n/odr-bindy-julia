# =============================================================================
# Recreate the top panel of Fig. 4 of Fung (2025), arXiv:2507.23426: ODR-BINDy
# recovering Lorenz63 from data with 20% noise, and denoising it.
#
#     julia --project=examples/figures -e "using Pkg; Pkg.instantiate()"   # once
#     julia --project=examples/figures examples/figures/lorenz_fig4_top.jl
#
# Data settings follow Lorenz.m, the MATLAB script that draws the paper's figure:
# x0 = [-8, 8, 27], dt = 0.01, T = 5 (N = 500), noise = 0.2 * std of all the
# clean data, one scalar for every state. Solver settings are those of the Fig. 4
# benchmark (benchmarks/lorenz_paper.jl). The noise draw is not the paper's:
# MATLAB's rng(12) and Julia's MersenneTwister(12) are different streams.
#
# The fit is cached next to this script (lorenz_fig4_top_*.csv), so restyling
# the figure does not refit. Delete those files to refit.
# =============================================================================

using ODRBINDy
using LinearAlgebra, Random, Statistics, Printf, DelimitedFiles
using Plots

const SIGMA, RHO, BETA = 10.0, 28.0, 8 / 3
const X0 = [-8.0, 8.0, 27.0]
const DT, T, NOISE = 0.01, 5.0, 0.2
const FD_ORDER = 6

const STATES_CSV = joinpath(@__DIR__, "lorenz_fig4_top_states.csv")
const XI_CSV = joinpath(@__DIR__, "lorenz_fig4_top_xi.csv")
const OUT = joinpath(@__DIR__, "..", "..", "docs", "images", "lorenz_fig4_top.png")

lorenz(x) = [SIGMA * (x[2] - x[1]), x[1] * (RHO - x[3]) - x[2], x[1] * x[2] - BETA * x[3]]

"Classical RK4, `nsub` substeps per sample; the first row is `x0`."
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

function truth()
    Xi = zeros(10, 3)                       # 1 x y z x^2 xy xz y^2 yz z^2
    Xi[2, 1], Xi[3, 1] = -SIGMA, SIGMA
    Xi[2, 2], Xi[3, 2], Xi[7, 2] = RHO, -1.0, -1.0
    Xi[4, 3], Xi[6, 3] = -BETA, 1.0
    return Xi
end

# --- fit (or load the cached one) ----------------------------------------------

N = round(Int, T / DT)
lib = PolynomialLibrary(3, 2; varnames = ["x", "y", "z"])

if isfile(STATES_CSV) && isfile(XI_CSV)
    S = readdlm(STATES_CSV, ','; skipstart = 1)
    Xclean, Xdata, Xhat = S[:, 1:3], S[:, 4:6], S[:, 7:9]
    Xi = readdlm(XI_CSV, ',')
    println("loaded cached fit (delete $(basename(STATES_CSV)) to refit)")
else
    rng = MersenneTwister(12)
    Xclean = integrate(lorenz, X0, DT, N)
    sx = NOISE * std(vec(Xclean))
    Xdata = Xclean .+ sx .* randn(rng, N, 3)

    IMat, DMat = finite_difference_matrices(N, FD_ORDER, DT)
    hyper = ODRHyperParameters(sigma_x = sx, sigma_y = 1e-3, sigma_p = 100.0,
                               Nx = N, Neq = size(IMat, 1), M = nterms(lib), D = 3)
    prob = ODRProblem(Xdata, lib, IMat, DMat, hyper)
    opts = ODROptions(n_multistart = 4, lm_maxiter_refine = 16000, lm_damping = :levenberg,
                      lm_accel = true, verbose = 1, rng = rng)

    @printf("Lorenz63: N = %d, dt = %.2f, noise = %.0f%% (sigma_x = %.4f)\n\n",
            N, DT, 100NOISE, sx)
    t0 = time()
    res = odr_bindy(prob, opts)
    @printf("\nelapsed: %.1f s\n", time() - t0)
    Xhat, Xi = res.X, res.Xi

    open(STATES_CSV, "w") do io
        println(io, "x_clean,y_clean,z_clean,x_data,y_data,z_data,x_odr,y_odr,z_odr")
        writedlm(io, hcat(Xclean, Xdata, Xhat), ',')
    end
    writedlm(XI_CSV, Xi, ',')
end

println("\ndiscovered:")
print_model(Xi, lib)
@printf("\nsupport recovered exactly: %s\n", (Xi .!= 0) == (truth() .!= 0) ? "yes" : "NO")
@printf("state RMSE: data %.3f -> denoised %.3f\n",
        sqrt(mean(abs2, Xdata .- Xclean)), sqrt(mean(abs2, Xhat .- Xclean)))

# --- figure (layout of Lorenz.m: 3D view on the left, x, y, z against t) --------

gr()
default(fontfamily = "Computer Modern", framestyle = :box, grid = true, gridalpha = 0.25)
C_DATA, C_TRUTH, C_ODR = RGB(0.85, 0.2, 0.15), :black, RGB(0.0, 0.45, 0.74)
t = DT .* (0:N-1)

p3 = scatter3d(Xdata[:, 1], Xdata[:, 2], Xdata[:, 3]; marker = :xcross, ms = 2.5,
               msw = 0.6, color = C_DATA, alpha = 0.5, label = "Data")
plot3d!(p3, Xclean[:, 1], Xclean[:, 2], Xclean[:, 3]; color = C_TRUTH, lw = 1.2, label = "Truth")
plot3d!(p3, Xhat[:, 1], Xhat[:, 2], Xhat[:, 3]; color = C_ODR, lw = 1.8, ls = :dash,
        label = "ODR-BINDy")
plot!(p3; xlabel = "x", ylabel = "y", zlabel = "z", camera = (40, 20), legend = :topleft)

ts = map(1:3) do i
    p = scatter(t, Xdata[:, i]; marker = :xcross, ms = 2, msw = 0.6, color = C_DATA,
                alpha = 0.5, label = "")
    plot!(p, t, Xclean[:, i]; color = C_TRUTH, lw = 1.2, label = "")
    plot!(p, t, Xhat[:, i]; color = C_ODR, lw = 1.8, ls = :dash, label = "")
    plot!(p; ylabel = ("x", "y", "z")[i], xlabel = i == 3 ? "t" : "",
          xformatter = i == 3 ? :auto : (_ -> ""))
end

fig = plot(p3, ts...; layout = @layout([a{0.42w} grid(3, 1)]), size = (1250, 500),
           dpi = 150, left_margin = 4Plots.mm, bottom_margin = 4Plots.mm)
mkpath(dirname(OUT))
savefig(fig, OUT)
println("\nwritten: ", normpath(OUT))
