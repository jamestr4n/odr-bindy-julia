# =============================================================================
# Reproduce the ODR-BINDy panel of Fig. 4 of Fung (2025), arXiv:2507.23426:
# the success rate of recovering Lorenz63 from noisy data, as a function of
# noise level and data length.
#
#     julia --project=. -t auto benchmarks/lorenz_paper.jl                  # headline cells
#     julia --project=. -t auto benchmarks/lorenz_paper.jl --T 10 --noise 0.1,0.2,0.3 --trials 64
#
# Every setting follows the paper (Table 1) and its script Lorenz_heatmap.m:
#
#   system      sigma = 10, rho = 28, beta = 8/3, x0 = [-8, 8, 27]
#   sampling    dt = 0.01, samples at t = dt, 2dt, ..., T  (N = T/dt), x0 first
#   noise       sigma_x = noise * std(vec(X_clean)), one scalar for all states,
#               where X_clean is the T = 10 trajectory (the "signal power")
#   model       2nd-order polynomial library, 6th-order finite differences
#   hyper       sigma_y = 1e-3, sigma_p = 100
#   success     selected support == true support, exactly
#   solver      lm_maxiter_refine = 16000, n_multistart = 4. MATLAB's initial fit
#               makes at least 4 attempts (MinInitialTrial) and doubles the
#               iteration cap from 1000 after each one, and later refits inherit
#               the doubled cap (1000 * 2^4 = 16000). At sigma_y = 1e-3 the full
#               library fit needs a few thousand LM steps, so the package default
#               of 1000 fails every run.
#               lm_damping = :levenberg. With Marquardt (diag(J'J)) damping,
#               almost every greedy trial hits its 100-step cap at
#               sigma_y = 1e-3, so no term can ever be removed. See the
#               docstring of `levenberg_marquardt`.
#               lm_accel = true: geodesic acceleration. Without it the
#               Gauss-Newton steps creep (gain ratio ~0.6), trials need 150-450
#               LM steps instead of the ~50 they need with it, and the 100-step
#               trial cap (MATLAB's) makes the search stall with spurious terms.
#               Everything else is the package default. In particular the trial
#               starting point stays `trial_xi_init = :previous`; MATLAB's
#               `:regress` is available via --trial_init regress.
#
# One row per trial is appended to benchmarks/results/lorenz_paper_trials.csv.
# Trials already in that file are skipped, so an interrupted run resumes where
# it stopped. Summarise with benchmarks/summarise_lorenz_paper.jl.
# =============================================================================

using ODRBINDy
using LinearAlgebra, Random, Statistics, Printf

const SIGMA, RHO, BETA = 10.0, 28.0, 8 / 3
const X0 = [-8.0, 8.0, 27.0]
const DT = 0.01
const FD_ORDER = 6

const RESULTS = joinpath(@__DIR__, "results", "lorenz_paper_trials.csv")
const HEADER = "T,N,noise,trial,seed,sigma_y,success,n_wrong,model_error,nterms," *
               "nlevidence,rms_data,rms_denoised,runtime_s"

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

# --- one trial ----------------------------------------------------------------

function run_trial(Xlong, signal, T, noise, trial, sigma_y, opts_kw)
    N = round(Int, T / DT)
    Xclean = Xlong[1:N, :]
    seed = trial                             # same noise draw for trial k in every cell
    rng = MersenneTwister(seed)
    sx = noise * signal
    Xdata = Xclean .+ sx .* randn(rng, N, 3)

    IMat, DMat = finite_difference_matrices(N, FD_ORDER, DT)
    lib = PolynomialLibrary(3, 2; varnames = ["x", "y", "z"])
    hyper = ODRHyperParameters(sigma_x = sx, sigma_y = sigma_y, sigma_p = 100.0,
                               Nx = N, Neq = size(IMat, 1), M = nterms(lib), D = 3)
    prob = ODRProblem(Xdata, lib, IMat, DMat, hyper)
    opts = ODROptions(; verbose = 0, rng = rng, opts_kw...)

    Xi_true = truth()
    t0 = time()
    res = odr_bindy(prob, opts)
    runtime = time() - t0

    return (T = T, N = N, noise = noise, trial = trial, seed = seed, sigma_y = sigma_y,
            success = res.mask == (Xi_true .!= 0),
            n_wrong = count(res.mask .!= (Xi_true .!= 0)),
            model_error = norm(res.Xi .- Xi_true) / norm(Xi_true),
            nterms = count(res.mask),
            nlevidence = res.nlevidence,
            rms_data = sqrt(mean(abs2, Xdata .- Xclean)),
            rms_denoised = sqrt(mean(abs2, res.X .- Xclean)),
            runtime_s = runtime)
end

csvrow(r) = join((r.T, r.N, r.noise, r.trial, r.seed, r.sigma_y, Int(r.success), r.n_wrong,
                  @sprintf("%.6g", r.model_error), r.nterms, @sprintf("%.8g", r.nlevidence),
                  @sprintf("%.6g", r.rms_data), @sprintf("%.6g", r.rms_denoised),
                  @sprintf("%.2f", r.runtime_s)), ",")

"(T, noise, trial, sigma_y) keys already in the results file."
function done_keys()
    keys = Set{Tuple{Int,Float64,Int,Float64}}()
    isfile(RESULTS) || return keys
    for line in Iterators.drop(eachline(RESULTS), 1)
        f = split(line, ',')
        push!(keys, (parse(Int, f[1]), parse(Float64, f[3]), parse(Int, f[4]), parse(Float64, f[6])))
    end
    return keys
end

# --- command line -------------------------------------------------------------

function parse_args(args)
    a = Dict("T" => "10", "noise" => "0.1,0.2,0.3", "trials" => "64", "sigma_y" => "1e-3",
             "multistart" => "4", "maxiter_refine" => "16000",
             "damping" => "levenberg", "trial_init" => "previous",
             "maxiter" => "100", "seeds" => "", "accel" => "true")
    i = 1
    while i <= length(args)
        startswith(args[i], "--") || error("unexpected argument $(args[i])")
        a[args[i][3:end]] = args[i + 1]
        i += 2
    end
    nums(s) = parse.(Float64, split(s, ','))
    return (T = Int.(nums(a["T"])), noise = nums(a["noise"]), trials = parse(Int, a["trials"]),
            sigma_y = parse(Float64, a["sigma_y"]), multistart = parse(Int, a["multistart"]),
            maxiter_refine = parse(Int, a["maxiter_refine"]), damping = Symbol(a["damping"]),
            trial_init = Symbol(a["trial_init"]), maxiter = parse(Int, a["maxiter"]),
            seeds = isempty(a["seeds"]) ? nothing : parse.(Int, split(a["seeds"], ',')),
            accel = parse(Bool, a["accel"]))
end

function main(args)
    cfg = parse_args(args)
    BLAS.set_num_threads(1)                  # parallelism is over trials instead

    Xlong = integrate(lorenz, X0, DT, round(Int, 10 / DT))
    signal = std(vec(Xlong))                 # paper: std of the flattened T = 10 data

    mkpath(dirname(RESULTS))
    isfile(RESULTS) || write(RESULTS, HEADER * "\n")
    done = done_keys()
    # trial-major order, so a run stopped early leaves every cell equally filled
    ks = cfg.seeds === nothing ? (1:cfg.trials) : cfg.seeds
    todo = [(T, e, k) for k in ks for T in cfg.T for e in cfg.noise
            if (T, e, k, cfg.sigma_y) ∉ done]

    @printf("Lorenz63, paper settings: signal std = %.3f, sigma_y = %g, %d threads\n",
            signal, cfg.sigma_y, Threads.nthreads())
    @printf("%d trials to run (%d already done)\n", length(todo),
            length(cfg.T) * length(cfg.noise) * length(ks) - length(todo))

    lk = ReentrantLock()
    ndone = Threads.Atomic{Int}(0)
    t0 = time()
    Threads.@threads :dynamic for (T, e, k) in todo
        r = run_trial(Xlong, signal, T, e, k, cfg.sigma_y, (n_multistart = cfg.multistart,
                                                                  lm_maxiter_refine = cfg.maxiter_refine,
                                                                  lm_damping = cfg.damping,
                                                                  trial_xi_init = cfg.trial_init,
                                                                  lm_maxiter = cfg.maxiter,
                                                                  lm_accel = cfg.accel))
        lock(lk) do
            open(io -> println(io, csvrow(r)), RESULTS, "a")
            n = Threads.atomic_add!(ndone, 1) + 1
            @printf("[%4d/%d  %6.0fs]  T=%-2d noise=%.3f trial=%-3d  %s  (%d terms, %.0fs)\n",
                    n, length(todo), time() - t0, T, e, k, r.success ? "ok  " : "FAIL",
                    r.nterms, r.runtime_s)
            flush(stdout)
        end
    end
end

main(ARGS)
