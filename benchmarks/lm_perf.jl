# =============================================================================
# Per-step cost of the Levenberg-Marquardt solver on a paper-size problem.
#
#     julia --project=. benchmarks/lm_perf.jl [label]
#
# Lorenz63, N = 1000 samples (T = 10), 30-term library, sigma_y = 1e-3. Times
# a fixed number of LM steps (tolerances set to zero so it never stops early)
# and the pieces each step is made of. Appends one block per run to
# benchmarks/results/lm_perf.txt, tagged with `label`, so each optimisation can
# be compared against the saved baseline on the same machine.
#
# Standard library only: each figure is the median of `REPS` timed repetitions
# after one warm-up call.
# =============================================================================

using ODRBINDy
using ODRBINDy: cost
using LinearAlgebra, SparseArrays, Random, Statistics, Printf

const REPS = 5
const STEPS = 50
const OUT = joinpath(@__DIR__, "results", "lm_perf.txt")

lorenz(x) = [10 * (x[2] - x[1]), x[1] * (28 - x[3]) - x[2], x[1] * x[2] - 8 / 3 * x[3]]

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

"Median time (s) and allocated bytes of `f()` over `REPS` runs, after a warm-up."
function measure(f)
    f()
    runs = [@timed(f()) for _ in 1:REPS]
    return median(r.time for r in runs), median(r.bytes for r in runs)
end

function main(label)
    BLAS.set_num_threads(1)
    N, dt = 1000, 0.01
    Xc = integrate(lorenz, [-8.0, 8.0, 27.0], dt, N)
    sx = 0.2 * std(vec(Xc))
    Xd = Xc .+ sx .* randn(MersenneTwister(1), N, 3)
    IMat, DMat = finite_difference_matrices(N, 6, dt)
    lib = PolynomialLibrary(3, 2)
    hyper = ODRHyperParameters(sigma_x = sx, sigma_y = 1e-3, sigma_p = 100.0,
                               Nx = N, Neq = size(IMat, 1), M = nterms(lib), D = 3)
    prob = ODRProblem(Xd, lib, IMat, DMat, hyper)
    mask = trues(nterms(lib), 3)
    z0 = vcat(vec(Xd), bootstrap_ridge(prob, mask, Xd; rng = MersenneTwister(2)))

    fr = z -> residual(prob, mask, z)
    fJ = z -> jacobian(prob, mask, z)
    J = fJ(z0)
    H = J' * J

    rows = [
        ("residual", measure(() -> fr(z0))),
        ("jacobian", measure(() -> fJ(z0))),
        ("J'J", measure(() -> J' * J)),
        ("cholesky(J'J + lambda I)", measure(() -> cholesky(Symmetric(H + 1e-3I)))),
    ]
    configs = [(:marquardt, false), (:levenberg, false), (:levenberg, true)]
    for (damping, accel) in configs
        run = () -> levenberg_marquardt(fr, fJ, z0; maxiter = STEPS, ftol = 0.0, xtol = 0.0,
                                        gtol = 0.0, damping = damping, accel = accel)
        t, b = measure(run)
        push!(rows, ("LM step ($damping$(accel ? "+accel" : ""))", (t / STEPS, b / STEPS)))
    end

    # What actually matters: wall time for the 30-term fit to converge. Timed
    # once each, since without acceleration this takes minutes.
    fits = String[]
    for (damping, accel) in configs
        t = @elapsed(res = levenberg_marquardt(fr, fJ, z0; maxiter = 16000,
                                               damping = damping, accel = accel))
        push!(fits, @sprintf("  %-28s %8.1f s   %5d steps  converged = %s  cost = %.4f",
                             "full fit ($damping$(accel ? "+accel" : ""))", t,
                             res.iterations, res.converged, res.cost))
    end

    mkpath(dirname(OUT))
    open(OUT, "a") do io
        for dest in (io, stdout)
            @printf(dest, "\n## %s  (julia %s, %s, N = %d, %d unknowns)\n", label, VERSION,
                    Sys.CPU_NAME, N, length(z0))
            for (name, (t, b)) in rows
                @printf(dest, "  %-28s %8.2f ms  %8.2f MiB\n", name, 1e3t, b / 2^20)
            end
            foreach(l -> println(dest, l), fits)
        end
    end
end

main(isempty(ARGS) ? "unlabelled" : ARGS[1])
