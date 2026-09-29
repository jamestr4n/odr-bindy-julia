# =============================================================================
# Regression fixture for refactors: the results must not change *at all*.
#
#     julia --project=. test/refactor_fixture.jl save    # before a refactor
#     julia --project=. test/refactor_fixture.jl check   # after it
#
# Runs odr_bindy on fixed-seed data and writes every number that matters (the
# selected mask, Xi, the -log(E) path, a checksum of the denoised states) at
# full precision to test/data/refactor_fixture.txt. `check` recomputes and
# compares the text exactly, so any change in floating-point behaviour shows.
#
# Uses only the original API (ODRProblem from matrices, ODROptions), which must
# keep working. Three runs:
#
#   A  Van der Pol, 2 states, cubic library, default options
#   B  the same data with every non-default solver/search option switched on,
#      so each option is checked to still reach the code that uses it
#   C  examples/lorenz.jl (N = 500, 20% noise, 4 starts): slow, ~6 min
# =============================================================================

using ODRBINDy
using LinearAlgebra, Random, Statistics, Printf

const FIXTURE = joinpath(@__DIR__, "data", "refactor_fixture.txt")

"Classical RK4, `nsub` substeps per sample. Returns an `N x D` matrix."
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

function report(io, name, res)
    println(io, "== ", name)
    println(io, "mask ", join(Int.(vec(res.mask)), ""))
    println(io, "nlevidence ", @sprintf("%.17g", res.nlevidence))
    println(io, "Xi ", join((@sprintf("%.17g", v) for v in vec(res.Xi)), " "))
    println(io, "sumX ", @sprintf("%.17g", sum(res.X)))
    println(io, "fit_iterations ", res.fit.iterations)
    for h in res.history
        println(io, "history ", h.nterms, " ", @sprintf("%.17g", h.nlevidence))
    end
end

function vanderpol_problem()
    mu = 1.5
    f(x) = [x[2], mu * (1 - x[1]^2) * x[2] - x[1]]
    N, dt, D = 400, 0.02, 2
    rng = MersenneTwister(42)
    Xtrue = integrate(f, [2.0, 0.0], dt, N)
    sx = 0.05 .* vec(std(Xtrue; dims = 1))
    Xdata = Xtrue .+ randn(rng, N, D) .* sx'
    IMat, DMat = finite_difference_matrices(N, 6, dt)
    lib = PolynomialLibrary(D, 3; varnames = ["x", "y"])
    hyper = ODRHyperParameters(repeat(sx', N, 1), fill(1e-2, size(IMat, 1), D),
                               fill(1e2, nterms(lib), D))
    return ODRProblem(Xdata, lib, IMat, DMat, hyper)
end

function lorenz_problem()
    SIGMA, RHO, BETA = 10.0, 28.0, 8 / 3
    f(x) = [SIGMA * (x[2] - x[1]), x[1] * (RHO - x[3]) - x[2], x[1] * x[2] - BETA * x[3]]
    N, dt, D = 500, 0.01, 3
    rng = MersenneTwister(20240717)
    Xtrue = integrate(f, [-8.0, 7.0, 27.0], dt, N)
    sx = 0.20 .* vec(std(Xtrue; dims = 1))
    Xdata = Xtrue .+ randn(rng, N, D) .* sx'
    IMat, DMat = finite_difference_matrices(N, 6, dt)
    lib = PolynomialLibrary(D, 2; varnames = ["x", "y", "z"])
    hyper = ODRHyperParameters(repeat(sx', N, 1), fill(1e-2, size(IMat, 1), D),
                               fill(1e2, nterms(lib), D))
    return ODRProblem(Xdata, lib, IMat, DMat, hyper), rng
end

function run_all(io)
    prob = vanderpol_problem()

    t = @elapsed res = odr_bindy(prob, ODROptions(verbose = 0, rng = MersenneTwister(1)))
    report(io, "A vanderpol defaults", res)
    @printf("A done in %.1f s\n", t)

    optsB = ODROptions(verbose = 0, rng = MersenneTwister(2), n_multistart = 3,
                       bootstrap_samples = 50, bragging = false,
                       lm_maxiter = 60, lm_maxiter_refine = 3000,
                       lm_damping = :levenberg, lm_accel = true,
                       ftol = 1e-9, xtol = 1e-13, gtol = 1e-11,
                       trial_xi_init = :regress, refine_after_removal = false,
                       stop_after_rises = 3)
    t = @elapsed res = odr_bindy(prob, optsB)
    report(io, "B vanderpol all options changed", res)
    @printf("B done in %.1f s\n", t)

    prob, rng = lorenz_problem()               # rng continues from the noise draw,
    t = @elapsed res = odr_bindy(prob, ODROptions(n_multistart = 4, verbose = 0, rng = rng))
    report(io, "C examples/lorenz.jl", res)    # exactly as in the example
    @printf("C done in %.1f s\n", t)
end

mode = isempty(ARGS) ? "check" : ARGS[1]
buf = IOBuffer()
run_all(buf)
out = String(take!(buf))

if mode == "save"
    mkpath(dirname(FIXTURE))
    write(FIXTURE, out)
    println("saved ", FIXTURE)
elseif mode == "check"
    expected = read(FIXTURE, String)
    if out == expected
        println("PASS: identical to ", FIXTURE)
    else
        println("FAIL: results differ from ", FIXTURE)
        for (a, b) in zip(split(expected, '\n'), split(out, '\n'))
            a == b || println("  expected: ", a, "\n  got:      ", b)
        end
        exit(1)
    end
else
    error("usage: refactor_fixture.jl save|check")
end
