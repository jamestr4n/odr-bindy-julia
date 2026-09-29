# =============================================================================
# Small fixed-seed test problems, shared by the test files. Each returns the
# data, the sample times and the true sparsity pattern for the library it names.
# =============================================================================

using Random, Statistics

"Classical RK4 with `nsub` substeps between consecutive samples in `t`."
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

"Add Gaussian noise with standard deviation `level` times each state's std."
function add_noise(Xtrue, level, rng)
    sx = level .* vec(std(Xtrue; dims = 1))
    return Xtrue .+ randn(rng, size(Xtrue)...) .* sx', sx
end

"Increasing sample times on [0, T], jittered by up to ±`jitter` of the mean spacing."
function uneven_times(N, T, jitter, rng)
    h = T / (N - 1)
    t = collect(range(0, T; length = N))
    t[2:(end - 1)] .+= jitter * h .* (2 .* rand(rng, N - 2) .- 1)
    return t
end

"""
Damped linear oscillator `x' = y, y' = -x - 0.25y`: 2 states and, with
`PolynomialLibrary(2, 1)` (terms `1, x, y`), 49 sparsity patterns. Small enough
for `Exhaustive`.
"""
function oscillator_data(; N = 200, T = 10.0, noise = 0.02, seed = 11)
    rng = MersenneTwister(seed)
    t = collect(range(0, T; length = N))
    Xtrue = integrate(x -> [x[2], -x[1] - 0.25x[2]], [2.0, 0.0], t)
    Xdata, sx = add_noise(Xtrue, noise, rng)
    true_mask = BitMatrix([0 0; 0 1; 1 1])            # rows 1, x, y
    return Xdata, t, sx, true_mask
end

"""
Van der Pol `x' = y, y' = mu (1 - x^2) y - x` with `mu = 1.5`, as in
test/refactor_fixture.jl. `true_mask` is for `PolynomialLibrary(2, 3)`.
"""
function vanderpol_data(; t = collect(0:0.02:7.98), noise = 0.05, seed = 42)
    rng = MersenneTwister(seed)
    mu = 1.5
    Xtrue = integrate(x -> [x[2], mu * (1 - x[1]^2) * x[2] - x[1]], [2.0, 0.0], t)
    Xdata, sx = add_noise(Xtrue, noise, rng)
    # 1 x y x^2 xy y^2 x^3 x^2y xy^2 y^3
    true_mask = falses(10, 2)
    true_mask[3, 1] = true                            # x' = y
    true_mask[[2, 3, 8], 2] .= true                   # y' = -x + mu y - mu x^2 y
    return Xdata, t, sx, true_mask
end

"""
Damped pendulum `θ' = ω, ω' = -2 sin θ - 0.2 ω`, released at `θ = 2.5` rad so
that `sin θ` is far from `θ`. `true_mask` is for
`CombinedLibrary(PolynomialLibrary(2, 1), FourierLibrary(2, 1))`, whose terms
are `1, θ, ω, sin θ, cos θ, sin ω, cos ω`.
"""
function pendulum_data(; t = collect(range(0, 15; length = 300)), noise = 0.05, seed = 7)
    rng = MersenneTwister(seed)
    Xtrue = integrate(x -> [x[2], -2sin(x[1]) - 0.2x[2]], [2.5, 0.0], t)
    Xdata, sx = add_noise(Xtrue, noise, rng)
    true_mask = falses(7, 2)
    true_mask[3, 1] = true                            # θ' = ω
    true_mask[[3, 4], 2] .= true                      # ω' = -0.2 ω - 2 sin θ
    return Xdata, t, sx, true_mask, Xtrue
end
