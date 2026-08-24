using DataDrivenDiffEq, ModelingToolkit, OrdinaryDiffEq
using DataDrivenSparse, LinearAlgebra, StableRNGs, Plots

rng = StableRNG(1000)

function f(u, p, t)
    x, y = u
    dx = 2.0 * x * y
    dy = 1
    return [dx, dy]
end

u0 = [1.0; 0]
tspan = (0.0, 2.0)
dt = 0.0001
prob = ODEProblem(f, u0, tspan)
sol = solve(prob, Tsit5(), saveat = dt)

# add noise
X = sol[:, :]
ts = sol.t

ddprob = ContinuousDataDrivenProblem(X, ts, GaussianKernel())

@variables u[1:2]
u = collect(u)
h = Num[1; polynomial_basis(u, 2); u]
basis = Basis(h, u)

opt = STLSQ(exp10.(-5:0.1:0))
res = solve(ddprob, basis, opt, options = DataDrivenCommonOptions(digits = 2))

println(get_basis(res))
println(get_parameter_map(get_basis(res)))

display(plot(plot(ddprob), plot(res), layout = (1, 2)))