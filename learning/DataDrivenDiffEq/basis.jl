using ModelingToolkit, DataDrivenDiffEq
using OrdinaryDiffEq, DataDrivenSparse

@variables t x(t) y(t)
u = [x; y]
basis = Basis(polynomial_basis(u, 5), u, iv = t)

expr1 = x^2
expr2 = sin(x * y)
println(expr1)
println(expr2)

h = [x, y, x^2, y^2, x*y, sin(x*y)]
basis = Basis(h, u, iv = t)
basis.eqs        # your 6 expressions, unchanged
basis.unknowns   # [x(t), y(t)]
basis.iv         # t
basis.ps         # empty — you didn't give it any parameters


@parameters w1 w2
@variables t u[1:2](t)
u = [x, y]
w = [w1, w2]

Ψ = Basis([u; sin.(w .* u)], u, parameters = w, iv = t)


function toy(u, p, t)
    x, y = u
    return [x*y, -y]         # true system: dx = xy, dy = -y
end

u0 = [1.0, 1.0]
prob = ODEProblem(toy, u0, (0.0, 10.0))
sol = solve(prob, Tsit5(), saveat = 0.01)

ddprob = DataDrivenProblem(sol)
opt = STLSQ(exp10.(-5:0.1:-1))
res = solve(ddprob, basis, opt, options = DataDrivenCommonOptions(digits = 2))
println(get_basis(res))
println(get_parameter_map(get_basis(res)))

