# =============================================================================
# Bottom-up verification of every derivative in the package.
#
#     julia --project=. examples/check_derivatives.jl
#
# Run this first, and after any change to libraries.jl, differentiation.jl,
# residual.jl or evidence.jl. Four checks, each against an independent
# numerical estimate:
#
#   1. dTheta          vs central differences of Theta
#   2. IMat, DMat      vs a function whose derivative is known
#   3. jacobian        vs central differences of residual, column by column
#   4. reduced_hessian vs second differences of the X-profiled loss
#
# These are intended to become test/runtests.jl by wrapping each `@printf` in
# an `@test`.
# =============================================================================

using ODRBINDy
using ODRBINDy: cost, unpack, active_indices
using LinearAlgebra, SparseArrays, Random, Printf, Statistics

rng = MersenneTwister(1234)
pass = true
report(name, err, tol) = (global pass &= err < tol;
                          @printf("  %-42s %9.2e  %s\n", name, err,
                                  err < tol ? "ok" : "FAIL (tol $tol)"))

# --- 1. library derivatives ---------------------------------------------------
# Central differences are O(h^2) accurate with a floating-point floor around
# sqrt(eps) ~ 1e-8, so expect ~1e-9 here.

println("1. library derivatives vs central differences")

lib1 = PolynomialLibrary(3, 2; varnames = ["x", "y", "z"])
X1 = randn(rng, 9, 3)
dTh = dtheta(lib1, X1)
h = 1e-5
for e in 1:3
    Xp = copy(X1); Xp[:, e] .+= h
    Xm = copy(X1); Xm[:, e] .-= h
    num = (theta(lib1, Xp) .- theta(lib1, Xm)) ./ (2h)
    report("d(Theta)/dx$e", maximum(abs, num .- dTh[:, :, e]), 1e-7)
end

# A third-order library in 2D exercises the exponent bookkeeping harder.
lib2 = PolynomialLibrary(2, 3)
X2 = randn(rng, 6, 2)
dTh2 = dtheta(lib2, X2)
for e in 1:2
    Xp = copy(X2); Xp[:, e] .+= h
    Xm = copy(X2); Xm[:, e] .-= h
    num = (theta(lib2, Xp) .- theta(lib2, Xm)) ./ (2h)
    report("2D order-3: d(Theta)/dx$e", maximum(abs, num .- dTh2[:, :, e]), 1e-7)
end

# --- 2. finite-difference operators ------------------------------------------
# f(t) = sin(3t): IMat must reproduce f at the collocation points exactly, DMat
# must match 3cos(3t) to the order of the stencil.

println("\n2. FD operators on sin(3t), dt = 0.005")

tv = collect(range(0.0, 2.0; length = 401))
dt = tv[2] - tv[1]
f = sin.(3 .* tv)
for n in (2, 4, 6, 8)
    IM, DM = finite_difference_matrices(length(tv), n, dt)
    tc = collocation_times(tv, n)
    report("order $n: IMat*f  vs  f(tc)", maximum(abs, IM * f .- sin.(3 .* tc)), 1e-13)
    report("order $n: DMat*f  vs  3cos(3tc)",
           maximum(abs, DM * f .- 3 .* cos.(3 .* tc)), 10.0^(-n / 2 - 1))
end

# --- 3. residual Jacobian -----------------------------------------------------
# Small problem so the dense numerical Jacobian is affordable. The mask is
# deliberately uneven: a uniform one would not catch an off-by-one in
# param_ranges, which is the easiest bug here to introduce and the hardest to
# see.

println("\n3. residual Jacobian vs central differences")

Nx, D, n = 26, 3, 6
lib = PolynomialLibrary(D, 2)
M = nterms(lib)
IMat, DMat = finite_difference_matrices(Nx, n, 0.01)
Neq = size(IMat, 1)

hyper = ODRHyperParameters(sigma_x = 0.5, sigma_y = 1e-2, sigma_p = 1e1,
                           Nx = Nx, Neq = Neq, M = M, D = D)
Xdata = randn(rng, Nx, D)
prob = ODRProblem(Xdata, lib, IMat, DMat, hyper)

mask = trues(M, D)
mask[5, 1] = false
mask[9, 2] = false
mask[2, 3] = false
mask[10, 3] = false

Np = count(mask)
z = vcat(vec(Xdata .+ 0.05 .* randn(rng, Nx, D)), 0.3 .* randn(rng, Np))

J = Matrix(jacobian(prob, mask, z))
Jnum = similar(J)
hz = 1e-6
for j in eachindex(z)
    zp = copy(z); zp[j] += hz
    zm = copy(z); zm[j] -= hz
    Jnum[:, j] = (residual(prob, mask, zp) .- residual(prob, mask, zm)) ./ (2hz)
end
report("max |J - Jnum| / max|Jnum|",
       maximum(abs, J .- Jnum) / maximum(abs, Jnum), 1e-6)
report("state columns   (A, C blocks)",
       maximum(abs, J[:, 1:(Nx * D)] .- Jnum[:, 1:(Nx * D)]) / maximum(abs, Jnum), 1e-6)
report("param columns   (B, P blocks)",
       maximum(abs, J[:, (Nx * D + 1):end] .- Jnum[:, (Nx * D + 1):end]) /
           maximum(abs, Jnum), 1e-6)

# --- 4. reduced Hessian -------------------------------------------------------
# H_red is d^2/dXi^2 of the *profiled* loss  Lprof(Xi) = min_X L(X, Xi).
# Estimate that numerically: re-minimise over X at perturbed Xi and take second
# differences. A few percent disagreement is expected -- Gauss-Newton genuinely
# drops terms that are nonzero away from a perfect fit. A factor of 2, or a sign
# flip, is a bug.

println("\n4. reduced Hessian vs the numerically profiled loss")

# Use data from an actual trajectory so the fit is meaningful.
lin(x) = [-0.9x[1] + 2.0x[2] - 0.3x[1] * x[3],
          -2.0x[1] - 0.9x[2],
           0.5 - 0.7x[3] + 0.2x[1] * x[2]]
function trajectory(f, x0, dt, N; nsub = 10)
    h = dt / nsub
    X = Matrix{Float64}(undef, N, length(x0))
    x = copy(x0)
    for i in 1:N
        X[i, :] = x
        for _ in 1:nsub
            k1 = f(x); k2 = f(x .+ h / 2 .* k1)
            k3 = f(x .+ h / 2 .* k2); k4 = f(x .+ h .* k3)
            x = x .+ (h / 6) .* (k1 .+ 2k2 .+ 2k3 .+ k4)
        end
    end
    return X
end

Xt = trajectory(lin, [0.8, 0.3, 0.5], 0.01, Nx)
Xn = Xt .+ 0.01 .* randn(rng, Nx, D)

hyper4 = ODRHyperParameters(sigma_x = 0.01, sigma_y = 1e-3, sigma_p = 1e1,
                            Nx = Nx, Neq = Neq, M = M, D = D)
prob4 = ODRProblem(Xn, lib, IMat, DMat, hyper4)

mask4 = falses(M, D)                       # a small, well-determined model
mask4[[2, 3], 1] .= true
mask4[[2, 3], 2] .= true
mask4[[1, 4, 6], 3] .= true

fit = fit_model(prob4, mask4, ODROptions(bootstrap_samples = 50, rng = rng);
                maxiter = 400)
xistar = fit.Xi[mask4]
@printf("  fit: converged = %s, L = %.6g\n", fit.converged, fit.cost)

nx = Nx * D
S = Matrix(reduced_hessian(prob4, jacobian(prob4, mask4, vcat(vec(fit.X), xistar))))

"min over X of L(X, xi), warm-started from X0."
function profiled_loss(prob, mask, xi, X0)
    nxl = prob.Nx * prob.D
    fr = v -> residual(prob, mask, vcat(v, xi))
    fJ = v -> jacobian(prob, mask, vcat(v, xi))[:, 1:nxl]
    r = levenberg_marquardt(fr, fJ, vec(Matrix(X0));
                            maxiter = 300, ftol = 1e-14, xtol = 1e-16, gtol = 1e-12)
    return r.cost
end

worst = 0.0
for (a, b) in ((1, 1), (2, 2), (5, 5), (1, 2), (1, 3), (3, 4), (5, 6))
    ha = 1e-3 * max(abs(xistar[a]), 1.0)
    hb = 1e-3 * max(abs(xistar[b]), 1.0)
    if a == b
        xp = copy(xistar); xp[a] += ha
        xm = copy(xistar); xm[a] -= ha
        num = (profiled_loss(prob4, mask4, xp, fit.X)
               - 2 * profiled_loss(prob4, mask4, xistar, fit.X)
               + profiled_loss(prob4, mask4, xm, fit.X)) / ha^2
    else
        L = Matrix{Float64}(undef, 2, 2)
        for (ia, sa) in enumerate((1, -1)), (ib, sb) in enumerate((1, -1))
            xpert = copy(xistar)
            xpert[a] += sa * ha
            xpert[b] += sb * hb
            L[ia, ib] = profiled_loss(prob4, mask4, xpert, fit.X)
        end
        num = (L[1, 1] - L[1, 2] - L[2, 1] + L[2, 2]) / (4 * ha * hb)
    end
    rel = abs(num - S[a, b]) / max(abs(num), abs(S[a, b]), 1.0)
    global worst = max(worst, rel)
    @printf("  H[%d,%d]  analytic %12.5g   numerical %12.5g   rel %7.2e\n",
            a, b, S[a, b], num, rel)
end
report("worst relative disagreement", worst, 5e-2)

println()
println(pass ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED")
exit(pass ? 0 : 1)
