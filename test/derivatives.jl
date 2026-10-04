# =============================================================================
# Every derivative in the package, each against an independent numerical
# estimate (ported from examples/check_derivatives.jl):
#
#   1. dtheta          vs central differences of theta
#   2. IMat, DMat      vs a function whose derivative is known
#   3. jacobian        vs central differences of residual, column by column
#   4. reduced_hessian vs second differences of the X-profiled loss
#
# Included by test/runtests.jl, before anything that relies on these.
# =============================================================================

using ODRBINDy: residual, jacobian

"min over X of L(X, xi), warm-started from X0."
function profiled_loss(prob, mask, xi, X0)
    nx = prob.Nx * prob.D
    fr = v -> residual(prob, mask, vcat(v, xi))
    fJ = v -> jacobian(prob, mask, vcat(v, xi))[:, 1:nx]
    r = levenberg_marquardt(fr, fJ, vec(Matrix(X0));
                            maxiter = 300, ftol = 1e-14, xtol = 1e-16, gtol = 1e-12)
    return r.cost
end

@testset "derivative checks" begin

rng = MersenneTwister(1234)

@testset "library derivatives" begin
    test_library(PolynomialLibrary(3, 2), 3)
    test_library(PolynomialLibrary(2, 3), 2)    # order 3 exercises the exponents harder
end

# f(t) = sin(3t): IMat must reproduce f at the collocation points exactly, and
# DMat must match 3cos(3t) to the order of the stencil.
@testset "FD operators, order $n" for n in (2, 4, 6, 8)
    tv = collect(range(0.0, 2.0; length = 401))
    f = sin.(3 .* tv)
    IM, DM = finite_difference_matrices(length(tv), n, tv[2] - tv[1])
    tc = collocation_times(tv, n)
    @test maximum(abs, IM * f .- sin.(3 .* tc)) < 1e-13
    @test maximum(abs, DM * f .- 3 .* cos.(3 .* tc)) < 10.0^(-n / 2 - 1)
end

# Small problem so the dense numerical Jacobian is affordable. The mask is
# deliberately uneven: a uniform one would not catch an off-by-one in
# param_ranges.
Nx, D = 26, 3
lib = PolynomialLibrary(D, 2)
M = nterms(lib)
IMat, DMat = finite_difference_matrices(Nx, 6, 0.01)
Neq = size(IMat, 1)

@testset "residual Jacobian" begin
    hyper = ODRHyperParameters(sigma_x = 0.5, sigma_y = 1e-2, sigma_p = 1e1,
                               Nx = Nx, Neq = Neq, M = M, D = D)
    Xdata = randn(rng, Nx, D)
    prob = ODRProblem(Xdata, lib, IMat, DMat, hyper)
    mask = trues(M, D)
    mask[5, 1] = mask[9, 2] = mask[2, 3] = mask[10, 3] = false

    z = vcat(vec(Xdata .+ 0.05 .* randn(rng, Nx, D)), 0.3 .* randn(rng, count(mask)))
    J = Matrix(jacobian(prob, mask, z))
    Jnum = similar(J)
    hz = 1e-6
    for j in eachindex(z)
        zp = copy(z); zp[j] += hz
        zm = copy(z); zm[j] -= hz
        Jnum[:, j] = (residual(prob, mask, zp) .- residual(prob, mask, zm)) ./ (2hz)
    end
    scale = maximum(abs, Jnum)
    nx = Nx * D
    @test maximum(abs, J[:, 1:nx] .- Jnum[:, 1:nx]) / scale < 1e-6          # A, C blocks
    @test maximum(abs, J[:, (nx + 1):end] .- Jnum[:, (nx + 1):end]) / scale < 1e-6  # B, P
end

# H_red is d^2/dXi^2 of the profiled loss Lprof(Xi) = min_X L(X, Xi), estimated
# by re-minimising over X at perturbed Xi. A few percent disagreement is
# expected, since Gauss-Newton drops terms that are nonzero away from a perfect
# fit. A factor of 2, or a sign flip, is a bug.
@testset "reduced Hessian" begin
    lin(x) = [-0.9x[1] + 2.0x[2] - 0.3x[1] * x[3],
              -2.0x[1] - 0.9x[2],
               0.5 - 0.7x[3] + 0.2x[1] * x[2]]
    Xt = integrate(lin, [0.8, 0.3, 0.5], (0:(Nx - 1)) .* 0.01)
    Xn = Xt .+ 0.01 .* randn(rng, Nx, D)
    hyper = ODRHyperParameters(sigma_x = 0.01, sigma_y = 1e-3, sigma_p = 1e1,
                               Nx = Nx, Neq = Neq, M = M, D = D)
    prob = ODRProblem(Xn, lib, IMat, DMat, hyper)

    mask = falses(M, D)                    # a small, well-determined model
    mask[[2, 3], 1] .= true
    mask[[2, 3], 2] .= true
    mask[[1, 4, 6], 3] .= true

    fit = fit_model(prob, mask, ODROptions(bootstrap_samples = 50, rng = rng);
                    maxiter = 400)
    @test fit.converged
    xistar = fit.Xi[mask]
    S = Matrix(reduced_hessian(prob, jacobian(prob, mask, vcat(vec(fit.X), xistar))))

    L0 = profiled_loss(prob, mask, xistar, fit.X)
    for (a, b) in ((1, 1), (2, 2), (5, 5), (1, 2), (1, 3), (3, 4), (5, 6))
        ha = 1e-3 * max(abs(xistar[a]), 1.0)
        hb = 1e-3 * max(abs(xistar[b]), 1.0)
        if a == b
            xp = copy(xistar); xp[a] += ha
            xm = copy(xistar); xm[a] -= ha
            num = (profiled_loss(prob, mask, xp, fit.X) - 2L0 +
                   profiled_loss(prob, mask, xm, fit.X)) / ha^2
        else
            Lab = Matrix{Float64}(undef, 2, 2)
            for (ia, sa) in enumerate((1, -1)), (ib, sb) in enumerate((1, -1))
                xpert = copy(xistar)
                xpert[a] += sa * ha
                xpert[b] += sb * hb
                Lab[ia, ib] = profiled_loss(prob, mask, xpert, fit.X)
            end
            num = (Lab[1, 1] - Lab[1, 2] - Lab[2, 1] + Lab[2, 2]) / (4ha * hb)
        end
        @test abs(num - S[a, b]) / max(abs(num), abs(S[a, b]), 1.0) < 5e-2
    end
end

end # testset
