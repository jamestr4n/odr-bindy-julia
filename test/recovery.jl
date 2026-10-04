# =============================================================================
# Recovery of known models, end to end through odr_bindy.
#
# Included by test/runtests.jl, after the derivative checks.
# =============================================================================

@testset "recovery" begin

# With no noise the selected model must be exactly right, the coefficients must
# match to the accuracy of the discretisation (about 1e-8 here), and the
# denoised states must be the data. sigma_x cannot be zero (the residual
# divides by it); a small value says the data are trusted.
@testset "noise-free: $name" for (name, data, lib, Xi_true) in (
        ("oscillator", oscillator_data(noise = 0), PolynomialLibrary(2, 1),
         [0 0; 0 -1; 1 -0.25]),
        ("Van der Pol", vanderpol_data(noise = 0), PolynomialLibrary(2, 3),
         [0 0; 0 -1; 1 1.5; 0 0; 0 0; 0 0; 0 0; 0 -1.5; 0 0; 0 0]),
        ("coupled 3-state", coupled_data(noise = 0), PolynomialLibrary(3, 2),
         [0 0 0.5; -0.9 -2 0; 2 -0.9 0; 0 0 -0.7; 0 0 0;
          0 0 0.2; -0.3 0 0; 0 0 0; 0 0 0; 0 0 0]))
    X, t, _, true_mask = data
    sx = 1e-3 .* vec(std(X; dims = 1))
    prob = ODRProblem(X, t, lib; sigma_x = sx, sigma_y = 1e-2, sigma_p = 1e2)
    res = odr_bindy(prob, quiet())
    @test res.mask == true_mask
    @test res.Xi ≈ Xi_true atol = 1e-6
    @test res.fit.X ≈ X atol = 1e-6
end

end # testset
