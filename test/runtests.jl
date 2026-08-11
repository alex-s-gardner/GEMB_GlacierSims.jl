using GEMB_GlacierSims
using Test

@testset "GEMB_GlacierSims.jl" begin
    # Basic module loading test
    @test isdefined(GEMB_GlacierSims, :era5_land_invariant)

    # Add more tests as needed
    # Note: Full simulation tests would require CDS API credentials
end
