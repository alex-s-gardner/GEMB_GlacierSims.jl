using GEMB_GlacierSims
using Test

@testset "GEMB_GlacierSims.jl" begin
    # Basic module loading test
    @test isdefined(GEMB_GlacierSims, :era5_land_invariant)
    @test isdefined(GEMB_GlacierSims, :gemb_glacier_elevation_class_runfile)

    # Add more tests as needed
    # Note: Full simulation tests would require CDS API credentials and network
    # access to the Copernicus DEM, so the run file builder is only checked for
    # definition/export here.
end
