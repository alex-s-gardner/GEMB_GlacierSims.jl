# GEMB at a firn-core site: the snowfall scaling, the run on synthetic forcing, and the profile read
# onto core depths. Included from `runtests.jl`; uses `_tiny_band_forcing` from `test_pipeline_audit.jl`.

# A cold snowy site: 10 years of 3-hourly forcing, 0.1 kg m-2 per step at 258 K (all snow), with a
# reference elevation 200 m below the site.
_firn_site_forcing() = _tiny_band_forcing(; years = 10, temperature = 258.15, precipitation = 0.1,
                                          shortwave = 120.0, longwave = 200.0, elevation = 1800.0)

@testset "firn site run" begin
    @testset "FirnSite validates its fields" begin
        kw = (; record_id = "s", latitude = 60.0, longitude = -150.0, elevation = 2000.0,
              date = DateTime(2008, 5, 1), accumulation = 500.0, depth = 12.0)
        @test FirnSite(; kw...).depth == 12.0
        @test_throws "needs a positive measured accumulation" FirnSite(; kw..., accumulation = 0.0)
        @test_throws "has no elevation" FirnSite(; kw..., elevation = NaN)
        @test_throws "core depth must be positive" FirnSite(; kw..., depth = 0.0)
        @test_throws "latitude must be in [-90, 90]" FirnSite(; kw..., latitude = 91.0)
    end

    @testset "the snowfall scaling matches the accumulation" begin
        forcing = _firn_site_forcing()
        mp = initialize_parameters()
        # 0.1 kg m-2 every 3 h, all below the rain threshold, is 292.2 kg m-2 per 365.25-day year.
        snow = 0.1 * 8 * 365.25
        @test site_snowfall_scaling(forcing, 2 * snow, mp) ≈ 2.0 rtol = 1e-3
        # Rain does not count: above the threshold the forcing has no snowfall to scale.
        warm = _tiny_band_forcing(; years = 2, temperature = 280.0, precipitation = 0.1)
        @test_throws "has no snowfall" site_snowfall_scaling(warm, 500.0, mp)
    end

    @testset "a site run reaches the core and densifies with depth" begin
        forcing = _firn_site_forcing()
        mp = initialize_parameters(output_frequency = :monthly)
        site = FirnSite(; record_id = "synthetic", latitude = 60.0, longitude = -150.0,
                        elevation = 2000.0, date = DateTime(2009, 6, 15), accumulation = 400.0,
                        depth = 8.0)
        r = firn_site_run(site, forcing, mp; spinup_window = (DateTime(2000), DateTime(2003) - Hour(3)),
                          simulation_years_maximum = 30)
        @test r.time == DateTime(2009, 5, 31, 21)
        @test r.snowfall ≈ 400.0 rtol = 1e-10
        @test sum(r.profile.dz) >= site.depth + GEMB_GlacierSims.FIRN_SITE_DEPTH_MARGIN
        @test r.spinup.cycles >= 1
        # Lapsed 200 m upward at 6.5 K km-1: the column is colder than the 258.15 K forcing.
        @test maximum(r.profile.temperature) < 258.15
        ρ = firn_profile_on_grid(r.profile, [0.5, 4.0, 8.0])
        @test ρ[1] < ρ[2] < ρ[3] <= mp.density_ice
        @test isnan(only(firn_profile_on_grid(r.profile, [sum(r.profile.dz) + 1.0])))
        @test_throws "depths must be non-negative" firn_profile_on_grid(r.profile, [-1.0])
        @test_throws "outside the forcing" firn_site_run(
            FirnSite(; record_id = "late", latitude = 60.0, longitude = -150.0, elevation = 2000.0,
                     date = DateTime(2020, 1, 1), accumulation = 400.0, depth = 8.0), forcing, mp)
    end
end
