# GEMB at a firn-core site: the snowfall scaling, the run on synthetic forcing, and the profile read
# onto core depths. Included from `runtests.jl`; uses `_tiny_band_forcing` and `_flat_applied` from
# `test_pipeline_audit.jl`, and `_cp_forcing`, `_cp_table` and `_CP_TIME` from `runtests.jl`.

# A cold snowy site: 10 years of 3-hourly forcing, 0.1 kg m-2 per step at 258 K (all snow), with a
# reference elevation 200 m below the site.
_firn_site_forcing() = _tiny_band_forcing(; years = 10, temperature = 258.15, precipitation = 0.1,
                                          shortwave = 120.0, longwave = 200.0, elevation = 1800.0)

@testset "firn site run" begin
    @testset "FirnSite validates its fields" begin
        kw = (; record_id = "s", latitude = 60.0, longitude = -150.0, elevation = 2000.0,
              date = DateTime(2008, 5, 1), accumulation = 500.0, depth = 12.0)
        @test FirnSite(; kw...).depth == 12.0
        @test_throws "a measured accumulation must be positive" FirnSite(; kw..., accumulation = 0.0)
        @test isnan(FirnSite(; Base.structdiff(kw, (; accumulation = 0))...).accumulation)
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

    @testset "site forcing is the band forcing of a one-cell tile" begin
        # A glaciated cell (glm 0.6) at 1000 m, warm enough that the decoupling and the glacier
        # correction act, taken to a 1550 m band by `elevation_interval_forcing` and to a 1550 m site by
        # `site_forcing`, under one applied downscaling: the two must agree exactly.
        n = 6
        temperature = [271.0, 274.0, 276.0, 279.0, 268.0, 281.0]
        cell = (lon = 10.0, lat = 46.0, z = 1000.0, glm = 0.6, bins = [(1550, 2.0)])
        forcing = _cp_forcing(cell.lat, cell.lon, cell.z, temperature)
        applied = _flat_applied(_CP_TIME[1:n], [(1500, 1600)]; k = 0.7, lapse_rate = 5.5)
        # The flat resolution carries no donor correction; give the cell a `k` series of its own.
        applied = GEMB_GlacierSims.AppliedDownscaling(applied.time, applied.lapse_rate,
            applied.lapse_rate_source, applied.bands, applied.basis, applied.settings,
            applied.provenance, z -> fill(0.8, n))
        band = only(collect(elevation_interval_forcing(
            _cp_table([cell]), applied; climate_model = :synthetic,
            time_range = (_CP_TIME[1], _CP_TIME[n]), forcing_loader = (_m, lat, lon; kw...) -> forcing)))
        site = site_forcing(forcing, applied, cell.glm)
        for v in (:temperature_air, :vapor_pressure, :longwave_downward, :pressure_air, :precipitation)
            @test collect(site[v]) == collect(band.forcing[v])
        end
        # The lapse and the decoupling both acted: the band is colder than the cell at every step.
        @test all(collect(site[:temperature_air]) .< temperature)
        @test_throws "applies no wind correction" site_forcing(forcing,
            GEMB_GlacierSims.AppliedDownscaling(applied.time, applied.lapse_rate,
                applied.lapse_rate_source, applied.bands, applied.basis,
                Dict{String,Any}("downscaling_wind_scaling" => 2.0), applied.provenance,
                applied.donor_decoupling_factor_at), cell.glm)
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

    @testset "a site run without accumulation matching keeps the forcing's snowfall" begin
        forcing = _firn_site_forcing()
        mp = initialize_parameters(output_frequency = :monthly)
        site = FirnSite(; record_id = "unmatched", latitude = 60.0, longitude = -150.0,
                        elevation = 2000.0, date = DateTime(2009, 6, 15), depth = 8.0)
        window = (DateTime(2000), DateTime(2003) - Hour(3))
        @test_throws "has no measured accumulation to match" firn_site_run(site, forcing, mp;
                                                                           spinup_window = window)
        r = firn_site_run(site, forcing, mp; match_accumulation = false, spinup_window = window,
                          simulation_years_maximum = 30)
        @test r.precipitation_scaling == 1
        # All of the forcing's 0.1 kg m-2 per 3 h falls as snow: 292.2 kg m-2 per 365.25-day year.
        @test r.snowfall ≈ 0.1 * 8 * 365.25 rtol = 1e-3
        @test sum(r.profile.dz) >= site.depth + GEMB_GlacierSims.FIRN_SITE_DEPTH_MARGIN
        # A measured accumulation is ignored without matching: the run is the same.
        r2 = firn_site_run(FirnSite(; record_id = "unmatched", latitude = 60.0, longitude = -150.0,
                                    elevation = 2000.0, date = DateTime(2009, 6, 15),
                                    accumulation = 400.0, depth = 8.0),
                           forcing, mp; match_accumulation = false, spinup_window = window,
                           simulation_years_maximum = 30)
        @test r2.profile.density == r.profile.density
    end
end
