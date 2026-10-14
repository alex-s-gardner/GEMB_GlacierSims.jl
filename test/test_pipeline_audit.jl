# Analytical tests: every expected value is a closed form, a hand calculation or a conservation law
# worked out independently of the code under test. Included from `runtests.jl`, whose fixtures
# (`_cp_forcing`, `_cp_table`, `_cp_loader`, `_fake_tile_run`, …) these use.

# An `AppliedDownscaling` with one band per `(lo, hi)` pair, constant `k` and lapse rate, for driving
# `elevation_interval_forcing` without a fit.
function _flat_applied(time, edges; k = 1.0, lapse_rate = 6.5)
    n = length(time)
    bands = [(; lo, hi, center = (lo + hi) / 2, decoupling_factor = fill(Float64(k), n),
              decoupling_factor_source = fill(Int8(1), n), n_fit_in_domain = n, n_fit_held = 0,
              albedo_ice = 0.48, albedo_ice_source = :default) for (lo, hi) in edges]
    return GEMB_GlacierSims.AppliedDownscaling(collect(time), fill(Float64(lapse_rate), n),
                                               fill(Int8(1), n), bands, :fitted,
                                               Dict{String,Any}(), Dict{String,Any}(),
                                               z -> fill(Float64(k), n))
end

@testset "pipeline audit: analytical" begin
    @testset "band shortwave layers" begin
        # Two cells holding 1 and 3 km² of one band. The zenith angle is the angle whose cosine is
        # the shortwave-weighted mean cosine; diffuse shortwave and optical thickness are plain area
        # means; a step with no shortwave in either cell has the sun at the horizon.
        cells = [(lon = 10.0, lat = 46.0, z = 1000.0, glm = 0.0, bins = [(1050, 1.0)]),
                 (lon = 10.1, lat = 46.0, z = 1000.0, glm = 0.0, bins = [(1050, 3.0)])]
        n = 3
        sw = ([100.0, 0.0, 50.0], [300.0, 0.0, 150.0])
        sza = ([30.0, 90.0, 50.0], [70.0, 90.0, 80.0])
        diffuse = ([10.0, 0.0, 5.0], [60.0, 0.0, 20.0])
        tau = ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        loader = function (_model, lat, lon; kwargs...)
            i = findfirst(c -> c.lat == lat && c.lon == lon, cells)
            _cp_forcing(lat, lon, cells[i].z, fill(270.0, n);
                        shortwave_downward = sw[i], solar_zenith_angle = sza[i],
                        shortwave_downward_diffuse = diffuse[i], cloud_optical_thickness = tau[i])
        end
        applied = _flat_applied(_CP_TIME[1:n], [(1000, 1100)])
        band = only(collect(elevation_interval_forcing(
            _cp_table(cells), applied; climate_model = :synthetic,
            time_range = (_CP_TIME[1], _CP_TIME[n]), forcing_loader = loader)))
        w = (1.0, 3.0)
        for t in 1:n
            total_sw = w[1] * sw[1][t] + w[2] * sw[2][t]
            expected_sza = total_sw > 0 ?
                acosd((w[1] * sw[1][t] * cosd(sza[1][t]) + w[2] * sw[2][t] * cosd(sza[2][t])) /
                      total_sw) : 90.0
            @test band.forcing[:solar_zenith_angle][t] ≈ expected_sza atol = 1e-10
            @test band.forcing[:shortwave_downward_diffuse][t] ≈
                  (diffuse[1][t] + 3 * diffuse[2][t]) / 4 atol = 1e-12
            @test band.forcing[:cloud_optical_thickness][t] ≈ (tau[1][t] + 3 * tau[2][t]) / 4 atol = 1e-12
            @test band.forcing[:shortwave_downward][t] ≈ total_sw / 4 atol = 1e-12
        end
    end
end
