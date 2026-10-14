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

@testset "pipeline audit: per-cell decoupling" begin
    # One cell at 1000 m with glacier fraction 0.6, forced 1500 m up with k = 0.7 and Γ = 5 K/km.
    # Its observed excess is `[1 - glm(1 - k)]·E_ambient`, so the on-glacier temperature at the bin
    # is `T_ref + k·(E_ambient(1000) - Γ·1.5)` with `E_ambient = E_obs / (1 - glm(1 - k))`.
    k, glm, Γ = 0.7, 0.6, 5.0
    T_obs = [280.0, 275.0, 270.0]
    f0 = _cp_forcing(46.0, 10.0, 1000.0, T_obs)
    f = DimensionalData.rebuild(f0; metadata = merge(DimensionalData.metadata(f0), Dict{String,Any}(
        "temperature_air_mean" => 275.0, "wind_speed_mean" => 3.0, "precipitation_mean" => 876.0)))
    cf = forcing_at_elevation(f, 1500.0; lapse_rate = Γ, decoupling_factor = k,
                              glacier_fraction = glm)
    E_amb = @. max(T_obs - 273.15, 0) / (1 - glm * (1 - k)) + min(T_obs - 273.15, 0)
    E_bin = E_amb .- Γ * 1.5
    expected = @. 273.15 + ifelse(E_bin > 0, k * E_bin, E_bin)
    @test cf.temperature_air ≈ expected atol = 1e-10
    # No glacier in the cell: the forcing is already ambient, and only the band's k acts.
    cf0 = forcing_at_elevation(f, 1500.0; lapse_rate = Γ, decoupling_factor = k)
    E0 = (T_obs .- 273.15) .- Γ * 1.5
    @test cf0.temperature_air ≈ @.(273.15 + ifelse(E0 > 0, k * E0, E0)) atol = 1e-10
end
