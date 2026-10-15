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
                                               z -> ones(n))
end

@testset "pipeline audit: analytical" begin
    @testset "band shortwave layers" begin
        # Two cells holding 1 and 3 km² of one band. The zenith angle is the angle whose cosine is
        # the shortwave-weighted mean cosine; diffuse shortwave is a plain area mean; the optical
        # thickness is the shortwave-weighted mean τ, which is the τ of the band's mean cloud
        # transmissivity. A step with no shortwave in either cell has the sun at the horizon and
        # τ = 0.
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
            expected_tau = total_sw > 0 ?
                (w[1] * sw[1][t] * tau[1][t] + w[2] * sw[2][t] * tau[2][t]) / total_sw : 0.0
            @test band.forcing[:solar_zenith_angle][t] ≈ expected_sza atol = 1e-10
            @test band.forcing[:shortwave_downward_diffuse][t] ≈
                  (diffuse[1][t] + 3 * diffuse[2][t]) / 4 atol = 1e-12
            @test band.forcing[:cloud_optical_thickness][t] ≈ expected_tau atol = 1e-12
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

@testset "pipeline audit: a donor where k(z) is near zero" begin
    # One fully glaciated cell (glm = 1) at 1800 m holding 5 km² of ice in the 1000-1100 m band, and
    # a fit, per timestep, with ambient excess α + βz of 8 K at the reference 1000 m and 4 K at
    # 1800 m, and glacier term γ + δz of -4 K at 1000 m (k = 0.5) and (k_d - 1)·4 K at 1800 m, so
    # k(1800) = k_d. The cell observes 2 K above melting. Its ambient excess is 2/k_d (glm = 1);
    # lapsed 750 m down at 6.5 K/km and decoupled with the band's k(1050) = 1 + g(1050)/E(1050), the
    # band temperature follows in closed form.
    n = length(_CP_TIME)
    ti = Ti(_CP_TIME)
    col(v) = DimArray(fill(Float64(v), n), ti)
    z̄, z_d, z_b = 1000.0, 1800.0, 1050.0
    β = (4.0 - 8.0) / (z_d - z̄)
    α = 8.0 - β * z̄
    function band_for(k_d)
        g_ref, g_d = -4.0, (k_d - 1) * 4.0
        δ = (g_d - g_ref) / (z_d - z̄)
        γ = g_ref - δ * z̄
        fit = (; time = _CP_TIME,
               decoupling = DimStack((decoupling_factor = col(0.5), coef_alpha = col(α),
                                      coef_beta = col(β), coef_gamma = col(γ),
                                      coef_delta = col(δ))),
               lapse_rate = DimStack((lapse_rate = col(6.5), n_cells = DimArray(fill(10, n), ti),
                                      elevation_spread = col(500.0))))
        intervals = [(; lo = 1000, hi = 1100, center = z_b)]
        applied = resolve_downscaling(fit, intervals, _CP_TIME; basis = :fitted)
        cells = [(lon = 10.0, lat = 46.0, z = z_d, glm = 1.0, bins = [(1050, 5.0)])]
        bands = collect(elevation_interval_forcing(
            _cp_table(cells), applied; climate_model = :synthetic,
            time_range = (_CP_TIME[1], _CP_TIME[end]),
            forcing_loader = _cp_loader(cells, c -> fill(273.15 + 2.0, n))))
        k_band = 1 + (γ + δ * z_b) / (α + β * z_b)
        return (; applied, bands, k_band)
    end
    lapse_down = 6.5 * (z_d - z_b) / 1000

    # k(z_d) = 1e-7: dividing by it would multiply the 2 K by 1e7. It lies below the donor limit,
    # so no timestep accepts it, there is no climatology either, and the donor is left as observed
    # (k = 1, the identity).
    near_zero = band_for(1e-7)
    @test near_zero.applied.donor_decoupling_factor_at(z_d) == ones(n)
    band = only(near_zero.bands)
    @test band.area ≈ 5.0
    @test all(t -> isapprox(t, 273.15 + near_zero.k_band * (2.0 + lapse_down); atol = 1e-9),
              band.forcing.temperature_air)

    # k(z_d) at the donor limit itself is accepted, and the conversion is the largest it can be:
    # 2/0.25 = 8 K, finite and bounded by the observed excess over the limit.
    k_min = GEMB_GlacierSims._DONOR_DECOUPLING_FACTOR_LIMITS[1]
    at_limit = band_for(k_min)
    @test all(k -> isapprox(k, k_min; atol = 1e-12), at_limit.applied.donor_decoupling_factor_at(z_d))
    band = only(at_limit.bands)
    @test all(t -> isapprox(t, 273.15 + at_limit.k_band * (2.0 / k_min + lapse_down); atol = 1e-9),
              band.forcing.temperature_air)
end

@testset "pipeline audit: band optical thickness" begin
    # A band's τ is that of its mean cloud transmissivity: with t(τ) = 1/(1 + 0.13τ) per cell, area
    # w and shortwave SW, the clear-sky shortwave is SW/t, t̄ = Σ w·SW / Σ w·SW/t and τ = (1/t̄ − 1)/0.13
    # (Lacis & Hansen 1974), capped at the ClimateForcing cap. Expected values below are those
    # closed forms; τ(0.3) = (1/0.3 − 1)/0.13 is a cell whose transmissivity is 0.3.
    cap = GEMB_ClimateForcing.CLOUD_OPTICAL_THICKNESS_MAX
    t_cap = 1 / (1 + 0.13 * cap)
    τ03 = (1 / 0.3 - 1) / 0.13
    S = 400.0                                   # clear-sky shortwave, W m-2
    # Five steps, each a case, for the equal-area pair (A, B):
    #   1. t = 1 and 0.3 under the same clear sky       → t̄ = 0.65
    #   2. no shortwave in either cell, τ 0 and the cap  → 0
    #   3. a clear cell and a capped cell with shortwave → t̄ = (1 + t_cap)/2
    #   4. a clear cell and a capped cell with none      → 0, the clear cell's τ
    #   5. one τ in both cells, different shortwave      → that τ exactly
    sw_A, τ_A = [S, 0.0, S, S, 250.0], [0.0, 0.0, 0.0, 0.0, 3.7]
    sw_B, τ_B = [0.3S, 0.0, t_cap * S, 0.0, 90.0], [τ03, cap, cap, cap, 3.7]
    expected_AB = [(1 / 0.65 - 1) / 0.13, 0.0, (2 / (1 + t_cap) - 1) / 0.13, 0.0, 3.7]
    # Areas 1 and 3 with t = 1 and 0.3 under the same clear sky: t̄ = (1 + 3·0.3)/4 = 0.475.
    sw_C, τ_C = fill(S, 5), zeros(5)
    sw_D, τ_D = fill(0.3S, 5), fill(τ03, 5)
    # A one-cell band carries its cell's τ bit for bit, including a capped step with no shortwave
    # and a night step.
    sw_E, τ_E = [S, 0.0, 37.0, 0.0, 212.0], [0.0, cap, 6.25, 0.0, τ03 / 3]
    cells = [(lon = 10.0, lat = 46.0, z = 1000.0, glm = 0.0, bins = [(1050, 1.0)], sw = sw_A, τ = τ_A),
             (lon = 10.1, lat = 46.0, z = 1000.0, glm = 0.0, bins = [(1050, 1.0)], sw = sw_B, τ = τ_B),
             (lon = 10.2, lat = 46.0, z = 2000.0, glm = 0.0, bins = [(2050, 1.0)], sw = sw_C, τ = τ_C),
             (lon = 10.3, lat = 46.0, z = 2000.0, glm = 0.0, bins = [(2050, 3.0)], sw = sw_D, τ = τ_D),
             (lon = 10.4, lat = 46.0, z = 3000.0, glm = 0.0, bins = [(3050, 2.0)], sw = sw_E, τ = τ_E)]
    n = 5
    loader = function (_model, lat, lon; kwargs...)
        c = cells[findfirst(c -> c.lat == lat && c.lon == lon, cells)]
        _cp_forcing(lat, lon, c.z, fill(270.0, n); shortwave_downward = c.sw,
                    shortwave_downward_diffuse = 0.3 .* c.sw, cloud_optical_thickness = c.τ)
    end
    applied = _flat_applied(_CP_TIME[1:n], [(1000, 1100), (2000, 2100), (3000, 3100)])
    bands = collect(elevation_interval_forcing(
        _cp_table(cells), applied; climate_model = :synthetic,
        time_range = (_CP_TIME[1], _CP_TIME[n]), forcing_loader = loader))
    @test [b.center for b in bands] == [1050.0, 2050.0, 3050.0]
    τ_band(b) = collect(b.forcing[:cloud_optical_thickness])

    @test all(isapprox.(τ_band(bands[1]), expected_AB; rtol = 1e-12, atol = 0))
    @test τ_band(bands[1])[2] == 0.0
    @test τ_band(bands[1])[4] == 0.0
    @test τ_band(bands[1])[5] == 3.7
    # A plain area mean would have given (0 + 17.95)/2 = 8.97 at step 1 and cap/2 at step 4.
    @test τ_band(bands[1])[1] < τ03 / 2
    @test all(<=(cap), τ_band(bands[1]))
    @test all(isapprox.(τ_band(bands[2]), (1 / 0.475 - 1) / 0.13; rtol = 1e-12))
    @test τ_band(bands[3]) == τ_E

    # The native run forces each cell on its own: a one-cell band of that cell, so its τ is the
    # cell's τ exactly, as in the tile run's one-cell band above.
    for c in cells
        own = collect(elevation_interval_forcing(
            _cp_table([c]), applied; climate_model = :synthetic,
            time_range = (_CP_TIME[1], _CP_TIME[n]), forcing_loader = loader))
        @test τ_band(only(own)) == c.τ
    end

    # The band transmissivity of finite sums is a weighted mean of values in (0, 1], so anything
    # above 1 beyond rounding is refused rather than inverted.
    @test_throws "band cloud transmissivity" GEMB_GlacierSims._band_cloud_optical_thickness(
        [1.0], [1.1], [0.0], [1.0])
    @test GEMB_GlacierSims._band_cloud_optical_thickness([1.0], [1.0 + 1e-13], [0.0], [1.0]) == [0.0]
end

# Forcing for a cell on an arbitrary time axis, shaped like `climate_forcing` output.
function _audit_forcing(times, lat, lon, elevation, temperature)
    ti = Ti(times)
    n = length(times)
    layer(v) = DimArray(fill(Float64(v), n), (ti,))
    return DimStack((temperature_air = DimArray(collect(Float64, temperature), (ti,)),
                     pressure_air = layer(8e4), vapor_pressure = layer(300.0),
                     wind_speed = layer(3.0), precipitation = layer(0.1),
                     shortwave_downward = layer(100.0), longwave_downward = layer(250.0),
                     solar_zenith_angle = layer(60.0), shortwave_downward_diffuse = layer(30.0),
                     cloud_optical_thickness = layer(2.0));
                    metadata = Dict{String,Any}(
                        "latitude" => Float64(lat), "longitude" => Float64(lon),
                        "elevation" => Float64(elevation), "dataset" => "synthetic",
                        "temperature_observation_height" => 2.0, "wind_observation_height" => 10.0))
end

@testset "pipeline audit: donor conversion under the climatology basis" begin
    # A region that follows the fit's model exactly, resolved as production resolves it: the fits
    # come from one window, the run is another, and every applied parameter is a month-by-hour
    # median (basis = :climatology). Thirty cells at 300–3000 m with glacier fractions from 0 to 1,
    # ambient excess E(z, t) = a_t − 0.006 z, and a decoupling factor of 0.8 at 12 UTC and 0.7 at
    # 15 UTC. A cell above melting observes T_ref + [1 − glm·(1 − k)]·E and one at or below it
    # observes T_ref + E. The band temperature at z_b is then T_ref + k·E(z_b) where E(z_b) > 0 and
    # T_ref + E(z_b) otherwise, from every donor — including donors that sit above the melting
    # level on some steps of the fit window (where the fit's glacier term γ + δz changes sign) and
    # donors held at the edge of the ambient excess.
    T_ref = 273.15
    z = collect(range(300.0, 3000.0; length = 30))
    glm = [clamp(((zi - 300) / 2700)^1.5 + 0.25 * sin(3i), 0.0, 1.0) for (i, zi) in enumerate(z)]
    glm[1], glm[15], glm[30] = 0.0, 0.5, 1.0
    k_at(t) = hour(t) == 12 ? 0.8 : 0.7
    E(zi, a) = a - 0.006 * zi
    T_obs(zi, g, a, t) = (e = E(zi, a); e > 0 ? T_ref + (1 - g * (1 - k_at(t))) * e : T_ref + e)
    T_true(zb, a, t) = (e = E(zb, a); e > 0 ? T_ref + k_at(t) * e : T_ref + e)

    days(year) = [DateTime(year, 7, d, h) for d in 1:31 for h in (12, 15)]
    fit_time, run_time = days(2001), days(2005)
    # Fit window: every step has a measurable k (enough cells above melting and > 0.5 K of ambient
    # excess at the reference elevation), and the 3000 m donor is below melting at a_t < 18 and
    # held (0 < E < 0.5 K) at a_t = 18.3.
    a_fit = [18.0 + 7.0 * sin(1.3i) for i in eachindex(fit_time)]
    a_fit[5] = 18.3
    # Run window: a different series, with the 3000 m donor held at a_t = 18.2 and every cell below
    # melting at a_t = 1.
    a_run = [15.0 + 11.0 * sin(0.7i) for i in eachindex(run_time)]
    a_run[3], a_run[4] = 18.2, 1.0

    # Every cell holds ice in its own 100 m bin; the 300 m (glm 0), 1603 m (glm 0.5) and 3000 m
    # (glm 1) donors also force bands 550 m above them, and the 3000 m donor one 450 m below.
    own_bin(zi) = 100 * floor(Int, zi / 100) + 50
    cells = [(lon = 10.0 + 0.01i, lat = 46.0, z = z[i], glm = glm[i],
              bins = [(own_bin(z[i]), 1.0);
                      i in (1, 15) ? [(own_bin(z[i] + 550), 2.0)] : Tuple{Int,Float64}[];
                      i == 30 ? [(3550, 2.0), (2550, 1.5)] : Tuple{Int,Float64}[]])
             for i in eachindex(z)]
    table = _cp_table(cells)
    loader(times, a) = function (_model, lat, lon; kwargs...)
        c = cells[findfirst(c -> c.lat == lat && c.lon == lon, cells)]
        _audit_forcing(times, lat, lon, c.z, [T_obs(c.z, c.glm, a[i], times[i]) for i in eachindex(times)])
    end

    p = derive_downscaling_parameters(:synthetic, (first(fit_time), last(fit_time)), table;
                                      token = nothing, cache_path = nothing,
                                      forcing_loader = loader(fit_time, a_fit))
    @test DimensionalData.metadata(p.decoupling)["n_fitted"] == length(fit_time)
    # The premise: the top donor is above the fitted melting level on some fit steps, so the
    # glacier term there has the opposite sign to the one on its warm steps.
    g_top = collect(p.decoupling.coef_gamma) .+ collect(p.decoupling.coef_delta) .* 3000.0
    @test any(>(0), g_top) && any(<(0), g_top)

    fit = (; time = p.time, p.decoupling, p.lapse_rate, p.lapse_rate_uncertainty)
    applied = resolve_downscaling(fit, hypsometry_intervals(table), run_time; basis = :climatology)
    # The donor k is the model's k at each hour, from every donor elevation, held or not.
    for zi in (z[1], z[15], z[30])
        @test all(isapprox.(applied.donor_decoupling_factor_at(zi), k_at.(run_time); atol = 1e-12))
    end

    run_range = (first(run_time), last(run_time))
    check(bands) = for b in bands
        truth = [T_true(b.center, a_run[i], run_time[i]) for i in eachindex(run_time)]
        @test maximum(abs.(collect(b.forcing.temperature_air) .- truth)) <= 1e-9
    end
    all_bands = collect(elevation_interval_forcing(table, applied; climate_model = :synthetic,
                                                   time_range = run_range,
                                                   forcing_loader = loader(run_time, a_run)))
    @test length(all_bands) == length(hypsometry_intervals(table))
    check(all_bands)
    # Each of the glm 0, 0.5 and 1 donors alone.
    for i in (1, 15, 30)
        check(collect(elevation_interval_forcing(table[[i], :], applied; climate_model = :synthetic,
                                                 time_range = run_range,
                                                 forcing_loader = loader(run_time, a_run))))
    end
end

# Band forcing for a short GEMB run: `years` of 3-hourly steps from 2000, every layer constant.
# Shortwave dominates the surface energy budget of a melting surface: the air is at the melting point
# and saturated, so the turbulent fluxes vanish over a surface at 0 °C, and the downward longwave
# equals what such a surface emits.
function _tiny_band_forcing(; years = 3, shortwave = 300.0, temperature = 273.15, precipitation = 0.0,
                            longwave = 0.97 * 5.670374419e-8 * 273.15^4, elevation = 1050.0)
    time = collect(DateTime(2000, 1, 1):Hour(3):(DateTime(2000 + years, 1, 1) - Hour(3)))
    n = length(time)
    ti = Ti(time)
    e_sat = 611.2                                # Pa, saturation vapor pressure at 0 °C
    layers = (temperature_air = fill(temperature, n), pressure_air = fill(9e4, n),
              vapor_pressure = fill(e_sat, n), wind_speed = fill(2.0, n),
              precipitation = fill(precipitation, n), shortwave_downward = fill(shortwave, n),
              longwave_downward = fill(longwave, n), solar_zenith_angle = fill(45.0, n),
              shortwave_downward_diffuse = fill(0.3 * shortwave, n),
              cloud_optical_thickness = fill(1.0, n))
    return DimStack(NamedTuple(k => DimArray(v, (ti,)) for (k, v) in pairs(layers));
                    metadata = Dict{String,Any}(
                        "elevation" => elevation, "dataset" => "synthetic",
                        "temperature_observation_height" => 2.0, "wind_observation_height" => 10.0,
                        "temperature_air_mean" => temperature, "wind_speed_mean" => 2.0,
                        "precipitation_mean" => precipitation * 8 * 365))
end

# A tile run of one band per entry of `albedos`, all on the same forcing. `core_bins` is the tile's
# hypsometry, `(center, area)` per 100 m bin, and `extra_edges` adds intervals to `applied` that have
# no band.
function _tiny_tile_run(albedos; forcing = _tiny_band_forcing(), mp = initialize_parameters(
                            output_frequency = :monthly), core_bins = [(1050, 10.0)],
                        extra_edges = Tuple{Int,Int}[], kwargs...)
    n = length(dims(forcing, Ti))
    bands = [(; lo = 1000 + 100 * (i - 1), hi = 1100 + 100 * (i - 1), center = 1050.0 + 100 * (i - 1),
              area = 10.0 * i, n_cells = 1, n_cells_substituted = 0, area_substituted = 0.0,
              area_unrecovered = 0.0, max_donor_distance_km = 0.0,
              albedo_ice = a, albedo_ice_source = :observed, forcing)
             for (i, a) in enumerate(albedos)]
    applied = _flat_applied(collect(dims(forcing, Ti)),
                            sort!([[(b.lo, b.hi) for b in bands]; extra_edges]))
    tile = (; index = (10, 46), name = "N46_E010.nc",
            bounds = (lon_min = 10.0, lon_max = 12.0, lat_min = 46.0, lat_max = 48.0),
            core = _cp_table([(lon = 10.0, lat = 46.0, z = 1000.0, glm = 0.0, bins = core_bins)]))
    window = (first(lookup(forcing, Ti)), last(lookup(forcing, Ti)))
    return gemb_glacier_tile(tile, applied, bands, mp; spinup_window = window,
                             simulation_years_maximum = 3, threaded = false, kwargs...)
end

# Two bands on one shortwave-dominated melting forcing, at bare-ice albedos 0.3 and 0.5. Shared by
# the height-closure and bare-ice albedo testsets, since a GEMB run is the expensive part of both.
const TINY_RUN = _tiny_tile_run([0.3, 0.5])

@testset "pipeline audit: height components close" begin
    # Spec 64: `dh` and its decomposition share the run-start epoch, so the stored series satisfy
    # dh = dh_mass + dh_water + dh_firn + dh_residual at every output time, and the residual — what
    # GEMB's own mass budget leaves — sits at the rounding floor.
    s = TINY_RUN.bands_series
    @test s[:dh] ≈ s[:dh_mass] .+ s[:dh_water] .+ s[:dh_firn] .+ s[:dh_residual] atol = 1e-12
    @test maximum(abs, s[:dh_residual]) < 1e-9
    # The first output interval is in every term, not only in `dh`.
    @test all(!iszero, s[:dh_mass][1, :, 1, 1])
    # dv carries the same epoch as the per-band dh it is weighted from.
    areas = [b.area for b in TINY_RUN.bands]
    @test TINY_RUN.totals[:dv][:, 1, 1] ≈ tile_volume_change(s[:dh][:, :, 1, 1], areas)
    @test TINY_RUN.totals[:dv_mass][:, 1, 1] ≈ tile_volume_change(s[:dh_mass][:, :, 1, 1], areas)
end

@testset "pipeline audit: record-mean rate" begin
    # Spec 68: hourly forcing from 2000-01-01, weekly output stamped at the last hour of each week,
    # and dv rising by r every week. The record covers N whole weeks, so the rate is r·365.25/7 per
    # year, not r·N/(N-1)·365.25/7.
    r, N = 0.25, 6
    time = [DateTime(2000, 1, 1) + Day(7i) - Hour(1) for i in 1:N]
    run = _fake_tile_run(; time)
    run.totals[:dv] .= reshape([r * i for i in 1:N], N, 1, 1)
    run.provenance["time_coverage_start"] = DateTime(2000, 1, 1)
    run.provenance["forcing_time_step_seconds"] = 3600.0
    @test GEMB_GlacierSims.record_years(run) ≈ 7N / 365.25
    @test run.totals[:dv][end, 1, 1] / GEMB_GlacierSims.record_years(run) ≈ r * 365.25 / 7
end

@testset "pipeline audit: append continuity" begin
    # Spec 63: a continuation's cumulative series start from its seam; appended, they continue the
    # stored record. dh [-1, -2] then a continuation [-1, -2] reads [-1, -2, -3, -4], and the totals
    # likewise; the state total_fac and the per-interval fluxes are written as they are.
    dir = mktempdir()
    first_run = _fake_tile_run(; time = [DateTime(2000, 1, 31), DateTime(2000, 2, 29)])
    later = _fake_tile_run(; time = [DateTime(2000, 3, 31), DateTime(2000, 4, 30)])
    for r in (first_run, later)
        r.bands_series[:dh] .= reshape([-1.0, -2.0], 2, 1, 1, 1)
        r.totals[:dv] .= reshape([-1.0, -2.0], 2, 1, 1)
        r.totals[:runoff] .= reshape([3.0, 7.0], 2, 1, 1)
        r.totals[:fac] .= reshape([5.0, 4.0], 2, 1, 1)
        for p in r.band_provenance
            p["temperature_lapse_rate_n_fitted"] = 10
            p["n_timesteps_above_freezing"] = 4
        end
    end
    for p in later.band_provenance
        p["temperature_lapse_rate"] = 7.2
        p["temperature_lapse_rate_n_fitted"] = 30
    end
    later.provenance["max_dh_residual"] = 3.0e-13
    path = write_glacier_tile_netcdf(joinpath(dir, "t.nc"), first_run; precision = Float64)
    append_glacier_tile_netcdf(path, later)
    NCDatasets.NCDataset(path, "r") do ds
        @test ds["dh"][:, 1, 1, 1] == [-1.0, -2.0, -3.0, -4.0]
        @test ds["dh"][:, 3, 2, 2] == [-1.0, -2.0, -3.0, -4.0]
        @test ds["total_dv"][:, 1, 1] == [-1.0, -2.0, -3.0, -4.0]
        @test ds["total_runoff"][:, 1, 1] == [3.0, 7.0, 10.0, 14.0]
        @test ds["total_fac"][:, 1, 1] == [5.0, 4.0, 5.0, 4.0]
        @test ds["melt"][3:4, :, :, :] == later.bands_series[:melt]
        # Record-long band summaries: a step-weighted mean of 6.2 over 10 steps and 7.2 over 30, and
        # counts summed across the seam.
        @test ds["band_temperature_lapse_rate"][1] ≈ (6.2 * 10 + 7.2 * 30) / 40
        @test ds["band_temperature_lapse_rate_n_fitted"][1] == 40
        @test ds["band_n_timesteps_above_freezing"][1] == 8
        @test ds.attrib["max_dh_residual"] == 3.0e-13
    end
end

@testset "pipeline audit: software provenance in every file" begin
    sw = software_provenance()
    @test issetequal(keys(sw), ["software_$(p)_$(f)" for p in ("gemb", "gemb_climateforcing",
                                                            "gemb_glaciersims")
                                for f in ("version", "tree")])
    @test all(v -> v isa AbstractString && !isempty(v), values(sw))
    path = write_glacier_tile_netcdf(joinpath(mktempdir(), "t.nc"), TINY_RUN)
    stored = read_glacier_tile_status(path).parameters
    @test all(stored[k] == v for (k, v) in sw)
    # A file from different code is a different experiment.
    other = merge(stored, Dict{String,Any}("software_gemb_tree" => "0"^40))
    @test haskey(run_parameter_differences(stored, other), "software_gemb_tree")
end

@testset "pipeline audit: threaded unreachable flags" begin
    # Spec 69: one cell holding ice in 64 bands that all fail the forcing validator (a 200 K/km lapse
    # rate is outside the range `climate_adjust_for_elevation` accepts). The flags are set from
    # `Threads.@threads`, and every one must survive on every repetition: each band is reported as
    # unreachable, none as merely empty.
    n = 3
    bins = [(1050 + 100i, 1.0) for i in 0:63]
    table = _cp_table([(lon = 10.0, lat = 46.0, z = 1000.0, glm = 0.0, bins)])
    applied = _flat_applied(_CP_TIME[1:n], [(c - 50, c + 50) for (c, _) in bins]; lapse_rate = 200.0)
    loader = (m, lat, lon; kw...) -> _cp_forcing(lat, lon, 1000.0, fill(270.0, n))
    unreachable_counts = map(1:100) do _
        logs, _ = Test.collect_test_logs() do
            collect(elevation_interval_forcing(table, applied; climate_model = :synthetic,
                                               time_range = (_CP_TIME[1], _CP_TIME[n]),
                                               forcing_loader = loader))
        end
        hit = filter(l -> occursin("no donor cell could be adjusted", l.message), logs)
        isempty(hit) ? 0 : only(hit).kwargs[:n_dropped]
    end
    @test all(==(64), unreachable_counts)
end

@testset "pipeline audit: a failed forcing load propagates" begin
    cells = [(lon = 10.0 + 0.1i, lat = 46.0, z = 1000.0 + 500i, glm = 0.2i, bins = [(1050 + 500i, 1.0)])
             for i in 0:3]
    inner = _cp_loader(cells, c -> fill(280.0 - 5 * c.z / 1000, length(_CP_TIME)))
    flaky = (m, lat, lon; kw...) -> lon == 10.1 ? error("CDS request timed out") : inner(m, lat, lon; kw...)
    @test_throws "CDS request timed out" derive_downscaling_parameters(
        :synthetic, (_CP_TIME[1], _CP_TIME[end]), _cp_table(cells); token = nothing,
        cache_path = nothing, min_cells = 2, forcing_loader = flaky)
end

@testset "pipeline audit: merge keeps per-source attributes" begin
    dir = mktempdir()
    a = _fake_tile_run(; deltas = [0.0, 1.0])
    b = _fake_tile_run(; deltas = [-1.0])
    a.provenance["max_dh_residual"] = 1.0e-14
    b.provenance["max_dh_residual"] = 4.0e-13
    a.provenance["n_bands_dropped"] = 0
    b.provenance["n_bands_dropped"] = 2
    b.provenance["spinup_converged"] = true          # `a` carries false
    pa = write_glacier_tile_netcdf(joinpath(dir, "a.nc"), a)
    pb = write_glacier_tile_netcdf(joinpath(dir, "b.nc"), b)
    out = merge_tile_perturbations(joinpath(dir, "m.nc"), [pa, pb])
    NCDatasets.NCDataset(out, "r") do ds
        @test ds["delta_temperature"][:] == [-1.0, 0.0, 1.0]
        @test ds["source_max_dh_residual"][:] == [4.0e-13, 1.0e-14, 1.0e-14]
        @test ds["source_n_bands_dropped"][:] == [2.0, 0.0, 0.0]
        @test ds["source_spinup_converged"][:] == ["true", "false", "false"]
        @test ds.attrib["max_dh_residual"] == 4.0e-13
        @test ds.attrib["spinup_converged"] == "false"
        @test !haskey(ds.attrib, "n_bands_dropped")
        @test issubset(["max_dh_residual", "n_bands_dropped", "spinup_converged"],
                       split(ds.attrib["source_attributes"]))
        # Shared attributes are carried as they are.
        @test ds.attrib["geotile_id"] == a.geotile_id
    end
end

@testset "pipeline audit: elevation datum and annual precipitation" begin
    path = write_glacier_tile_netcdf(joinpath(mktempdir(), "t.nc"), _fake_tile_run())
    NCDatasets.NCDataset(path, "r") do ds
        for name in ("band_center", "band_lower", "band_upper")
            @test ds[name].attrib["standard_name"] == "height_above_mean_sea_level"
            @test ds[name].attrib["vertical_datum"] == "EGM2008"
        end
    end
    # Hourly forcing at 0.1 kg m-2 per step is 876 kg m-2 per 365-day year.
    cells = [(lon = 10.0, lat = 46.0, z = 1000.0, glm = 0.0, bins = [(1050, 1.0)])]
    applied = _flat_applied(_CP_TIME, [(1000, 1100)])
    band = only(collect(elevation_interval_forcing(
        _cp_table(cells), applied; climate_model = :synthetic,
        time_range = (_CP_TIME[1], _CP_TIME[end]),
        forcing_loader = _cp_loader(cells, c -> fill(270.0, length(_CP_TIME))))))
    @test DimensionalData.metadata(band.forcing)["precipitation_mean"] ≈ 0.1 * 8760
end

@testset "pipeline audit: band validation grid" begin
    # A band at 200 K survives ΔT = 0 but not ΔT = -30 (170 K is below the validator's 180 K floor).
    # Run at ΔT = 0 alone it is kept; validated against a grid reaching -30 K it is dropped, as it is
    # from a run over that grid.
    cold = _tiny_band_forcing(; years = 1, temperature = 200.0)
    band = (; lo = 1000, hi = 1100, center = 1050.0, area = 1.0, forcing = cold)
    kept, _ = GEMB_GlacierSims._runnable_bands([band], [0.0], [1.0])
    @test length(kept) == 1
    kept, dropped = GEMB_GlacierSims._runnable_bands([band], [-30.0, 0.0], [1.0])
    @test isempty(kept) && only(dropped).delta == -30.0
    @test_throws ForcingUnavailable _tiny_tile_run([0.4]; forcing = cold,
                                                   validation_grid = ([-30.0], [1.0]))
end

@testset "pipeline audit: intervals without band forcing" begin
    # The tile holds 99.5 km² at 1050 m, which has a band, and 0.5 km² at 1150 m, which does not —
    # an interval `elevation_interval_forcing` dropped. 0.5 % of the tile's ice is under the 1 %
    # default: the tile runs, and the interval is counted with its area and written to the file.
    forcing = _tiny_band_forcing()
    run = _tiny_tile_run([0.4]; forcing, core_bins = [(1050, 99.5), (1150, 0.5)],
                         extra_edges = [(1100, 1200)])
    p = run.provenance
    @test length(run.bands) == 1
    @test p["n_bands_dropped"] == 1
    @test p["n_intervals_without_forcing"] == 1
    @test p["intervals_without_forcing_area_km2"] ≈ 0.5
    @test p["dropped_area_km2"] ≈ 0.5
    @test p["hypsometry_area_km2"] ≈ 100.0
    path = write_glacier_tile_netcdf(joinpath(mktempdir(), "t.nc"), run)
    NCDatasets.NCDataset(path, "r") do ds
        @test ds.attrib["n_bands_dropped"] == 1
        @test ds.attrib["intervals_without_forcing_area_km2"] ≈ 0.5
    end

    # 2 km² of 100 is above the 1 % default, so the tile is refused before any GEMB run; a looser
    # stated fraction admits it.
    @test_throws "above the 1.0 % allowed by `dropped_interval_area_fraction_maximum`" _tiny_tile_run(
        [0.4]; forcing, core_bins = [(1050, 98.0), (1150, 2.0)], extra_edges = [(1100, 1200)])
    loose = _tiny_tile_run([0.4]; forcing, core_bins = [(1050, 98.0), (1150, 2.0)],
                           extra_edges = [(1100, 1200)],
                           dropped_interval_area_fraction_maximum = 0.05)
    @test loose.provenance["intervals_without_forcing_area_km2"] ≈ 2.0
    @test_throws "dropped_interval_area_fraction_maximum must be in [0, 1]" _tiny_tile_run(
        [0.4]; forcing, dropped_interval_area_fraction_maximum = 1.5)

    # Every interval with a band: nothing is counted.
    @test TINY_RUN.provenance["n_intervals_without_forcing"] == 0
    @test TINY_RUN.provenance["n_bands_dropped"] == 0
end

@testset "pipeline audit: unconverged spinups are recorded" begin
    # `TINY_RUN` spins up for one 3-year cycle, fewer than the drift window needs, so neither column
    # can converge: both are kept, counted, and their bands' area recorded.
    p = TINY_RUN.provenance
    @test p["spinup_converged"] == false
    @test p["spinup_n_unconverged"] == 2
    @test p["spinup_unconverged_area_km2"] ≈ sum(b.area for b in TINY_RUN.bands)
    @test all(==(0.0), TINY_RUN.run_provenance["spinup_converged"])
end

@testset "pipeline audit: hypsometry areas" begin
    # Spec 66: two fully glaciated 0.1° cells at 46-46.1°N on a flat 1050 m DEM. Each cell's area in
    # the 1000-1100 m bin is the spherical box R²·Δλ·(sin φ₂ - sin φ₁); the two sum to the 0.2° box;
    # a pixel 10 m below the geoid is counted (in the lowest bin), not dropped.
    GI = GeoDataFrames.GeoInterface
    Rs = GEMB_GlacierSims.Rasters
    dem_values = fill(1050.0f0, 120, 120)
    dem = Rs.Raster(dem_values, (X(9.905:0.01:11.095), Y(45.905:0.01:47.095)); crs = Rs.EPSG(4326))
    lake = (findfirst(≈(10.055), collect(9.905:0.01:11.095)), findfirst(≈(46.055), collect(45.905:0.01:47.095)))
    dem[lake...] = -10.0f0
    glaciers = Rs.Raster(zeros(Float32, 10, 10), (X(10.05:0.1:10.95), Y(46.05:0.1:46.95)))
    geoms = [GI.Polygon([GI.LinearRing([(9.5, 45.5), (11.5, 45.5), (11.5, 47.5), (9.5, 47.5),
                                        (9.5, 45.5)])])]
    gtree = GEMB_GlacierSims.SortTileRecursiveTree.STRtree(geoms)
    gp = (; geometry = [GI.Point(10.05, 46.05), GI.Point(10.15, 46.05)],
          index = [CartesianIndex(1, 1), CartesianIndex(2, 1)])
    edges = 0:100:10000
    H = zeros(2, length(edges) - 1)
    GEMB_GlacierSims.tile_hypsometry!(H, gp, 1:2, dem, geoms, gtree, glaciers, (0.1, 0.1), edges,
                                      length(edges) - 1, Base.Semaphore(1))
    R = 6371.0088                                   # km, mean Earth radius
    box(Δλ) = R^2 * deg2rad(Δλ) * (sind(46.1) - sind(46.0))
    @test sum(H[1, :]) ≈ box(0.1) rtol = 1e-3
    @test sum(H[2, :]) ≈ box(0.1) rtol = 1e-3
    @test sum(H) ≈ box(0.2) rtol = 1e-3
    @test H[1, 1] > 0                               # the -10 m pixel, in the lowest bin
    @test H[2, 1] == 0
    @test sum(H[:, 11]) ≈ box(0.2) - H[1, 1] rtol = 1e-12
end

@testset "pipeline audit: perturbation chain" begin
    # Spec 61: T = 263.15 K at 80 % relative humidity with respect to ice, LW = 0.8σT⁴, P = 1 kg m-2
    # per step. At ΔT = +2 and pscale 1.5 the chain gives T' = T + 2, e' = 0.8·eₛ,ᵢ(T') (relative
    # humidity is held against ice below 273.16 K), LW' = (ε_cs(e', T') + Δε)σT'⁴ with
    # Δε = 0.8 - ε_cs(e, T), and P' = 1.5; every other layer is untouched.
    CF = GEMB_GlacierSims.GEMB_ClimateForcing
    σ = 5.67e-8                     # the Stefan–Boltzmann value `temperature_adjust` documents
    # IFS Cy45r1 Part IV eq. 7.5 over ice (AERKi constants), written out independently of the package
    es_ice(T) = 611.21 * exp(22.587 * (T - 273.16) / (T + 0.7))
    T, n = 263.15, 4
    e = 0.8 * es_ice(T)
    f = _cp_forcing(46.0, 10.0, 1000.0, fill(T, n); vapor_pressure = e, longwave_downward = 0.8σ * T^4,
                    precipitation = 1.0)
    g = GEMB_GlacierSims._perturbed_forcing(f, 2.0, 1.5)
    T′ = T + 2
    e′ = 0.8 * es_ice(T′)
    Δε = 0.8 - CF.konzelmann_clear_sky_emissivity(e, T)
    @test all(≈(T′), g[:temperature_air])
    @test all(x -> isapprox(x, e′; rtol = 1e-12), g[:vapor_pressure])
    @test all(x -> isapprox(x, (CF.konzelmann_clear_sky_emissivity(e′, T′) + Δε) * σ * T′^4;
                            rtol = 1e-12), g[:longwave_downward])
    @test all(==(1.5), g[:precipitation])
    for v in (:pressure_air, :wind_speed, :shortwave_downward, :solar_zenith_angle,
              :shortwave_downward_diffuse, :cloud_optical_thickness)
        @test g[v] == f[v]
    end
    # At the identity point a supersaturated input (relative humidity 1.2) comes back bit for bit.
    wet = _cp_forcing(46.0, 10.0, 1000.0, fill(T, n);
                      vapor_pressure = 1.2 * CF.saturation_vapor_pressure(T),
                      longwave_downward = 0.8σ * T^4)
    same = GEMB_GlacierSims._perturbed_forcing(wet, 0.0, 1.0)
    @test all(v -> same[v] == wet[v], GEMB_GlacierSims._FORCING_VARIABLES)
end

@testset "pipeline audit: tile totals by hand" begin
    # Spec 62: bands of 10 and 30 km², runoff [1,2,3] and [4,5,6] kg m-2 per interval, dh
    # [-0.1,-0.2,-0.3] and [0,-0.1,-0.2] m.
    series = Dict{Symbol,Array{Float64,4}}()
    for v in (TILE_MASS_VARIABLES..., TILE_HEIGHT_VARIABLES...)
        series[v] = zeros(3, 2, 1, 1)
    end
    series[:runoff][:, :, 1, 1] = [1 4; 2 5; 3 6]
    series[:dh][:, :, 1, 1] = [-0.1 0.0; -0.2 -0.1; -0.3 -0.2]
    totals = GEMB_GlacierSims._tile_totals(series, [10.0, 30.0])
    @test totals[:runoff][:, 1, 1] ≈ [1.3, 3.0, 5.1] .* 1e-4
    @test totals[:dv][:, 1, 1] ≈ [-0.001, -0.005, -0.009]
    @test totals[:dm][:, 1, 1] ≈ -totals[:runoff][:, 1, 1]
end

@testset "pipeline audit: bare-ice albedo reaches GEMB" begin
    # Spec 65: on bare ice under a shortwave-dominated melting forcing, melt is proportional to the
    # absorbed shortwave, so the bands at albedo_ice 0.3 and 0.5 melt in the ratio 0.7/0.5.
    m = TINY_RUN.bands_series[:melt]
    @test sum(m[:, 1, 1, 1]) / sum(m[:, 2, 1, 1]) ≈ 0.7 / 0.5 rtol = 0.1
end

@testset "pipeline audit: time integration" begin
    # Spec 67: 1 kg m-2 of precipitation every hour for three weeks, weekly output: 168 kg m-2 per
    # interval, stamped at the last hour of days 7, 14 and 21, and the tile total over bands of 10 and
    # 30 km² accumulates 168·k·40·1e6 kg = 168·k·40·1e-6 Gt by interval k. GEMB's weeks are calendar
    # weeks ending on Sunday, so the record starts on a Monday to make the three intervals whole.
    start = DateTime(2000, 1, 3)
    time = collect(start:Hour(1):(start + Day(21) - Hour(1)))
    n = length(time)
    ti = Ti(time)
    f = DimStack(NamedTuple(k => DimArray(fill(v, n), (ti,)) for (k, v) in pairs((
            temperature_air = 260.0, pressure_air = 8e4, vapor_pressure = 150.0, wind_speed = 3.0,
            precipitation = 1.0, shortwave_downward = 100.0, longwave_downward = 220.0,
            solar_zenith_angle = 60.0, shortwave_downward_diffuse = 30.0,
            cloud_optical_thickness = 2.0)));
        metadata = Dict{String,Any}("temperature_air_mean" => 260.0, "wind_speed_mean" => 3.0,
                                    "precipitation_mean" => 8760.0,
                                    "temperature_observation_height" => 2.0,
                                    "wind_observation_height" => 10.0))
    mp = initialize_parameters(output_frequency = :weekly)
    cf = GEMB.initialize_forcing(f)
    out = GEMB.gemb(GEMB.initialize_profile(mp, cf), cf, mp)
    @test collect(lookup(out, Ti)) == [start + Day(d - 1) + Hour(23) for d in (7, 14, 21)]
    @test collect(out[:precipitation]) ≈ fill(168.0, 3)
    series = Dict{Symbol,Array{Float64,4}}(v => zeros(3, 2, 1, 1)
                                           for v in (TILE_MASS_VARIABLES..., TILE_HEIGHT_VARIABLES...))
    series[:precipitation][:, 1, 1, 1] = out[:precipitation]
    series[:precipitation][:, 2, 1, 1] = out[:precipitation]
    totals = GEMB_GlacierSims._tile_totals(series, [10.0, 30.0])
    @test totals[:precipitation][:, 1, 1] ≈ [168.0 * k * 40 * 1e-6 for k in 1:3]
end
