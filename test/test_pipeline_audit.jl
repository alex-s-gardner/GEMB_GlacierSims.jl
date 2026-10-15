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
                                               z -> zeros(n))
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

@testset "pipeline audit: a donor where k(z) is near zero" begin
    # One fully glaciated cell (glm = 1) at 1800 m holding 5 km² of ice in the 1000-1100 m band, and
    # a fit, per timestep, with ambient excess α + βz of 8 K at the reference 1000 m and 4 K at
    # 1800 m, and glacier term γ + δz of -4 K at 1000 m (k = 0.5) and -4·(1 - 1e-7) K at 1800 m
    # (k = 1e-7). The cell observes 2 K above melting where the fit predicts 4e-7 K, a +2 K
    # residual. Its ambient excess is the fit's 4 K plus that residual,
    # 2 + 4·(1 - 1e-7) K; lapsed 750 m down at 6.5 K/km and decoupled with the band's
    # k(1050) = 1 + g(1050)/E(1050), the band temperature follows in closed form. Inverting the
    # observed excess with k at the donor instead multiplies it by 1e7, which no validator accepts,
    # and the band is dropped.
    n = length(_CP_TIME)
    ti = Ti(_CP_TIME)
    col(v) = DimArray(fill(Float64(v), n), ti)
    z̄, z_d, z_b = 1000.0, 1800.0, 1050.0
    β = (4.0 - 8.0) / (z_d - z̄)
    α = 8.0 - β * z̄
    g_ref, g_d = -4.0, (1e-7 - 1) * 4.0
    δ = (g_d - g_ref) / (z_d - z̄)
    γ = g_ref - δ * z̄
    fit = (; time = _CP_TIME,
           decoupling = DimStack((decoupling_factor = col(0.5), coef_alpha = col(α),
                                  coef_beta = col(β), coef_gamma = col(γ), coef_delta = col(δ))),
           lapse_rate = DimStack((lapse_rate = col(6.5), n_cells = DimArray(fill(10, n), ti),
                                  elevation_spread = col(500.0))))
    intervals = [(; lo = 1000, hi = 1100, center = z_b)]
    applied = resolve_downscaling(fit, intervals, _CP_TIME; basis = :fitted)
    cells = [(lon = 10.0, lat = 46.0, z = z_d, glm = 1.0, bins = [(1050, 5.0)])]
    T_obs = 273.15 + 2.0
    bands = collect(elevation_interval_forcing(
        _cp_table(cells), applied; climate_model = :synthetic,
        time_range = (_CP_TIME[1], _CP_TIME[end]),
        forcing_loader = _cp_loader(cells, c -> fill(T_obs, n))))
    @test length(bands) == 1
    band = only(bands)
    @test band.area ≈ 5.0
    E_ambient = 2.0 + 4.0 * (1 - 1e-7)
    E_band = E_ambient + 6.5 * (z_d - z_b) / 1000
    k_band = 1 + (γ + δ * z_b) / (α + β * z_b)
    @test all(t -> isapprox(t, 273.15 + k_band * E_band; atol = 1e-6),
              band.forcing.temperature_air)
end

@testset "pipeline audit: band optical thickness at the cap" begin
    # Three cells at the per-cell cap whose area-weighted mean rounds one ulp above it, as the
    # accumulation forms it: the band mean is the cap itself, which the forcing validator accepts.
    cap = GEMB_GlacierSims.CLOUD_OPTICAL_THICKNESS_MAX
    w = [0.1, 0.1, 0.7]
    total = foldl((acc, x) -> acc + x * cap, w; init = 0.0)
    @test total / foldl(+, w) > cap
    sums = Dict(:cloud_optical_thickness => [total], :shortwave_downward => [1.0])
    @test GEMB_GlacierSims._band_layer(:cloud_optical_thickness, sums, foldl(+, w)) == [cap]
    @test_throws "above the per-cell cap" GEMB_GlacierSims._band_layer(
        :cloud_optical_thickness, Dict(:cloud_optical_thickness => [1.01 * cap]), 1.0)
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
    # Spec 61: T = 263.15 K at 80 % relative humidity, LW = 0.8σT⁴, P = 1 kg m-2 per step. At
    # ΔT = +2 and pscale 1.5 the chain gives T' = T + 2, e' = 0.8·eₛ(T'), LW' = (ε_cs(e', T') + Δε)σT'⁴
    # with Δε = 0.8 - ε_cs(e, T), and P' = 1.5; every other layer is untouched.
    CF = GEMB_GlacierSims.GEMB_ClimateForcing
    σ = 5.67e-8                     # the Stefan–Boltzmann value `temperature_adjust` documents
    T, n = 263.15, 4
    e = 0.8 * CF.saturation_vapor_pressure(T)
    f = _cp_forcing(46.0, 10.0, 1000.0, fill(T, n); vapor_pressure = e, longwave_downward = 0.8σ * T^4,
                    precipitation = 1.0)
    g = GEMB_GlacierSims._perturbed_forcing(f, 2.0, 1.5)
    T′ = T + 2
    e′ = 0.8 * CF.saturation_vapor_pressure(T′)
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
