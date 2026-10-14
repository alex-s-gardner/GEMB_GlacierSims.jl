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

# A tile run of one band per entry of `albedos`, all on the same forcing.
function _tiny_tile_run(albedos; forcing = _tiny_band_forcing(), mp = initialize_parameters(
                            output_frequency = :monthly), kwargs...)
    n = length(dims(forcing, Ti))
    bands = [(; lo = 1000 + 100 * (i - 1), hi = 1100 + 100 * (i - 1), center = 1050.0 + 100 * (i - 1),
              area = 10.0 * i, n_cells = 1, n_cells_substituted = 0, area_substituted = 0.0,
              area_unrecovered = 0.0, max_donor_distance_km = 0.0,
              albedo_ice = a, albedo_ice_source = :observed, forcing)
             for (i, a) in enumerate(albedos)]
    applied = _flat_applied(collect(dims(forcing, Ti)), [(b.lo, b.hi) for b in bands])
    tile = (; index = (10, 46), name = "N46_E010.nc",
            bounds = (lon_min = 10.0, lon_max = 12.0, lat_min = 46.0, lat_max = 48.0),
            core = _cp_table([(lon = 10.0, lat = 46.0, z = 1000.0, glm = 0.0, bins = [(1050, 10.0)])]))
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
