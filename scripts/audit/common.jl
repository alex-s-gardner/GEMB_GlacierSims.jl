# Synthetic inputs shared by the audit scripts. Uses only API present on both
# `feat/native-resolution-run` and `fix/pipeline-audit`, so each script runs on either branch.
using GEMB_GlacierSims, DataFrames, Dates, DimensionalData
import GEMB, GeoDataFrames
const GS = GEMB_GlacierSims
const GI = GeoDataFrames.GeoInterface

# A band-forcing stack: every layer constant over `time`, all ten layers `climate_forcing` returns.
function synthetic_forcing(time; temperature = 273.15, shortwave = 300.0, precipitation = 0.0,
                           longwave = 0.97 * 5.670374419e-8 * 273.15^4, elevation = 1050.0)
    n = length(time)
    layers = (temperature_air = temperature, pressure_air = 9e4, vapor_pressure = 611.2,
              wind_speed = 2.0, precipitation = precipitation, shortwave_downward = shortwave,
              longwave_downward = longwave, solar_zenith_angle = 45.0,
              shortwave_downward_diffuse = 0.3 * shortwave, cloud_optical_thickness = 1.0)
    return DimStack(NamedTuple(k => DimArray(fill(Float64(v), n), (Ti(time),)) for (k, v) in pairs(layers));
                    metadata = Dict{String,Any}(
                        "elevation" => elevation, "latitude" => 46.0, "longitude" => 10.0,
                        "dataset" => "synthetic", "temperature_observation_height" => 2.0,
                        "wind_observation_height" => 10.0, "temperature_air_mean" => temperature,
                        "wind_speed_mean" => 2.0, "precipitation_mean" => precipitation * 8760))
end

# An `AppliedDownscaling` with one band per `(lo, hi)`, k = 1 and a constant lapse rate.
function flat_applied(time, edges; lapse_rate = 6.5)
    n = length(time)
    bands = [(; lo, hi, center = (lo + hi) / 2, decoupling_factor = ones(n),
              decoupling_factor_source = fill(Int8(1), n), n_fit_in_domain = n, n_fit_held = 0,
              albedo_ice = 0.48, albedo_ice_source = :default) for (lo, hi) in edges]
    return GS.AppliedDownscaling(collect(time), fill(Float64(lapse_rate), n), fill(Int8(1), n),
                                 bands, :fitted, Dict{String,Any}())
end

# A one-row glacier elevation-class table: one cell with `bins = [(center, area_km2), ...]`.
function one_cell_table(; lon = 10.0, lat = 46.0, glm = 0.0, bins)
    d = Dict{Symbol,Any}(:geometry => [GI.Point(lon, lat)], :glm => [glm], :latitude => [lat],
                         :longitude => [lon], :chunk_id => [1])
    for name in GS._hyps_colnames(0:100:10_000)
        d[name] = [0.0]
    end
    for (c, a) in bins
        d[only(GS._hyps_colnames([c - 50, c + 50]))] = [Float64(a)]
    end
    return DataFrame(d)
end

# A tile run of one band per albedo, all on `forcing`, spun up for one 3-year cycle.
function tiny_tile_run(forcing, albedos; mp = GEMB.initialize_parameters(output_frequency = :monthly))
    time = collect(lookup(forcing, Ti))
    bands = [(; lo = 1000 + 100 * (i - 1), hi = 1100 + 100 * (i - 1), center = 1050.0 + 100 * (i - 1),
              area = 10.0 * i, n_cells = 1, n_cells_substituted = 0, area_substituted = 0.0,
              area_unrecovered = 0.0, max_donor_distance_km = 0.0, albedo_ice = a,
              albedo_ice_source = :observed, forcing) for (i, a) in enumerate(albedos)]
    applied = flat_applied(time, [(b.lo, b.hi) for b in bands])
    tile = (; index = (10, 46), name = "N46_E010.nc",
            bounds = (lon_min = 10.0, lon_max = 12.0, lat_min = 46.0, lat_max = 48.0),
            core = one_cell_table(; bins = [(1050, 10.0)]))
    return gemb_glacier_tile(tile, applied, bands, mp; spinup_window = (first(time), last(time)),
                             simulation_years_maximum = 3, threaded = false)
end

three_years_3hourly() = collect(DateTime(2000, 1, 3):Hour(3):(DateTime(2003, 1, 3) - Hour(3)))

# A one-band, one-perturbation `GlacierTileRun` with `dh` and `total_dv` set to `dv` and every other
# series zero, built without running GEMB.
function minimal_tile_run(time; dv = [-1.0, -2.0], provenance = Dict{String,Any}())
    n_t = length(time)
    series = Dict{Symbol,Array{Float64,4}}(v => zeros(n_t, 1, 1, 1)
                                           for v in (TILE_MASS_VARIABLES..., TILE_HEIGHT_VARIABLES...))
    series[:dh][:, 1, 1, 1] = dv
    totals = Dict{Symbol,Array{Float64,3}}(v => zeros(n_t, 1, 1)
                                           for v in (TILE_MASS_VARIABLES..., :dm, :dv, :dv_mass, :dv_firn, :fac))
    totals[:dv][:, 1, 1] = dv
    profiles = Array{Union{Nothing,DimStack}}(nothing, 1, 1, 1)
    profiles[1, 1, 1] = DimStack(NamedTuple(v => DimArray(fill(1.0, 3), (GEMB.Z(1:3),))
                                            for v in GS.PROFILE_VARIABLES))
    prov = [Dict{String,Any}("temperature_lapse_rate" => 6.5, "temperature_lapse_rate_n_fitted" => 10)]
    return GlacierTileRun((10, 46), "N46_E010.nc", "lat[+46+48]lon[+010+012]",
                          (lon_min = 10.0, lon_max = 12.0, lat_min = 46.0, lat_max = 48.0), 1, 1,
                          [(; lo = 1000, hi = 1100, center = 1050.0, area = 10.0, n_cells = 1)],
                          [0.0], [1.0], time, series, totals, profiles, prov,
                          Dict{String,Array{Float64,3}}(), Dict{String,Any}(), provenance)
end
