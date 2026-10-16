# Finding: tile spinups were unconverged at the 100-simulated-year ceiling (11% of glacier area at
# 0 K, 41% at -3 K in the merged tile runs). This measures how long the current physics actually
# needs: every band of one tile at ΔT = 0 and -3 K (pscale 1), spun up on the sweep's 1950-1980
# `:representative` climatology to a generous ceiling, with the sweep's drift criterion.
#
# Reads the ERA5-Land Zarr cache and the downscaling parameter tile under CLIMATE_CACHE (read only;
# a cell missing from the cache would be fetched from CDS into it), and writes one TSV row per column.
#
# Run from the repository root (about 15-30 core-minutes per tile):
#   julia -t 2 --project=<env> scripts/audit/spinup_convergence.jl N82_W048 1500 out_dir
# Tiles measured: N82_W048, S78_E168, N66_W156, N62_W142.
# Expected on fix/pipeline-audit: every column converged; slowest 285 simulated years (N82_W048,
# 250 m band, 0 K); otherwise at most 168 years.
using GEMB_GlacierSims, GEMB_ClimateForcing, GEMB, DataFrames, Dates, DimensionalData
import GeoDataFrames
const GS = GEMB_GlacierSims
const NAME, CEILING, OUT_DIR = ARGS[1], parse(Float64, ARGS[2]), ARGS[3]
const CLIMATE_CACHE = get(ENV, "CLIMATE_CACHE", "/mnt/bylot-r3/data/era5land")
const WINDOW = (DateTime(1950, 1, 2), DateTime(1980, 12, 31))
const TIME_RANGE = (DateTime(1950, 1, 2), DateTime(1981, 1, 1))     # ERA5-Land starts 1950-01-02

table = GeoDataFrames.read(joinpath(@__DIR__, "..", "..", "data", "era5land_glacier_elevation_classes.parquet"))
table[!, :longitude] = GeoDataFrames.GeoInterface.x.(table.geometry)
table[!, :latitude] = GeoDataFrames.GeoInterface.y.(table.geometry)
tile = only(filter(t -> t.name == NAME * ".nc", downscaling_tiles(table; tile_size = 2, buffer = 1)))
token = GEMB_ClimateForcing.get_cds_api_key()
cache = joinpath(CLIMATE_CACHE, "cache")

fit = read_downscaling_tile(joinpath(CLIMATE_CACHE, "downscaling_parameters", tile.name))
intervals = hypsometry_intervals(tile.core)
row = first(tile.core)
probe_time = collect(dims(climate_forcing(:era5land, row.latitude, row.longitude;
                                          time_range = TIME_RANGE, token, cache_path = cache), Ti))
applied = resolve_downscaling(fit, intervals, probe_time; basis = :climatology,
                              decoupling_factor_prior = decoupling_factor_prior(tile.core))
bands = collect(elevation_interval_forcing(tile.core, applied; climate_model = :era5land,
                                           time_range = TIME_RANGE, token, cache_path = cache,
                                           elevation_interval_batch = 0, donor_cells = tile.buffered))

mp = GEMB.initialize_parameters(output_frequency = :weekly)
tasks = [(ib, dT) for dT in (0.0, -3.0) for ib in eachindex(bands)]
mkpath(OUT_DIR)
out = joinpath(OUT_DIR, NAME * ".tsv")
open(io -> println(io, join(["tile", "band_center", "band_area_km2", "delta_temperature", "converged",
                             "cycles", "years_simulated", "seconds"], '\t')), out, "w")
lk = ReentrantLock()
Threads.@threads :dynamic for t in eachindex(tasks)
    ib, dT = tasks[t]
    band = bands[ib]
    t0 = time()
    cf = initialize_forcing(precipitation_adjust(temperature_adjust(band.forcing, dT), 1.0))
    mp_band = GS._with_albedo_ice(mp, band.albedo_ice)
    cf_spinup = GS._spinup_climatology(cf, WINDOW, mp_band)
    initial = initialize_profile(mp_band, cf_spinup)
    profile = gemb_spinup(initial, cf_spinup, mp_band; simulation_years_maximum = CEILING,
                          drift_window = GS.SPINUP_DRIFT_WINDOW, convergence_delta_density = nothing,
                          convergence_drift_density = GS._spinup_drift_tolerance(
                              initial, cf_spinup, mp_band, GS.SPINUP_DRIFT_FAC))
    m = DimensionalData.metadata(profile)
    line = join(Any[NAME, band.center, band.area, dT, m.spinup_converged, m.spinup_cycles,
                    m.spinup_years_simulated, round(time() - t0; digits = 1)], '\t')
    lock(() -> open(io -> println(io, line), out, "a"), lk)
end
rows = [split(l, '\t') for l in readlines(out)[2:end]]
for dT in ("0.0", "-3.0")
    r = filter(x -> x[4] == dT, rows)
    years = parse.(Float64, getindex.(r, 7))
    println(NAME, " ΔT = ", dT, ": ", length(r), " columns, ", count(x -> x[5] != "true", r),
            " unconverged, slowest ", round(Int, maximum(years)), " simulated years")
end
