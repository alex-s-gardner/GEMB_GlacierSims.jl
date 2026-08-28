# Check that the tile path's two threaded stages reproduce their serial results exactly.
#
# Two stages are threaded, for different reasons, and each has its own way of being wrong:
#
#   1. `elevation_interval_forcing` threads over bands within each cell's contribution. Each thread
#      writes only its own band's accumulator, so the addition order within a band stays the cell order.
#      Threading over *cells* instead would have several threads adding into one band, and
#      floating-point addition is not associative — so a mismatch here means the loop was parallelized
#      along the wrong axis.
#   2. `gemb_glacier_tile` threads the (band x delta x scaling) simulations and reduces them serially in
#      task order afterwards. A mismatch here means state is leaking between concurrent `gemb` calls
#      rather than rounding.
#
# Bit-for-bit is the right bar for both. Approximate agreement would hide exactly the failures this
# exists to catch.
#
# Run with more than one thread or it proves nothing:
#   julia --project=. -t 8 scripts/check_tile_threaded_equivalence.jl [tile_name]

using Dates
using DataFrames
using DimensionalData
using GEMB
using GEMB_GlacierSims
using GEMB_ClimateForcing
import GeoDataFrames
const GI = GeoDataFrames.GeoInterface

Threads.nthreads() == 1 &&
    @warn "Running with 1 thread; the threaded paths will not actually run concurrently"

const CLIMATE_MODEL = :era5land
const PARQUET = joinpath(@__DIR__, "..", "data", "$(CLIMATE_MODEL)_glacier_elevation_classes.parquet")
const CLIMATE_CACHE = get(ENV, "CLIMATE_CACHE", joinpath("/mnt/bylot-r3/data", string(CLIMATE_MODEL)))

# A small tile, run over a short window: this is about equivalence, not about a realistic record. The
# default is Iceland's Vatnajökull edge — few enough bands and cells to run the whole thing twice.
const TILE_NAME = length(ARGS) >= 1 ? ARGS[1] : "N64_W018"
# Must span more than a year: the spinup averages a climatological year out of this record, and a
# shorter window is refused (`SpinupWindowUnavailable`) rather than averaged from a fragment.
const TIME_RANGE = (DateTime(2018, 1, 1), DateTime(2019, 6, 1))

const failures = String[]
function report(name, pass)
    pass || push!(failures, name)
    println((pass ? "  PASS  " : "  FAIL  ") * name)
    return pass
end

token = GEMB_ClimateForcing.get_cds_api_key()
cache = joinpath(CLIMATE_CACHE, "cache")

table = GeoDataFrames.read(PARQUET)
table[!, :longitude] = GI.x.(table.geometry)
table[!, :latitude] = GI.y.(table.geometry)
tiles = downscaling_tiles(table; tile_size = 2, buffer = 1)
tile = only(filter(t -> t.name == TILE_NAME * ".nc", tiles))

fit = read_downscaling_tile(joinpath(CLIMATE_CACHE, "downscaling_parameters", TILE_NAME * ".nc"))
intervals = hypsometry_intervals(tile.core)

# The fits cover a different window than this short one, so the climatology basis is the applicable
# one — and it makes the resolution independent of the run length, which keeps this check cheap.
probe = climate_forcing(CLIMATE_MODEL, first(tile.core).latitude, first(tile.core).longitude;
                        time_range = TIME_RANGE, token, cache_path = cache)
applied = resolve_downscaling(fit, intervals, collect(dims(probe, Ti)); basis = :climatology,
                              decoupling_factor_prior = decoupling_factor_prior(tile.core))

@info "Tile under test" TILE_NAME cells=nrow(tile.core) bands=length(applied.bands) threads=Threads.nthreads()

# --- stage 1: band forcing ---------------------------------------------------------------------
#
# There is no serial switch to compare against — the accumulation is threaded unconditionally, because
# a flag whose off-path nobody runs is a second implementation waiting to drift. So the check is that
# two independent builds agree bit-for-bit: a race in the accumulator would be timing-dependent and
# would show up as a difference between runs.
t0 = time(); bands_a = collect(elevation_interval_forcing(tile.core, applied;
                                                          climate_model = CLIMATE_MODEL,
                                                          time_range = TIME_RANGE, token,
                                                          cache_path = cache,
                                                          elevation_interval_batch = 0))
t_forcing = time() - t0
bands_b = collect(elevation_interval_forcing(tile.core, applied;
                                             climate_model = CLIMATE_MODEL,
                                             time_range = TIME_RANGE, token, cache_path = cache,
                                             elevation_interval_batch = 0))

report("band count identical", length(bands_a) == length(bands_b))
report("band areas bit-for-bit identical", [b.area for b in bands_a] == [b.area for b in bands_b])
forcing_same = let same = true
    for (a, b) in zip(bands_a, bands_b), v in keys(a.forcing)
        same &= collect(a.forcing[v]) == collect(b.forcing[v])
    end
    same
end
report("band forcing bit-for-bit identical across two builds", forcing_same)

# Batching must not change the answer either: it changes how many passes over the cells are made, not
# what is accumulated, and it interacts with the threading because each batch threads over its own
# subset of bands.
bands_c = collect(elevation_interval_forcing(tile.core, applied;
                                             climate_model = CLIMATE_MODEL,
                                             time_range = TIME_RANGE, token, cache_path = cache,
                                             elevation_interval_batch = 3))
batch_same = length(bands_a) == length(bands_c) && let same = true
    for (a, c) in zip(bands_a, bands_c), v in keys(a.forcing)
        same &= collect(a.forcing[v]) == collect(c.forcing[v])
    end
    same
end
report("band forcing identical under batching", batch_same)

# --- stage 2: the simulations ------------------------------------------------------------------
mp = initialize_parameters(output_frequency = :monthly)
kw = (; delta_temperatures = [0.0, 1.0], precipitation_scalings = [1.0, 1.5],
      max_iterations = 20, convergence_delta_density = 0.01)

t0 = time(); serial = gemb_glacier_tile(tile, applied, bands_a, mp; kw..., threaded = false)
t_serial = time() - t0
t0 = time(); threaded = gemb_glacier_tile(tile, applied, bands_a, mp; kw..., threaded = true)
t_threaded = time() - t0

report("time axis identical", serial.time == threaded.time)
report("bands identical", serial.bands == threaded.bands)

for v in sort!(collect(keys(serial.bands_series)))
    a, b = serial.bands_series[v], threaded.bands_series[v]
    same = a == b                     # bit-for-bit, not isapprox
    report("bands_series[$v] bit-for-bit identical", same)
    same || println("          max abs diff = $(maximum(abs, a .- b))")
end

for v in sort!(collect(keys(serial.totals)))
    a, b = serial.totals[v], threaded.totals[v]
    same = a == b
    report("totals[$v] bit-for-bit identical", same)
    same || println("          max abs diff = $(maximum(abs, a .- b))")
end

prof_same = let same = true
    for idx in eachindex(serial.profiles)
        ps, pt = serial.profiles[idx], threaded.profiles[idx]
        if ps === nothing || pt === nothing
            same &= (ps === nothing) == (pt === nothing)
        else
            for v in PROFILE_VARIABLES
                same &= collect(ps[v]) == collect(pt[v])
            end
        end
    end
    same
end
report("all restart profiles bit-for-bit identical", prof_same)

println()
println("band forcing : $(round(t_forcing, digits = 1)) s on $(Threads.nthreads()) threads")
println("serial sims  : $(round(t_serial, digits = 1)) s")
println("threaded sims: $(round(t_threaded, digits = 1)) s   " *
        "($(round(t_serial / t_threaded, digits = 2))x)")
println()
if isempty(failures)
    println("RESULT: the tile path's threaded stages == their serial results (bit-for-bit)")
else
    println("RESULT: MISMATCH — a threaded stage is not equivalent:")
    foreach(f -> println("  - ", f), failures)
    exit(1)
end
