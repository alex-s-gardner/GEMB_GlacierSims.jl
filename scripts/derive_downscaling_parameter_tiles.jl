# Derive downscaling parameters for every glacier grid cell on Earth, tile by tile.
#
# Tiles the cached glacier elevation-class table onto a `TILE_SIZE`° grid, fits each tile's
# decoupling factor and lapse rate from a `BUFFER`°-widened cell selection, and writes one
# CF-compliant netCDF-4 (HDF5) file per tile. Every cell of the table belongs to exactly one tile, and
# `tiles_index.parquet` records that mapping, so "parameters for all cells" is checkable with a join.
#
# The sweep is **resumable**: a tile whose file already covers the requested window with the same
# settings is skipped from metadata alone, with no forcing read. So an interrupted run is resumed by
# re-running the same command, and only the incomplete tiles cost anything.
#
# Needs CDS credentials (`~/.cdsapirc` or `ENV["CDS_API_KEY"]`) and network. Forcing is cached as Zarr
# chunks under `CLIMATE_CACHE` and shared between tiles, which is why tiles are visited in chunk order.
#
# Run:  julia --project=. scripts/derive_downscaling_parameter_tiles.jl [start_year] [end_year]
#
# Sizing, measured on the 47,121-cell global table at the 2°/1° default: 819 non-empty tiles,
# ~170,000 cell forcing loads, ~60 MB per tile of fits before compression at the full 1950-2026
# record. Turning on the elevation-interval forcing (see RETAIN_ELEVATION_INTERVAL_FORCING) adds
# ~300 GB globally at that window and at least doubles the forcing reads, so it is off here.

using GEMB_GlacierSims
using GEMB_ClimateForcing
using DataFrames
using Dates
import GeoDataFrames

include(joinpath(@__DIR__, "run_settings.jl"))

const BASE_CLIMATE_DIR = "/mnt/bylot-r3/data";
const CLIMATE_MODEL = :era5land

# The grid. 2° tiles with a 1° buffer means each tile's fits see a 4°×4° neighbourhood: both fits are
# regressions across cells, and a bare 2° tile often carries too little elevation and glacier-fraction
# range to constrain them. `TILE_SIZE` must be a whole number of degrees dividing 360, so tile edges
# land exactly on ±180 and only the buffer of the seam tile crosses the antimeridian.
const TILE_SIZE = 2
const BUFFER = 1

# Fewest cells in the buffered window before either fit is attempted. At 2°/1° on the global table,
# 62 of 819 tiles fall below the default 8; each still gets a file, recording its cell count with no
# time axis, so a missing file always means "not yet run" rather than "could not be fitted".
const MIN_CELLS = 8

# Skip cells holding less than this total glacier area (km²). 0 keeps every cell, which is what makes
# the tile files a complete partition of the table.
const AREA_MINIMUM = 0.0

# Derive each tile's observed bare-ice albedo profile from MODIS and store it in the tile file, so every
# elevation class runs at a measured `albedo_ice`? Off, every band falls back to the tuned constant
# `albedo_ice` — a tile file written that way carries no profile, and nothing downstream can recover one.
# Needs the Copernicus DEM and so network, but no CDS token.
const DERIVE_ALBEDO = get(ENV, "DERIVE_ALBEDO", "1") == "1"

# Also store each tile's glacier-area weighted per-elevation-interval forcing? This is the expensive
# half — most of the output volume, plus at least one extra pass over every cell's forcing — and the
# fits stored here are sufficient to regenerate it later. Turn it on only when the downstream
# bias-correction sweep needs to read the forcing rather than re-derive it.
const RETAIN_ELEVATION_INTERVAL_FORCING = false

# Default window: two full years, enough for a seasonal cycle with a second year to show it repeats.
# Widen to the full record (1950, 2027) once a global pass at this window looks right — a wider window
# re-derives every tile, since a pooled cross-cell regression cannot be appended to.
const START_YEAR = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2018
const END_YEAR = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2020
const TIME_RANGE = (DateTime(START_YEAR, 1, 1), DateTime(END_YEAR, 1, 1))


const GI = GeoDataFrames.GeoInterface

# Where the forcing cache and the tile outputs live. Kept outside the repo and off `tempdir()`: the
# ERA5-Land chunks are tens of GB and expensive to re-fetch, so they must survive a reboot. Override
# with `ENV["CLIMATE_CACHE"]` on a machine without this mount.
const CLIMATE_CACHE = get(ENV, "CLIMATE_CACHE",
                          joinpath(BASE_CLIMATE_DIR, string(CLIMATE_MODEL)))

const PARQUET = joinpath(@__DIR__, "..", "data", "$(CLIMATE_MODEL)_glacier_elevation_classes.parquet")
# Overridable so a timing probe can share the forcing cache without writing files the production sweep
# would then treat as current and skip.
const OUTPUT_DIR = get(ENV, "OUTPUT_DIR", joinpath(CLIMATE_CACHE, "downscaling_parameters"))

# In-memory budget for the cross-tile forcing cache, in GiB.
#
# Each tile fits over a `BUFFER`-widened window, so at 2°/1° every cell falls inside four tiles'
# buffered sets: a global sweep asks for 170,474 cells to derive 47,121 distinct ones, a factor of 3.62.
# The Zarr cache keeps the compressed bytes local but each load still decompresses them, extracts the
# record across four variable groups, derives vapour pressure and wind speed, and validates units —
# ~200 ms per cell, which is 99.8% of a tile's time. Holding recently-loaded cells removes most of that
# repeat, because tiles are visited in chunk order so the tiles sharing a cell are neighbours in visit
# order.
#
# Sized against the working set rather than the whole sweep: a few tiles' buffered windows. Set to 0 to
# disable, e.g. on a machine where the memory is needed elsewhere.
const CACHE_BUDGET_GIB = parse(Float64, get(ENV, "FORCING_CACHE_GIB", "8"))

# Which slice of the tile list this process owns, 1-based. Over the full record the fit is far too long
# to run serially, so the sweep scales out across processes: each drains its own contiguous block of the
# chunk-ordered list — preserving the forcing-cache locality that ordering buys — and then steals the
# heaviest tile still unclaimed, so none idles through the tail. Tiles are claimed by an atomic mkdir
# under `OUTPUT_DIR/claims`, which is the whole coordination mechanism.
const TILE_BLOCKS = haskey(ENV, "TILE_BLOCKS") ? parse(Int, ENV["TILE_BLOCKS"]) : 1
const TILE_BLOCK = haskey(ENV, "TILE_BLOCK") ? parse(Int, ENV["TILE_BLOCK"]) : 1
# Must exceed the longest a single tile's fit can take: reclaiming a running tile would put two writers
# on one file. `0` disables reclaiming.
const CLAIM_STALE_AFTER = Hour(parse(Int, get(ENV, "CLAIM_STALE_HOURS", "24")))

# Stop after this many tiles. For a timing probe before committing to a global pass, which the sizing note
# above asks for; the sweep is resumable, so a bounded run is a prefix of the full one and not a detour.
const TILE_LIMIT = haskey(ENV, "TILE_LIMIT") ? parse(Int, ENV["TILE_LIMIT"]) : typemax(Int)

# Restrict the sweep to named tiles, as `gemb_tile_sweep.jl` does. A region can then be derived and run
# end to end before the globe is committed to — which is the only way to size the global pass on real
# tiles, since the sparse ones cost nothing and measure nothing.
const TILE_NAMES = haskey(ENV, "TILE_NAMES") ? Set(split(ENV["TILE_NAMES"], ",")) : Set{String}()

# What this run is configured to do, for the log and for the terminal that launched it. Built before any
# precondition is checked, so a run that stops on a missing table or a missing key still says what it was
# going to do.
function settings_rows()
    return [
        setting("time range", TIME_RANGE, length(ARGS) >= 1 ? "ARGS" : "default"),
        setting("tile size / buffer", "$(TILE_SIZE)° / $(BUFFER)°"),
        setting("minimum cells to fit", MIN_CELLS),
        setting("minimum cell glacier area", "$(AREA_MINIMUM) km²"),
        env_setting("observed bare-ice albedo", DERIVE_ALBEDO, "DERIVE_ALBEDO"),
        setting("elevation-interval forcing", RETAIN_ELEVATION_INTERVAL_FORCING),
        env_setting("worker blocks", TILE_BLOCKS, "TILE_BLOCKS"),
        env_setting("claim stale after", "$(Dates.value(CLAIM_STALE_AFTER)) h", "CLAIM_STALE_HOURS"),
        env_setting("tile limit", TILE_LIMIT == typemax(Int) ? "all" : TILE_LIMIT, "TILE_LIMIT"),
        env_setting("tiles", isempty(TILE_NAMES) ? "all" : join(sort!(collect(TILE_NAMES)), ", "),
                    "TILE_NAMES"),
        env_setting("forcing cache budget", "$(CACHE_BUDGET_GIB) GiB", "FORCING_CACHE_GIB"),
        env_setting("climate cache", CLIMATE_CACHE, "CLIMATE_CACHE"),
        env_setting("DEM tile cache", get(ENV, "GEMB_CACHE_PATH",
                                          "GEMB_ClimateForcing data/ (package default)"),
                    "GEMB_CACHE_PATH"),
        env_setting("output", OUTPUT_DIR, "OUTPUT_DIR"),
    ]
end

function main()
    print_settings("Downscaling-parameter derivation — settings in force", settings_rows())
    SHOW_SETTINGS && return nothing

    token = GEMB_ClimateForcing.get_cds_api_key()
    token === nothing && error("no CDS API key; set ENV[\"CDS_API_KEY\"] or write ~/.cdsapirc")
    cache = joinpath(CLIMATE_CACHE, "cache")

    isfile(PARQUET) || error("no glacier elevation-class table at $PARQUET; build it with " *
                             "src/era5_example.jl first")
    table = GeoDataFrames.read(PARQUET)

    # The table carries only the Point geometry; the forcing loader is keyed on these columns. Left in
    # the native 0-359.9°E convention, which is what `climate_forcing` expects — the tiler wraps
    # internally for its own geometry and writes wrapped values into the tile files.
    table[!, :longitude] = GI.x.(table.geometry)
    table[!, :latitude] = GI.y.(table.geometry)

    # Hourly, so the record length follows from the window. Used only to turn the memory budget into a
    # cell count, which is what the cache is sized in.
    n_time = round(Int, Dates.value(TIME_RANGE[2] - TIME_RANGE[1]) / 3_600_000)
    loader = climate_forcing
    if CACHE_BUDGET_GIB > 0
        capacity = forcing_cache_capacity(CACHE_BUDGET_GIB * 2^30, n_time)
        loader = CachedForcingLoader(climate_forcing; capacity)
        @info "Cross-tile forcing cache" budget_GiB=CACHE_BUDGET_GIB capacity_cells=capacity MiB_per_cell=round(CACHE_BUDGET_GIB * 1024 / capacity, digits = 1)
    end

    @info "Global downscaling-parameter sweep" cells=nrow(table) tile_size=TILE_SIZE buffer=BUFFER time_range=TIME_RANGE output=OUTPUT_DIR block="$TILE_BLOCK/$TILE_BLOCKS"

    # Claim-based work stealing across processes. The gate is asked per tile, in the visit order the
    # sweep itself chooses, so this process takes whatever it wins and skips the rest.
    #
    # Affinity is expressed by *declining* tiles outside this worker's block until it has drained them:
    # `derive_downscaling_parameter_tiles` owns the visit order, so a worker cannot reorder it — but it
    # can pass on a tile now and take it on a second pass. One pass with stealing enabled from the start
    # is simpler and costs only that a worker may take a neighbour's tile early, which the claim makes
    # safe either way.
    claim_dir = joinpath(OUTPUT_DIR, "claims")
    mkpath(claim_dir)
    function gate(tile, path)
        isempty(TILE_NAMES) || replace(tile.name, ".nc" => "") in TILE_NAMES || return false
        TILE_BLOCKS == 1 && return true          # sole worker: nothing to claim against
        return claim_tile!(claim_dir, tile.name, path; stale_after = CLAIM_STALE_AFTER)
    end

    t0 = time()
    summary = derive_downscaling_parameter_tiles(CLIMATE_MODEL, TIME_RANGE, table, OUTPUT_DIR;
                                                token, cache_path = cache,
                                                forcing_loader = loader,
                                                tile_size = TILE_SIZE,
                                                buffer = BUFFER,
                                                min_cells = MIN_CELLS,
                                                area_minimum = AREA_MINIMUM,
                                                derive_albedo = DERIVE_ALBEDO,
                                                retain_elevation_interval_forcing =
                                                    RETAIN_ELEVATION_INTERVAL_FORCING,
                                                institution = "NASA Jet Propulsion Laboratory",
                                                tile_limit = TILE_LIMIT,
                                                tile_gate = (TILE_BLOCKS == 1 && isempty(TILE_NAMES)) ?
                                                            nothing : gate)
    @info "Sweep finished" minutes=round((time() - t0) / 60, digits = 1)

    # The hit rate against the 3.62 requests per cell the tiling implies: the ceiling is ~0.72, and a
    # rate well below it means the capacity is smaller than the working set the visit order produces.
    loader isa CachedForcingLoader &&
        @info "Forcing cache" forcing_cache_report(loader)...

    # The fitted fraction is the thing to look at before trusting a global pass: a tile whose `k` was
    # fitted at almost no timestep is a tile whose forcing carried no warm excess to damp, which is
    # normal for cold high-latitude ice and not normal in the mid-latitudes.
    done = summary[summary.status .∈ Ref(["written", "skipped"]), :]
    if !isempty(done) && any(>(0), done.n_timesteps)
        frac = [t > 0 ? k / t : NaN for (k, t) in zip(done.n_decoupling_factor_fitted, done.n_timesteps)]
        finite = filter(isfinite, frac)
        isempty(finite) || @info "Decoupling factor fitted fraction across tiles" median=round(sort(finite)[cld(length(finite), 2)], digits = 3) min=round(minimum(finite), digits = 3) max=round(maximum(finite), digits = 3)
    end

    return summary
end

main()
