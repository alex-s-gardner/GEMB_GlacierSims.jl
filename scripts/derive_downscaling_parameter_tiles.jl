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
const OUTPUT_DIR = joinpath(CLIMATE_CACHE, "downscaling_parameters")

function main()
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

    @info "Global downscaling-parameter sweep" cells=nrow(table) tile_size=TILE_SIZE buffer=BUFFER time_range=TIME_RANGE output=OUTPUT_DIR

    t0 = time()
    summary = derive_downscaling_parameter_tiles(CLIMATE_MODEL, TIME_RANGE, table, OUTPUT_DIR;
                                                token, cache_path = cache,
                                                tile_size = TILE_SIZE,
                                                buffer = BUFFER,
                                                min_cells = MIN_CELLS,
                                                area_minimum = AREA_MINIMUM,
                                                retain_elevation_interval_forcing =
                                                    RETAIN_ELEVATION_INTERVAL_FORCING,
                                                institution = "NASA Jet Propulsion Laboratory")
    @info "Sweep finished" minutes=round((time() - t0) / 60, digits = 1)

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
