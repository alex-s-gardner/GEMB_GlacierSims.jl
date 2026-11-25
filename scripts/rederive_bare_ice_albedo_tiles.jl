# Re-derive the bare-ice albedo profile of every downscaling tile file, then fill each profile's
# unresolved bins from the nearest tile with an observed bin (`rederive_bare_ice_albedo_tiles`).
#
# Only the albedo part of each file is rewritten; the forcing fits stay as they are, so this needs the
# Copernicus DEM but no CDS token. A tile whose profile was already derived with the current settings
# is skipped, so the pass is resumable.
#
#   julia --project=. -t 8 scripts/rederive_bare_ice_albedo_tiles.jl
#
# Across N processes: run N workers with TILE_BLOCKS=N TILE_BLOCK=i FILL=0, then one process with
# FILL_ONLY=1 once they have all finished. The fill rewrites every tile file and must not overlap a
# worker.

using GEMB_GlacierSims
using DataFrames
using Dates
import GeoDataFrames

include(joinpath(@__DIR__, "run_settings.jl"))

const CLIMATE_MODEL = :era5land
const TILE_SIZE = 2
const BUFFER = 1
const CLIMATE_CACHE = get(ENV, "CLIMATE_CACHE", joinpath("/mnt/bylot-r3/data", string(CLIMATE_MODEL)))
const PARAMETER_DIR = get(ENV, "OUTPUT_DIR", joinpath(CLIMATE_CACHE, "downscaling_parameters"))
const PARQUET = joinpath(@__DIR__, "..", "data", "$(CLIMATE_MODEL)_glacier_elevation_classes.parquet")

const DEM_ARCHIVE = let root = get(ENV, "COPERNICUS_DEM_ARCHIVE", nothing)
    root === nothing || isfile(joinpath(root, "tileList.txt")) ||
        error("COPERNICUS_DEM_ARCHIVE=$root holds no tileList.txt; it must be a mirror of the copernicus-dem-30m bucket")
    root
end
const DEM_CACHE = get(ENV, "GEMB_CACHE_PATH", nothing)

const TILE_BLOCKS = haskey(ENV, "TILE_BLOCKS") ? parse(Int, ENV["TILE_BLOCKS"]) : 1
const TILE_BLOCK = haskey(ENV, "TILE_BLOCK") ? parse(Int, ENV["TILE_BLOCK"]) : 1
const CLAIM_STALE_AFTER = Hour(parse(Int, get(ENV, "CLAIM_STALE_HOURS", "6")))
const FORCE = get(ENV, "FORCE", "0") == "1"
const FILL = get(ENV, "FILL", TILE_BLOCKS == 1 ? "1" : "0") == "1"
const FILL_ONLY = get(ENV, "FILL_ONLY", "0") == "1"

function settings_rows()
    return [
        setting("tile size / buffer", "$(TILE_SIZE)° / $(BUFFER)°"),
        setting("bare-ice albedo", "white-sky, MODIS cells ≥ $(BARE_ICE_ALBEDO_MIN_GLACIER_FRACTION) glacier"),
        env_setting("re-derive current profiles", FORCE, "FORCE"),
        env_setting("neighbor fill", FILL_ONLY ? "only" : FILL, FILL_ONLY ? "FILL_ONLY" : "FILL"),
        env_setting("worker blocks", TILE_BLOCKS, "TILE_BLOCKS"),
        env_setting("Copernicus DEM", DEM_ARCHIVE === nothing ? "network (copernicus-dem-30m bucket)" :
                                      "local archive $(DEM_ARCHIVE)", "COPERNICUS_DEM_ARCHIVE"),
        env_setting("parameter tiles", PARAMETER_DIR, "OUTPUT_DIR"),
    ]
end

function main()
    print_settings("Bare-ice albedo re-derivation — settings in force", settings_rows())
    SHOW_SETTINGS && return nothing
    isdir(PARAMETER_DIR) || error("no downscaling parameters at $PARAMETER_DIR")

    if FILL_ONLY
        fill_bare_ice_albedo_neighbors(PARAMETER_DIR)
        return nothing
    end

    isfile(PARQUET) || error("no glacier elevation-class table at $PARQUET")
    table = GeoDataFrames.read(PARQUET)
    GI = GeoDataFrames.GeoInterface
    table[!, :longitude] = GI.x.(table.geometry)
    table[!, :latitude] = GI.y.(table.geometry)

    claim_dir = joinpath(PARAMETER_DIR, "albedo_claims")
    mkpath(claim_dir)
    # A tile is asked for only while its profile is not current, so a claim whose owner died is
    # reclaimed by age and a finished tile is never asked for again. The tile file always exists, so
    # it cannot serve as the claim's "finished" marker; a path that is never written stands in for it.
    gate = TILE_BLOCKS == 1 ? nothing :
           (tile, path) -> claim_tile!(claim_dir, tile.name, joinpath(claim_dir, "never-written");
                                       stale_after = CLAIM_STALE_AFTER)
    summary = rederive_bare_ice_albedo_tiles(PARAMETER_DIR, table; tile_size = TILE_SIZE,
                                             buffer = BUFFER, dem_archive = DEM_ARCHIVE,
                                             dem_cache_path = DEM_CACHE, force = FORCE,
                                             tile_gate = gate, fill = FILL)
    failed = summary[summary.status .== "failed", :]
    isempty(failed) || (@error "Bare-ice albedo failed for some tiles" names = collect(failed.name); exit(1))
    return nothing
end

main()
