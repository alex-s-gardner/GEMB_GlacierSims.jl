# Run GEMB at every firn core of the fit set and save the modeled profile on each drilling date.
#
#     julia --project=. -t <n> scripts/firn_site_sweep.jl <library> <parameters> <output.jls> [record_id ...]
#
# `<library>` is a FirnCoreLibrary checkout; `<parameters>` the directory of downscaling-parameter tiles
# (`derive_downscaling_parameter_tiles.jl`) holding each site's 2° tile, under `<tile>/<tile>.nc` or
# `<tile>.nc`. The fit set is every record that
# `extracted/USABLE.tsv` marks usable and that carries a measured mean accumulation; a record
# without a site elevation is skipped with a message, since the forcing is lapsed to it. Passing
# record ids restricts the run to those.
#
# `MATCH_ACCUMULATION=false` runs the end-to-end test instead: every usable record with a site
# elevation, measured accumulation or not, on the forcing's own snowfall (`firn_site_run` with
# `match_accumulation = false`). Each result records which test it belongs to in `match_accumulation`.
#
# Each site runs `firn_site_run` under the physics member `PHYSICS_MEMBER` (default
# "S-kfit-table", the production snow set) with snowfall matched to the core's accumulation, from the
# start of the ERA5-Land record to the drilling date. The forcing at the site is the production band
# forcing: the site's tile resolved for one band at the site elevation (`site_downscaling`, with the
# tile's Shaw et al. prior, the member's band decoupling and the production glacier correction),
# applied to the site's own cell with its `glm` (`site_forcing`). A site whose tile holds no
# glacierized cell, so has no fit, is lapsed at 6.5 K km-1 and left ambient, and says so in the log. A
# record whose `position.ice_sheet` is set is forced the same way: ERA5-Land models the ice-sheet surface
# itself, so its cell needs no glacier correction, and the glacier-band fit of a peripheral tile does
# not apply to it. The spinup climatology is 1951-1980, or the
# first 30 years of whatever record a site has. A date given only to the month is taken as its 15th,
# and one given only to the year as 1 July.
#
# `MODEL_OVERRIDES`, when set, is a Julia `NamedTuple` literal of `ModelParameters` fields applied on
# top of the member's snow settings, e.g. `(; crocus_handoff_density = 500.0)`, for a sensitivity run
# under otherwise production settings. The overrides are recorded with each result.
#
# Results are written with `Serialization` to `<output.jls>` as a `Dict(record_id => result)`, where a result is
# `firn_site_run`'s NamedTuple or, for a site that failed, `(; error = <message>)`. The file is
# rewritten after each site, so an interrupted sweep keeps what it finished and a rerun skips those.
#
# Forcing is read serially (the Zarr cache is shared), then sites run in parallel across Julia
# threads, one column per thread. With `FORCING_DIR` set, each site's raw forcing is read from
# `<FORCING_DIR>/<record_id>.jls` instead, so a machine without the cache or CDS access runs the same
# sites; `PREFETCH_FORCING=1` writes those files from the cache and exits without running GEMB.

using GEMB_GlacierSims, GEMB, GEMB_ClimateForcing
using Dates, Serialization, TOML, DelimitedFiles, DataFrames
import GeoDataFrames, GeoInterface

length(ARGS) >= 3 ||
    error("usage: firn_site_sweep.jl <library> <parameters> <output.jls> [record_id ...]")
const LIBRARY = ARGS[1]
const PARAMETERS = ARGS[2]
const OUTPUT = ARGS[3]
const ONLY = Set(ARGS[4:end])
const MEMBER = GEMB_GlacierSims.physics_member(get(ENV, "PHYSICS_MEMBER", "S-kfit-table"))
const OVERRIDES = let s = get(ENV, "MODEL_OVERRIDES", "(;)")
    v = eval(Meta.parse(s))
    v isa NamedTuple || error("MODEL_OVERRIDES must be a NamedTuple literal, got $s")
    v
end
const MATCH = let s = get(ENV, "MATCH_ACCUMULATION", "true")
    s in ("true", "false") || error("MATCH_ACCUMULATION must be true or false, got $s")
    s == "true"
end
const CACHE = joinpath(get(ENV, "CLIMATE_CACHE", "/mnt/bylot-r3/data/era5land"), "cache")
const FORCING_DIR = get(ENV, "FORCING_DIR", "")
const PREFETCH = get(ENV, "PREFETCH_FORCING", "0") == "1"
PREFETCH && isempty(FORCING_DIR) && error("PREFETCH_FORCING=1 needs FORCING_DIR")
const RECORD_START = DateTime(1950, 1, 2)
const SPINUP_WINDOW = (DateTime(1951, 1, 1), DateTime(1980, 12, 31, 23))

# A date as `YYYY-MM-DD`, `YYYY-MM` (taken as the 15th) or `YYYY` (taken as 1 July).
function core_date(s::AbstractString)
    parts = split(s, '-')
    length(parts) == 3 && return DateTime(parse.(Int, parts)...)
    length(parts) == 2 && return DateTime(parse(Int, parts[1]), parse(Int, parts[2]), 15)
    length(parts) == 1 && return DateTime(parse(Int, parts[1]), 7, 1)
    error("unrecognized core date \"$s\"")
end

function fit_set(library)
    ex = joinpath(library, "extracted")
    table, header = readdlm(joinpath(ex, "USABLE.tsv"), '\t', String; header = true)
    col(n) = findfirst(==(n), vec(header))
    usable = [table[i, col("record_id")] for i in axes(table, 1)
              if table[i, col("dir")] == "extracted" && table[i, col("usable")] == "yes"]
    sites, skipped, ice_sheet = FirnSite[], String[], Set{String}()
    for id in usable
        isempty(ONLY) || id in ONLY || continue
        meta = TOML.parsefile(joinpath(ex, id, "meta.toml"))
        # A record holds only the sections its sources fill, as in the library's own scripts.
        acc = get(get(meta, "accumulation", Dict()), "mean_accumulation_kg_m2_yr", NaN)
        measured = acc isa Real && isfinite(acc) && acc > 0
        MATCH && !measured && continue
        pos = meta["position"]
        elev = get(pos, "elevation_m", NaN)
        if !(elev isa Real && isfinite(elev))
            push!(skipped, "$id: no site elevation")
            continue
        end
        isempty(get(pos, "ice_sheet", "")) || push!(ice_sheet, id)
        push!(sites, FirnSite(; record_id = id, latitude = pos["latitude"], longitude = pos["longitude"],
                              elevation = elev, date = core_date(meta["timing"]["date"]),
                              accumulation = measured ? acc : NaN, depth = meta["density"]["max_depth_m"]))
    end
    return sites, skipped, ice_sheet
end

# The site's ERA5-Land cell in the glacier elevation-class table, for its `glm`, and its 2° tile.
function site_cells(sites, ice_sheet)
    table = GeoDataFrames.read(joinpath(@__DIR__, "..", "data", "era5land_glacier_elevation_classes.parquet"))
    lon = GeoInterface.x.(table.geometry)
    lat = GeoInterface.y.(table.geometry)
    tiles = Dict(t.name => t for t in downscaling_tiles(
        transform(table, :geometry => ByRow(GeoInterface.x) => :longitude,
                  :geometry => ByRow(GeoInterface.y) => :latitude); tile_size = 2, buffer = 1))
    out = Dict{String,Any}()
    for s in sites
        x = mod(s.longitude, 360)
        i = argmin(@. abs(lat - s.latitude) + abs(mod(lon - x + 180, 360) - 180))
        near = abs(lat[i] - s.latitude) <= 0.051 && abs(mod(lon[i] - x + 180, 360) - 180) <= 0.051
        name = tile_output_name(tile_index(s.longitude, s.latitude))
        sheet = s.record_id in ice_sheet
        out[s.record_id] = (; glm = near && !sheet ? GEMB_GlacierSims._row_glm(table[i, :]) : 0.0,
                            tile = sheet ? nothing : get(tiles, name, nothing), name = sheet ? "ice_sheet" : name)
    end
    return out
end

function tile_file(name)
    for p in (joinpath(PARAMETERS, replace(name, ".nc" => ""), name), joinpath(PARAMETERS, name))
        isfile(p) && return p
    end
    return nothing
end

function read_forcing(site, token)
    isempty(FORCING_DIR) || return deserialize(joinpath(FORCING_DIR, site.record_id * ".jls"))
    return climate_forcing(:era5land, site.latitude, site.longitude;
                           time_range = (RECORD_START, site.date + Month(2)), token, cache_path = CACHE)
end

function prefetch(sites)
    mkpath(FORCING_DIR)
    token = GEMB_ClimateForcing.get_cds_api_key()
    n_fail = 0
    for site in sites
        path = joinpath(FORCING_DIR, site.record_id * ".jls")
        isfile(path) && continue
        try
            f = climate_forcing(:era5land, site.latitude, site.longitude;
                                time_range = (RECORD_START, site.date + Month(2)), token, cache_path = CACHE)
            serialize(path * ".part", f)
            mv(path * ".part", path)
            println("prefetched ", site.record_id)
        catch e
            e isa InterruptException && rethrow()
            n_fail += 1
            println("FAILED ", site.record_id, ": ", first(sprint(showerror, e), 300))
        end
        flush(stdout)
    end
    println(length(sites), " sites, ", n_fail, " failed; forcing in ", FORCING_DIR)
    n_fail == 0 || exit(1)
end

function main()
    sites, skipped, ice_sheet = fit_set(LIBRARY)
    foreach(s -> println("skip ", s), skipped)
    PREFETCH && return prefetch(sites)
    results = isfile(OUTPUT) ? deserialize(OUTPUT) : Dict{String,Any}()
    todo = [s for s in sites if !haskey(results, s.record_id)]
    println(length(sites), MATCH ? " sites in the fit set, " : " sites, unmatched, ",
            length(todo), " to run, under ", MEMBER.id,
            isempty(OVERRIDES) ? "" : " with $OVERRIDES", " on ", Threads.nthreads(), " threads")
    mp = GEMB.initialize_parameters(; MEMBER.model..., OVERRIDES...)
    MEMBER.wind_scaling == 1 && MEMBER.wind_correction === :scaling ||
        error("physics member $(MEMBER.id) corrects the wind, which a site run does not apply")
    token = isempty(FORCING_DIR) ? GEMB_ClimateForcing.get_cds_api_key() : nothing
    cells = site_cells(todo, ice_sheet)
    # From the cache, forcing is read serially before the threaded runs; from `FORCING_DIR`, each thread
    # reads its own site's file, so only the running sites' records are held in memory.
    load(site) = try
        read_forcing(site, token)
    catch e
        e isa InterruptException && rethrow()
        (; error = sprint(showerror, e))
    end
    forcings = isempty(FORCING_DIR) ? Dict(site.record_id => load(site) for site in todo) : nothing
    lock = ReentrantLock()
    Threads.@threads for site in todo
        t0 = time()
        forcing = forcings === nothing ? load(site) : forcings[site.record_id]
        result = forcing isa NamedTuple ? forcing : try
            # The fixed window where the record holds it; otherwise `firn_site_run`'s default, the
            # first 30 whole years of the record.
            times = GEMB.lookup(forcing, GEMB.Ti)
            window = first(times) <= SPINUP_WINDOW[1] && SPINUP_WINDOW[2] <= last(times) ?
                     SPINUP_WINDOW : nothing
            c = cells[site.record_id]
            path = c.tile === nothing ? nothing : tile_file(c.name)
            c.tile !== nothing && path === nothing &&
                error("no downscaling parameters for tile $(c.name) under $PARAMETERS")
            downscaling = path === nothing ? nothing :
                site_downscaling(read_downscaling_tile(path), site, times;
                                 decoupling_factor_prior = shaw_decoupling_prior(c.tile.bounds),
                                 band_decoupling = MEMBER.band_rule,
                                 glacier_correction = DEFAULT_GLACIER_CORRECTION)
            r = firn_site_run(site, forcing, mp; match_accumulation = MATCH, spinup_window = window,
                              downscaling, glacier_fraction = c.glm)
            merge(r, (; tile = c.name, downscaled = downscaling !== nothing, glm = c.glm,
                      overrides = OVERRIDES, match_accumulation = MATCH))
        catch e
            e isa InterruptException && rethrow()
            (; error = sprint(showerror, e))
        end
        Base.@lock lock begin
            results[site.record_id] = result
            serialize(OUTPUT, results)
            status = haskey(result, :error) ? "FAILED: " * first(result.error, 200) :
                     "spinup $(result.spinup.cycles) cycles, converged $(result.spinup.converged)" *
                     (result.downscaled ? "" : result.tile == "ice_sheet" ?
                      ", ice sheet: lapsed at 6.5 K/km, ambient" : ", no tile fit: lapsed at 6.5 K/km, ambient")
            println(rpad(site.record_id, 48), round(Int, time() - t0), " s  ", status)
            flush(stdout)
        end
    end
    n_fail = count(r -> haskey(r, :error), values(results))
    println(length(results), " results in ", OUTPUT, "; ", n_fail, " failed")
end

main()
