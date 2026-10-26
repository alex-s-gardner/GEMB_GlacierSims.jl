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
# Each site runs `firn_site_run` under the physics member `PHYSICS_MEMBER` (default
# "S-kfit-table", the production snow set) with snowfall matched to the core's accumulation, from the
# start of the ERA5-Land record to the drilling date. The forcing at the site is the production band
# forcing: the site's tile resolved for one band at the site elevation (`site_downscaling`, with the
# tile's Shaw et al. prior, the member's band decoupling and the production glacier correction),
# applied to the site's own cell with its `glm` (`site_forcing`). A site whose tile holds no
# glacierized cell, so has no fit, is lapsed at 6.5 K km-1 and left ambient, and says so in the log. The spinup climatology is 1951-1980, or the
# first 30 years of whatever record a site has. A date given only to the month is taken as its 15th,
# and one given only to the year as 1 July.
#
# Results are written with `Serialization` to `<output.jls>` as a `Dict(record_id => result)`, where a result is
# `firn_site_run`'s NamedTuple or, for a site that failed, `(; error = <message>)`. The file is
# rewritten after each site, so an interrupted sweep keeps what it finished and a rerun skips those.
#
# Forcing is read serially (the Zarr cache is shared), then sites run in parallel across Julia
# threads, one column per thread.

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
const CACHE = joinpath(tempdir(), ".cache", "era5land")
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
    sites, skipped = FirnSite[], String[]
    for id in usable
        isempty(ONLY) || id in ONLY || continue
        meta = TOML.parsefile(joinpath(ex, id, "meta.toml"))
        # A record holds only the sections its sources fill, as in the library's own scripts.
        acc = get(get(meta, "accumulation", Dict()), "mean_accumulation_kg_m2_yr", NaN)
        (acc isa Real && isfinite(acc) && acc > 0) || continue
        pos = meta["position"]
        elev = get(pos, "elevation_m", NaN)
        if !(elev isa Real && isfinite(elev))
            push!(skipped, "$id: no site elevation")
            continue
        end
        push!(sites, FirnSite(; record_id = id, latitude = pos["latitude"], longitude = pos["longitude"],
                              elevation = elev, date = core_date(meta["timing"]["date"]),
                              accumulation = acc, depth = meta["density"]["max_depth_m"]))
    end
    return sites, skipped
end

# The site's ERA5-Land cell in the glacier elevation-class table, for its `glm`, and its 2° tile.
function site_cells(sites)
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
        out[s.record_id] = (; glm = near ? GEMB_GlacierSims._row_glm(table[i, :]) : 0.0,
                            tile = get(tiles, name, nothing), name)
    end
    return out
end

function tile_file(name)
    for p in (joinpath(PARAMETERS, replace(name, ".nc" => ""), name), joinpath(PARAMETERS, name))
        isfile(p) && return p
    end
    return nothing
end

function main()
    sites, skipped = fit_set(LIBRARY)
    foreach(s -> println("skip ", s), skipped)
    results = isfile(OUTPUT) ? deserialize(OUTPUT) : Dict{String,Any}()
    todo = [s for s in sites if !haskey(results, s.record_id)]
    println(length(sites), " sites in the fit set, ", length(todo), " to run, under ", MEMBER.id,
            " on ", Threads.nthreads(), " threads")
    mp = GEMB_GlacierSims.physics_model_parameters(MEMBER)
    MEMBER.wind_scaling == 1 && MEMBER.wind_correction === :scaling ||
        error("physics member $(MEMBER.id) corrects the wind, which a site run does not apply")
    token = GEMB_ClimateForcing.get_cds_api_key()
    cells = site_cells(todo)
    forcings = Dict{String,Any}()
    for site in todo
        forcings[site.record_id] = try
            climate_forcing(:era5land, site.latitude, site.longitude;
                            time_range = (RECORD_START, site.date + Month(2)), token, cache_path = CACHE)
        catch e
            e isa InterruptException && rethrow()
            (; error = sprint(showerror, e))
        end
    end
    lock = ReentrantLock()
    Threads.@threads for site in todo
        t0 = time()
        forcing = forcings[site.record_id]
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
            r = firn_site_run(site, forcing, mp; spinup_window = window, downscaling,
                              glacier_fraction = c.glm)
            merge(r, (; tile = c.name, downscaled = downscaling !== nothing, glm = c.glm))
        catch e
            e isa InterruptException && rethrow()
            (; error = sprint(showerror, e))
        end
        Base.@lock lock begin
            results[site.record_id] = result
            serialize(OUTPUT, results)
            status = haskey(result, :error) ? "FAILED: " * first(result.error, 200) :
                     "spinup $(result.spinup.cycles) cycles, converged $(result.spinup.converged)" *
                     (result.downscaled ? "" : ", no tile fit: lapsed at 6.5 K/km, ambient")
            println(rpad(site.record_id, 48), round(Int, time() - t0), " s  ", status)
            flush(stdout)
        end
    end
    n_fail = count(r -> haskey(r, :error), values(results))
    println(length(results), " results in ", OUTPUT, "; ", n_fail, " failed")
end

main()
