# Run GEMB for the elevation bands of every glacierized 2° tile, tile by tile.
#
# Consumes the downscaling parameters `derive_downscaling_parameter_tiles.jl` produced and writes one
# CF-compliant netCDF per tile holding, over the full (temperature offset × precipitation scaling)
# grid: per-band surface height change and mass fluxes, and the tile-integrated volume and mass change
# in the units the 2° altimetry products use. That grid is what the downstream fit searches for the
# scalings that reproduce the measured volume change.
#
# The sweep is **resumable**: a tile whose file already covers the requested window with the same
# settings is skipped from metadata alone, with no forcing read and no simulation. So an interrupted run
# is resumed by re-running the same command, and only the incomplete tiles cost anything.
#
# Needs CDS credentials (`~/.cdsapirc` or `ENV["CDS_API_KEY"]`) and the parameter tiles. Forcing is read
# from the shared Zarr cache under `CLIMATE_CACHE`, so tiles are visited in chunk order.
#
# Run:  julia --project=. -t 64 scripts/gemb_tile_sweep.jl [start_year] [end_year]
#
# Environment overrides:  TILE_LIMIT, TILE_NAMES (comma-separated), FORCE=1, SPINUP_MAX_ITERATIONS
#
# Sizing: a 60-band tile over the full 8x8 grid is 3,840 simulations. At the 2018-2020 window that is
# minutes per tile on this machine; at the full record it is hours, so start with a narrow window and a
# TILE_LIMIT before committing to a global pass.

using GEMB_GlacierSims
using GEMB_ClimateForcing
using DataFrames
using Dates
using DimensionalData
using Statistics
import GEMB
import GeoDataFrames
const GI = GeoDataFrames.GeoInterface

const CLIMATE_MODEL = :era5land
const PARQUET = joinpath(@__DIR__, "..", "data", "$(CLIMATE_MODEL)_glacier_elevation_classes.parquet")
const CLIMATE_CACHE = get(ENV, "CLIMATE_CACHE", joinpath("/mnt/bylot-r3/data", string(CLIMATE_MODEL)))
const PARAMETER_DIR = joinpath(CLIMATE_CACHE, "downscaling_parameters")
const OUTPUT_DIR = joinpath(CLIMATE_CACHE, "tile_runs")

# Must match the gridding the parameter tiles were derived on, or a tile's parameters describe a
# different neighbourhood than its cells. The parameter files record their own `tile_size`/`tile_buffer`,
# and the pre-flight below compares them.
const TILE_SIZE = 2
const BUFFER = 1

const START_YEAR = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2018
const END_YEAR = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2020
const TIME_RANGE = (DateTime(START_YEAR, 1, 1), DateTime(END_YEAR, 1, 1))

# The precipitation scalings the earlier MATLAB GEMB sweep used, so a fitted `pscale` is directly
# comparable to the published one, against eight temperature offsets spanning the plausible reanalysis
# bias in both directions.
const PRECIPITATION_SCALINGS = [0.25, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
const DELTA_TEMPERATURES = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]

const SPINUP_MAX_ITERATIONS = parse(Int, get(ENV, "SPINUP_MAX_ITERATIONS", "400"))
const SPINUP_CONVERGENCE = 0.01

const TILE_LIMIT = haskey(ENV, "TILE_LIMIT") ? parse(Int, ENV["TILE_LIMIT"]) : typemax(Int)
const TILE_NAMES = haskey(ENV, "TILE_NAMES") ? split(ENV["TILE_NAMES"], ",") : String[]
const FORCE = get(ENV, "FORCE", "0") == "1"

function main()
    token = GEMB_ClimateForcing.get_cds_api_key()
    token === nothing && error("no CDS API key; set ENV[\"CDS_API_KEY\"] or write ~/.cdsapirc")
    cache = joinpath(CLIMATE_CACHE, "cache")

    isfile(PARQUET) || error("no glacier elevation-class table at $PARQUET")
    isdir(PARAMETER_DIR) || error("no downscaling parameters at $PARAMETER_DIR; run " *
                                  "scripts/derive_downscaling_parameter_tiles.jl first")

    table = GeoDataFrames.read(PARQUET)
    table[!, :longitude] = GI.x.(table.geometry)
    table[!, :latitude] = GI.y.(table.geometry)

    # `order = :chunk` visits tiles sharing an ERA5-Land download chunk consecutively, which is what
    # keeps the Zarr cache warm across neighbours. The band forcing pass is the I/O cost here.
    tiles = downscaling_tiles(table; tile_size = TILE_SIZE, buffer = BUFFER, order = :chunk)
    isempty(TILE_NAMES) ||
        (tiles = filter(t -> replace(t.name, ".nc" => "") in TILE_NAMES, tiles))
    selected = first(tiles, min(TILE_LIMIT, length(tiles)))

    mkpath(OUTPUT_DIR)
    mp = GEMB.initialize_parameters(output_frequency = :monthly)

    @info "GEMB tile sweep" tiles=length(selected) of=length(tiles) time_range=TIME_RANGE perturbations="$(length(DELTA_TEMPERATURES))x$(length(PRECIPITATION_SCALINGS))" threads=Threads.nthreads() output=OUTPUT_DIR

    rows = NamedTuple[]
    t_start = time()
    for (i, tile) in enumerate(selected)
        name = replace(tile.name, ".nc" => "")
        path = joinpath(OUTPUT_DIR, tile.name)
        t0 = time()
        try
            push!(rows, run_tile(i, tile, name, path, mp; token, cache))
        catch e
            e isa InterruptException && rethrow()
            # A broken caller fails identically for every tile, so it must not be recorded as a
            # property of this one.
            GEMB_GlacierSims.is_caller_error(e) && rethrow()
            @error "Tile failed; continuing" tile=i name exception=(e, catch_backtrace())
            push!(rows, summary_row(tile, name, :failed, time() - t0; error = sprint(showerror, e)))
        end
    end

    if isempty(rows)
        @warn "No tiles were selected; nothing to sweep" TILE_LIMIT TILE_NAMES
        return DataFrame()
    end

    summary = DataFrame(rows)
    GEMB_GlacierSims.Parquet2.writefile(joinpath(OUTPUT_DIR, "tile_runs_summary.parquet"), summary)

    @info "Sweep finished" minutes=round((time() - t_start) / 60; digits = 1) written=count(==("written"), summary.status) skipped=count(==("skipped"), summary.status) empty=count(==("empty"), summary.status) no_parameters=count(==("no_parameters"), summary.status) failed=count(==("failed"), summary.status)

    done = summary[summary.status .== "written", :]
    if !isempty(done)
        @info "Closure across written tiles" max_dh_residual=maximum(done.max_dh_residual) worst_tile=done.name[argmax(done.max_dh_residual)]
        @info "Volume change rate across written tiles (baseline, km3 i.e./yr)" median=round(median(skipmissing(done.dv_rate_baseline)); digits = 4) min=round(minimum(skipmissing(done.dv_rate_baseline)); digits = 4) max=round(maximum(skipmissing(done.dv_rate_baseline)); digits = 4)
    end
    return summary
end

function run_tile(i, tile, name, path, mp; token, cache)
    t0 = time()
    parameter_path = joinpath(PARAMETER_DIR, tile.name)
    if !isfile(parameter_path)
        @warn "No downscaling parameters for this tile; skipping" tile=i name
        return summary_row(tile, name, :no_parameters, time() - t0)
    end

    intervals = hypsometry_intervals(tile.core)
    if isempty(intervals)
        # A tile can carry cells with no populated hypsometry at all — the tiling is a partition of the
        # table, so a cell with zero glacier area still belongs to a tile.
        @info "Tile has no populated elevation bands; nothing to run" tile=i name
        return summary_row(tile, name, :empty, time() - t0)
    end

    fit = read_downscaling_tile(parameter_path)

    # --- pre-flight, before any forcing I/O --------------------------------------------------------
    # Resolving the parameters costs one small netCDF read plus a Shaw-table lookup per cell — a couple
    # of seconds — against a forcing pass and thousands of simulations. So it happens first, and its
    # result is both the skip test and the input to the run.
    probe_time = probe_run_time(tile, token, cache)
    basis = length(fit.time) == length(probe_time) && collect(fit.time) == probe_time ?
            :fitted : :climatology
    prior = decoupling_factor_prior(tile.core)
    applied = resolve_downscaling(fit, intervals, probe_time; basis, decoupling_factor_prior = prior)
    requested = tile_run_parameters(mp, applied;
                                   spinup_window = default_spinup_window(probe_time))

    status = read_glacier_tile_status(path)
    if !FORCE && status !== nothing && tile_run_is_current(status, requested, probe_time, intervals)
        @info "Tile already covers the request; skipping" tile=i name last_time=status.time
        return summary_row(tile, name, :skipped, time() - t0)
    end

    bands = collect(elevation_interval_forcing(tile.core, applied;
                                               climate_model = CLIMATE_MODEL,
                                               time_range = TIME_RANGE, token, cache_path = cache,
                                               elevation_interval_batch = 0))
    t_forcing = time() - t0

    run = gemb_glacier_tile(tile, applied, bands, mp;
                            delta_temperatures = DELTA_TEMPERATURES,
                            precipitation_scalings = PRECIPITATION_SCALINGS,
                            max_iterations = SPINUP_MAX_ITERATIONS,
                            convergence_delta_density = SPINUP_CONVERGENCE)
    t_gemb = time() - t0 - t_forcing

    write_glacier_tile_netcdf(path, run; institution = "NASA Jet Propulsion Laboratory")

    @info "Wrote tile" tile=i name bands=length(run.bands) simulations=length(run.bands)*length(DELTA_TEMPERATURES)*length(PRECIPITATION_SCALINGS) forcing_s=round(t_forcing; digits = 1) gemb_s=round(t_gemb; digits = 1) basis
    return summary_row(tile, name, :written, time() - t0; run, t_forcing, t_gemb, basis)
end

# The authoritative run time axis, from one cell's forcing rather than assumed to be hourly. Also the
# earliest point a wrong token or a cold cache surfaces, which is worth paying before a tile-wide pass.
function probe_run_time(tile, token, cache)
    row = first(tile.core)
    fd = climate_forcing(CLIMATE_MODEL, row.latitude, row.longitude;
                         time_range = TIME_RANGE, token, cache_path = cache)
    return collect(dims(fd, Ti))
end

# The first 30 complete years of the run window, matching what `gemb_glacier_tile` derives when no
# window is given. Computed here so the pre-flight's parameter comparison uses the same value the run
# will, rather than a value the run then recomputes differently.
default_spinup_window(run_time) =
    (DateTime(year(first(run_time)), 1, 1), DateTime(year(first(run_time)) + 29, 12, 31))

# Whether an existing tile file already answers this request. Decided from the file's coordinates and
# attributes only — the point of the check is to avoid the forcing pass, so it must not read one.
function tile_run_is_current(status, requested, run_time, intervals)
    status.n_timesteps == 0 && return false
    # The window: the file must already reach the end of what was asked for.
    status.time === nothing && return false
    status.time < last(run_time) && return false
    # The run grid: a changed band set or perturbation grid means the stored arrays describe something
    # else entirely.
    status.band_centers == [x.center for x in intervals] || return false
    status.delta_temperatures == DELTA_TEMPERATURES || return false
    status.precipitation_scalings == PRECIPITATION_SCALINGS || return false
    # The settings: a changed model parameter or downscaling policy makes it a different experiment.
    isempty(status.parameters) && return false
    return isempty(run_parameter_differences(status.parameters, requested))
end

function summary_row(tile, name, status, seconds; run = nothing,
                     t_forcing = missing, t_gemb = missing, basis = missing, error = "")
    i_dt = findfirst(==(0.0), DELTA_TEMPERATURES)
    i_ps = findfirst(==(1.0), PRECIPITATION_SCALINGS)
    years = run === nothing ? missing :
            (last(run.time) - first(run.time)).value / (365.25 * 86_400_000)
    return (; name,
            geotile_id = geotile_id(tile.bounds),
            index_lon = tile.index[1], index_lat = tile.index[2],
            n_cells_core = nrow(tile.core),
            n_bands = run === nothing ? 0 : length(run.bands),
            glacier_area_km2 = run === nothing ? sum(glacier_area_column(tile.core)) :
                               run.provenance["glacier_area_km2"],
            n_timesteps = run === nothing ? 0 : length(run.time),
            basis = basis === missing ? "" : string(basis),
            max_dh_residual = run === nothing ? 0.0 : run.provenance["max_dh_residual"],
            dv_rate_baseline = (run === nothing || i_dt === nothing || i_ps === nothing) ? missing :
                               run.totals[:dv][end, i_dt, i_ps] / years,
            dm_rate_baseline = (run === nothing || i_dt === nothing || i_ps === nothing) ? missing :
                               run.totals[:dm][end, i_dt, i_ps] / years,
            # A string rather than a `Symbol`, because that is what Parquet can hold — and it matches
            # the `status` column `derive_downscaling_parameter_tiles` writes, so the two summaries
            # join without a conversion.
            status = string(status),
            seconds = round(seconds; digits = 1),
            forcing_seconds = t_forcing === missing ? missing : round(t_forcing; digits = 1),
            gemb_seconds = t_gemb === missing ? missing : round(t_gemb; digits = 1),
            error)
end

main()
