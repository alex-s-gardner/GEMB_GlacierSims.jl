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
# Run one block:  julia --project=. -t 1 scripts/gemb_tile_sweep.jl [start_year] [end_year]
# Run the sweep:  scripts/run_tile_sweep.sh [start_year] [end_year]
#
# Environment overrides:  TILE_BLOCKS, TILE_BLOCK, TILE_LIMIT, TILE_NAMES (comma-separated), FORCE=1,
# SPINUP_MAX_ITERATIONS, PRECIPITATION_SCALINGS, DELTA_TEMPERATURES, FORCING_CACHE_GIB
#
# Sizing: a 60-band tile over the 7x7 grid is 2,940 simulations, and one simulation is the spinup plus
# the transient. Over a 7-year record that is about 10 s each, so the largest tiles are hours and the
# global pass is days. Start with a TILE_LIMIT or a TILE_NAMES subset before committing to one.

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

# The perturbation grid the downstream fit searches. Both axes are dense near their identity and sparse
# at the extremes: the optimum is expected close to 1.0 / 0 K, so resolution matters there, while the
# far points exist to bracket it and to show the response is monotonic.
#
# `1.0` and `0.0` must be present. They are the baseline the reports and the reference discharge locate
# with `findfirst`, and a grid without them silently loses both.
#
# Overridable so a timing probe or a re-fit does not need the file edited, but the defaults are the run:
# they are what a re-run reproduces, and they are in git.
const PRECIPITATION_SCALINGS = haskey(ENV, "PRECIPITATION_SCALINGS") ?
    parse.(Float64, split(ENV["PRECIPITATION_SCALINGS"], ",")) :
    [0.25, 0.75, 0.8, 1.0, 1.25, 1.5, 4.0]
const DELTA_TEMPERATURES = haskey(ENV, "DELTA_TEMPERATURES") ?
    parse.(Float64, split(ENV["DELTA_TEMPERATURES"], ",")) :
    [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0]

# Ceiling on spinup cycles, not the convergence test: bands exit on the drift criterion well inside this
# on glacier firn, so the ceiling only binds on an outlier and a generous one costs nothing.
const SPINUP_MAX_ITERATIONS = parse(Int, get(ENV, "SPINUP_MAX_ITERATIONS", "400"))
# Spinup exits when the FAC trend flattens; see `SPINUP_DRIFT_FAC` in `glacier_run.jl`, including why this
# value is too loose for an ice-sheet plateau.
const SPINUP_DRIFT_FAC = 1e-2

const TILE_LIMIT = haskey(ENV, "TILE_LIMIT") ? parse(Int, ENV["TILE_LIMIT"]) : typemax(Int)
const TILE_NAMES = haskey(ENV, "TILE_NAMES") ? split(ENV["TILE_NAMES"], ",") : String[]
const FORCE = get(ENV, "FORCE", "0") == "1"

# Which slice of the tile list this process owns, as `TILE_BLOCK` of `TILE_BLOCKS`, 1-based.
#
# **Parallelise across processes, not threads.** A GEMB spinup cycle allocates about 100 MiB, and
# Julia's garbage collector is per-process and stops every thread, so threads inside one process contend
# for it rather than for cores: the collector takes 9% of one thread's wall clock, 38% of eight and 66%
# of thirty-two. Tile throughput therefore peaks near 16 threads and *falls* beyond it, and per-core
# throughput is highest at one thread per process:
#
#   threads/process     1     2     4     8    16    32    64
#   throughput/core  1.00  0.91  0.67  0.48  0.25  0.11  0.05
#
# Separate processes have separate heaps and do not contend, so the width to run is one process per
# physical core. Each holds about 2.5 GB — a fixed Julia runtime, the 1 GiB forcing cache, the
# elevation-class table, and the current tile's band forcing — and that total barely moves with the
# record length, since only the band forcing scales with it.
#
# `--heap-size-hint` does not shift any of this, and `Threads.@threads` and `Threads.@spawn` perform
# identically, so neither is a place to look for headroom.
#
# Within-tile threading is for the *latency* of a single tile — what `gemb_tile_e2e.jl` wants — not for
# throughput. At 25% efficiency, 16 threads still finish one tile 4x sooner.
#
# **Contiguous blocks, not a stride.** Tiles are ordered by forcing chunk so neighbours share cells; a
# modulo split would scatter neighbours across processes and each process's cache would miss what
# another already holds. A contiguous block keeps one process's run spatially coherent.
#
# `run_tile_sweep.sh` launches the blocks and collects their logs. Each process writes its own summary
# parquet, suffixed with its block, so they merge afterwards rather than overwriting.
const TILE_BLOCKS = haskey(ENV, "TILE_BLOCKS") ? parse(Int, ENV["TILE_BLOCKS"]) : 1
const TILE_BLOCK = haskey(ENV, "TILE_BLOCK") ? parse(Int, ENV["TILE_BLOCK"]) : 1

function main()
    token = GEMB_ClimateForcing.get_cds_api_key()
    token === nothing && error("no CDS API key; set ENV[\"CDS_API_KEY\"] or write ~/.cdsapirc")
    cache = joinpath(CLIMATE_CACHE, "cache")

    # The baseline corner has to exist: it is what the summary's `dv_rate_baseline` and every downstream
    # anomaly are measured against, and `findfirst` returning `nothing` would drop them silently rather
    # than fail.
    1.0 in PRECIPITATION_SCALINGS || error(
        "PRECIPITATION_SCALINGS must include 1.0, the unperturbed baseline; got " *
        string(PRECIPITATION_SCALINGS))
    0.0 in DELTA_TEMPERATURES || error(
        "DELTA_TEMPERATURES must include 0.0, the unperturbed baseline; got " *
        string(DELTA_TEMPERATURES))
    all(>=(0), PRECIPITATION_SCALINGS) || error(
        "a negative precipitation scaling is not a scenario; got " * string(PRECIPITATION_SCALINGS))

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
    selected = _tile_block(selected)

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
    # One summary per block, or concurrent processes would overwrite each other's. A single-block run
    # keeps the unsuffixed name so nothing that reads it has to know about blocks.
    suffix = TILE_BLOCKS == 1 ? "" : "_block$(lpad(TILE_BLOCK, 3, '0'))of$(TILE_BLOCKS)"
    GEMB_GlacierSims.Parquet2.writefile(
        joinpath(OUTPUT_DIR, "tile_runs_summary$suffix.parquet"), summary)

    @info "Sweep finished" minutes=round((time() - t_start) / 60; digits = 1) written=count(==("written"), summary.status) skipped=count(==("skipped"), summary.status) empty=count(==("empty"), summary.status) no_parameters=count(==("no_parameters"), summary.status) failed=count(==("failed"), summary.status)

    done = summary[summary.status .== "written", :]
    if !isempty(done)
        @info "Closure across written tiles" max_dh_residual=maximum(done.max_dh_residual) worst_tile=done.name[argmax(done.max_dh_residual)]
        @info "Volume change rate across written tiles (baseline, km3 i.e./yr)" median=round(median(skipmissing(done.dv_rate_baseline)); digits = 4) min=round(minimum(skipmissing(done.dv_rate_baseline)); digits = 4) max=round(maximum(skipmissing(done.dv_rate_baseline)); digits = 4)
    end
    return summary
end

# This process's contiguous slice of the tile list, balanced by the work each tile carries rather than
# by tile count.
#
# The sweep is embarrassingly parallel across blocks and finishes when the slowest block does, so what
# matters is the *cost* of a block. A tile's cost is its band count times the perturbation grid, and band
# count is not uniform: over the 816 runnable tiles it runs 1 to 63 with a median of 12 and a 95th
# percentile of 35. Splitting the list into equal-length runs therefore leaves the heaviest block doing
# 2.25x the mean at 32 blocks, which idles most of the machine through the tail. Equalising total band
# count instead brings that to 1.14x.
#
# Contiguous runs, still: tiles are ordered by forcing chunk so neighbours share donor cells, and a
# scattered assignment would make each process miss what another already holds. A contiguous
# weight-balanced partition keeps both properties, and beats a dynamic work queue, whose balance is
# floored by the single 63-band tile it cannot subdivide.
#
# Every process computes the same partition from the same inputs, so no coordination is needed.
function _tile_block(tiles)
    TILE_BLOCKS >= 1 || error("TILE_BLOCKS must be at least 1, got $TILE_BLOCKS")
    1 <= TILE_BLOCK <= TILE_BLOCKS ||
        error("TILE_BLOCK must be in 1..$TILE_BLOCKS, got $TILE_BLOCK")
    TILE_BLOCKS == 1 && return tiles

    # Band count is the cost proxy: the perturbation grid is the same for every tile, and the record
    # length is too, so simulations per tile is proportional to it. Cheap to compute — hypsometry comes
    # from the already-loaded table, with no forcing read.
    weights = Float64[length(hypsometry_intervals(t.core)) for t in tiles]
    block = _weight_balanced_blocks(weights, TILE_BLOCKS)[TILE_BLOCK]
    @info "Tile block" block="$TILE_BLOCK/$TILE_BLOCKS" tiles=length(block) range="$block of $(length(tiles))" bands=Int(sum(weights[block])) bands_mean_per_block=round(sum(weights) / TILE_BLOCKS; digits = 1)
    return tiles[block]
end

# Split `1:length(w)` into `k` contiguous ranges of as near equal total weight as the item granularity
# allows, by closing each range as the running total crosses its share. Ranges may be empty when `k`
# exceeds the number of non-zero weights, which is why callers must tolerate an empty block.
function _weight_balanced_blocks(w, k::Int)
    total = sum(w)
    blocks = UnitRange{Int}[]
    lo = 1
    acc = zero(eltype(w))
    for i in eachindex(w)
        acc += w[i]
        if length(blocks) < k - 1 && acc >= total * (length(blocks) + 1) / k
            push!(blocks, lo:i)
            lo = i + 1
        end
    end
    push!(blocks, lo:length(w))
    return blocks
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
    if !FORCE && status !== nothing && tile_run_is_current(status, requested, probe_time, intervals, mp)
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
                            convergence_drift_fac = SPINUP_DRIFT_FAC)
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

# The longest span one output period can cover. An upper bound is the safe direction for the window
# test below: understating it would declare a complete file short and re-run it, which is the failure
# this exists to prevent. `nothing` means the output axis lands on the forcing axis, so the window end
# can be compared exactly.
function output_period_bound(mp)
    f = mp.output_frequency
    f === :monthly && return Day(31)
    f === :daily && return Day(1)
    f === :yearly && return Day(366)
    return nothing
end

# Whether an existing tile file already answers this request. Decided from the file's coordinates and
# attributes only — the point of the check is to avoid the forcing pass, so it must not read one.
function tile_run_is_current(status, requested, run_time, intervals, mp)
    status.n_timesteps == 0 && return false
    status.time === nothing && return false
    # The window: the file must cover it to within one output period. The stored time is the last
    # *output* sample while `run_time` is the last *forcing* step, and a coarser output grid never
    # reaches the forcing's final step — monthly output for a window ending 2023-01-01T00:00 lands on
    # 2022-12-31T23:00. Comparing the two directly is what made every file look stale.
    #
    # The tolerance means a window extended by less than one output period counts as covered. That is
    # the intended reading: such an extension spans no further output interval, so re-running would
    # reproduce the same series.
    period = output_period_bound(mp)
    if period === nothing
        status.time < last(run_time) && return false
    else
        status.time + period < last(run_time) && return false
    end
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
