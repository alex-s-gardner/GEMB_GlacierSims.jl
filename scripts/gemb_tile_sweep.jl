# Run GEMB for the elevation bands of every glacierized 2° tile, tile by tile.
#
# Consumes the downscaling parameters `derive_downscaling_parameter_tiles.jl` produced and writes one
# CF-compliant netCDF per tile holding, over the full (temperature offset × precipitation scaling)
# grid: per-band surface height change and mass fluxes, and the tile-integrated volume and mass change
# in the units the 2° altimetry products use. That grid is what the downstream fit searches for the
# scalings that reproduce the measured volume change.
#
# The sweep is **resumable and extendable**. A tile whose file already covers the requested window with
# the same settings is skipped from metadata alone, with no forcing read and no simulation. A tile whose
# settings match but whose record has since grown is *appended* to: it resumes from the firn state in the
# file's restart group, fetches only the forcing after it, and skips the spinup entirely. So an
# interrupted run is resumed by re-running the same command, and a run against a longer ERA5-Land record
# costs the new months rather than the whole record.
#
# Needs CDS credentials (`~/.cdsapirc` or `ENV["CDS_API_KEY"]`) and the parameter tiles. Forcing is read
# from the shared Zarr cache under `CLIMATE_CACHE`, so tiles are visited in chunk order.
#
# Run one block:  julia --project=. -t 1 scripts/gemb_tile_sweep.jl [start_year] [end_year]
# Run the sweep:  scripts/run_tile_sweep.sh [start_year] [end_year]
#
# Environment overrides:  TILE_BLOCKS, TILE_BLOCK, TILE_LIMIT, TILE_NAMES (comma-separated), FORCE=1,
# SPINUP_SIMULATION_YEARS_MAXIMUM, SPINUP_CLIMATOLOGY_START, SPINUP_CLIMATOLOGY_STOP,
# PRECIPITATION_SCALINGS, DELTA_TEMPERATURES, FORCING_CACHE_GIB, DONOR_MAX_DISTANCE_KM
#
# Sizing: a 60-band tile over the 7x7 grid is 2,940 simulations, and one simulation is the spinup plus
# the transient. Both scale with the record: over a 7-year record a simulation was about 10 s, and the
# full ERA5-Land record is eleven times longer with a `:representative` spinup costing about 2.2x the
# model-years of an averaged one. Measure with a TILE_LIMIT or TILE_NAMES subset before committing to a
# global pass.

using GEMB_GlacierSims
using GEMB_ClimateForcing
using DataFrames
using Dates
using DimensionalData
using Statistics
import GEMB
import GeoDataFrames

include(joinpath(@__DIR__, "run_settings.jl"))

const GI = GeoDataFrames.GeoInterface
const CLIMATE_MODEL = :era5land
const PARQUET = joinpath(@__DIR__, "..", "data", "$(CLIMATE_MODEL)_glacier_elevation_classes.parquet")
const CLIMATE_CACHE = get(ENV, "CLIMATE_CACHE", joinpath("/mnt/bylot-r3/data", string(CLIMATE_MODEL)))
const PARAMETER_DIR = joinpath(CLIMATE_CACHE, "downscaling_parameters")
# Overridable so a timing probe on a reduced perturbation grid can share the forcing cache without
# writing files the production sweep would then have to recognise as a different experiment and rebuild.
const OUTPUT_DIR = get(ENV, "OUTPUT_DIR", joinpath(CLIMATE_CACHE, "tile_runs"))

# Must match the gridding the parameter tiles were derived on, or a tile's parameters describe a
# different neighbourhood than its cells. The parameter files record their own `tile_size`/`tile_buffer`,
# and the pre-flight below compares them.
const TILE_SIZE = 2
const BUFFER = 1

const START_YEAR = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1950
const END_YEAR = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2027
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

# Ceiling on the spinup, not the convergence test: bands exit on the drift criterion well inside this on
# glacier firn, so the ceiling only binds on an outlier.
#
# A span of simulated years rather than a cycle count, because a cycle count means a different amount of
# spinup on a different climatology. `gemb_spinup` divides by the measured cycle length to get its own
# ceiling: under `SPINUP_CLIMATOLOGY_METHOD = :representative` a cycle is `SPINUP_CLIMATOLOGY_N_YEARS`
# years, so 100 years is about 34 cycles. `SPINUP_DRIFT_WINDOW` cycles must run before the drift criterion
# can judge a slope, so the ceiling has to leave room for that — 34 does, 10 would not.
const SPINUP_SIMULATION_YEARS_MAXIMUM =
    parse(Float64, get(ENV, "SPINUP_SIMULATION_YEARS_MAXIMUM", "100"))
# Spinup exits when the FAC trend flattens; see `SPINUP_DRIFT_FAC` in `glacier_run.jl`, including why this
# value is too loose for an ice-sheet plateau. Stated per **year** and scaled to the cycle by
# `_spinup_drift_tolerance`, so 1e-2 is 1 cm of firn air per year whatever the cycle length.
const SPINUP_DRIFT_FAC = 1e-2

# The climatology the spinup repeats, fixed rather than derived from the run window. Fixing it is what
# makes an appended update defensible: were it tied to the record, extending the record would change the
# spinup and the new segment would not continue the old one. 1950-1980 is the earliest three decades
# ERA5-Land offers, so it is the closest this forcing comes to a pre-industrial reference. It must span at
# least `SPINUP_CLIMATOLOGY_N_YEARS`, which `_spinup_climatology` checks.
const SPINUP_CLIMATOLOGY_WINDOW = (
    DateTime(parse(Int, get(ENV, "SPINUP_CLIMATOLOGY_START", "1950")), 1, 1),
    DateTime(parse(Int, get(ENV, "SPINUP_CLIMATOLOGY_STOP", "1980")), 12, 31),
)

# How far a cell that carries ice but falls outside ERA5-Land's land mask may borrow forcing from; see
# `elevation_interval_forcing`. Donors are drawn from the tile's *buffered* neighbourhood rather than its
# core, so a tile whose every core cell is masked — the whole Antarctic coast, several Arctic
# archipelagos — still runs instead of failing. `0` drops those cells' ice instead.
const DONOR_MAX_DISTANCE_KM = parse(Float64, get(ENV, "DONOR_MAX_DISTANCE_KM", "50.0"))

# Slack on the forcing an append fetches, before the saved output time. The run trims to strictly after
# that time, so this only guards against the forcing grid not landing on it; one step would do, and a day
# costs nothing against a fetch measured in months.
# How often the transient is sampled. Weekly resolves the melt season, which monthly averages across:
# ablation is concentrated in a few weeks and a monthly mean spreads it over the whole month, so a
# comparison against altimetry binned finer than a month cannot see it. `output_period_bound` must know
# the period of whatever is chosen here.
#
# The cost is series size, not simulation time: the timestep is the forcing's either way, so this only
# changes how much of it is written — about 4.4x more steps than monthly over the full record.
const OUTPUT_FREQUENCY = Symbol(get(ENV, "OUTPUT_FREQUENCY", "weekly"))

const RESTART_FETCH_OVERLAP = Day(1)

# Where tile claims live, and how long before one is treated as abandoned. Under `OUTPUT_DIR` so a claim
# shares the filesystem of the file it guards — a claim on a different mount could outlive an output the
# sweep can no longer see.
#
# The bound has to exceed the longest a single tile can legitimately take: reclaiming a running tile would
# put two processes on one netCDF. A 63-band tile over the 7x7 grid and the full record is the worst case,
# so this is deliberately generous. Set `CLAIM_STALE_HOURS=0` to disable reclaiming entirely.
const CLAIM_SUBDIR = "claims"
const CLAIM_STALE_AFTER = Hour(parse(Int, get(ENV, "CLAIM_STALE_HOURS", "24")))

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

# Whether the parameter tiles this sweep reads carry an observed bare-ice albedo profile. Without one
# every band runs at the tuned constant `albedo_ice`, which the run records but does not warn about — so
# it belongs in the settings table, where a sweep about to reproduce the constant-albedo experiment is
# visible before it starts rather than afterwards.
#
# Sampled, not counted: storing a profile is a property of the derivation, which either did it for every
# tile it could or was run with it off, and a full census would be 819 header reads in each of 96 workers.
function albedo_profile_sample(n = 12)
    isdir(PARAMETER_DIR) || return "unknown — no parameter tiles yet"
    files = sort!(filter(endswith(".nc"), readdir(PARAMETER_DIR)))
    isempty(files) && return "unknown — no parameter tiles yet"
    # Sparse tiles are skipped, not counted against the total: they carry no fits at all and return
    # before the albedo is derived, so a profile is not something they can be missing. Counting them
    # would report a complete derivation as "mixed" purely by where the alphabet puts them.
    fitted = 0
    with = 0
    for f in files
        GEMB_GlacierSims.NCDatasets.NCDataset(joinpath(PARAMETER_DIR, f), "r") do ds
            get(ds.attrib, "sparse", "false") == "true" && return
            fitted += 1
            haskey(ds.dim, "bare_ice_albedo_bin") && (with += 1)
        end
        fitted >= n && break
    end
    fitted == 0 && return "unknown — the tiles sampled are all sparse, which carry no albedo"
    with == fitted && return "observed, MODIS ($(with)/$(fitted) fitted tiles sampled)"
    with == 0 && return "none stored — every band falls back to the default"
    return "mixed — $(with) of $(fitted) fitted tiles sampled carry a profile"
end

# What this run is configured to do, for the log and for the terminal that launched it.
function settings_rows()
    return [
        setting("time range", TIME_RANGE, length(ARGS) >= 1 ? "ARGS" : "default"),
        setting("tile size / buffer", "$(TILE_SIZE)° / $(BUFFER)°"),
        env_setting("output frequency", OUTPUT_FREQUENCY, "OUTPUT_FREQUENCY"),
        env_setting("temperature offsets (K)", DELTA_TEMPERATURES, "DELTA_TEMPERATURES"),
        env_setting("precipitation scalings", PRECIPITATION_SCALINGS, "PRECIPITATION_SCALINGS"),
        setting("perturbations per band",
                "$(length(DELTA_TEMPERATURES)) × $(length(PRECIPITATION_SCALINGS)) = " *
                "$(length(DELTA_TEMPERATURES) * length(PRECIPITATION_SCALINGS))"),
        setting("bare-ice albedo", albedo_profile_sample(), "parameter tiles"),
        setting("spinup climatology", SPINUP_CLIMATOLOGY_WINDOW,
                haskey(ENV, "SPINUP_CLIMATOLOGY_START") || haskey(ENV, "SPINUP_CLIMATOLOGY_STOP") ?
                "ENV[SPINUP_CLIMATOLOGY_START/STOP]" : "default"),
        env_setting("spinup ceiling", "$(SPINUP_SIMULATION_YEARS_MAXIMUM) simulated years",
                    "SPINUP_SIMULATION_YEARS_MAXIMUM"),
        setting("spinup drift tolerance", "$(SPINUP_DRIFT_FAC) m firn air / year"),
        env_setting("masked-cell donor range", "$(DONOR_MAX_DISTANCE_KM) km", "DONOR_MAX_DISTANCE_KM"),
        env_setting("worker blocks", TILE_BLOCKS, "TILE_BLOCKS"),
        # From JLOptions, not ENV: this is the value the collector acts on, so a hint lost on its way to
        # the worker shows up here as "none" instead of being reported as if it had taken effect.
        let hint = Base.JLOptions().heap_size_hint
            hint == 0 ?
                setting("GC heap hint", "none (target sized from physical memory)") :
                setting("GC heap hint", "$(round(hint / 2^30; digits = 1)) GiB", "--heap-size-hint")
        end,
        env_setting("claim stale after", "$(Dates.value(CLAIM_STALE_AFTER)) h", "CLAIM_STALE_HOURS"),
        env_setting("tile limit", TILE_LIMIT == typemax(Int) ? "all" : TILE_LIMIT, "TILE_LIMIT"),
        env_setting("tiles", isempty(TILE_NAMES) ? "all" : join(TILE_NAMES, ", "), "TILE_NAMES"),
        env_setting("force rebuild of current tiles", FORCE, "FORCE"),
        env_setting("climate cache", CLIMATE_CACHE, "CLIMATE_CACHE"),
        setting("parameters", PARAMETER_DIR),
        env_setting("output", OUTPUT_DIR, "OUTPUT_DIR"),
    ]
end

function main()
    print_settings("GEMB tile sweep — settings in force", settings_rows())
    SHOW_SETTINGS && return nothing

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
    # Band count is what the block split is balanced on, and what stealing orders by: the perturbation
    # grid is the same for every tile, so simulations per tile is proportional to it, and hypsometry
    # comes from the already-loaded table with no forcing read. It spans 1 to 63 across the runnable
    # tiles (median 12), so balancing on it beats equal-length blocks substantially.
    #
    # It is a proxy, not the cost: a tile's wall time also carries a CDS download and a spinup whose
    # cycle count varies per column, neither proportional to band count. That gap is what
    # `claim_order`'s steal pass covers, and why the block split alone must not be relied on to
    # equalise workers.
    weights = Float64[length(hypsometry_intervals(t.core)) for t in selected]
    order = claim_order(weights, TILE_BLOCK, TILE_BLOCKS)
    if TILE_BLOCKS > 1
        affinity = weight_balanced_blocks(weights, TILE_BLOCKS)[TILE_BLOCK]
        @info "Tile block" block="$TILE_BLOCK/$TILE_BLOCKS" affinity=length(affinity) range="$affinity of $(length(selected))" bands=Int(sum(weights[affinity])) bands_mean_per_block=round(sum(weights) / TILE_BLOCKS; digits = 1) steal_candidates=(length(selected) - length(affinity))
    end

    mkpath(OUTPUT_DIR)
    claim_dir = joinpath(OUTPUT_DIR, CLAIM_SUBDIR)
    mkpath(claim_dir)
    mp = GEMB.initialize_parameters(output_frequency = OUTPUT_FREQUENCY)

    @info "GEMB tile sweep" candidates=length(order) of=length(tiles) block="$TILE_BLOCK/$TILE_BLOCKS" time_range=TIME_RANGE output_frequency=OUTPUT_FREQUENCY perturbations="$(length(DELTA_TEMPERATURES))x$(length(PRECIPITATION_SCALINGS))" threads=Threads.nthreads() output=OUTPUT_DIR

    rows = NamedTuple[]
    t_start = time()
    for i in order
        tile = selected[i]
        name = replace(tile.name, ".nc" => "")
        path = joinpath(OUTPUT_DIR, tile.name)
        # Whoever creates the claim owns the tile. Losing the race costs a directory lookup.
        claim_tile!(claim_dir, tile.name, path; stale_after = CLAIM_STALE_AFTER) || continue
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
        @warn "This worker claimed no tiles; every candidate was already taken" TILE_LIMIT TILE_NAMES block="$TILE_BLOCK/$TILE_BLOCKS"
        return DataFrame()
    end

    summary = DataFrame(rows)
    # One summary per block, or concurrent processes would overwrite each other's. A single-block run
    # keeps the unsuffixed name so nothing that reads it has to know about blocks.
    suffix = TILE_BLOCKS == 1 ? "" : "_block$(lpad(TILE_BLOCK, 3, '0'))of$(TILE_BLOCKS)"
    GEMB_GlacierSims.Parquet2.writefile(
        joinpath(OUTPUT_DIR, "tile_runs_summary$suffix.parquet"), summary)

    @info "Sweep finished" minutes=round((time() - t_start) / 60; digits = 1) written=count(==("written"), summary.status) extended=count(==("extended"), summary.status) skipped=count(==("skipped"), summary.status) empty=count(==("empty"), summary.status) no_parameters=count(==("no_parameters"), summary.status) failed=count(==("failed"), summary.status)

    # Both outcomes produced a run this pass, so both carry closure and volume diagnostics worth
    # reporting; an extended tile's are for its new segment only.
    done = summary[in.(summary.status, Ref(("written", "extended"))), :]
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
    # Must carry the same settings the run below records, or a file written under one compares as current
    # against another and the change never takes effect. `run_parameter_differences` only compares keys
    # present in *both* dicts, so a setting omitted here is a setting the skip test cannot notice —
    # which is why the spinup ceiling and tolerance are passed and not left to default to `nothing`.
    requested = tile_run_parameters(mp, applied;
                                   spinup_window = SPINUP_CLIMATOLOGY_WINDOW,
                                   simulation_years_maximum = SPINUP_SIMULATION_YEARS_MAXIMUM,
                                   convergence_drift_fac = SPINUP_DRIFT_FAC,
                                   donor_max_distance_km = DONOR_MAX_DISTANCE_KM,
                                   climatology_method = GEMB_GlacierSims.SPINUP_CLIMATOLOGY_METHOD,
                                   climatology_n_years = GEMB_GlacierSims.SPINUP_CLIMATOLOGY_N_YEARS)

    status = read_glacier_tile_status(path)
    disposition = FORCE ? :rebuild :
                  tile_run_disposition(status, requested, probe_time, intervals, mp)
    if disposition === :current
        @info "Tile already covers the request; skipping" tile=i name last_time=status.time
        return summary_row(tile, name, :skipped, time() - t0)
    end

    # An extendable file is only extendable if it actually carries a firn state; one written before the
    # restart group existed has to be rebuilt.
    restart = disposition === :extendable ? read_glacier_tile_restart(path) : nothing
    appending = restart !== nothing && restart.time !== nothing
    if disposition === :extendable && !appending
        @info "Tile carries no restart state; rebuilding instead of appending" tile=i name
    end

    # Appending needs only the forcing after the saved state. `RESTART_FETCH_OVERLAP` of slack costs one
    # extra step and covers the forcing grid not landing exactly on the saved output time; the run trims
    # to `t > restart.time` regardless. This is the whole point of the restart path — a few months of
    # forcing and no spinup, against 76 years and a spinup on a rebuild.
    fetch_range = appending ?
                  (max(first(TIME_RANGE), restart.time - RESTART_FETCH_OVERLAP), last(TIME_RANGE)) :
                  TIME_RANGE

    # Area comes from the core cells, donors from the buffered neighbourhood: a masked core cell keeps
    # its ice by borrowing a neighbour's forcing, and the neighbour contributes no area of its own, so
    # nothing is double counted against the adjacent tile.
    bands = collect(elevation_interval_forcing(tile.core, applied;
                                               climate_model = CLIMATE_MODEL,
                                               time_range = fetch_range, token, cache_path = cache,
                                               elevation_interval_batch = 0,
                                               donor_cells = tile.buffered,
                                               donor_max_distance_km = DONOR_MAX_DISTANCE_KM))
    t_forcing = time() - t0

    local run
    try
        run = gemb_glacier_tile(tile, applied, bands, mp;
                                delta_temperatures = DELTA_TEMPERATURES,
                                precipitation_scalings = PRECIPITATION_SCALINGS,
                                spinup_window = SPINUP_CLIMATOLOGY_WINDOW,
                                simulation_years_maximum = SPINUP_SIMULATION_YEARS_MAXIMUM,
                                convergence_drift_fac = SPINUP_DRIFT_FAC,
                                donor_max_distance_km = DONOR_MAX_DISTANCE_KM,
                                restart = appending ? restart : nothing)
    catch err
        # The record grew by less than one usable step. Not a failure: the file already says everything
        # this forcing can.
        err isa ForcingUpToDate || rethrow()
        @info "Tile is already up to date with the forcing; skipping" tile=i name last_time=restart.time
        return summary_row(tile, name, :skipped, time() - t0)
    end
    t_gemb = time() - t0 - t_forcing

    if appending
        append_glacier_tile_netcdf(path, run)
    else
        write_glacier_tile_netcdf(path, run; institution = "NASA Jet Propulsion Laboratory")
    end

    @info(appending ? "Extended tile" : "Wrote tile", tile=i, name, bands=length(run.bands), simulations=length(run.bands)*length(DELTA_TEMPERATURES)*length(PRECIPITATION_SCALINGS), steps=length(run.time), forcing_s=round(t_forcing; digits = 1), gemb_s=round(t_gemb; digits = 1), basis, substituted_area_km2=round(run.provenance["substituted_area_km2"]; digits = 2), unrecovered_area_km2=round(run.provenance["unrecovered_area_km2"]; digits = 2))
    return summary_row(tile, name, appending ? :extended : :written, time() - t0;
                       run, t_forcing, t_gemb, basis)
end

# The authoritative run time axis, from one cell's forcing rather than assumed to be hourly. Also the
# earliest point a wrong token or a cold cache surfaces, which is worth paying before a tile-wide pass.
function probe_run_time(tile, token, cache)
    row = first(tile.core)
    fd = climate_forcing(CLIMATE_MODEL, row.latitude, row.longitude;
                         time_range = TIME_RANGE, token, cache_path = cache)
    return collect(dims(fd, Ti))
end

# The longest span one output period can cover. An upper bound is the safe direction for the window test
# below: understating it would declare a complete file short and re-run it, which is the failure this
# exists to prevent.
#
# `nothing` is reserved for `:all`, where every output step *is* a forcing step and the window end compares
# exactly. Any coarser frequency must name its period here: a frequency that falls through to `nothing`
# would be compared exactly against the last forcing step, which a coarser grid never reaches, so every
# such file would read as short and be re-run or re-appended forever.
function output_period_bound(mp)
    f = mp.output_frequency
    f === :monthly && return Day(31)
    f === :weekly && return Day(7)
    f === :daily && return Day(1)
    f === :all && return nothing
    # `:last` leaves a single step and is only used inside the spinup, never for a tile run; anything else
    # is a frequency added to GEMB without a period given here, and silently comparing it exactly is the
    # bug described above.
    error("output_period_bound: no period known for output_frequency = :$f; add one before using it " *
          "for a tile run, or the skip test will treat every file as short")
end

# What an existing tile file is worth to this request:
#
#   `:current`    — it already covers the record; nothing to do.
#   `:extendable` — the same experiment, but the record has grown since. Resume from its firn state and
#                   append, rather than repeating a spinup and a whole transient to add a few months.
#   `:rebuild`    — a different experiment, or nothing usable.
#
# Decided from the file's coordinates and attributes only: the point of the check is to avoid the forcing
# pass, so it must not read one. Splitting `:extendable` out from `:rebuild` is what makes an update
# cheap; before, both were "not current" and both rebuilt.
function tile_run_disposition(status, requested, run_time, intervals, mp)
    status === nothing && return :rebuild
    status.n_timesteps == 0 && return :rebuild
    status.time === nothing && return :rebuild
    # The run grid: a changed band set or perturbation grid means the stored arrays describe something
    # else entirely, and no seam could join them.
    status.band_centers == [x.center for x in intervals] || return :rebuild
    status.delta_temperatures == DELTA_TEMPERATURES || return :rebuild
    status.precipitation_scalings == PRECIPITATION_SCALINGS || return :rebuild
    # The settings: a changed model parameter or downscaling policy makes it a different experiment, which
    # must not be spliced onto an old record.
    isempty(status.parameters) && return :rebuild
    isempty(run_parameter_differences(status.parameters, requested)) || return :rebuild

    # Same experiment. Does it already reach the end of the record? The stored time is the last *output*
    # sample while `run_time` is the last *forcing* step, and a coarser output grid never reaches the
    # forcing's final step — monthly output for a window ending 2023-01-01T00:00 lands on
    # 2022-12-31T23:00. Comparing the two directly is what made every file look stale.
    #
    # The tolerance means a record extended by less than one output period counts as covered: such an
    # extension spans no further output interval, so re-running would reproduce the same series.
    period = output_period_bound(mp)
    covered = period === nothing ? status.time >= last(run_time) :
              status.time + period >= last(run_time)
    return covered ? :current : :extendable
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
            # The tile's ice according to its hypsometry, whether or not it was modelled. Recorded
            # alongside the modelled area so the shortfall is one subtraction in the summary rather
            # than a join against the elevation-class table, and so it is visible for every cause —
            # including a band dropped whole, which reaches no band-level counter.
            hypsometry_area_km2 = sum(glacier_area_column(tile.core)),
            substituted_area_km2 = run === nothing ? 0.0 :
                                   run.provenance["substituted_area_km2"],
            unrecovered_area_km2 = run === nothing ? 0.0 :
                                   run.provenance["unrecovered_area_km2"],
            max_donor_distance_km = run === nothing ? 0.0 :
                                    run.provenance["max_donor_distance_km"],
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
