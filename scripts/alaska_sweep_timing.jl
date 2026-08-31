# Timed GEMB sweep over Alaska glacier grid cells.
#
# Sweep: precipitation scalings [0.5, 1.0, 2.0] x temperature offsets [-1.0, 0.0, 1.0]
# (9 perturbations) for every ERA5-Land grid cell in the Alaska bounding box holding at least
# `cell_area_minimum` of glacier ice, over the hypsometry bins covering `hypsometry_coverage`
# of each cell's glacier area.
#
# The point of this script is the timing: each cell is split into forcing download, GEMB run
# (spinup + integration for bins x 9), and NetCDF write, and each phase is timed separately.
# It prints a per-cell line and, at the end, per-simulation costs and an extrapolation to the
# full Alaska cell count.
#
# Cells are grouped by `chunk_id` before the parallel loop. `chunk_id` is the ERA5-Land Zarr
# *chunk* a cell's forcing lives in (`GEMB_ClimateForcing/src/datasets/era5_land.jl`, where it
# is built straight from the Zarr array's chunk shape), and the `CachingStore` under
# `cache_path` holds one file per chunk. Grouping therefore buys two things:
#
#   1. Cache reuse. Every cell in a group reads the same chunk files, so the chunk is fetched
#      once and the remaining cells in the group hit warm cache. In Alaska that is a mean of
#      13.6 cells per download instead of one.
#   2. No cache conflicts. Distinct groups touch disjoint chunk files, so running one group per
#      worker means no two workers ever write the same cache entry — the `CachingStore` has no
#      cross-process locking, so two workers pulling the same chunk concurrently would race on
#      the same path.
#
# Parallelism is therefore over *groups*, with the cells inside a group run serially: that is
# what keeps a chunk owned by exactly one worker.
#
# Usage:
#   ALASKA_CELL_LIMIT=4 julia --project scripts/alaska_sweep_timing.jl
#   ALASKA_CELL_LIMIT=all julia -t 8 --project scripts/alaska_sweep_timing.jl   # whole region
#
# Env:
#   ALASKA_CELL_LIMIT   number of cells to run, or "all"  (default 3)
#   ALASKA_OUTPUT_DIR   where the per-cell NetCDFs go     (default data/gemb_runs/alaska_timing)
#   ALASKA_SAMPLE       "first" (default) or "random": how a partial cell set is drawn
#   ALASKA_SEED         RNG seed for the random draw       (default 1234)
#   ALASKA_PARALLEL     "cell" (default) or "sim": which level gets the threads
#   CLIMATE_CACHE       root of the persistent forcing cache (default /mnt/bylot-r3/data/era5land)
#
# Threads come from `julia -t N`, and go to exactly one of two levels (never both, which would
# oversubscribe the machine):
#
#   ALASKA_PARALLEL=cell  threads run different cells, one chunk group each. Use for a big sweep.
#   ALASKA_PARALLEL=sim   threads run the (bin x delta x scaling) simulations inside one cell.
#                         Use when there are fewer cells than cores — a single cell, or a handful.

using Dates
using Printf
using Random
using Statistics
using GeoDataFrames
using GEMB
using GEMB_GlacierSims
using GEMB_ClimateForcing
using DimensionalData
using Rasters

# --- sweep configuration ---------------------------------------------------------------------
const CLIMATE_MODEL           = :era5land
const PRECIPITATION_SCALINGS  = [0.5, 1.0, 2.0]     # precipitation multipliers (1)
const DELTA_TEMPERATURES      = [-1.0, 0.0, 1.0]    # air temperature offsets (K)
const HYPSOMETRY_COVERAGE     = 0.95
const CELL_AREA_MINIMUM       = 1.0                 # km² of glacier ice per cell
const FORCING_TIME_RANGE      = (DateTime(1950, 1, 1), DateTime(2026, 8, 1))

# Alaska bounding box (deg). Wider than RGI region 01 proper at the eastern edge, so it also
# picks up the Yukon/NW-British-Columbia cells that share the same icefields.
const ALASKA_BBOX = (lon = (-170.0, -129.0), lat = (54.0, 72.0))

const CELL_LIMIT = let v = get(ENV, "ALASKA_CELL_LIMIT", "3")
    lowercase(v) in ("all", "0", "nothing") ? nothing : parse(Int, v)
end

const CLASSES_FILE = joinpath(@__DIR__, "..", "data",
                              "$(CLIMATE_MODEL)_glacier_elevation_classes.parquet")
const OUTPUT_DIR = get(ENV, "ALASKA_OUTPUT_DIR",
                       joinpath(@__DIR__, "..", "data", "gemb_runs", "alaska_timing"))
# Shared persistent forcing cache: the ERA5-Land chunks are tens of GB and expensive to re-fetch, so
# they live off `tempdir()` and survive a reboot.
const CLIMATE_CACHE = get(ENV, "CLIMATE_CACHE",
                          joinpath("/mnt/bylot-r3/data", string(CLIMATE_MODEL)))
const FORCING_CACHE = joinpath(CLIMATE_CACHE, "cache")

# Same filename convention as `src/era5_example.jl`, so a cell file is traceable to its cell.
_degrees_tag(x) = replace(string(round(x, digits = 3)), '.' => 'p', '-' => 'm')
cell_output_path(r) = joinpath(OUTPUT_DIR,
    "gemb_cell_" * lpad(r.chunk_id, 6, '0') * "_" *
    _degrees_tag(r.latitude) * "_" * _degrees_tag(wrap_lon(r.longitude)) * ".nc")

isfile(CLASSES_FILE) || error("missing $(CLASSES_FILE); run src/era5_example.jl to build it")
mkpath(OUTPUT_DIR)

@info "Loading glacier elevation-class table" CLASSES_FILE
classes = GeoDataFrames.read(CLASSES_FILE)
classes[!, :longitude] = GeoDataFrames.GeoInterface.x.(classes.geometry)
classes[!, :latitude]  = GeoDataFrames.GeoInterface.y.(classes.geometry)

# --- cell selection --------------------------------------------------------------------------
rows = collect(eachrow(classes))
in_alaska(r) = ALASKA_BBOX.lon[1] <= wrap_lon(r.longitude) <= ALASKA_BBOX.lon[2] &&
               ALASKA_BBOX.lat[1] <= r.latitude <= ALASKA_BBOX.lat[2]

alaska_all = [i for i in eachindex(rows) if in_alaska(rows[i])]
qualifying = [i for i in alaska_all if glacier_area_total(rows[i]) >= CELL_AREA_MINIMUM]

# Bin counts drive the simulation count, so report them before running anything.
bins_per_cell = [length(glacier_hypsometry_coverage(rows[i]; coverage = HYPSOMETRY_COVERAGE).modeled)
                 for i in qualifying]
n_perturbations = length(DELTA_TEMPERATURES) * length(PRECIPITATION_SCALINGS)

@info "Alaska sweep scope" cells_in_bbox=length(alaska_all) qualifying=length(qualifying) glacier_area_km2=round(sum(glacier_area_total(rows[i]) for i in qualifying), digits=1) bins_total=sum(bins_per_cell) bins_median=median(bins_per_cell) bins_max=maximum(bins_per_cell) perturbations=n_perturbations simulations_total=n_perturbations*sum(bins_per_cell)

# The qualifying cells are in table order, which is spatially clustered (the table is built
# chunk by chunk), so `first(...)` samples one corner of the region. For a timing estimate that
# extrapolates to all of Alaska, prefer a random draw: spinup cost varies with the column, and
# maritime and interior columns are not interchangeable.
sample_mode = lowercase(get(ENV, "ALASKA_SAMPLE", "first"))
sample_mode in ("first", "random") ||
    error("ALASKA_SAMPLE must be \"first\" or \"random\", got $(sample_mode)")

positions = if CELL_LIMIT === nothing
    eachindex(qualifying)
elseif sample_mode == "random"
    sort(randperm(MersenneTwister(parse(Int, get(ENV, "ALASKA_SEED", "1234"))),
                  length(qualifying))[1:min(CELL_LIMIT, length(qualifying))])
else
    1:min(CELL_LIMIT, length(qualifying))
end
selected = qualifying[positions]

# --- group by chunk_id -----------------------------------------------------------------------
# One group per ERA5-Land Zarr chunk. The group is the unit of parallelism: its cells share
# cache files, so they must stay on one worker (see the header note).
chunk_groups = let groups = Dict{Int,Vector{Int}}()
    for i in selected
        push!(get!(groups, Int(rows[i].chunk_id), Int[]), i)
    end
    # Largest group first: with a few long groups among many short ones, descending order keeps
    # the tail from being a single huge group finishing alone after every worker has drained.
    sort!(collect(groups); by = g -> -length(g.second))
end

cells_per_group = [length(g.second) for g in chunk_groups]

# Where to spend the threads. There are two independent levels of parallelism and using both at
# once oversubscribes the machine, so pick one:
#
#   :cell (default) — threads run different cells (one chunk group each). Best for a large sweep:
#                     there are far more cells than cores, so every thread stays busy, and each
#                     chunk stays owned by one worker.
#   :sim            — cells run one at a time, threads run the (bin x delta x scaling)
#                     simulations within each cell. Best for a small sweep (fewer cells than
#                     cores, or a single cell), where cell-level threading would idle most cores.
#
# Only one level is ever active: with :cell the inner `gemb_glacier_cell` is called with
# `threaded=false`, and with :sim the group loop is serial.
const PARALLEL_LEVEL = Symbol(lowercase(get(ENV, "ALASKA_PARALLEL", "cell")))
PARALLEL_LEVEL in (:cell, :sim) ||
    error("ALASKA_PARALLEL must be \"cell\" or \"sim\", got $(PARALLEL_LEVEL)")

const THREAD_CELLS = PARALLEL_LEVEL === :cell
const THREAD_SIMS  = PARALLEL_LEVEL === :sim

@info "Running" n_cells=length(selected) sample=sample_mode threads=Threads.nthreads() parallel_level=PARALLEL_LEVEL chunk_groups=length(chunk_groups) cells_per_group_mean=round(mean(cells_per_group), digits=1) cells_per_group_max=maximum(cells_per_group) simulations=n_perturbations*sum(bins_per_cell[positions])

# --- timed sweep -----------------------------------------------------------------------------
fmt_hms_inline(s) = (h = floor(Int, s / 3600); m = floor(Int, (s % 3600) / 60);
                     @sprintf("%dh %02dm %02ds", h, m, round(Int, s % 60)))

mp = initialize_parameters(output_frequency = :monthly)

records = NamedTuple[]      # one entry per cell that produced output
skipped = NamedTuple[]      # cells with no land forcing / already up to date / failed
results_lock = ReentrantLock()   # `records`/`skipped` are appended from every worker
done = Threads.Atomic{Int}(0)

# The CDS key is read once here rather than per cell: `get_cds_api_key` touches ~/.cdsapirc, and
# every worker calling it inside the loop would hammer the same file for a value that never
# changes.
const CDS_TOKEN = GEMB_ClimateForcing.get_cds_api_key()

t_sweep = time()

# One chunk group: its cells run serially so the group's cache files stay owned by one worker.
function run_chunk_group(chunk_id, group_cells)
    t_group = time()

    for i in group_cells
    r = rows[i]
    path = cell_output_path(r)
    try
        restart = read_glacier_cell_restart(path)

        t0 = time()
        forcing_data = climate_forcing(CLIMATE_MODEL, r.latitude, r.longitude;
                                       time_range = FORCING_TIME_RANGE,
                                       token = CDS_TOKEN,
                                       cache_path = FORCING_CACHE)
        t_forcing = time() - t0

        t0 = time()
        run = gemb_glacier_cell(r, forcing_data, mp;
                                delta_temperatures = DELTA_TEMPERATURES,
                                precipitation_scalings = PRECIPITATION_SCALINGS,
                                coverage = HYPSOMETRY_COVERAGE,
                                restart, threaded = THREAD_SIMS)
        t_run = time() - t0

        t0 = time()
        if restart === nothing
            write_glacier_cell_netcdf(path, run;
                                      institution = "NASA Jet Propulsion Laboratory")
        else
            append_glacier_cell_netcdf(path, run)
        end
        t_write = time() - t0

        n_sim = length(run.bins) * n_perturbations
        rec = (; cell = i, chunk_id, path, bins = length(run.bins), n_sim,
               area_km2 = sum(run.weights), steps = length(run.time),
               resumed = restart !== nothing,
               t_forcing, t_run, t_write, t_total = t_forcing + t_run + t_write)

        n = Threads.atomic_add!(done, 1) + 1
        @lock results_lock begin
            push!(records, rec)
            # Printed under the lock so concurrent workers cannot interleave one line.
            @printf("[%d/%d] chunk %6d cell %6d  %2d bins x %d = %3d sims | forcing %6.1fs  run %8.1fs  write %5.2fs  total %8.1fs  (%.2f s/sim)\n",
                    n, length(selected), chunk_id, i, rec.bins, n_perturbations, n_sim,
                    t_forcing, t_run, t_write, rec.t_total, t_run / n_sim)
            flush(stdout)
        end
    catch e
        e isa InterruptException && rethrow()
        reason = e isa ForcingUnavailable ? :no_land_forcing :
                 e isa ForcingUpToDate    ? :up_to_date : :failed
        Threads.atomic_add!(done, 1)
        @lock results_lock push!(skipped, (; cell = i, chunk_id, reason))
        if reason === :failed
            @error "Cell failed; continuing" cell=i path exception=(e, catch_backtrace())
        else
            @info "Skipping cell" cell=i reason lat=r.latitude lon=r.longitude
        end
    end
    end  # cells within group

    @lock results_lock begin
        @printf("  ↳ chunk %6d complete: %2d cells in %s\n",
                chunk_id, length(group_cells), fmt_hms_inline(time() - t_group))
        flush(stdout)
    end
    return nothing
end

if THREAD_CELLS
    # `:dynamic` (the default) lets a worker that finishes a short group pick up the next one,
    # which matters because per-cell cost spans 10x.
    Threads.@threads for gi in eachindex(chunk_groups)
        run_chunk_group(chunk_groups[gi]...)
    end
else
    # Cells serial; the threads are inside `gemb_glacier_cell` instead.
    for gi in eachindex(chunk_groups)
        run_chunk_group(chunk_groups[gi]...)
    end
end
t_sweep = time() - t_sweep

# --- timing summary --------------------------------------------------------------------------
const fmt_hms = fmt_hms_inline

println("\n", "="^92)
println("TIMING SUMMARY — Alaska, ΔT ∈ $(DELTA_TEMPERATURES) K, P × $(PRECIPITATION_SCALINGS)")
println("="^92)

if isempty(records)
    println("No cell produced output; nothing to time. Skipped: ", skipped)
else
    n_cells = length(records)
    sims    = sum(r.n_sim for r in records)
    tf, tr, tw = sum(r.t_forcing for r in records), sum(r.t_run for r in records), sum(r.t_write for r in records)

    @printf("cells run                 : %d (%d skipped: %s)\n", n_cells, length(skipped),
            isempty(skipped) ? "-" : join(unique(s.reason for s in skipped), ", "))
    @printf("simulations               : %d  (%d bins x %d perturbations)\n",
            sims, sum(r.bins for r in records), n_perturbations)
    @printf("output steps per run      : %d\n", records[1].steps)
    @printf("wall clock                : %s  (%d threads, parallel over %s, %d chunk groups)\n",
            fmt_hms(t_sweep), Threads.nthreads(),
            THREAD_CELLS ? "cells" : "sims within a cell", length(chunk_groups))
    println("-"^92)

    # Cache effectiveness: within a chunk group only the first cell pays the download, so the
    # first-cell and later-cell forcing times should differ by orders of magnitude. If they do
    # not, the grouping is not buying what it should.
    by_chunk = Dict{Int,Vector{Float64}}()
    for r in records
        push!(get!(by_chunk, r.chunk_id, Float64[]), r.t_forcing)
    end
    firsts = [first(v) for v in values(by_chunk)]
    laters = reduce(vcat, [v[2:end] for v in values(by_chunk) if length(v) > 1]; init = Float64[])
    @printf("forcing, first cell/chunk : %7.2f s  (n=%d, cold chunk fetch)\n",
            mean(firsts), length(firsts))
    if isempty(laters)
        println("forcing, later cells      :       -    (no chunk group had >1 cell in this sample)")
    else
        @printf("forcing, later cells      : %7.2f s  (n=%d, warm cache — %.0fx faster)\n",
                mean(laters), length(laters), mean(firsts) / max(mean(laters), 1e-9))
    end
    println("-"^92)
    # Percentages are of summed *worker* time, not wall clock: with N threads the phases sum to
    # roughly N x wall clock, so dividing by wall clock would exceed 100%.
    t_work = tf + tr + tw
    @printf("(phase shares are of %s summed worker time%s)\n", fmt_hms(t_work),
            THREAD_CELLS ? " across $(Threads.nthreads()) threads" : "")
    @printf("forcing download/read     : %8.1f s  (%5.1f%%)  %7.2f s/cell\n", tf, 100tf/t_work, tf/n_cells)
    @printf("GEMB runs (spinup + integ): %8.1f s  (%5.1f%%)  %7.2f s/cell  %6.3f s/sim\n",
            tr, 100tr/t_work, tr/n_cells, tr/sims)
    @printf("NetCDF write              : %8.1f s  (%5.1f%%)  %7.3f s/cell\n", tw, 100tw/t_work, tw/n_cells)
    println("-"^92)
    @printf("per cell   : mean %7.1f s   median %7.1f s   min %7.1f s   max %7.1f s\n",
            mean(r.t_total for r in records), median([r.t_total for r in records]),
            minimum(r.t_total for r in records), maximum(r.t_total for r in records))
    @printf("per sim    : mean %7.3f s   median %7.3f s\n",
            mean(r.t_run / r.n_sim for r in records), median([r.t_run / r.n_sim for r in records]))

    # Extrapolate on simulation count, not cell count: bins per cell vary 1..38, so cost scales
    # with bins, and the sampled cells are not a random draw from that distribution. Forcing is
    # charged per *chunk group* rather than per cell, since that is what a download costs once
    # cells are grouped.
    s_per_sim   = tr / sims
    s_per_chunk = mean(firsts)
    sims_all    = n_perturbations * sum(bins_per_cell)
    all_groups  = length(unique(Int(rows[i].chunk_id) for i in qualifying))
    est_all     = s_per_sim * sims_all + s_per_chunk * all_groups
    println("-"^92)
    @printf("EXTRAPOLATION to all of Alaska (%d cells, %d groups, %d bins, %d sims):\n",
            length(qualifying), all_groups, sum(bins_per_cell), sims_all)
    @printf("  serial, 1 thread         : %s\n", fmt_hms(est_all))
    for np in (8, 16, 32, 64)
        # Groups are the parallel unit, so speedup is capped by the group count, not the core
        # count — irrelevant for Alaska's 268 groups, but it bites on a small region.
        eff = min(np, all_groups)
        @printf("  %2d-way parallel (ideal)  : %s%s\n", np, fmt_hms(est_all / eff),
                eff < np ? "  (capped: only $(all_groups) groups)" : "")
    end
    println("  (grouping by chunk_id charges each chunk download once, not once per cell)")
end
println("="^92)
