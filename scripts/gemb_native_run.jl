# Run GEMB once, globally, at the native resolution of the forcing: every glacierized ERA5-Land cell on
# its own forcing and its own hypsometry, under the calibrated forcing correction of its 2° geotile.
#
# Each cell's elevation bands are a GEMB column of their own, forced with that cell's forcing lapsed to
# the band center and decoupled by its tile's downscaling (`gemb_native_cell`). The correction — the
# precipitation scaling and temperature offset GlobalGlacierAnalysis.jl's `gemb_calibration` fitted per
# geotile against altimetry, over the (ΔT × pscale) grid of the tile sweep — is applied after the
# downscaling, as it was in the runs it was fitted to. So this is the tile sweep at one perturbation
# point per tile, with the cells run separately instead of averaged.
#
# One tile-format netCDF per cell, `OUTPUT_DIR/<tile>/<cell>.nc`, whose one-element `delta_temperature`
# and `precipitation_scaling` axes hold the correction. The unit of work is a run of consecutive cells of
# one tile holding at most UNIT_COLUMNS columns, claimed by an atomic mkdir; a unit gains a
# `<unit>.complete` marker in its tile's directory once every cell is written or accounted for. The
# marker records the unit's ice and the part of it left unmodelled.
#
# Each cell runs the bands its tile's reference run modelled, so the native and tile runs drop the same
# intervals. The 1 % lost-ice rule is applied to the tile, not to each cell: once every unit of a tile
# has its marker, the unit that completes it sums the markers, and a tile above the rule is reported as
# failed and gains a `<tile>.lost_ice_exceeded` file beside the markers, which `run_native_sweep.sh`
# counts as a failure on every launch until the tile's markers are removed and it passes.
#
# Resumable and extendable as the tile sweep is: a cell file that already covers the request under the
# same settings and correction is skipped from its metadata, one whose record has since grown is appended
# to from its saved firn state, and anything else is rebuilt.
#
# Settings must match the tile sweep the correction was fitted against — anything else applies the
# correction to a different model. Every tile is therefore checked against its file in
# REFERENCE_TILE_DIR before any forcing is read, and a disagreement throws.
#
# Run one block:  julia --project=. -t 1 scripts/gemb_native_run.jl [start] [end]
# Run them all:   scripts/run_native_sweep.sh [start] [end]
#
# `start` and `end` are read as the tile sweep reads them (`gemb_tile_sweep.jl`): a year or an ISO
# date-time, defaulting to 1950 and the end of the cached ERA5-Land record. The process exits with
# status 1 when any cell or tile it ran failed.
#
# Environment overrides:  NATIVE_BLOCKS, NATIVE_BLOCK, TILE_LIMIT, TILE_NAMES (comma-separated), FORCE=1,
# FORCING_CORRECTIONS, REFERENCE_TILE_DIR, OUTPUT_DIR, CORRECTION_MAX_DISTANCE_KM, CLAIM_STALE_HOURS,
# UNIT_COLUMNS, DONOR_MAX_DISTANCE_KM

using GEMB_GlacierSims
using GEMB_ClimateForcing
using DataFrames
using Dates
using DimensionalData
using Statistics
import GEMB
import GeoDataFrames
import TOML

include(joinpath(@__DIR__, "run_settings.jl"))

const GI = GeoDataFrames.GeoInterface
const CLIMATE_MODEL = :era5land
const PARQUET = joinpath(@__DIR__, "..", "data", "$(CLIMATE_MODEL)_glacier_elevation_classes.parquet")
const CLIMATE_CACHE = get(ENV, "CLIMATE_CACHE", joinpath("/mnt/bylot-r3/data", string(CLIMATE_MODEL)))
const PARAMETER_DIR = joinpath(CLIMATE_CACHE, "downscaling_parameters")
const OUTPUT_DIR = get(ENV, "OUTPUT_DIR", joinpath(CLIMATE_CACHE, "native_runs"))

# The calibration to apply: the reference ensemble member's `gemb_calibration` output.
const FORCING_CORRECTIONS = get(ENV, "FORCING_CORRECTIONS",
    "/mnt/bylot-r3/data/binned/2deg/" *
    "glacier_rgi7_dh_best_cc_nmad3_v01_filled_ac_p6_synthesized_gembfit.arrow")
# The tile sweep that correction was fitted against, whose stored run parameters this run must match.
const REFERENCE_TILE_DIR = get(ENV, "REFERENCE_TILE_DIR", joinpath(CLIMATE_CACHE, "tile_runs_merged"))
# How far a geotile the calibration did not fit may borrow a neighbour's correction.
const CORRECTION_MAX_DISTANCE_KM = parse(Float64, get(ENV, "CORRECTION_MAX_DISTANCE_KM", "1000"))

# Every setting below is the tile sweep's (`gemb_tile_sweep.jl`, which documents each); they are what
# the reference check compares, so a change here is a change of experiment.
const TILE_SIZE = 2
const BUFFER = 1
const RUN_TIME = run_time_range(ARGS, era5_land_record_span(joinpath(CLIMATE_CACHE, "cache")))
const TIME_RANGE = RUN_TIME.time_range
const SPINUP_SIMULATION_YEARS_MAXIMUM = 500.0
const SPINUP_DRIFT_FAC = 1e-2
const SPINUP_CLIMATOLOGY_WINDOW = (DateTime(1950, 1, 1), DateTime(1980, 12, 31))
const DONOR_MAX_DISTANCE_KM = parse(Float64, get(ENV, "DONOR_MAX_DISTANCE_KM", "50.0"))
const OUTPUT_FREQUENCY = :weekly
const RESTART_FETCH_OVERLAP = Day(1)

# Claims are kept alive by a heartbeat while their unit runs, as in the tile sweep (which explains it).
const CLAIM_SUBDIR = "claims"
const CLAIM_STALE_AFTER = Hour(parse(Int, get(ENV, "CLAIM_STALE_HOURS", "1")))
const CLAIM_HEARTBEAT = Minute(5)

# The most columns one work unit holds. A unit runs serially on one single-threaded worker, so its size
# bounds the tail of the run: the largest tile holds about 5,000 columns, over 100 hours whole, and the
# run cannot end before its last unit does. 200 columns at about 80 s each is about 4.5 hours.
const UNIT_COLUMNS = parse(Int, get(ENV, "UNIT_COLUMNS", "200"))

const TILE_LIMIT = haskey(ENV, "TILE_LIMIT") ? parse(Int, ENV["TILE_LIMIT"]) : typemax(Int)
const TILE_NAMES = haskey(ENV, "TILE_NAMES") ? split(ENV["TILE_NAMES"], ",") : String[]
const FORCE = get(ENV, "FORCE", "0") == "1"
# One single-threaded process per core, for the reason given at `TILE_BLOCKS` in `gemb_tile_sweep.jl`.
const NATIVE_BLOCKS = haskey(ENV, "NATIVE_BLOCKS") ? parse(Int, ENV["NATIVE_BLOCKS"]) : 1
const NATIVE_BLOCK = haskey(ENV, "NATIVE_BLOCK") ? parse(Int, ENV["NATIVE_BLOCK"]) : 1

function settings_rows()
    return [
        setting("time range", TIME_RANGE, RUN_TIME.source),
        setting("tile size / buffer", "$(TILE_SIZE)° / $(BUFFER)°"),
        setting("output frequency", OUTPUT_FREQUENCY),
        env_setting("forcing corrections", FORCING_CORRECTIONS, "FORCING_CORRECTIONS"),
        env_setting("correction borrow range", "$(CORRECTION_MAX_DISTANCE_KM) km",
                    "CORRECTION_MAX_DISTANCE_KM"),
        env_setting("reference tile sweep", REFERENCE_TILE_DIR, "REFERENCE_TILE_DIR"),
        setting("spinup climatology", SPINUP_CLIMATOLOGY_WINDOW),
        setting("spinup ceiling", "$(SPINUP_SIMULATION_YEARS_MAXIMUM) simulated years"),
        setting("spinup drift tolerance", "$(SPINUP_DRIFT_FAC) m firn air / year"),
        env_setting("masked-cell donor range", "$(DONOR_MAX_DISTANCE_KM) km", "DONOR_MAX_DISTANCE_KM"),
        env_setting("columns per work unit", UNIT_COLUMNS, "UNIT_COLUMNS"),
        env_setting("worker blocks", NATIVE_BLOCKS, "NATIVE_BLOCKS"),
        let hint = Base.JLOptions().heap_size_hint
            hint == 0 ?
                setting("GC heap hint", "none (target sized from physical memory)") :
                setting("GC heap hint", "$(round(hint / 2^30; digits = 1)) GiB", "--heap-size-hint")
        end,
        env_setting("claim stale after", "$(Dates.value(CLAIM_STALE_AFTER)) h without a heartbeat",
                    "CLAIM_STALE_HOURS"),
        setting("claim heartbeat", "every $(Dates.value(CLAIM_HEARTBEAT)) min"),
        env_setting("tile limit", TILE_LIMIT == typemax(Int) ? "all" : TILE_LIMIT, "TILE_LIMIT"),
        env_setting("tiles", isempty(TILE_NAMES) ? "all" : join(TILE_NAMES, ", "), "TILE_NAMES"),
        env_setting("force rebuild of current cells", FORCE, "FORCE"),
        env_setting("climate cache", CLIMATE_CACHE, "CLIMATE_CACHE"),
        setting("parameters", PARAMETER_DIR),
        env_setting("output", OUTPUT_DIR, "OUTPUT_DIR"),
    ]
end

function main()
    print_settings("GEMB native-resolution run — settings in force", settings_rows())
    SHOW_SETTINGS && return nothing

    token = GEMB_ClimateForcing.get_cds_api_key()
    token === nothing && error("no CDS API key; set ENV[\"CDS_API_KEY\"] or write ~/.cdsapirc")
    cache = joinpath(CLIMATE_CACHE, "cache")

    isfile(PARQUET) || error("no glacier elevation-class table at $PARQUET")
    isdir(PARAMETER_DIR) || error("no downscaling parameters at $PARAMETER_DIR")
    isdir(REFERENCE_TILE_DIR) || error("no reference tile sweep at $REFERENCE_TILE_DIR")
    corrections = read_forcing_corrections(FORCING_CORRECTIONS)

    table = GeoDataFrames.read(PARQUET)
    table[!, :longitude] = GI.x.(table.geometry)
    table[!, :latitude] = GI.y.(table.geometry)

    tiles = downscaling_tiles(table; tile_size = TILE_SIZE, buffer = BUFFER, order = :chunk)
    isempty(TILE_NAMES) ||
        (tiles = filter(t -> replace(t.name, ".nc" => "") in TILE_NAMES, tiles))
    selected = first(tiles, min(TILE_LIMIT, length(tiles)))

    # Columns per unit are the cost the blocks are balanced on: one GEMB column per populated
    # (cell, band), all at the same single perturbation.
    units = reduce(vcat, (work_units(t) for t in selected); init = NamedTuple[])
    weights = Float64[u.columns for u in units]
    order = claim_order(weights, NATIVE_BLOCK, NATIVE_BLOCKS)

    mkpath(OUTPUT_DIR)
    claim_dir = joinpath(OUTPUT_DIR, CLAIM_SUBDIR)
    mkpath(claim_dir)
    mp = GEMB.initialize_parameters(output_frequency = OUTPUT_FREQUENCY)

    @info "GEMB native-resolution run" units=length(order) tiles=length(selected) of=length(tiles) columns=Int(sum(weights)) block="$NATIVE_BLOCK/$NATIVE_BLOCKS" time_range=TIME_RANGE corrections=FORCING_CORRECTIONS output=OUTPUT_DIR

    rows = NamedTuple[]
    t_start = time()
    for i in order
        unit = units[i]
        claim_tile!(claim_dir, unit.name, unit.marker; stale_after = CLAIM_STALE_AFTER) || continue
        append!(rows, with_claim_heartbeat(claim_dir, unit.name; interval = CLAIM_HEARTBEAT) do
            run_native_unit(unit, mp, corrections; token, cache)
        end)
    end

    if isempty(rows)
        @warn "This worker claimed no tiles; every candidate was already taken" block="$NATIVE_BLOCK/$NATIVE_BLOCKS"
        return DataFrame()
    end

    summary = DataFrame(rows)
    suffix = NATIVE_BLOCKS == 1 ? "" : "_block$(lpad(NATIVE_BLOCK, 3, '0'))of$(NATIVE_BLOCKS)"
    GEMB_GlacierSims.Parquet2.writefile(joinpath(OUTPUT_DIR, "native_runs_summary$suffix.parquet"), summary)
    @info "Native run finished" minutes=round((time() - t_start) / 60; digits = 1) cells=nrow(summary) written=count(==("written"), summary.status) extended=count(==("extended"), summary.status) skipped=count(==("skipped"), summary.status) no_reference=count(==("no_reference"), summary.status) no_correction=count(==("no_correction"), summary.status) no_forcing=count(==("no_forcing"), summary.status) dropped=count(==("dropped"), summary.status) empty=count(==("empty"), summary.status) failed=count(==("failed"), summary.status)
    return summary
end

# A tile's cells in table order, cut into runs of consecutive cells holding at most `UNIT_COLUMNS`
# columns each (a single cell larger than that is a unit of its own). A tile that fits in one unit keeps
# the tile's name. Consecutive, so a unit stays spatially compact and its cells share forcing chunks.
function work_units(tile)
    tile_name = replace(tile.name, ".nc" => "")
    tile_dir = joinpath(OUTPUT_DIR, tile_name)
    intervals = hypsometry_intervals(tile.core)
    cells = native_cells(tile)
    columns = [count(>(0), GEMB_GlacierSims._interval_areas(c, intervals)) for c in eachrow(tile.core)]

    groups = UnitRange{Int}[]
    start, total = 1, 0
    for i in eachindex(cells)
        if total > 0 && total + columns[i] > UNIT_COLUMNS
            push!(groups, start:(i - 1))
            start, total = i, 0
        end
        total += columns[i]
    end
    isempty(cells) || push!(groups, start:lastindex(cells))

    n = length(groups)
    return [let name = n == 1 ? tile_name : "$(tile_name)_part$(lpad(k, 3, '0'))of$(lpad(n, 3, '0'))"
                (; tile, tile_name, tile_dir, name, cells = cells[g], columns = sum(columns[g]),
                 marker = joinpath(tile_dir, name * ".complete"))
            end
            for (k, g) in enumerate(groups)]
end

# Every cell of one work unit, in table order. Returns one summary row per cell. The tile-level steps —
# the downscaling resolution, the correction lookup and the reference check — run once per unit.
function run_native_unit(unit, mp, corrections; token, cache)
    (; tile, tile_name, tile_dir, cells) = unit
    parameter_path = joinpath(PARAMETER_DIR, tile.name)
    if !isfile(parameter_path)
        # The tile sweep skips such a tile too, so it carries no correction-bearing reference either.
        @warn "No downscaling parameters for this tile; skipping its cells" tile=tile_name
        return [summary_row(c, tile_name, :no_parameters, 0.0) for c in cells]
    end
    intervals = hypsometry_intervals(tile.core)
    isempty(intervals) && return [summary_row(c, tile_name, :empty, 0.0) for c in cells]

    gid = geotile_id(tile.bounds)
    correction = try
        forcing_correction(gid, corrections; max_distance_km = CORRECTION_MAX_DISTANCE_KM)
    catch e
        # Every ice-bearing geotile but one is within the cap of a fitted neighbour (checked across
        # the full elevation-class table); a tile beyond it has no defensible correction to borrow,
        # so its cells are skipped and counted rather than treated as a worker failure.
        e isa ArgumentError && occursin("has no forcing correction", e.msg) || rethrow()
        @warn "No forcing correction within reach of this geotile; skipping its cells" tile=tile_name geotile=gid error=e.msg
        return [summary_row(c, tile_name, :no_correction, 0.0) for c in cells]
    end
    correction.distance_km > 0 &&
        @info "Geotile has no fitted correction; borrowing the nearest" tile=tile_name geotile=gid donor=correction.donor distance_km=round(correction.distance_km; digits = 1)

    fit = read_downscaling_tile(parameter_path)
    probe_time = probe_run_time(tile, token, cache)
    basis = length(fit.time) == length(probe_time) && collect(fit.time) == probe_time ?
            :fitted : :climatology
    applied = resolve_downscaling(fit, intervals, probe_time; basis,
                                  decoupling_factor_prior = decoupling_factor_prior(tile.core))
    requested = tile_run_parameters(mp, applied;
                                   spinup_window = SPINUP_CLIMATOLOGY_WINDOW,
                                   simulation_years_maximum = SPINUP_SIMULATION_YEARS_MAXIMUM,
                                   convergence_drift_fac = SPINUP_DRIFT_FAC,
                                   drift_window = GEMB_GlacierSims.SPINUP_DRIFT_WINDOW,
                                   donor_max_distance_km = DONOR_MAX_DISTANCE_KM,
                                   climatology_method = GEMB_GlacierSims.SPINUP_CLIMATOLOGY_METHOD,
                                   climatology_n_years = GEMB_GlacierSims.SPINUP_CLIMATOLOGY_N_YEARS)
    # A tile the reference sweep never wrote has no settings to check against, and no calibrated tile
    # behind its correction: skipped and counted rather than run unverified.
    reference = joinpath(REFERENCE_TILE_DIR, tile.name)
    if !isfile(reference)
        @warn "No reference tile run for this tile; skipping its cells" tile=tile_name reference
        return [summary_row(c, tile_name, :no_reference, 0.0; correction) for c in cells]
    end
    assert_matches_reference(reference, tile.name, requested)
    # The bands the reference tile run modelled: an interval it dropped is dropped from every cell, so
    # the two runs drop the same intervals.
    modelled_intervals = reference_status(reference).band_edges
    prior = decoupling_factor_prior(tile.core)

    mkpath(tile_dir)
    rows = NamedTuple[]
    for cell in cells
        t0 = time()
        try
            push!(rows, run_cell(cell, tile_name, tile_dir, applied, requested, correction, mp;
                                 token, cache, fit, intervals, basis, prior, modelled_intervals))
        catch e
            e isa InterruptException && rethrow()
            GEMB_GlacierSims.is_caller_error(e) && rethrow()
            e isa RestartParameterMismatch && rethrow()
            @error "Cell failed; continuing" tile=tile_name cell=cell.name exception=(e, catch_backtrace())
            push!(rows, summary_row(cell, tile_name, :failed, time() - t0; correction,
                                    hypsometry_area_km2 = cell_ice(cell, intervals),
                                    error = sprint(showerror, e)))
        end
    end

    # Marked complete only when no cell failed, so a stale claim on a unit with a failure is reclaimed
    # and the failed cells retried, while the finished ones skip from their metadata.
    any(r -> r.status == "failed", rows) && return rows
    write_unit_marker(unit.marker, (; hypsometry_area_km2 = sum(r.hypsometry_area_km2 for r in rows),
                                    lost_area_km2 = sum(r.lost_area_km2 for r in rows)))
    exceeded = check_tile_lost_area(tile, tile_name, intervals)
    exceeded === nothing || push!(rows, exceeded)
    return rows
end

# A cell's ice over its tile's elevation intervals, in km².
cell_ice(cell, intervals) =
    sum(GEMB_GlacierSims._interval_areas(first(eachrow(cell.core)), intervals); init = 0.0)

# The unit's lost-ice accounting, written to a temporary file and renamed into place, so a marker that
# exists is a whole one.
function write_unit_marker(path, accounting)
    tmp = path * ".part"
    open(tmp, "w") do io
        println(io, "completed = \"", now(), "\"")
        println(io, "hypsometry_area_km2 = ", repr(Float64(accounting.hypsometry_area_km2)))
        println(io, "lost_area_km2 = ", repr(Float64(accounting.lost_area_km2)))
    end
    Base.Filesystem.rename(tmp, path)
    return path
end

function read_unit_marker(path)
    d = TOML.parsefile(path)
    (haskey(d, "hypsometry_area_km2") && haskey(d, "lost_area_km2")) || error(
        "unit marker $path records no lost-ice accounting; delete it so the unit is run again")
    return (; hypsometry_area_km2 = Float64(d["hypsometry_area_km2"]),
            lost_area_km2 = Float64(d["lost_area_km2"]))
end

# The 1 % rule on the tile's total, once every unit of the tile is complete; `nothing` until then and
# when the tile is within the rule. A tile above it is recorded in its `.lost_ice_exceeded` file, which
# stays until the rule passes, and is returned as a failed summary row.
function check_tile_lost_area(tile, tile_name, intervals)
    markers = [u.marker for u in work_units(tile)]
    all(isfile, markers) || return nothing
    failure = joinpath(OUTPUT_DIR, tile_name, tile_name * ".lost_ice_exceeded")
    total = native_tile_lost_area(read_unit_marker.(markers))
    hypsometry = sum(c -> cell_ice(c, intervals), native_cells(tile); init = 0.0)
    isapprox(total.hypsometry_area_km2, hypsometry; rtol = 1e-9) || error(
        "the unit markers of $tile_name account for $(total.hypsometry_area_km2) km² of ice, but " *
        "the tile holds $hypsometry km²; a marker is from a different hypsometry, so delete them")
    total.fraction <= LOST_AREA_FRACTION_MAXIMUM && (rm(failure; force = true); return nothing)
    message = "native run of tile $tile_name leaves $(round(total.lost_area_km2; digits = 2)) km² of " *
              "its $(round(hypsometry; digits = 2)) km² of ice unmodelled " *
              "($(round(100 * total.fraction; digits = 2)) %), above the " *
              "$(100 * LOST_AREA_FRACTION_MAXIMUM) % allowed; the per-cell summary rows give the " *
              "cells and bands"
    @error message tile=tile_name
    write(failure, message, "\n")
    return summary_row(nothing, tile_name, :failed, 0.0; tile, hypsometry_area_km2 = hypsometry,
                       lost_area_km2 = total.lost_area_km2, error = message)
end

# A reference tile file's status, which must be a whole file: the native run takes its settings and
# its band set from it.
function reference_status(reference)
    status = read_glacier_tile_status(reference)
    status.complete || error("the reference tile run $reference is not a complete file " *
                             "($(status.reason)); rebuild it in the tile sweep")
    return status
end

# The settings this tile is about to run under, against those stored in the tile sweep the correction
# was fitted to. `run_parameter_differences` reports a key either side lacks as well as a changed value,
# so this catches a changed model parameter, spinup policy, downscaling resolution or package version,
# which would apply the correction to a model it was not fitted for.
function assert_matches_reference(reference, name, requested)
    status = reference_status(reference)
    differences = run_parameter_differences(status.parameters, requested)
    isempty(differences) || error(
        "run settings for $name differ from the reference sweep the correction was fitted " *
        "against ($reference): $(join(differences, "; "))")
    return nothing
end

function run_cell(cell, tile_name, tile_dir, applied, requested, correction, mp; token, cache,
                  fit, intervals, basis, prior, modelled_intervals)
    t0 = time()
    path = joinpath(tile_dir, cell.name)
    row = first(eachrow(cell.core))
    ice = cell_ice(cell, intervals)
    populated = [iv for (iv, a) in
                 zip(applied.bands, GEMB_GlacierSims._interval_areas(row, applied.bands)) if a > 0]
    isempty(populated) && return summary_row(cell, tile_name, :empty, time() - t0; correction)
    # Every band of this cell lies in an interval the reference tile run dropped: the whole cell's ice
    # is dropped with them, and counted against the tile.
    modelled = Set(modelled_intervals)
    if !any(iv -> (Float64(iv.lo), Float64(iv.hi)) in modelled, populated)
        @warn "Every band of this cell was dropped by the reference tile run; its ice is unmodelled" tile=tile_name cell=cell.name area_km2=ice
        return summary_row(cell, tile_name, :dropped, time() - t0; correction,
                           hypsometry_area_km2 = ice, lost_area_km2 = ice)
    end

    status = read_glacier_tile_status(path)
    # The cell's own lost-ice share is not bounded; the rule applies to the tile.
    disposition = FORCE ? :rebuild :
                  tile_run_disposition(status, requested, applied.time, populated;
                                       delta_temperatures = [correction.ΔT],
                                       precipitation_scalings = [correction.pscale],
                                       output_frequency = mp.output_frequency,
                                       lost_area_fraction_maximum = 1.0)
    disposition === :current &&
        return summary_row(cell, tile_name, :skipped, time() - t0; correction,
                           hypsometry_area_km2 = status.hypsometry_area_km2,
                           lost_area_km2 = status.lost_area_km2)

    # Per-timestep fits cover the whole record and cannot be resolved onto an appended segment.
    disposition === :extendable && basis === :fitted && (disposition = :rebuild)
    restart = disposition === :extendable ? read_glacier_tile_restart(path) : nothing
    appending = restart !== nothing && restart.time !== nothing
    fetch_range = appending ?
                  (max(first(TIME_RANGE), restart.time - RESTART_FETCH_OVERLAP), last(TIME_RANGE)) :
                  TIME_RANGE
    # The band forcing is built on the fetched steps, so the applied parameters are resolved onto them.
    if appending
        applied = resolve_downscaling(fit, intervals, probe_run_time(cell, token, cache, fetch_range);
                                      basis, decoupling_factor_prior = prior)
    end

    local run
    try
        run = gemb_native_cell(cell, applied, mp; correction,
                               climate_model = CLIMATE_MODEL, time_range = fetch_range,
                               token, cache_path = cache,
                               donor_max_distance_km = DONOR_MAX_DISTANCE_KM,
                               spinup_window = SPINUP_CLIMATOLOGY_WINDOW,
                               simulation_years_maximum = SPINUP_SIMULATION_YEARS_MAXIMUM,
                               convergence_drift_fac = SPINUP_DRIFT_FAC,
                               modelled_intervals,
                               restart = appending ? restart : nothing)
    catch err
        err isa ForcingUpToDate &&
            return summary_row(cell, tile_name, :skipped, time() - t0; correction,
                               hypsometry_area_km2 = status.hypsometry_area_km2,
                               lost_area_km2 = status.lost_area_km2)
        # Ice with no forcing of its own and no donor in reach, which the tile run counts as
        # unrecovered: recorded with its area in the summary, counted against the tile, and not
        # retried.
        err isa ForcingUnavailable || rethrow()
        @warn "Cell has no usable forcing; its ice is unrecovered" tile=tile_name cell=cell.name area_km2=ice
        return summary_row(cell, tile_name, :no_forcing, time() - t0; correction,
                           hypsometry_area_km2 = ice, lost_area_km2 = ice)
    end
    t_run = time() - t0

    if appending
        append_glacier_tile_netcdf(path, run)
    else
        write_glacier_tile_netcdf(path, run; institution = "NASA Jet Propulsion Laboratory")
    end
    @info(appending ? "Extended cell" : "Wrote cell", tile=tile_name, cell=cell.name,
          bands=length(run.bands), seconds=round(t_run; digits = 1))
    return summary_row(cell, tile_name, appending ? :extended : :written, time() - t0;
                       run, correction, t_run)
end

function probe_run_time(tile, token, cache, time_range = TIME_RANGE)
    row = first(tile.core)
    fd = climate_forcing(CLIMATE_MODEL, row.latitude, row.longitude;
                         time_range, token, cache_path = cache)
    return collect(dims(fd, Ti))
end

# One row per cell, or with `cell = nothing` one row for the tile (its `tile` is then required).
# `hypsometry_area_km2` is the ice over the tile's elevation intervals and `lost_area_km2` the part of
# it left unmodelled; a run's own accounting replaces both.
function summary_row(cell, tile_name, status, seconds; run = nothing, correction = nothing,
                     t_run = missing, error = "", tile = nothing, hypsometry_area_km2 = 0.0,
                     lost_area_km2 = 0.0)
    years = run === nothing ? missing : GEMB_GlacierSims.record_years(run)
    p = run === nothing ? Dict{String,Any}() : run.provenance
    where = cell === nothing ? tile : cell
    return (; tile = tile_name, cell = cell === nothing ? "" : replace(cell.name, ".nc" => ""),
            geotile_id = geotile_id(where.bounds),
            latitude = cell === nothing ? missing : cell.latitude,
            longitude = cell === nothing ? missing : Float64(wrap_lon(cell.longitude)),
            hypsometry_area_km2 = Float64(get(p, "hypsometry_area_km2", hypsometry_area_km2)),
            lost_area_km2 = Float64(get(p, "lost_area_km2", lost_area_km2)),
            glacier_area_km2 = get(p, "glacier_area_km2", 0.0),
            n_bands = run === nothing ? 0 : length(run.bands),
            pscale = correction === nothing ? missing : correction.pscale,
            delta_temperature = correction === nothing ? missing : correction.ΔT,
            correction_donor = correction === nothing ? "" : correction.donor,
            correction_distance_km = correction === nothing ? missing : correction.distance_km,
            substituted_area_km2 = get(p, "substituted_area_km2", 0.0),
            unrecovered_area_km2 = get(p, "unrecovered_area_km2", 0.0),
            dropped_area_km2 = get(p, "dropped_area_km2", 0.0),
            max_dh_residual = get(p, "max_dh_residual", 0.0),
            dv_rate = run === nothing ? missing : run.totals[:dv][end, 1, 1] / years,
            dm_rate = run === nothing ? missing : run.totals[:dm][end, 1, 1] / years,
            runoff_rate = run === nothing ? missing : run.totals[:runoff][end, 1, 1] / years,
            status = string(status),
            seconds = round(seconds; digits = 1),
            run_seconds = t_run === missing ? missing : round(t_run; digits = 1),
            error)
end

exit_on_failures(main(), "cells or tiles")
