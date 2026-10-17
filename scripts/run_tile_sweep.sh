#!/usr/bin/env bash
#
# Run the whole (temperature offset x precipitation scaling) sweep over every glacierized 2 degree
# tile, as TILE_BLOCKS concurrent single-threaded processes.
#
#   scripts/run_tile_sweep.sh [start] [end]
#
# `start` and `end` are each a year, an ISO date-time, or `record-start` / `record-end`; without them the
# sweep runs from the first to the last step of the cached ERA5-Land record (see gemb_tile_sweep.jl). Exits non-zero when any worker exits non-zero,
# which a worker does when any tile it claimed failed, and reports how many tiles failed.
#
# One thread per process, one process per physical core. A GEMB spinup cycle allocates about 100 MiB and
# Julia's collector is per-process and stops every thread, so threads inside one process contend for the
# collector while separate processes do not: tile throughput peaks near 16 threads and falls beyond it.
# The measured curve is in the TILE_BLOCKS comment in gemb_tile_sweep.jl. A worker holds about 5 GB and
# the heaviest tiles peak near 8 GB, so at 120 blocks the set occupies roughly 600 GB: wide enough that
# memory, not collector contention, is what bounds the block count on a 1 TB machine. HEAP_SIZE_HINT
# bounds it if that becomes tight. Each worker keeps a small `sh` process beside it that refreshes the
# heartbeat of its tile claim, so a tile that runs for days is not reclaimed by another worker.
#
# Each process drains its own contiguous, weight-balanced block of the chunk-ordered tile list first —
# which is what keeps the forcing cache warm across neighbours — and then steals the heaviest tile still
# unclaimed. Tiles are claimed by an atomic mkdir under $OUTPUT_DIR/claims, so nothing coordinates beyond the
# filesystem, and a worker that finishes early takes work instead of idling. That matters more than it used
# to: with forcing fetched on demand a tile's wall time is partly a CDS download, so band count alone no
# longer predicts cost.
#
# Resumable and extendable: a tile whose file already covers the request is skipped from its metadata
# alone, and one whose settings match but whose record has grown is appended to from its saved firn state.
# Re-running the same command after an interruption therefore costs only the tiles that did not finish, and
# re-running against a longer ERA5-Land record costs the new months. Every process writes its own log and
# its own summary parquet.
#
# Before launching anything it runs the sweep script once with SHOW_SETTINGS=1, which prints the settings
# in force — including whether the parameter tiles carry an observed bare-ice albedo profile — and exits.
# That table is the record of what this run was configured to do, and the pass that prints it also loads
# the package, so a syntax error or a missing dependency surfaces once here rather than 96 times in the
# logs.
#
# Environment: TILE_BLOCKS (default 96), CLIMATE_CACHE, PHYSICS_MEMBER (default B-kfit-table, as in
# run_settings.jl), OUTPUT_DIR (default $CLIMATE_CACHE/tile_runs_v2/$PHYSICS_MEMBER), EFFECTIVE_LOADING_TABLE,
# JULIA, FORCING_CACHE_GIB, TILE_NAMES, FORCE,
# CLAIM_STALE_HOURS, SPINUP_SIMULATION_YEARS_MAXIMUM, HEAP_SIZE_HINT.

set -uo pipefail

# The run's ends, passed through to every worker; the workers read the cached record and resolve
# `record-start` and an absent end from it.
run_args=("${1:-record-start}")
[ -n "${2:-}" ] && run_args+=("$2")
# One process per core, less headroom. 96 of 128 leaves room for other work on the machine and for the
# page cache the Zarr reads live in; raise it when the machine is otherwise idle.
BLOCKS="${TILE_BLOCKS:-96}"
JULIA="${JULIA:-$HOME/.juliaup/bin/julia}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLIMATE_CACHE="${CLIMATE_CACHE:-/mnt/bylot-r3/data/era5land}"
# The sweep's own default (`tile_run_dir` in run_settings.jl: `TILE_RUN_DIR` and the physics member),
# which the native run takes as its reference. Exported so the workers run as this member and write
# where the logs and claims below are kept.
PHYSICS_MEMBER="${PHYSICS_MEMBER:-B-kfit-table}"
export PHYSICS_MEMBER
OUTPUT_DIR="${OUTPUT_DIR:-$CLIMATE_CACHE/tile_runs_v2/$PHYSICS_MEMBER}"
export OUTPUT_DIR
LOG_DIR="$OUTPUT_DIR/logs"

[ -x "$JULIA" ] || { echo "no julia at $JULIA; set JULIA=/path/to/julia" >&2; exit 1; }
[ -f "$REPO/data/era5land_glacier_elevation_classes.parquet" ] ||
    { echo "no elevation-class table under $REPO/data" >&2; exit 1; }
[ -d "$CLIMATE_CACHE/downscaling_parameters" ] ||
    { echo "no downscaling parameters under $CLIMATE_CACHE; run derive_downscaling_parameter_tiles.jl first" >&2; exit 1; }

# Put the invariant geopotential file in place before the workers start, or they race to download it.
. "$REPO/scripts/restore_invariant.sh"
restore_invariant "$JULIA" "$REPO" "$CLIMATE_CACHE" || { echo "nothing launched" >&2; exit 1; }

# Julia sizes its GC heap target from total physical memory, so each of TILE_BLOCKS workers defers
# collection as though it owned the machine alone and the collective high-water mark ratchets up over a
# multi-day run. A hint bounds each worker instead, trading more frequent collection for a predictable
# footprint. Unset leaves Julia's default; set it below the ~8 GB a heavy tile peaks at to cap the set.
# The settings pass takes the same flag so the table reports what the workers will actually run with.
heap_args=()
[ -n "${HEAP_SIZE_HINT:-}" ] && heap_args=(--heap-size-hint="$HEAP_SIZE_HINT")

TILE_BLOCKS="$BLOCKS" SHOW_SETTINGS=1 "$JULIA" --project="$REPO" --startup-file=no -t 1 ${heap_args[@]+"${heap_args[@]}"} \
    "$REPO/scripts/gemb_tile_sweep.jl" "${run_args[@]}" ||
    { echo "settings pass failed; nothing launched" >&2; exit 1; }

mkdir -p "$LOG_DIR"

# Claims are advisory and only meaningful while their worker lives, so a launcher that starts the whole
# set of workers starts from a clean slate. Without this, a sweep killed mid-run leaves every tile claimed
# and the next launch finds nothing to do until the staleness bound expires hours later.
#
# Refused while workers are alive, because the claims are the only thing stopping two of them from
# running the same tile into the same output file. A second launch — a subset re-run alongside a full
# sweep, say — would otherwise clear the first sweep's claims and let its next steal attempt take a
# tile already in progress. Wait for the running sweep, or point this one at a different output.
if pgrep -f "$REPO/scripts/gemb_tile_sweep.jl" > /dev/null; then
    echo "a sweep is already running against $CLIMATE_CACHE; its claims are what prevents two" >&2
    echo "workers from writing one tile, so this launch will not clear them. Wait for it to finish:" >&2
    pgrep -fa "$REPO/scripts/gemb_tile_sweep.jl" | head -3 >&2
    exit 1
fi
rm -rf "$OUTPUT_DIR/claims"
echo "sweep ${run_args[*]} over $BLOCKS blocks; logs in $LOG_DIR"

pids=()
for i in $(seq 1 "$BLOCKS"); do
    log="$LOG_DIR/block_$(printf '%03d' "$i")of$BLOCKS.log"
    TILE_BLOCKS="$BLOCKS" TILE_BLOCK="$i" FORCING_CACHE_GIB="${FORCING_CACHE_GIB:-1}" \
        "$JULIA" --project="$REPO" --startup-file=no -t 1 ${heap_args[@]+"${heap_args[@]}"} \
        "$REPO/scripts/gemb_tile_sweep.jl" "${run_args[@]}" > "$log" 2>&1 &
    pids+=($!)
done

echo "launched ${#pids[@]} processes; follow with:  tail -f $LOG_DIR/block_001of$BLOCKS.log"

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=$((failed + 1))
done

# Counted from the logs rather than the per-block summary parquets, so this needs no Julia process of
# its own and still reports when a block died before writing its summary.
echo
echo "blocks finished: $(( ${#pids[@]} - failed )) ok, $failed non-zero exit"
printf 'tiles written  : %s\n' "$(grep -ah 'Wrote tile' "$LOG_DIR"/block_*of"$BLOCKS".log | wc -l)"
printf 'tiles extended : %s\n' "$(grep -ah 'Extended tile' "$LOG_DIR"/block_*of"$BLOCKS".log | wc -l)"
printf 'tiles skipped  : %s\n' "$(grep -ah 'already covers the request' "$LOG_DIR"/block_*of"$BLOCKS".log | wc -l)"
tiles_failed=$(grep -ah 'Tile failed' "$LOG_DIR"/block_*of"$BLOCKS".log | wc -l | tr -d ' ')
printf 'tiles failed   : %s\n' "$tiles_failed"
echo "summaries      : $OUTPUT_DIR/tile_runs_summary_block*of$BLOCKS.parquet"

if [ "$failed" -ne 0 ] || [ "$tiles_failed" -ne 0 ]; then
    echo "sweep incomplete: $tiles_failed tile(s) failed, $failed block(s) exited non-zero" >&2
    exit 1
fi
