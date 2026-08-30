#!/usr/bin/env bash
#
# Run the whole (temperature offset x precipitation scaling) sweep over every glacierized 2 degree
# tile, as TILE_BLOCKS concurrent single-threaded processes.
#
#   scripts/run_tile_sweep.sh [start_year] [end_year]
#
# One thread per process, one process per physical core. A GEMB spinup cycle allocates about 100 MiB and
# Julia's collector is per-process and stops every thread, so threads inside one process contend for the
# collector while separate processes do not: tile throughput peaks near 16 threads and falls beyond it.
# The measured curve is in the TILE_BLOCKS comment in gemb_tile_sweep.jl. Each process holds about
# 2.5 GB, nearly all of it independent of the record length, so memory is not what bounds the width.
#
# Blocks are contiguous runs of the chunk-ordered tile list carrying equal total band count, so they
# cost the same without breaking the forcing-cache locality that chunk order buys. Each process derives
# the same partition, so nothing coordinates.
#
# Resumable: a tile whose file already covers the request is skipped from its metadata alone, with no
# forcing read and no simulation. Re-running the same command after an interruption therefore costs only
# the tiles that did not finish. Every process writes its own log and its own summary parquet.
#
# Environment: TILE_BLOCKS (default 32), CLIMATE_CACHE, JULIA, FORCING_CACHE_GIB, TILE_NAMES, FORCE.

set -uo pipefail

START_YEAR="${1:-1950}"
END_YEAR="${2:-2026}"
BLOCKS="${TILE_BLOCKS:-64}"
JULIA="${JULIA:-$HOME/.juliaup/bin/julia}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLIMATE_CACHE="${CLIMATE_CACHE:-/mnt/bylot-r3/data/era5land}"
LOG_DIR="$CLIMATE_CACHE/tile_runs/logs"

[ -x "$JULIA" ] || { echo "no julia at $JULIA; set JULIA=/path/to/julia" >&2; exit 1; }
[ -f "$REPO/data/era5land_glacier_elevation_classes.parquet" ] ||
    { echo "no elevation-class table under $REPO/data" >&2; exit 1; }
[ -d "$CLIMATE_CACHE/downscaling_parameters" ] ||
    { echo "no downscaling parameters under $CLIMATE_CACHE; run derive_downscaling_parameter_tiles.jl first" >&2; exit 1; }

mkdir -p "$LOG_DIR"
echo "sweep $START_YEAR-$END_YEAR over $BLOCKS blocks; logs in $LOG_DIR"

pids=()
for i in $(seq 1 "$BLOCKS"); do
    log="$LOG_DIR/block_$(printf '%03d' "$i")of$BLOCKS.log"
    TILE_BLOCKS="$BLOCKS" TILE_BLOCK="$i" FORCING_CACHE_GIB="${FORCING_CACHE_GIB:-1}" \
        "$JULIA" --project="$REPO" --startup-file=no -t 1 \
        "$REPO/scripts/gemb_tile_sweep.jl" "$START_YEAR" "$END_YEAR" > "$log" 2>&1 &
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
printf 'tiles skipped  : %s\n' "$(grep -ah 'already covers the request' "$LOG_DIR"/block_*of"$BLOCKS".log | wc -l)"
printf 'tiles failed   : %s\n' "$(grep -ah 'Tile failed' "$LOG_DIR"/block_*of"$BLOCKS".log | wc -l)"
echo "summaries      : $CLIMATE_CACHE/tile_runs/tile_runs_summary_block*of$BLOCKS.parquet"

[ "$failed" -eq 0 ] || exit 1
