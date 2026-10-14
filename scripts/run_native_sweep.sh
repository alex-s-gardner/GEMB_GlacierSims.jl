#!/usr/bin/env bash
#
# Run GEMB at native ERA5-Land resolution over every glacierized cell, under the calibrated forcing
# corrections, as NATIVE_BLOCKS concurrent single-threaded processes.
#
#   scripts/run_native_sweep.sh [start_year] [end_year]
#
# The process layout is the tile sweep's (`run_tile_sweep.sh`, which explains it): one thread per process,
# one process per core less headroom, each draining a weight-balanced block of the chunk-ordered tile list
# and then stealing, with tiles claimed by an atomic mkdir under native_runs/claims.
#
# Resumable: a cell whose file already covers the request is skipped from its metadata alone, so
# re-running the same command after an interruption costs only the cells that did not finish.
#
# Environment: NATIVE_BLOCKS (default 96), CLIMATE_CACHE, JULIA, TILE_NAMES, FORCE, FORCING_CORRECTIONS,
# REFERENCE_TILE_DIR, OUTPUT_DIR, CLAIM_STALE_HOURS, HEAP_SIZE_HINT.

set -uo pipefail

START_YEAR="${1:-1950}"
END_YEAR="${2:-2027}"
BLOCKS="${NATIVE_BLOCKS:-96}"
JULIA="${JULIA:-$HOME/.juliaup/bin/julia}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLIMATE_CACHE="${CLIMATE_CACHE:-/mnt/bylot-r3/data/era5land}"
OUTPUT_DIR="${OUTPUT_DIR:-$CLIMATE_CACHE/native_runs}"
export OUTPUT_DIR
LOG_DIR="$OUTPUT_DIR/logs"
SCRIPT="$REPO/scripts/gemb_native_run.jl"

[ -x "$JULIA" ] || { echo "no julia at $JULIA; set JULIA=/path/to/julia" >&2; exit 1; }
[ -f "$REPO/data/era5land_glacier_elevation_classes.parquet" ] ||
    { echo "no elevation-class table under $REPO/data" >&2; exit 1; }
[ -d "$CLIMATE_CACHE/downscaling_parameters" ] ||
    { echo "no downscaling parameters under $CLIMATE_CACHE" >&2; exit 1; }

# The invariant geopotential file is cached under the package's install directory, which a dependency
# bump replaces; restore it before the workers start, or they race to download it (see CLAUDE.md).
PKGDIR=$("$JULIA" --project="$REPO" --startup-file=no -e \
    'using GEMB_ClimateForcing; print(pkgdir(GEMB_ClimateForcing))') ||
    { echo "could not load GEMB_ClimateForcing" >&2; exit 1; }
INVARIANT="geo_1279l4_0.1x0.1.grib2_v4_unpack.nc"
if [ ! -f "$PKGDIR/data/invariant/era5_land/$INVARIANT" ]; then
    mkdir -p "$PKGDIR/data/invariant/era5_land"
    cp -a "$CLIMATE_CACHE/invariant/era5_land/$INVARIANT" "$PKGDIR/data/invariant/era5_land/" ||
        { echo "could not restore the invariant geopotential file" >&2; exit 1; }
fi

heap_args=()
[ -n "${HEAP_SIZE_HINT:-}" ] && heap_args=(--heap-size-hint="$HEAP_SIZE_HINT")

NATIVE_BLOCKS="$BLOCKS" SHOW_SETTINGS=1 "$JULIA" --project="$REPO" --startup-file=no -t 1 "${heap_args[@]}" \
    "$SCRIPT" "$START_YEAR" "$END_YEAR" ||
    { echo "settings pass failed; nothing launched" >&2; exit 1; }

mkdir -p "$LOG_DIR"

# Claims only mean something while their workers live, so a fresh launch clears them — but never while
# workers are running, since the claims are what stop two of them writing one tile.
if pgrep -f "$SCRIPT" > /dev/null; then
    echo "a native run is already running; wait for it to finish:" >&2
    pgrep -fa "$SCRIPT" | head -3 >&2
    exit 1
fi
rm -rf "$OUTPUT_DIR/claims"
echo "native run $START_YEAR-$END_YEAR over $BLOCKS blocks; logs in $LOG_DIR"

pids=()
for i in $(seq 1 "$BLOCKS"); do
    log="$LOG_DIR/block_$(printf '%03d' "$i")of$BLOCKS.log"
    NATIVE_BLOCKS="$BLOCKS" NATIVE_BLOCK="$i" \
        "$JULIA" --project="$REPO" --startup-file=no -t 1 "${heap_args[@]}" \
        "$SCRIPT" "$START_YEAR" "$END_YEAR" > "$log" 2>&1 &
    pids+=($!)
done

echo "launched ${#pids[@]} processes; follow with:  tail -f $LOG_DIR/block_001of$BLOCKS.log"

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=$((failed + 1))
done

echo
echo "blocks finished: $(( ${#pids[@]} - failed )) ok, $failed non-zero exit"
printf 'cells written  : %s\n' "$(grep -ah 'Wrote cell' "$LOG_DIR"/block_*of"$BLOCKS".log | wc -l)"
printf 'cells extended : %s\n' "$(grep -ah 'Extended cell' "$LOG_DIR"/block_*of"$BLOCKS".log | wc -l)"
printf 'cells failed   : %s\n' "$(grep -ah 'Cell failed' "$LOG_DIR"/block_*of"$BLOCKS".log | wc -l)"
printf 'units complete : %s\n' "$(ls "$OUTPUT_DIR"/*/*.complete 2>/dev/null | wc -l)"
echo "summaries      : $OUTPUT_DIR/native_runs_summary_block*of$BLOCKS.parquet"

[ "$failed" -eq 0 ] || exit 1
