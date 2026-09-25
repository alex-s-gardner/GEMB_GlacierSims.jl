#!/usr/bin/env bash
#
# Derive downscaling parameters for every glacierized 2 degree tile, as TILE_BLOCKS concurrent
# single-threaded processes.
#
#   scripts/run_derive_parameters.sh [start_year] [end_year]
#
# The fit window must match the window the tile runs will use, or `gemb_tile_sweep.jl` resolves the
# parameters on the climatology path rather than the fitted one: it compares the parameter file's own time
# axis against the run's, and a mismatch is not repairable downstream. So a full-record sweep needs
# full-record parameters, which is what this exists to produce.
#
# One thread per process, one process per core less headroom. Same reasoning as run_tile_sweep.sh: a
# Julia collector is per-process and stops every thread, so separate processes scale where threads inside
# one do not. Each process drains its own contiguous block of the chunk-ordered tile list, keeping the
# forcing cache warm across neighbours, then steals the heaviest tile still unclaimed so none idles
# through the tail. Tiles are claimed by an atomic mkdir under the output directory's claims/, so nothing
# coordinates beyond the filesystem.
#
# Resumable: a tile whose file already covers the requested window with the same settings is skipped from
# its metadata alone, with no forcing read. Re-running the same command after an interruption therefore
# costs only the tiles that did not finish.
#
# Before launching anything it runs the derivation script once with SHOW_SETTINGS=1, which prints the
# settings in force and exits. That table is the record of what this run was configured to do, and the
# pass that prints it also loads the package, so a syntax error or a missing dependency surfaces once here
# rather than 96 times in the logs.
#
# Environment: TILE_BLOCKS (default 96), CLIMATE_CACHE, JULIA, FORCING_CACHE_GIB, CLAIM_STALE_HOURS,
# DERIVE_ALBEDO.

set -uo pipefail

START_YEAR="${1:-1950}"
END_YEAR="${2:-2027}"
BLOCKS="${TILE_BLOCKS:-96}"
JULIA="${JULIA:-$HOME/.juliaup/bin/julia}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLIMATE_CACHE="${CLIMATE_CACHE:-/mnt/bylot-r3/data/era5land}"
LOG_DIR="$CLIMATE_CACHE/downscaling_parameters/logs"

[ -x "$JULIA" ] || { echo "no julia at $JULIA; set JULIA=/path/to/julia" >&2; exit 1; }
[ -f "$REPO/data/era5land_glacier_elevation_classes.parquet" ] ||
    { echo "no elevation-class table under $REPO/data" >&2; exit 1; }

mkdir -p "$LOG_DIR"

# Claims are advisory and only meaningful while their worker lives, so a launcher that starts the whole
# set of workers starts from a clean slate. Without this, a sweep killed mid-run leaves every tile claimed
# and the next launch finds nothing to do until the staleness bound expires hours later.
rm -rf "$CLIMATE_CACHE/downscaling_parameters/claims"

TILE_BLOCKS="$BLOCKS" SHOW_SETTINGS=1 "$JULIA" --project="$REPO" --startup-file=no -t 1 \
    "$REPO/scripts/derive_downscaling_parameter_tiles.jl" "$START_YEAR" "$END_YEAR" ||
    { echo "settings pass failed; nothing launched" >&2; exit 1; }

echo
echo "deriving downscaling parameters $START_YEAR-$END_YEAR as $BLOCKS processes"
echo "logs: $LOG_DIR"

pids=()
for i in $(seq 1 "$BLOCKS"); do
    log="$LOG_DIR/block_$(printf '%03d' "$i")of$BLOCKS.log"
    TILE_BLOCKS="$BLOCKS" TILE_BLOCK="$i" FORCING_CACHE_GIB="${FORCING_CACHE_GIB:-1}" \
        "$JULIA" --project="$REPO" --startup-file=no -t 1 \
        "$REPO/scripts/derive_downscaling_parameter_tiles.jl" "$START_YEAR" "$END_YEAR" > "$log" 2>&1 &
    pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=$((failed + 1))
done

# Counted from the logs rather than a per-block summary, so this needs no Julia process of its own and
# still reports when a block died before finishing.
echo
echo "blocks finished: $(( ${#pids[@]} - failed )) ok, $failed non-zero exit"
printf 'tiles written  : %s\n' "$(grep -ah 'Wrote tile\|status = :written' "$LOG_DIR"/block_*of"$BLOCKS".log | wc -l)"
printf 'tiles skipped  : %s\n' "$(grep -ah 'already covers the request' "$LOG_DIR"/block_*of"$BLOCKS".log | wc -l)"
printf 'tiles sparse   : %s\n' "$(grep -ah 'below_min_cells\|:sparse' "$LOG_DIR"/block_*of"$BLOCKS".log | wc -l)"
printf 'tiles failed   : %s\n' "$(grep -ah 'Tile failed' "$LOG_DIR"/block_*of"$BLOCKS".log | wc -l)"

[ "$failed" -eq 0 ] || exit 1
