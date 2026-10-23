# Sourced by the launchers (run_tile_sweep.sh, run_native_sweep.sh, run_derive_parameters.sh), which call
#
#   restore_invariant "$JULIA" "$REPO" "$CLIMATE_CACHE"
#
# before starting any worker.
#
# Every worker reads the ERA5-Land geopotential invariant (the reanalysis elevation of each cell, which
# the lapse to an elevation band starts from). GEMB_ClimateForcing caches it at the path its
# `_default_invariant_cache(:era5_land)` resolves: under `$GEMB_CACHE_PATH` when that is set, otherwise
# under the installed package's own `data/` directory, which a dependency bump replaces with an empty one.
# Workers that start together on a missing file all download it to one shared `.part` path and the losers
# fail their first tile, so the file is put in place once, here: copied from the preserved copy at
# `$CLIMATE_CACHE/invariant/era5_land/` when there is one, downloaded by a single process otherwise. The
# copy is written beside its target and renamed into place.
#
# Returns non-zero, with the reason on stderr, when the file cannot be put in place.
restore_invariant() {
    local julia="$1" repo="$2" climate_cache="$3"
    local target source
    target=$("$julia" --project="$repo" --startup-file=no -e '
        import GEMB_ClimateForcing as CF
        print(joinpath(CF._default_invariant_cache(:era5_land),
                       CF._INVARIANT_REGISTRY[:era5_land][:z].file))') ||
        { echo "could not load GEMB_ClimateForcing to locate the invariant cache" >&2; return 1; }
    [ -f "$target" ] && return 0
    source="$climate_cache/invariant/era5_land/$(basename "$target")"
    mkdir -p "$(dirname "$target")" ||
        { echo "could not create $(dirname "$target")" >&2; return 1; }
    if [ -f "$source" ]; then
        cp -p "$source" "$target.part" && mv -f "$target.part" "$target" ||
            { echo "could not copy the invariant file from $source to $target" >&2; return 1; }
        echo "restored the ERA5-Land invariant file to $target"
    else
        echo "no preserved invariant file at $source; downloading it once to $target"
        "$julia" --project="$repo" --startup-file=no -e '
            import GEMB_ClimateForcing as CF
            CF.climate_model_invariant(; model = :era5_land, parameter = :z)' > /dev/null ||
            { echo "could not download the ERA5-Land invariant file" >&2; return 1; }
        [ -f "$target" ] ||
            { echo "the download did not produce $target" >&2; return 1; }
    fi
}
