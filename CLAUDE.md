# GEMB_GlacierSims.jl

## CDS credentials (ERA5-Land forcing)

Credentials are already configured on this machine — **do not ask the user for a key**. Resolve one
with:

```julia
using GEMB_ClimateForcing
token = GEMB_ClimateForcing.get_cds_api_key()   # reads ENV["CDS_API_KEY"], else ~/.cdsapirc
```

It reads `ENV["CDS_API_KEY"]` first and falls back to `~/.cdsapirc` (which exists here), so a
plain `julia --project=.` run picks it up with no extra setup. Pass the result as the `token`
keyword to `climate_forcing`, `climate_chunk_map`, `derive_downscaling_parameters`, and
`derive_downscaling_parameter_tiles`.

Downloads are cached as Zarr chunks and the cache is shared across cells and tiles, so re-runs and
neighbouring cells are nearly free. Use the same cache path the scripts do:

```julia
climate_cache = get(ENV, "CLIMATE_CACHE", "/mnt/bylot-r3/data/era5land")
cache = joinpath(climate_cache, "cache")
```

This lives on shared storage rather than `tempdir()` on purpose: the global cache is already tens of
GB and a reboot would otherwise throw it away. **Never delete it casually** — refetching means
waiting on the CDS queue for hours. `$CLIMATE_CACHE/downscaling_parameters` holds the tile outputs
from `derive_downscaling_parameter_tiles.jl` for the same reason.

One cache is *not* covered by this: `climate_forcing` does not thread a path through to the
invariant geopotential file, so `GEMB_ClimateForcing` caches it under its own package install
directory — `_default_invariant_cache` resolves to
`<pkgdir(GEMB_ClimateForcing)>/data/invariant/era5_land/`. That path is keyed to the package's
content-hash slug (`~/.julia/packages/GEMB_ClimateForcing/<slug>/`), so **any `Pkg.update` or
`Pkg.resolve` that changes the resolved commit orphans the cached file**: the new slug's
`data/invariant/era5_land/` starts empty, and every worker that starts after the bump tries to
redownload it. Starting several workers at once at that moment hits a known race
(`rename(...): no such file or directory`) — one wins, the rest fail that tile and exit. A copy
is preserved at `$CLIMATE_CACHE/invariant/era5_land/`. The three launchers (`run_tile_sweep.sh`,
`run_native_sweep.sh`, `run_derive_parameters.sh`) put it into the *current* path before starting any
worker (`scripts/restore_invariant.sh`, which copies the preserved file or, failing that, downloads it
once). Workers started by hand after a dependency bump (or a reboot, which has the same effect on other
`tempdir()`-cached state in this repo) need the same done first:

```sh
. scripts/restore_invariant.sh
restore_invariant ~/.juliaup/bin/julia . /mnt/bylot-r3/data/era5land
```

Low stakes either way — it is one 50 MB file from a plain HTTP URL, not a CDS-queued request.

This means network-dependent work **can** be verified directly rather than deferred — run a real
cell or tile rather than stopping at the offline tests. A first fetch of a fresh region is slow
(CDS queues the request), so give such commands a long timeout and prefer running them in the
background.

## Data

- `data/era5land_glacier_elevation_classes.parquet` — the cached global glacier elevation-class
  table (47,121 cells). Built by `src/era5_example.jl` only when absent; building it takes hours,
  so never delete or regenerate it casually.
- The table stores **native 0–359.9°E longitudes** and carries no `:latitude`/`:longitude` columns;
  callers add them from the Point geometry and leave them native, because that is what
  `climate_forcing` is keyed on. `wrap_lon` converts to `(-180, 180]` for geometry.
- `data/background_black_carbon.csv` — background BC (and dust) in snow per RGI 7.0 o1 region, with
  method, season and source per row; read by `background_black_carbon`. The scripts apply it only
  when `BACKGROUND_BLACK_CARBON` is `bc` or `bc_dust` (default `off`).
- `data/rgi7_o1regions_simplified.geojson` — RGI 7.0 first-order region outlines (GTN-G 2023,
  doi:10.5904/gtng-glacreg-2023-07), simplified at 0.01°, used to assign a tile its region.

## Testing

`julia --project=. test/runtests.jl` is fully offline — the derivation paths take an injected
`forcing_loader`, so no token or network is needed. Keep it that way when adding tests.
