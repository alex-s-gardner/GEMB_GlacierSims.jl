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
cache = joinpath(tempdir(), ".cache", "era5land")
```

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

## Testing

`julia --project=. test/runtests.jl` is fully offline — the derivation paths take an injected
`forcing_loader`, so no token or network is needed. Keep it that way when adding tests.
