# Audit scripts

Each script reproduces one GEMB_GlacierSims finding of the 2026-10-13 GEMB pipeline review, on
synthetic inputs unless stated otherwise. Run them from the repository root against either branch;
the header of each script states the finding and the output expected on each.

| Script | Finding |
|---|---|
| `append_continuity.jl` | appended cumulative series restart at zero at the seam |
| `height_components_closure.jl` | `dh` and `dh_mass + dh_water + dh_firn + dh_residual` use different epochs (runs GEMB, about a minute) |
| `dv_rate.jl` | the record-mean rate counts N output intervals as N - 1 |
| `unreachable_race.jl` | `BitVector` flags written from `Threads.@threads` lose writes (run with `-t 8`) |
| `hypsometry_drop.jl` | a glacier pixel outside the bin edges is dropped from the hypsometry |
| `parameter_differences.jl` | no package versions recorded, and keys on one side only are ignored (reads output files) |
| `spinup_unconverged_outputs.jl` | unconverged area at the 100-year spinup ceiling in a merged sweep (reads output files) |
| `spinup_convergence.jl` | simulated years the current physics needs to converge, per tile (reads the ERA5-Land cache) |

`common.jl` holds the synthetic inputs, built only from API present on both branches.

## Environments

- `feat/native-resolution-run`: the branch's committed `Manifest.toml`, which pins GEMB and
  GEMB_ClimateForcing to their `main` branches.
- `fix/pipeline-audit`: an environment that `Pkg.develop`s this checkout, GEMB.jl on
  `fix/physics-audit` and GEMB_ClimateForcing on `fix/forcing-audit`.

```sh
julia --project=<env> scripts/audit/append_continuity.jl
julia -t 8 --project=<env> scripts/audit/unreachable_race.jl
julia --project=<env> scripts/audit/spinup_unconverged_outputs.jl [tile_dir]
julia -t 2 --project=<env> scripts/audit/spinup_convergence.jl N82_W048 1500 <out_dir>
```

The three scripts that read data use `CLIMATE_CACHE` (default `/mnt/bylot-r3/data/era5land`) and
only read from it. `spinup_convergence.jl` reads the Zarr forcing cache through `climate_forcing`,
which fetches a missing chunk from CDS into the cache.
