# Batch GEMB run over glacier grid cells, forced with ERA5-Land via GEMB_ClimateForcing.jl.
#
# For every glacier grid cell holding at least `cell_area_minimum` of glacier ice, this runs
# GEMB over the full outer product of prescribed temperature deltas and precipitation scalings,
# for the hypsometry bins covering at least `hypsometry_coverage` of the cell's glacier area.
# The per-bin mass fluxes are area-weighted into per-cell mass totals (kg) and written to one
# CF-compliant NetCDF per cell, alongside the final firn profile of every run so the record can
# be extended when new forcing appears without repeating the spinup.
#
# Setup (required before running):
#   1. Install GEMB_ClimateForcing from GitHub:
#      using Pkg
#      Pkg.add(url="https://github.com/alex-s-gardner/GEMB_ClimateForcing.jl")
#   2. Get a CDS API key from: https://cds.climate.copernicus.eu/api-how-to
#   3. Set environment variable: ENV["CDS_API_KEY"] = "your-key-here"
#
# NOTE: This example requires GEMB_ClimateForcing.jl to be installed.
begin
    # cached glacier elevation-class table, will only be built if it doesn't already exist
    climate_model = :era5land

    # --- sweep configuration -----------------------------------------------------------------
    # Prescribed perturbations, run as a full outer product. Note the cost: each cell runs
    # (bins x deltas x scalings) GEMB simulations, each with its own spinup.
    delta_temperatures     = [0.0, 1.0, 2.0]        # air temperature offsets (K)
    precipitation_scalings = [0.8, 1.0, 1.2]        # precipitation multipliers (1)

    # Fraction of each cell's glacier area the modeled hypsometry bins must cover. Bins are
    # taken largest-area first; the area of the unmodeled remainder is folded into the nearest
    # modeled bin, so the full cell area always contributes to the mass totals.
    hypsometry_coverage = 0.95

    # Skip grid cells holding less than this total glacier area (km²).
    cell_area_minimum = 1.0

    # Restrict the sweep to the first N qualifying cells (`nothing` runs all of them). Start
    # small: a global sweep is many thousands of cells x the perturbation grid.
    cell_limit = 1

    # Extending an existing cell file requires that its stored run parameters (every
    # `ModelParameters` field, the hypsometry coverage, the lapse rate) match this sweep's;
    # otherwise the append is refused, because it would splice two different experiments into
    # one record. Set `true` to append across such a change anyway — the file's stored
    # parameters are then overwritten and the pre-seam record no longer reflects them. Changing
    # a model parameter normally means the cells should be rebuilt from scratch instead.
    force_restart = false

    using GEMB
    using Dates
    using Statistics
    using CairoMakie
    using Rasters               # also re-exports DimensionalData (dims, .val)
    using GeoDataFrames
    using GeoParquet            # backend for GeoDataFrames.write(...parquet)
    using GEMB_GlacierSims
    using GEMB_ClimateForcing
    using DimensionalData
    using Logging
    import NCDatasets              # to read the per-cell output files back for plotting

    # Set after the imports above: `DateTime` comes from Dates.
    forcing_time_range = (DateTime(1950, 1, 1), DateTime(2026, 8, 1))

    gemb_elevation_classes_file = joinpath(@__DIR__, "..", "data", "$(climate_model)_glacier_elevation_classes.parquet")
    output_dir = joinpath(@__DIR__, "..", "data", "gemb_runs", string(climate_model))
    forcing_cache = joinpath(tempdir(), ".cache", "$(climate_model)")

    #disable_logging(Logging.Info)

    # Get CDS API key (automatically reads from ENV or ~/.cdsapirc)
    cds_api_key = GEMB_ClimateForcing.get_cds_api_key()

    glacier_vector_file = get(ENV, "RGI_VECTOR_FILE",
        "/Users/gardnera/data/GlacierOutlines/RGI2000-v7.0-G-global-fix/rgi70_Global.gpkg")
end;

# Build the per-grid-cell glacier elevation-class table. For every chunk-map cell with
# fractional glacier cover above `glacier_cutoff`, this bins the ~30 m Copernicus DEM into a
# 100 m glacier hypsometry (glacier area, km², per elevation bin), attaches the invariant
# fields and chunk id, and returns a GeoParquet-ready DataFrame with a Point geometry column.
# The hypsometry is stored flat as one scalar column per bin (`hyps_<lo>_<hi>`); this keeps the
# Parquet columns 1-D so the cached file reads in ms rather than deserializing a nested list.
if !isfile(gemb_elevation_classes_file)

    @info "Building glacier elevation-class table (this may take several hours)..."

    glacier_polygons = GeoDataFrames.read(glacier_vector_file)

    # Lazy Copernicus 30 m DEM mosaic; cropped per tile and read on demand inside the runfile.
    dem = climate_model_invariant(model = :copernicus_dem_30m)

    # ERA5-Land download chunk map (:geo strategy); its grid defines the output cells.
    geo_chunk_map = climate_chunk_map(climate_model; chunk_strategy=:geo, token=cds_api_key)

    # Invariant fields carried through as columns. `era5_land_invariant` (from GEMB_GlacierSims)
    # reads them and rewraps native 0–359.9°E longitudes to the (-180, 180] grid; the runfile
    # regrids the stack onto its internal glacier grid as needed. Surface elevation is no longer
    # carried here: `climate_forcing` derives it per cell from the ERA5-Land geopotential invariant.
    era5_land_invariants = era5_land_invariant(parameter=(:glm, :lsm, :cl))

    glacier_elevation_classes = gemb_glacier_elevation_class_runfile(
        glacier_polygons, dem, geo_chunk_map, era5_land_invariants;
        elevation_bin_edges = 0:100:10000,
        glacier_cutoff = 0.0,
        oversample_factor = 10,
        dem_fetch_concurrency = 8,
    )

    ## Save glacier_elevation_classes as a GeoParquet file
    mkpath(dirname(gemb_elevation_classes_file))
    GeoDataFrames.write(gemb_elevation_classes_file, glacier_elevation_classes)
    @info "Saved $(nrow(glacier_elevation_classes)) glacier points to $(gemb_elevation_classes_file)"
else
    @info "Loading cached glacier elevation-class table from $(gemb_elevation_classes_file)"
    glacier_elevation_classes = GeoDataFrames.read(gemb_elevation_classes_file)
end;

# Add lat/lon from the Point geometry. Surface elevation is not derived here; `climate_forcing`
# provides the per-cell orthometric elevation in its output metadata.
begin
    glacier_elevation_classes[!,:longitude] = GeoDataFrames.GeoInterface.x.(glacier_elevation_classes.geometry)
    glacier_elevation_classes[!,:latitude] = GeoDataFrames.GeoInterface.y.(glacier_elevation_classes.geometry)
end;

## Sweep every qualifying glacier grid cell

# Model parameters, shared by every run. Monthly output keeps the per-cell files small over a
# 76-year record; `gemb_spinup` overrides the frequency to :last internally, so no separate
# spinup parameters are needed.
mp = initialize_parameters(output_frequency = :monthly);

# Cells holding enough glacier ice to be worth running. `glacier_area_total` sums the flat
# hyps_<lo>_<hi> columns; the bin selection itself (largest-area-first to `hypsometry_coverage`,
# with the unmodeled remainder folded into the nearest modeled bin so no area is dropped) happens
# inside `gemb_glacier_cell`, so the ~47,000-cell screen does not pay for it.
begin
    cell_rows = collect(eachrow(glacier_elevation_classes))
    qualifying = [i for i in eachindex(cell_rows)
                  if glacier_area_total(cell_rows[i]) >= cell_area_minimum]
    cell_limit === nothing || (qualifying = first(qualifying, cell_limit))
    @info "Cells to run" total=length(cell_rows) qualifying=length(qualifying) runs_per_cell="bins x $(length(delta_temperatures)) x $(length(precipitation_scalings))"
end;

# One NetCDF per cell, named by chunk id and cell center so a file is traceable to its cell.
# Degrees go into the name with '.' -> 'p' and '-' -> 'm' so the filename stays shell-safe.
_degrees_tag(x) = replace(string(round(x, digits = 3)), '.' => 'p', '-' => 'm')
cell_output_path(r) = joinpath(output_dir,
    "gemb_cell_" * lpad(r.chunk_id, 6, '0') * "_" *
    _degrees_tag(r.latitude) * "_" * _degrees_tag(wrap_lon(r.longitude)) * ".nc")

for i in qualifying
    r = cell_rows[i]
    path = cell_output_path(r)

    # One failing cell (a CDS timeout, an infeasible column grid) must not abort the sweep.
    try
        # An existing file carries the firn state and last time of the previous run; when present
        # the cell resumes from it over only the newer forcing and skips the spinup entirely.
        restart = read_glacier_cell_restart(path)

        # Download the full forcing time series for this cell from the Copernicus Climate Data
        # Store; cached locally so re-runs skip the download. The returned stack is
        # self-describing: its metadata carries the cell's absolute (orthometric) surface
        # elevation, which is the reference the per-bin lapse adjustment raises from.
        forcing_data = climate_forcing(climate_model, r.latitude, r.longitude;
                                       time_range = forcing_time_range,
                                       token = cds_api_key,
                                       cache_path = forcing_cache)

        # Runs (bins x deltas x scalings) simulations and area-weights the per-bin mass fluxes
        # (kg m-2) by the glacier area attributed to each bin, giving per-cell masses (kg).
        run = gemb_glacier_cell(r, forcing_data, mp;
                                delta_temperatures, precipitation_scalings,
                                coverage = hypsometry_coverage,
                                restart, force_restart)

        if restart === nothing
            write_glacier_cell_netcdf(path, run;
                                      institution = "NASA Jet Propulsion Laboratory")
        else
            append_glacier_cell_netcdf(path, run)
        end

        @info "Wrote cell" path bins=length(run.bins) area_km2=sum(run.weights) steps=length(run.time)
    catch e
        e isa InterruptException && rethrow()
        # ERA5-Land is land-only, so a cell whose reanalysis grid point falls on water (common
        # for coastal and island glaciers) has no forcing at all. Those cells are unrunnable
        # rather than failed, so skip them quietly instead of logging an error per cell.
        if e isa ForcingUnavailable
            @info "Skipping cell: no land forcing at this reanalysis grid point" cell=i lat=r.latitude lon=r.longitude
            continue
        end
        # A run-parameter change is a property of this sweep's configuration, not of one cell,
        # so it would fail identically for every remaining cell. Abort rather than log it
        # thousands of times; the message says how to proceed deliberately.
        e isa RestartParameterMismatch && rethrow()
        # An existing file that already spans the forcing is up to date, not broken; re-running
        # the sweep before new forcing is published hits this for every completed cell.
        if e isa ForcingUpToDate
            @info "Cell already up to date" cell=i path restart_time=e.restart_time
            continue
        end
        @error "Cell failed; continuing" cell=i path exception=(e, catch_backtrace())
    end
end

# Quick-look plot of the per-cell mass totals for the first cell written: cumulative mass change
# for every prescribed (temperature delta, precipitation scaling) combination.
begin
    files = isdir(output_dir) ? filter(endswith(".nc"), readdir(output_dir; join = true)) : String[]
    if !isempty(files)
        NCDatasets.NCDataset(first(files), "r") do ds
            # `datetime2decyear` takes the whole vector; GEMB and GEMB_ClimateForcing both
            # export it, so qualify which one.
            years = GEMB.datetime2decyear(collect(DateTime, ds["time"][:]))
            fig = Figure(size = (900, 500))
            ax = Axis(fig[1, 1]; xlabel = "year", ylabel = "cumulative mass change (Gt)",
                      title = basename(first(files)))
            for (j, dT) in enumerate(ds["delta_temperature"][:]),
                (k, ps) in enumerate(ds["precipitation_scaling"][:])
                # 1 Gt = 1e12 kg.
                lines!(ax, years, cumsum(ds["mass_change"][:, j, k]) .* 1e-12;
                       label = "ΔT=$(dT) K, P×$(ps)")
            end
            axislegend(ax; position = :lb, framevisible = false)
            display(fig)
        end
    end
end
