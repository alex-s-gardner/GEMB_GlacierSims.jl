# Example of running GEMB with ERA5 reanalysis data using GEMB_ClimateForcing.jl
#
# This example uses the GEMB_ClimateForcing.jl package to automatically download
# and format ERA5-Land climate data.
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

    # Skip hypsometry bins holding less than this glacier area (km²) when looping over elevation classes.
    glacier_area_minimum = 0

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

    gemb_elevation_classes_file = joinpath(@__DIR__, "..", "data", "$(climate_model)_glacier_elevation_classes.parquet")

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

# Coordinates for one glacier cell to hand to `climate_forcing`, read off its Point geometry
# (lon in (-180, 180], lat).
begin
    r = eachrow(glacier_elevation_classes)[3]


    ## Download ERA5-Land forcing data

    # Download the full climate forcing time series for the selected cell (lat, lon) from the
    # Copernicus Climate Data Store; cached locally so re-runs skip the download. The returned
    # stack is self-describing: its metadata carries the cell's absolute (orthometric) surface
    # elevation, derived from the ERA5-Land geopotential invariant. We use it as the reference
    # the per-bin lapse adjustment raises from, so the adjustment records an absolute target
    # elevation (= bin_center) that flows through to the GEMB output and the plot header banner.
    forcing_data = climate_forcing(climate_model, r.latitude, r.longitude;
                                    time_range=(DateTime(1950,1,1), DateTime(2026,8,1)),
                                    token=cds_api_key,
                                    cache_path=joinpath(tempdir(), ".cache", "$(climate_model)"));

    forcing_elevation = metadata(forcing_data)["elevation"]   # ERA5-Land cell surface elevation (m)


    ## Run GEMB once per populated glacier elevation bin

    # Populated hypsometry bins for this cell, decoded from the flat hyps_<lo>_<hi> columns by the
    # package (edges/center/area already parsed; bins below the area threshold dropped).
    bins = glacier_hypsometry(r; area_minimum = glacier_area_minimum);

    # Collect one GEMB output per elevation bin above the area threshold, keyed by bin-center elevation (m).
    bin_outputs = Dict{Float64, Any}();

    for bin in bins
        bin_center = bin.center                          # target elevation (m)

        # Lapse-rate-adjust the (once-downloaded) forcing to this bin's center, relative to the
        # forcing's own surface elevation, and convert it to a ClimateForcing. Adjusting from the
        # original `forcing_data` each time keeps the elevation deltas from compounding across bins.
        cf = forcing_at_elevation(forcing_data, bin_center - forcing_elevation)

        # Build a repeating climatological year from the forcing, used to spin the model up.
        cf_spinup = forcing_climatology(cf, (DateTime(1950,1,1), DateTime(1980,12,31)))

        # Model parameters; write output at monthly frequency for the transient run.
        mp = initialize_parameters(output_frequency=:monthly)

        # Initialize the firn column (layer geometry, density, temperature) from the spinup climate.
        # `initialize_profile` returns a possibly-adjusted `mp` (shrunken column limits when
        # `depth_autoadjust` fires for an ice column); thread it through spinup AND the transient run.
        (initial_profile, initial_mp) = initialize_profile(mp, cf_spinup)

        # Spin up on the climatology (keeping only the final state) until density converges or 1000 iters.
        # gemb_spinup internally forces output_frequency=:last, so no separate :last params are needed.
        profile_spunup = gemb_spinup(initial_profile, cf_spinup, initial_mp; max_iterations = 1000, convergence_delta_density = 0.01)
        
        # Run the transient simulation from the spun-up profile over the full forcing record.
        output = gemb(profile_spunup, cf, initial_mp)

        bin_outputs[bin_center] = output
        @info "Ran GEMB bin" bin_center area=bin.area elevation_delta=(bin_center - forcing_elevation)
    end
end

# Quick-look plot of the standard GEMB output fields for the highest-elevation bin that ran.
if !isempty(bin_outputs)
    gemb_plot_output(bin_outputs[maximum(keys(bin_outputs))])
end
