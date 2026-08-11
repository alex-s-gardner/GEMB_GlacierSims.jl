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
glacier_cutoff = 0.0;        # min fractional glacier cover for a cell to be included
dem_fetch_concurrency = 8;   # max concurrent GDAL /vsicurl DEM fetches (1 = fully serial)

using GEMB
using Dates
using Statistics
using CairoMakie
using Rasters               # also re-exports DimensionalData (dims, .val)
using GeoDataFrames
using GeoParquet            # backend for GeoDataFrames.write(...parquet)
using GEMB_GlacierSims

# Check if GEMB_ClimateForcing is available
try
    using GEMB_ClimateForcing
catch e
    @error """
    GEMB_ClimateForcing.jl not found!

    To run this example, install GEMB_ClimateForcing.jl from GitHub:
        using Pkg
        Pkg.add(url="https://github.com/alex-s-gardner/GEMB_ClimateForcing.jl")

    Then get a CDS API key from: https://cds.climate.copernicus.eu/api-how-to
    And set: ENV["CDS_API_KEY"] = "your-key-here"
    """
    rethrow(e)
end

# Get CDS API key (automatically reads from ENV or ~/.cdsapirc)
cds_api_key = GEMB_ClimateForcing.get_cds_api_key()

glacier_vector_file = get(ENV, "RGI_VECTOR_FILE",
    "/Users/gardnera/data/GlacierOutlines/RGI2000-v7.0-G-global-fix/rgi70_Global.gpkg")

glacier_polygons = GeoDataFrames.read(glacier_vector_file)

# Lazy Copernicus 30 m DEM mosaic; cropped per tile and read on demand inside the runfile.
dem = climate_model_invariant(model = :copernicus_dem_30m)

# ERA5-Land download chunk map (:geo strategy); its grid defines the output cells.
geo_chunk_map = climate_chunk_map(:era5land; chunk_strategy=:geo, token=cds_api_key)

# Invariant fields carried through as columns. `era5_land_invariant` (from GEMB_GlacierSims)
# reads them and rewraps native 0–359.9°E longitudes to the (-180, 180] grid; the runfile
# regrids the stack onto its internal glacier grid as needed.
era5_land_invariants = era5_land_invariant(parameter=(:glm, :z, :lsm))

# Build the per-grid-cell glacier elevation-class table. For every chunk-map cell with
# fractional glacier cover above `glacier_cutoff`, this bins the ~30 m Copernicus DEM into a
# 100 m glacier hypsometry vector (glacier area, km², per elevation bin), attaches the invariant
# fields and chunk id, and returns a GeoParquet-ready DataFrame with a Point geometry column.
@time glacier_points = gemb_glacier_elevation_class_runfile(
    glacier_polygons, dem, geo_chunk_map, era5_land_invariants;
    elevation_bin_edges = 0:100:10000,
    glacier_cutoff = glacier_cutoff,
    oversample_factor = 10,
    dem_fetch_concurrency = dem_fetch_concurrency,
)

## Save glacier_points as a GeoParquet file
glacier_points_file = joinpath(@__DIR__, "..", "data", "era5_land_glacier_elevation_classes.parquet")
mkpath(dirname(glacier_points_file))
GeoDataFrames.write(glacier_points_file, glacier_points)
println("Saved $(nrow(glacier_points)) glacier points to $(glacier_points_file)")



## Download ERA5-Land forcing data

# Download the full climate forcing time series for Summit Station, Greenland (72.58°N, 38.48°W)
# from the Copernicus Climate Data Store; cached locally so re-runs skip the download.
@time forcing_data = climate_forcing(:era5land, 72.58, -38.48;
                                time_range=(DateTime(1979,1,1), DateTime(2025,12,31)),
                                token=cds_api_key,
                                cache_path=joinpath(tempdir(), ".cache", "era5land"))

# Optionally lapse-rate-adjust the forcing to a different elevation (here +100 m) before running.
@time forcing_data = climate_adjust_for_elevation(
    forcing_data,
    100.0;
    lapse_rate=6.5,
    precip_scaling_method = nothing
)

# Convert the downloaded forcing into a GEMB ClimateForcing (automatic via package extension).
cf = GEMB.ClimateForcing(forcing_data)

## Run GEMB

# Model parameters; write output at daily frequency for the transient run.
mp = ModelParameters(output_frequency=:daily)

# Build a repeating climatological year from the forcing, used to spin the model up.
cf_spinup = forcing_climatology(cf)

# Initialize the firn column (layer geometry, density, temperature) from the spinup climate.
profile = initialize_profile(mp, cf_spinup)

# Spin up on the climatology (keeping only the final state) until density converges or 100 iters.
mp_spinup = ModelParameters(output_frequency=:last)
@time profile_spunup = gemb_spinup(profile, cf_spinup, mp_spinup; max_iterations = 100, convergence_delta_density = 0.01)

# Run the transient simulation from the spun-up profile over the full forcing record.
@time output = gemb(profile_spunup, cf, mp)

# Quick-look plot of the standard GEMB output fields.
gemb_plot_output(output)

## Post-processing examples

# Depth of each layer's center from its thickness (dz), for plotting against depth.
z_center = dz2z(output[:dz])

# Extract the surface (top-layer) temperature as a time series.
temp_surface = surface_timeseries(output[:temperature])

# Interpolate temperature onto the spun-up profile's fixed vertical grid so it can be heatmapped.
temp_gridded = gemb_interp(z_center, output[:density], profile_spunup[:z_center])

# Heatmap of the temperature profile over time (top 5 m of the column).
f = Figure()
ax = Axis(f[1,1], xlabel="Year", ylabel="Depth (m)", title="Temperature Profile")
hm = heatmap!(ax, GEMB.datetime2decyear(dims(temp_gridded,2).val), dims(temp_gridded,1).val, parent(temp_gridded'))
Colorbar(f[1,2], hm, label="Temperature (°C)")
ylims!(ax, -5, 0)
f

# Recover relative humidity from the model's vapor pressure and air temperature.
rh = vapor_pressure_to_relative_humidity(
    parent(cf.vapor_pressure), parent(cf.temperature_air))

println("Simulation complete!")
println("  Mean surface albedo: ", round(mean(parent(output[:albedo_surface])), digits=3))
println("  Mean firn air content: ", round(mean(parent(output[:firn_air_content])), digits=3), " m")

