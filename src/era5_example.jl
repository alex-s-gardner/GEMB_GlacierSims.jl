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

# cached glacier elevation-class table, will only be built if it doesn't already exist
gemb_elevation_classes_file = joinpath(@__DIR__, "..", "data", "era5_land_glacier_elevation_classes.parquet") 

using GEMB
using Dates
using Statistics
using CairoMakie
using Rasters               # also re-exports DimensionalData (dims, .val)
using GeoDataFrames
using GeoParquet            # backend for GeoDataFrames.write(...parquet)
using GEMB_GlacierSims
using GEMB_ClimateForcing

# Get CDS API key (automatically reads from ENV or ~/.cdsapirc)
cds_api_key = GEMB_ClimateForcing.get_cds_api_key()

glacier_vector_file = get(ENV, "RGI_VECTOR_FILE",
    "/Users/gardnera/data/GlacierOutlines/RGI2000-v7.0-G-global-fix/rgi70_Global.gpkg")


# Build the per-grid-cell glacier elevation-class table. For every chunk-map cell with
# fractional glacier cover above `glacier_cutoff`, this bins the ~30 m Copernicus DEM into a
# 100 m glacier hypsometry (glacier area, km², per elevation bin), attaches the invariant
# fields and chunk id, and returns a GeoParquet-ready DataFrame with a Point geometry column.
# The hypsometry is stored flat as one scalar column per bin (`hyps_<lo>_<hi>`); this keeps the
# Parquet columns 1-D so the cached file reads in ms rather than deserializing a nested list.
if !isfile(gemb_elevation_classes_file)

    println("Building glacier elevation-class table (this may take several hours)...")

    glacier_polygons = GeoDataFrames.read(glacier_vector_file)

    # Lazy Copernicus 30 m DEM mosaic; cropped per tile and read on demand inside the runfile.
    dem = climate_model_invariant(model = :copernicus_dem_30m)

    # ERA5-Land download chunk map (:geo strategy); its grid defines the output cells.
    geo_chunk_map = climate_chunk_map(:era5land; chunk_strategy=:geo, token=cds_api_key)

    # Invariant fields carried through as columns. `era5_land_invariant` (from GEMB_GlacierSims)
    # reads them and rewraps native 0–359.9°E longitudes to the (-180, 180] grid; the runfile
    # regrids the stack onto its internal glacier grid as needed.
    era5_land_invariants = era5_land_invariant(parameter=(:glm, :z, :lsm, :cl))

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
    println("Saved $(nrow(glacier_elevation_classes)) glacier points to $(gemb_elevation_classes_file)")
else
    println("Loading cached glacier elevation-class table from $(gemb_elevation_classes_file)")
    glacier_elevation_classes = GeoDataFrames.read(gemb_elevation_classes_file)
end

# Add lat/lon and orthometric height:                          
glacier_elevation_classes[!,:longitude] = GeoDataFrames.GeoInterface.x.(glacier_elevation_classes.geometry)
glacier_elevation_classes[!,:latitude] = GeoDataFrames.GeoInterface.y.(glacier_elevation_classes.geometry)
glacier_elevation_classes[!,:height_orthometric] = geopotential2height.(glacier_elevation_classes.z, glacier_elevation_classes[!,:longitude], glacier_elevation_classes[!,:latitude]; height_reference=:orthometric)      

# Coordinates for one glacier cell to hand to `climate_forcing`, read off its Point geometry
# (lon in (-180, 180], lat).
r = eachrow(glacier_elevation_classes)[1]


## Download ERA5-Land forcing data

# Download the full climate forcing time series for the selected cell (lat, lon) from the
# Copernicus Climate Data Store; cached locally so re-runs skip the download.
@time forcing_data = climate_forcing(:era5land, r.latitude, r.longitude;
                                time_range=(DateTime(1950,1,1), DateTime(2026,8,1)),
                                token=cds_api_key,
                                cache_path=joinpath(tempdir(), ".cache", "era5land"))

# Optionally lapse-rate-adjust the forcing to a different elevation (here +100 m) before running.
@time forcing_data = climate_adjust_for_elevation(
    forcing_data,
    100.0;
    lapse_rate=6.5,
    precip_scaling_method = nothing
)


# Build a repeating climatological year from the forcing, used to spin the model up.
cf_spinup = forcing_climatology(forcing_data, (DateTime(1950,1,1), DateTime(1980,12,31)))



## Run GEMB

# Model parameters; write output at daily frequency for the transient run.
mp = ModelParameters(output_frequency=:weekly)



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

