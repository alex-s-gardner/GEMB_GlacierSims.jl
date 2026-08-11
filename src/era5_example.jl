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
glacier_cutoff = 0.0;
land_cutoff = 0.0;
dem_fetch_concurrency = 8;   # max concurrent GDAL /vsicurl DEM fetches (1 = fully serial)

using GEMB
using DimensionalData
using Dates
using Statistics
using CairoMakie
using Rasters
using GeoDataFrames
using GeoParquet
using Proj
using ProgressMeter
using SortTileRecursiveTree
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

# geo_chunk_map
dem = climate_model_invariant(
      model  = :copernicus_dem_30m,
  )

geo_chunk_map  = climate_chunk_map(:era5land; chunk_strategy=:geo,  token=cds_api_key)

oversample_factor = 10
@time begin
    _fine = Rasters.disaggregate(geo_chunk_map, oversample_factor)
    _fine_binary = Raster(zeros(Float32, size(_fine)), dims(_fine); missingval=NaN32)
    rasterize!(_fine_binary, glacier_polygons; fill=1.0f0, reducer=last, boundary=:center)
    glaciers = Rasters.aggregate(mean, _fine_binary, oversample_factor)
end

# If you need the ERA5-Land glacier mask, you can read it in as follows.
# `era5_land_invariant` (from GEMB_GlacierSims) reads the field and rewraps its
# native 0–359.9°E longitudes to match the (-180, 180] grid used by `glaciers`.
era5_land_mask = era5_land_invariant(parameter=(:glm, :z, :lsm), to=glaciers)

# find all glacier grid cells in the ERA5-Land mask
glacier_points = DataFrame();
glacier_points[!, :index] = findall((glaciers .> glacier_cutoff) .& (era5_land_mask[:lsm] .> land_cutoff))
glacier_points[!, :chunk_id] = geo_chunk_map[glacier_points.index]
glacier_points[!, :glacier_frac] = glaciers[glacier_points.index]

for k in keys(era5_land_mask)
    glacier_points[!, k] = era5_land_mask[k][glacier_points.index]
end


# GeoParquet needs a real geometry column, and Parquet can't hold the CartesianIndex `index`
# column. Build a Point geometry from each cell's (lon, lat) up front; downstream code reads
# each cell's longitude/latitude straight off this geometry (see `cell_lonlat`). The
# `hypsometry` Vector{Float64} column round-trips fine as a list column.
glacier_points[!, :geometry] = [GeoDataFrames.GeoInterface.Point(xy...) for xy in  DimPoints(glaciers)[glacier_points.index]]
metadata!(glacier_points, "GEOINTERFACE:geometrycolumns", (:geometry,); style=:note)
metadata!(glacier_points, "GEOINTERFACE:crs", GeoDataFrames.GFT.EPSG(4326); style=:note)

sort!(glacier_points, :chunk_id)

# Glacier hypsometry within each ERA5-Land grid cell.
# For each cell we clip the ~30 m Copernicus DEM to the cell extent, rasterize the RGI
# glacier polygons onto that fine grid, weight each glacier pixel by its true (spherical)
# area, and bin those areas into 100 m elevation intervals from 0 to 10000 m.
#
# Performance: the DEM is tiled as 1°×1° COGs, so ~100 of these 0.1° cells share one tile.
# We group cells by tile, fetch each tile's DEM once, pre-filter the RGI polygons to the tile
# with an STR-tree spatial index, and then bin every contained cell from the in-memory tile —
# turning tens of thousands of remote reads into a few hundred.
const bin_edges = 0:100:10000      # 100 bins: [0,100), [100,200), … [9900,10000)
const n_bins    = length(bin_edges) - 1

# Record the elevation bin edges (m) on the :glacier_hypsometry column so each area vector's bins
# are self-describing: bin i spans [bin_edges[i], bin_edges[i+1]).
glacier_points[!, :glacier_hypsometry] = Vector{Union{Missing,Vector{Float64}}}(missing, nrow(glacier_points))
colmetadata!(glacier_points, :glacier_hypsometry, "bin_edges", collect(bin_edges); style=:note)

# Spatial index over all RGI polygons, built once. `query(gtree, box)` returns the indices of
# polygons whose bounding box intersects `box` (bbox over-inclusion is harmless — rasterize
# only burns pixels actually covered).
gcol  = first(GeoDataFrames.getgeometrycolumns(glacier_polygons))   # usually :geometry
geoms = glacier_polygons[!, gcol]
gtree = STRtree(geoms)

# Bounds how many GDAL /vsicurl DEM opens run at once (see tile_hypsometry!). Fully
# unbounded concurrency crashes GDAL with a NULL dataset handle; a small pool overlaps
# the network waits while staying within what GDAL tolerates. Set via
# `dem_fetch_concurrency` at the top of the file.
const DEM_FETCH_POOL = Base.Semaphore(dem_fetch_concurrency)

# ERA5-Land cell extent (from its CartesianIndex in `glaciers`) normalized to the DEM's
# −180…180°E convention. `glaciers` is a regular 0.1° grid, so we take the cell center ±
# half the grid step for the edges — its lookups use Points sampling, so `intervalbounds`
# would collapse to zero width. `glaciers` inherits the native 0–359.9°E ERA5 grid, so a
# cell whose center is east of 180° is shifted by −360° to line up with the Copernicus DEM.
# NOTE: a cell straddling the antimeridian (center ≈ 180°) is not handled specially.
function era5_cell_extent(I)
    xdim, ydim = dims(glaciers, X), dims(glaciers, Y)
    hx, hy = abs(step(xdim)) / 2, abs(step(ydim)) / 2
    xc, yc = xdim[I[1]], ydim[I[2]]
    shift = dem_lon(xc) - xc   # 0 west of 180°, −360 east of it (see wrap_lon)
    Extents.Extent(X = (xc - hx + shift, xc + hx + shift),
                   Y = (yc - hy, yc + hy))
end

# Longitude on the DEM's −180…180°E convention (glaciers are native 0–359.9°E).
dem_lon(lon) = float(wrap_lon(lon))

# (lon, lat) of a cell, read off its Point geometry (native 0–359.9°E, same as `glaciers`).
cell_lonlat(pt) = (GeoDataFrames.GeoInterface.x(pt), GeoDataFrames.GeoInterface.y(pt))

# O(1) bin lookup for the uniform range bin_edges = 0:100:10000; 0 means "outside 0–10000 m".
@inline binindex(z) = (b = floor(Int, z / 100) + 1; 1 <= b <= n_bins ? b : 0)

# Compute hypsometry for every glacier cell in `rows` (all sharing one 1° DEM tile) and write
# each result into gp[r, :glacier_hypsometry]. Fetches the padded tile DEM once and reuses it.
function tile_hypsometry!(gp, rows, geoms, gtree, grid_size)
    geomcol = gp.geometry
    lonlats = [cell_lonlat(geomcol[r]) for r in rows]   # decode each geometry once
    lons = [dem_lon(ll[1]) for ll in lonlats]
    lats = [float(ll[2])   for ll in lonlats]
    # Pad by one climate half-cell per axis so every cell's window (center ± half-cell) is fully
    # covered by the fetched tile. `grid_size` is (Δlon, Δlat) in degrees.
    padx, pady = grid_size[1] / 2, grid_size[2] / 2
    # Clamp to the DEM's valid range: the pad is only margin, and cell windows sit within it, so
    # clamping near the poles / antimeridian still covers every real cell.
    box  = Extents.Extent(X = (clamp(minimum(lons) - padx, -180.0, 180.0),
                               clamp(maximum(lons) + padx, -180.0, 180.0)),
                          Y = (clamp(minimum(lats) - pady,  -90.0,  90.0),
                               clamp(maximum(lats) + pady,  -90.0,  90.0)))

    idx = query(gtree, box)
    if isempty(idx)
        for r in rows                      # no glaciers over this tile → all-zero hypsometry
            gp[r, :glacier_hypsometry] = zeros(Float64, n_bins)
        end
        return
    end
    localgeoms = geoms[idx]

    # ~30 m Copernicus DEM (m, EPSG:4326) for the whole tile; only the covering tiles fetched.
    # GDAL's /vsicurl dataset open is not thread-safe under full concurrency, so cap concurrent
    # fetches with a semaphore; `read` materializes it into memory, so everything downstream
    # (view/rasterize/cellarea) then works on plain arrays outside the pool.
    #
    # Some ERA5-Land cells flagged as glacier land sit where the Copernicus DEM publishes no
    # tiles (it omits ocean tiles). The fetch throws ArgumentError there; treat such a tile as
    # having no DEM coverage → all-zero hypsometry, rather than crashing the whole @threads run.
    Base.acquire(DEM_FETCH_POOL)
    dem = try
        read(climate_model_invariant(model=:copernicus_dem_30m, extent=box, verbose=false))
    catch e
        # Only the "no published DEM tiles" case is expected here; anything else is a real
        # failure and must not be silently zero-filled.
        (e isa ArgumentError && occursin("No Copernicus DEM tiles", e.msg)) || rethrow(e)
        for r in rows
            gp[r, :glacier_hypsometry] = zeros(Float64, n_bins)
        end
        return
    finally
        Base.release(DEM_FETCH_POOL)
    end

    for r in rows
        ext = era5_cell_extent(gp.index[r])   # −180…180°E, half-step cell box
        subdem = view(dem, X = (ext.X[1] .. ext.X[2]), Y = (ext.Y[1] .. ext.Y[2]))
        # boolean glacier mask on the cell's DEM grid, from the pre-filtered polygons
        submask = rasterize(last, localgeoms; to=subdem, fill=true, missingval=false,
                            boundary=:center, progress=false, verbose=false)
        # cellarea needs Intervals sampling; the DEM lookups come in as Points, so retag them.
        # Area varies only with latitude, so we compute it over a single column (X(1:1)) and
        # get per-row (Y) km²; index by j.
        subint = set(subdem[X(1:1)],
            X => DimensionalData.Lookups.Intervals(DimensionalData.Lookups.Center()),
            Y => DimensionalData.Lookups.Intervals(DimensionalData.Lookups.Center()))
        area_by_row = vec(parent(cellarea(subint))) ./ 1e6   # length ny(Y), m² → km²

        D = parent(subdem); M = parent(submask)   # both nx × ny; iterate raw matrices
        hyps = zeros(Float64, n_bins)
        @inbounds for j in axes(D, 2), i in axes(D, 1)
            (M[i, j] && isfinite(D[i, j])) || continue
            b = binindex(D[i, j]); b == 0 && continue
            hyps[b] += area_by_row[j]
        end
        gp[r, :glacier_hypsometry] = hyps
    end
    return
end

# Climate grid cell size (Δlon, Δlat) in degrees, read from the glaciers grid so it stays
# correct if the grid changes. Used to pad each tile's DEM fetch by one half-cell.
climate_grid_size = (abs(step(dims(glaciers, X))), abs(step(dims(glaciers, Y))))

# Group the first N glacier cells by their 1° DEM tile (SW integer corner, −180…180°E).
N = nrow(glacier_points)
tile_key(lon, lat) = (floor(Int, dem_lon(lon)), floor(Int, lat))
tiles = Dict{Tuple{Int,Int},Vector{Int}}()
for i in 1:N
    push!(get!(tiles, tile_key(cell_lonlat(glacier_points.geometry[i])...), Int[]), i)
end
tilelist = collect(tiles)

# One DEM fetch per tile; threads over tiles. Writes target disjoint rows, so this is safe.
# The DEM fetch is capped by a semaphore inside tile_hypsometry! (GDAL /vsicurl tolerates
# only bounded concurrency); rasterize + binning run fully concurrently across tiles.
prog = Progress(length(tilelist); desc="Glacier hypsometry by tile: ", showspeed=true)
Threads.@threads for t in eachindex(tilelist)
    tile_hypsometry!(glacier_points, tilelist[t].second, geoms, gtree, climate_grid_size)
    next!(prog)
end
finish!(prog)

## Save glacier_points as a GeoParquet file
glacier_points_file = joinpath(@__DIR__, "..", "data", "era5_land_glacier_elevation_classes.parquet")
mkpath(dirname(glacier_points_file))
GeoDataFrames.write(glacier_points_file, select(glacier_points, Not(:index)))
println("Saved $(nrow(glacier_points)) glacier points to $(glacier_points_file)")






## Download ERA5-Land forcing data

# Download data for Summit Station, Greenland (72.58°N, 38.48°W)
# This will download data from the Copernicus Climate Data Store

@time forcing_data = climate_forcing(:era5land, 72.58, -38.48;
                                time_range=(DateTime(1979,1,1), DateTime(2025,12,31)),
                                token=cds_api_key,
                                cache_path=joinpath(tempdir(), ".cache", "era5land"))


# Optionally adjust the forcing to a different elevation (e.g. +100 m) before running.
@time forcing_data = climate_adjust_for_elevation(
    forcing_data,
    100.0;
    lapse_rate=6.5,
    precip_scaling_method = nothing
)

# Convert to GEMB ClimateForcing (automatic via package extension)
cf = GEMB.ClimateForcing(forcing_data)

## Run GEMB

# Initialize model parameters
mp = ModelParameters(output_frequency=:daily)

# Create climatological forcing for spinup
cf_spinup = forcing_climatology(cf)

# Initialize the firn column
profile = initialize_profile(mp, cf_spinup)

# Spin up for 100 years to reach quasi-steady state
mp_spinup = ModelParameters(output_frequency=:last)
@time profile_spunup = gemb_spinup(profile, cf_spinup, mp_spinup; max_iterations = 100, convergence_delta_density = 0.01)

# Run GEMB with transient forcing and the spun-up profile
@time output = gemb(profile_spunup, cf, mp)

gemb_plot_output(output)

## Post-processing examples

# Get grid cell centers for plotting
z_center = dz2z(output[:dz])

# Get surface temperature time series
temp_surface = surface_timeseries(output[:temperature])

# Regrid to fixed vertical coordinate for plotting
temp_gridded = gemb_interp(z_center, output[:density], profile_spunup[:z_center])

f = Figure()
ax = Axis(f[1,1], xlabel="Year", ylabel="Depth (m)", title="Temperature Profile")
hm = heatmap!(ax, GEMB.datetime2decyear(dims(temp_gridded,2).val), dims(temp_gridded,1).val, parent(temp_gridded'))
Colorbar(f[1,2], hm, label="Temperature (°C)")
ylims!(ax, -5, 0)
f


# Convert vapor pressure back to relative humidity
rh = vapor_pressure_to_relative_humidity(
    parent(cf.vapor_pressure), parent(cf.temperature_air))

println("Simulation complete!")
println("  Mean surface albedo: ", round(mean(parent(output[:albedo_surface])), digits=3))
println("  Mean firn air content: ", round(mean(parent(output[:firn_air_content])), digits=3), " m")

