# Glacier elevation-class run file builder.
#
# Turns RGI glacier polygons, a Copernicus 30 m DEM mosaic, an ERA5-Land chunk map, and a
# pre-read invariant RasterStack into a per-grid-cell table where each glacier cell carries a
# 100 m glacier hypsometry (glacier area, km², per elevation bin), stored flat as one scalar
# column per elevation bin.
#
# This is the reusable, global-free version of the script in `era5_example.jl`: every helper
# takes its dependencies as arguments rather than reading module-level globals.

"""
    gemb_glacier_elevation_class_runfile(glacier_polygons, dem, chunk_map, invariant_parameters;
                                         elevation_bin_edges = 0:100:10000,
                                         glacier_cutoff = 0.0,
                                         oversample_factor = 10,
                                         dem_fetch_concurrency = 8)

Build a per–grid-cell glacier elevation-class table.

For every cell of the `chunk_map` grid whose fractional glacier cover exceeds `glacier_cutoff`,
compute the glacier hypsometry — glacier area (km²) binned into `elevation_bin_edges` (m) — from
the Copernicus DEM, and attach the fields of `invariant_parameters` and the chunk id.

Arguments:
- `glacier_polygons`: glacier outlines (e.g. RGI v7) as a GeoDataFrame / table with a geometry
  column, in EPSG:4326.
- `dem`: a lazy Copernicus 30 m DEM mosaic (e.g. `climate_model_invariant(model=:copernicus_dem_30m)`),
  cropped per tile and read on demand.
- `chunk_map`: the climate download chunk map (`climate_chunk_map(...)`); its grid defines the
  output cells and supplies each cell's `:chunk_id`. Longitudes native 0–359.9°E.
- `invariant_parameters`: a pre-read `RasterStack` of invariant fields (e.g. from
  `era5_land_invariant(parameter=(:glm, :z, :lsm))`). Each layer becomes a column; regridded to
  the internal glacier grid if its dims differ.

Keywords:
- `elevation_bin_edges`: monotone elevation bin edges (m); bin `i` spans
  `[edges[i], edges[i+1])`. Each bin becomes a scalar `hyps_<lo>_<hi>` column (the column names are
  the authoritative, GeoParquet-persisted record of the binning). The full edge vector is also set
  as `"hypsometry_bin_edges"` table metadata for convenience, but GeoParquet does not serialize it.
- `glacier_cutoff`: minimum fractional glacier cover for a cell to be included.
- `oversample_factor`: factor by which the chunk-map grid is disaggregated before rasterizing
  the glacier polygons, to estimate fractional cover.
- `dem_fetch_concurrency`: max concurrent GDAL /vsicurl DEM reads (GDAL tolerates only bounded
  concurrency).
- `verbose`: show a per-tile progress bar while computing hypsometry (set `false` for headless
  or batch runs).

Returns the glacier-points DataFrame with columns `:chunk_id`, `:glacier_frac`, one column per
`invariant_parameters` layer, `:geometry` (Point, EPSG:4326), and one scalar `Float64` column per
elevation bin named `hyps_<lo>_<hi>` (m edges, zero-padded to 5 digits; the glacier area in km² for
that bin). The `:index` column is removed. The binning is recoverable from the `hyps_*` column
names; the edges are also set as `"hypsometry_bin_edges"` table metadata (not persisted by GeoParquet).
"""
function gemb_glacier_elevation_class_runfile(
    glacier_polygons, dem, chunk_map, invariant_parameters;
    elevation_bin_edges = 0:100:10000,
    glacier_cutoff = 0.0,
    oversample_factor = 10,
    dem_fetch_concurrency = 8,
    verbose = true,
)
    bin_edges = elevation_bin_edges
    n_bins = length(bin_edges) - 1
    n_bins >= 1 || throw(ArgumentError("elevation_bin_edges must have at least two edges"))

    # 1. Fractional glacier cover on the chunk-map grid.
    #    Disaggregate, burn the polygons onto the fine grid, then average back to the coarse grid.
    fine = Rasters.disaggregate(chunk_map, oversample_factor)
    fine_binary = Raster(zeros(Float32, size(fine)), dims(fine); missingval=NaN32)
    Rasters.rasterize!(fine_binary, glacier_polygons; fill=1.0f0, reducer=last, boundary=:center)
    glaciers = Rasters.aggregate(mean, fine_binary, oversample_factor)

    # 2. Regrid the invariant stack onto the glacier grid if it is not already aligned.
    stack = _align_stack_to(invariant_parameters, glaciers)

    # 3. Select glacier cells and assemble the DataFrame.
    glacier_points = DataFrame()
    glacier_points[!, :index] = findall(glaciers .> glacier_cutoff)

    # Point geometry from each cell's (lon, lat); Parquet cannot hold the CartesianIndex column,
    # and downstream code reads lon/lat off the geometry (see `_cell_lonlat`).
    pts = DimPoints(glaciers)
    glacier_points[!, :geometry] =
        [GeoDataFrames.GeoInterface.Point(pts[i]...) for i in glacier_points.index]
    metadata!(glacier_points, "GEOINTERFACE:geometrycolumns", (:geometry,); style=:note)
    metadata!(glacier_points, "GEOINTERFACE:crs", GeoDataFrames.GFT.EPSG(4326); style=:note)
    
    glacier_points[!, :chunk_id] = chunk_map[glacier_points.index]
    glacier_points[!, :glacier_frac] = glaciers[glacier_points.index]
    for k in keys(stack)
        glacier_points[!, k] = stack[k][glacier_points.index]
    end

    sort!(glacier_points, :chunk_id)

    # 4. Hypsometry results matrix (nrow × n_bins): bin i spans [bin_edges[i], bin_edges[i+1]).
    #    Stored flat as one scalar Float64 column per bin (see step 8); a shared matrix lets the
    #    threaded tile loop write disjoint rows without boxing.
    H = zeros(Float64, nrow(glacier_points), n_bins)

    # 5. Spatial index over all polygons + grouping into 1° DEM tiles.
    gcol = first(GeoDataFrames.getgeometrycolumns(glacier_polygons))
    geoms = glacier_polygons[!, gcol]
    gtree = STRtree(geoms)

    # Climate grid cell size (Δlon, Δlat) in degrees, read from the glacier grid.
    grid_size = (abs(step(dims(glaciers, X))), abs(step(dims(glaciers, Y))))

    tiles = Dict{Tuple{Int,Int},Vector{Int}}()
    for i in 1:nrow(glacier_points)
        key = _tile_key(_cell_lonlat(glacier_points.geometry[i])...)
        push!(get!(tiles, key, Int[]), i)
    end
    tilelist = collect(tiles)

    # 6. One DEM fetch per tile, threaded over tiles. Writes target disjoint rows, so this is safe.
    dem_pool = Base.Semaphore(dem_fetch_concurrency)
    prog = Progress(length(tilelist); desc="Glacier hypsometry by tile: ", showspeed=true, enabled=verbose)
    Threads.@threads for t in eachindex(tilelist)
        tile_hypsometry!(H, glacier_points, tilelist[t].second, dem, geoms, gtree, glaciers,
                         grid_size, bin_edges, n_bins, dem_pool)
        next!(prog)
    end
    finish!(prog)

    # 7. Splice the results matrix into the DataFrame as one scalar Float64 column per bin. Native
    #    flat Parquet columns read in ms, versus a nested LIST column that deserializes into boxed
    #    Vector{Any}. The binning is recoverable from the column names and from table metadata.
    for (b, name) in enumerate(_hyps_colnames(bin_edges))
        glacier_points[!, name] = H[:, b]
    end
    # Column names (hyps_<lo>_<hi>) are the authoritative record of the binning. GeoParquet only
    # serializes geo metadata, so this table metadata is convenience for the in-memory table and
    # will NOT survive a write/read round-trip — recover the edges from the column names instead.
    metadata!(glacier_points, "hypsometry_bin_edges", collect(bin_edges); style=:note)

    # 8. Drop the CartesianIndex column (not serializable) and return.
    return select(glacier_points, Not(:index))
end

# Regrid an invariant RasterStack onto `grid`'s X/Y so cell selection by CartesianIndex is valid.
# If the stack's X/Y dims already match `grid`, it is returned unchanged.
function _align_stack_to(stack, grid)
    gx, gy = dims(grid, X), dims(grid, Y)
    if dims(stack, X) == gx && dims(stack, Y) == gy
        return stack
    end
    return resample(stack; to=grid, method=:near)
end

# Longitude on the DEM's −180…180°E convention (climate grids are native 0–359.9°E).
_dem_lon(lon) = float(wrap_lon(lon))

# (lon, lat) of a cell, read off its Point geometry (native 0–359.9°E, same as the glacier grid).
_cell_lonlat(pt) = (GeoDataFrames.GeoInterface.x(pt), GeoDataFrames.GeoInterface.y(pt))

# 1° DEM tile SW integer corner (−180…180°E).
_tile_key(lon, lat) = (floor(Int, _dem_lon(lon)), floor(Int, lat))

# Bin lookup for monotone `bin_edges`; bin i spans [edges[i], edges[i+1]). Returns 0 when `z`
# falls outside the covered range. `searchsortedlast` handles both uniform and non-uniform edges.
@inline function _bin_index(z, bin_edges, n_bins)
    b = searchsortedlast(bin_edges, z)
    return 1 <= b <= n_bins ? b : 0
end

# Scalar per-bin column names for the flat hypsometry layout: bin i → :hyps_<lo>_<hi> with the
# elevation edges (m) zero-padded to 5 digits (e.g. :hyps_00000_00100 … :hyps_09900_10000). The
# binning is thus self-describing from the column names; the full edge vector is also stored in the
# "hypsometry_bin_edges" table metadata.
_hyps_colnames(bin_edges) =
    [Symbol("hyps_" * lpad(Int(bin_edges[i]), 5, '0') * "_" * lpad(Int(bin_edges[i+1]), 5, '0'))
     for i in 1:length(bin_edges) - 1]

# Inverse of the `hyps_<lo>_<hi>` naming: parse one column name back to its (lo, hi) edges (m).
# Returns `nothing` for names that are not hypsometry columns.
function _parse_hyps_colname(name)
    s = string(name)
    startswith(s, "hyps_") || return nothing
    parts = split(s, '_')
    length(parts) == 3 || return nothing
    lo = tryparse(Int, parts[2]); hi = tryparse(Int, parts[3])
    (lo === nothing || hi === nothing) && return nothing
    return (lo, hi)
end

"""
    hypsometry_bin_edges(df) -> Vector{Int}

Recover the elevation bin edges (m) from the `hyps_<lo>_<hi>` columns of a glacier
elevation-class table. This is the inverse of the flat-column encoding written by
[`gemb_glacier_elevation_class_runfile`](@ref) and is the authoritative way to recover the
binning after a GeoParquet round-trip (the `"hypsometry_bin_edges"` table metadata is not
persisted by GeoParquet).

The edges are read back from whichever `hyps_*` columns are present and sorted, so arbitrary
(including non-uniform) upstream `elevation_bin_edges` are handled transparently. Throws if the
recovered bins are not contiguous (each bin's `hi` must equal the next bin's `lo`).
"""
function hypsometry_bin_edges(df)
    bins = sort!(filter(!isnothing, map(_parse_hyps_colname, names(df))))
    isempty(bins) && throw(ArgumentError("no hyps_<lo>_<hi> columns found"))
    for i in 1:length(bins) - 1
        bins[i][2] == bins[i+1][1] ||
            throw(ArgumentError("non-contiguous hypsometry bins: $(bins[i]) then $(bins[i+1])"))
    end
    return Int[bins[1][1]; last.(bins)]
end

"""
    glacier_hypsometry(row; area_minimum = 0)

Decode the populated hypsometry bins of a single glacier elevation-class `row` (one element of
`eachrow(df)`). Returns a vector of `(; lo, hi, center, area)` named tuples — the elevation bin
edges (m), the bin-center elevation (m), and the glacier area (km²) — for every bin whose area is
strictly greater than `area_minimum`, sorted by elevation.

This replaces hand-parsing the `hyps_<lo>_<hi>` column names downstream; the binning contract
lives here alongside the encoder ([`gemb_glacier_elevation_class_runfile`](@ref)).
"""
function glacier_hypsometry(row; area_minimum = 0)
    out = @NamedTuple{lo::Int, hi::Int, center::Float64, area::Float64}[]
    for name in propertynames(row)
        edges = _parse_hyps_colname(name)
        edges === nothing && continue
        area = row[name]
        area > area_minimum || continue
        lo, hi = edges
        push!(out, (; lo, hi, center = (lo + hi) / 2, area = float(area)))
    end
    sort!(out; by = b -> b.center)
    return out
end

# Extent of the glacier-grid cell at CartesianIndex `I`, normalized to the DEM's −180…180°E
# convention. The grid is regular, so we take the cell center ± half the grid step (its lookups
# use Points sampling, so `intervalbounds` would collapse to zero width). A cell whose center is
# east of 180° is shifted by −360° to line up with the Copernicus DEM.
# NOTE: a cell straddling the antimeridian (center ≈ 180°) is not handled specially.
function _glacier_cell_extent(I, glaciers)
    xdim, ydim = dims(glaciers, X), dims(glaciers, Y)
    hx, hy = abs(step(xdim)) / 2, abs(step(ydim)) / 2
    xc, yc = xdim[I[1]], ydim[I[2]]
    shift = _dem_lon(xc) - xc   # 0 west of 180°, −360 east of it
    Extents.Extent(X = (xc - hx + shift, xc + hx + shift),
                   Y = (yc - hy, yc + hy))
end

# Compute hypsometry for every glacier cell in `rows` (all sharing one 1° DEM tile) and write each
# result into row `r` of the shared results matrix `H` (H[r, :]). `gp` supplies each cell's geometry
# and CartesianIndex. Crops and reads the padded tile DEM once and reuses it. Threads call this with
# disjoint `rows`, so the shared `H` writes never overlap.
function tile_hypsometry!(H, gp, rows, dem, geoms, gtree, glaciers, grid_size, bin_edges, n_bins, dem_pool)
    geomcol = gp.geometry
    lonlats = [_cell_lonlat(geomcol[r]) for r in rows]   # decode each geometry once
    lons = [_dem_lon(ll[1]) for ll in lonlats]
    lats = [float(ll[2])    for ll in lonlats]
    # Pad by one climate half-cell per axis so every cell's window (center ± half-cell) is fully
    # covered by the fetched tile. `grid_size` is (Δlon, Δlat) in degrees. Clamp to the DEM's valid
    # range: the pad is only margin, so clamping near the poles / antimeridian still covers every cell.
    padx, pady = grid_size[1] / 2, grid_size[2] / 2
    box = Extents.Extent(X = (clamp(minimum(lons) - padx, -180.0, 180.0),
                              clamp(maximum(lons) + padx, -180.0, 180.0)),
                         Y = (clamp(minimum(lats) - pady,  -90.0,  90.0),
                              clamp(maximum(lats) + pady,  -90.0,  90.0)))

    idx = query(gtree, box)
    if isempty(idx)
        return                             # no glaciers over this tile → H rows stay all-zero
    end
    localgeoms = geoms[idx]

    # Crop the passed DEM mosaic to the tile and read it into memory. GDAL's /vsicurl dataset open
    # is not thread-safe under full concurrency, so cap concurrent reads with the semaphore; once
    # materialized, everything downstream works on plain arrays outside the pool.
    #
    # Some cells flagged as glacier land sit where the Copernicus DEM publishes no tiles (it omits
    # ocean tiles). Reading throws ArgumentError there; treat such a tile as having no DEM coverage
    # → all-zero hypsometry, rather than crashing the whole @threads run.
    Base.acquire(dem_pool)
    tiledem = try
        read(view(dem, X = (box.X[1] .. box.X[2]), Y = (box.Y[1] .. box.Y[2])))
    catch e
        (e isa ArgumentError && occursin("No Copernicus DEM tiles", e.msg)) || rethrow(e)
        return                             # no DEM coverage → H rows stay all-zero
    finally
        Base.release(dem_pool)
    end

    for r in rows
        ext = _glacier_cell_extent(gp.index[r], glaciers)   # −180…180°E, half-step cell box
        subdem = view(tiledem, X = (ext.X[1] .. ext.X[2]), Y = (ext.Y[1] .. ext.Y[2]))
        # boolean glacier mask on the cell's DEM grid, from the pre-filtered polygons
        submask = rasterize(last, localgeoms; to=subdem, fill=true, missingval=false,
                            boundary=:center, progress=false, verbose=false)
        # cellarea needs Intervals sampling; the DEM lookups come in as Points, so retag them.
        # Area varies only with latitude, so we compute it over a single column (X(1:1)) and get
        # per-row (Y) km²; index by j.
        subint = set(subdem[X(1:1)],
            X => DimensionalData.Lookups.Intervals(DimensionalData.Lookups.Center()),
            Y => DimensionalData.Lookups.Intervals(DimensionalData.Lookups.Center()))
        area_by_row = vec(parent(cellarea(subint))) ./ 1e6   # length ny(Y), m² → km²

        D = parent(subdem); M = parent(submask)   # both nx × ny; iterate raw matrices
        hyps = zeros(Float64, n_bins)
        @inbounds for j in axes(D, 2), i in axes(D, 1)
            (M[i, j] && isfinite(D[i, j])) || continue
            b = _bin_index(D[i, j], bin_edges, n_bins); b == 0 && continue
            hyps[b] += area_by_row[j]
        end
        H[r, :] = hyps
    end
    return
end
