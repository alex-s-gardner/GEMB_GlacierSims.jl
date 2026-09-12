# A guided tour of bare-ice albedo as a function of elevation.
#
# Run it, or `include` it in a REPL and poke at the objects it leaves behind (`a`, `f`, `sweep`).
#
#   julia --project=. -t 8 scripts/try_bare_ice_albedo.jl              # default Alaska cell
#   julia --project=. -t 8 scripts/try_bare_ice_albedo.jl -141.4 60.5  # any glacier cell
#
# Environment switches:
#   SKIP_TILE=1   skip section 7 (the whole-tile sweep, ~25 s); everything else is a few seconds
#   SKIP_PLOT=1   skip the PNG
#
# Needs network for the Copernicus DEM. It does **not** need CDS or Earthdata credentials: the
# albedo comes from tables vendored in GEMB_ClimateForcing's `data/`, so no MODIS granule is
# downloaded.

using GEMB_GlacierSims
using GEMB_ClimateForcing
using Rasters
using DataFrames
using Statistics
import GeoDataFrames
import GeoInterface as GI

isdefined(GEMB_ClimateForcing, :bare_ice_albedo) || error("""
    This GEMB_ClimateForcing has no `bare_ice_albedo`. The Manifest is pointing at a checkout that
    predates it. Repoint it with

        julia --project=. -e 'using Pkg; Pkg.develop(path="/mnt/bylot-r3/data/GEMB_ClimateForcing.jl")'
    """)

# A cell on the Bagley Icefield unless told otherwise. Longitudes may be given in either
# convention; the elevation-class table stores 0-359.9°E and everything here wraps as needed.
lon = length(ARGS) >= 2 ? parse(Float64, ARGS[1]) : -141.4
lat = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 60.5
const BINS = 0:100:10000          # the 100 m elevation classes GEMB runs
const CELL = 0.1                  # ERA5-Land grid spacing

banner(n, title) = println("\n", "="^78, "\n ", n, ". ", title, "\n", "="^78)

# ---------------------------------------------------------------------------------------------
banner(1, "The geometry: one ERA5-Land cell")

geom = era5_land_cell_polygon(lon, lat; cell_size = CELL)
ring = collect(GI.getpoint(first(GI.getring(geom))))
println("cell centre         : ", (wrap_lon(lon), lat))
println("cell key            : ", era5_land_cell_key(lon, lat; cell_size = CELL))
println("polygon corners     : ", [(round(GI.x(p); digits = 3), round(GI.y(p); digits = 3))
                                   for p in ring[1:end - 1]])
println("\nAny GeoInterface geometry works — this one just happens to be a grid cell.")

# ---------------------------------------------------------------------------------------------
banner(2, "bare_ice_albedo(geometry)  ->  every MODIS cell inside it")

a = bare_ice_albedo(geom)
n = length(a[:albedo_bsa])
println("MCD43A3 cells selected : ", n)
println("glacierized (RGI 7.0)  : ", count(a[:in_product]), "   (:in_product == false is not glacier)")
println("with a resolved albedo : ", count(!isnan, a[:albedo_bsa]))
println("layers                 : ", keys(a))
resolved = filter(!isnan, collect(a[:albedo_bsa]))
if !isempty(resolved)
    println("albedo_bsa             : ", round(minimum(resolved); digits = 3), " … ",
            round(maximum(resolved); digits = 3), "  (median ",
            round(median(resolved); digits = 3), ")")
end
println("\nOne value per 463 m cell, pooled over 2000-2025. No elevation attached yet.")

# ---------------------------------------------------------------------------------------------
banner(3, "Elevation at those cells — Rasters does the geometry/raster work")

pts = bare_ice_albedo_points(a)
println("bare_ice_albedo_points(a) -> ", length(pts), "-element ", eltype(pts))
println("first point               : ", pts[1])

dem = climate_model_invariant(model = :copernicus_dem_30m, verbose = false)
# Crop BEFORE extracting. Sampling the lazy global mosaic costs one HTTP read per point: 258
# points took 89 s that way, and 5 ms from a window read once.
pad = CELL / 2 + 0.02
window = read(view(dem, X = (wrap_lon(lon) - pad .. wrap_lon(lon) + pad),
                        Y = (lat - pad .. lat + pad)))
println("DEM window                : ", size(window), " px")

e = extract(window, pts; geometry = false, progress = false)
println("extract(...) element      : ", e[1])
println("  (the field name is empty — that is the Copernicus DEM's own layer name)")
zs = [ismissing(first(r)) ? NaN : Float64(first(r)) for r in e]
finite = filter(isfinite, zs)
println("elevation                 : ", round(minimum(finite)), " … ", round(maximum(finite)),
        " m;  no DEM value at ", count(isnan, zs), " points")

# ---------------------------------------------------------------------------------------------
banner(4, "bare_ice_albedo_hyps(a, e, bins)  ->  a callable")

f = bare_ice_albedo_hyps(a, e, BINS)
show(stdout, MIME"text/plain"(), f)
println("\n")
println("f is callable. f(z) is the best estimate at elevation z:\n")
println("    z (m)     f(z)   source")
# Walk from a little below the cell's lowest ice to a little above its highest, so both the
# resolved interior and the unresolved ends are visible.
zlo = max(0, round(Int, minimum(finite) / 200) * 200 - 400)
zhi = round(Int, maximum(finite) / 200) * 200 + 400
for z in range(zlo, zhi; step = 200)
    v = f(z)
    println("    ", lpad(z, 5), "  ", isnan(v) ? "    NaN" : lpad(round(v; digits = 4), 7),
            "   ", bare_ice_albedo_source(f, z))
end
probe = round.(Int, range(minimum(finite), maximum(finite); length = 3))
println("\nand it broadcasts: f.(", probe, ") = ", round.(f.(probe); digits = 4))

# ---------------------------------------------------------------------------------------------
banner(5, "The profile, bin by bin")

t = bare_ice_albedo_table(f)
shown = findall(i -> t.n_cells[i] > 0 || !isnan(t.albedo[i]), eachindex(t.lo))
println("  bin (m)        albedo   observed   cells   retrievals   source")
for i in shown
    println("  ", lpad(Int(t.lo[i]), 5), "-", lpad(Int(t.hi[i]), 5), "  ",
            lpad(isnan(t.albedo[i]) ? "NaN" : string(round(t.albedo[i]; digits = 4)), 8), "   ",
            lpad(isnan(t.observed[i]) ? "-" : string(round(t.observed[i]; digits = 4)), 8), "  ",
            lpad(t.n_cells[i], 6), "   ", lpad(t.n_valid[i], 10), "   ", t.source[i])
end
println("""
  `albedo` is what f(z) returns. `observed` is the bin's own mean whatever its cell count, so a
  fit-filled bin can be checked against the few cells that were in it.""")

# ---------------------------------------------------------------------------------------------
banner(6, "The three knobs worth knowing")

println("min_cells — how many MODIS cells a bin needs before its own mean is trusted:")
for mc in (1, 3, 10)
    g = bare_ice_albedo_hyps(a, e, BINS; min_cells = mc)
    println("   min_cells = ", lpad(mc, 2), " -> ", lpad(count(==(:observed), g.sources), 3),
            " observed, ", lpad(count(==(:fit), g.sources), 3), " fit-filled")
end

println("\nextrapolate — how far (m) past the observed elevations a bin may be fit-filled:")
for ex in (0, 300, 1000)
    g = bare_ice_albedo_hyps(a, e, BINS; extrapolate = ex)
    v = filter(!isnan, g.albedo)
    println("   extrapolate = ", lpad(ex, 4), " -> ", lpad(count(!isnan, g.albedo), 3),
            " bins resolved, albedo ", isempty(v) ? "none" :
            string(round(minimum(v); digits = 3), " … ", round(maximum(v); digits = 3)))
end
println("""   At 0, bins outside the glacier's own elevation span stay NaN — there is no ice there
   to have an albedo, and a linear fit stretched kilometres past the data leaves 0-1 entirely.""")

println("\nsky — black-sky (what an energy-balance model wants) vs white-sky:")
for sky in (:bsa, :wsa)
    g = bare_ice_albedo_hyps(a, e, BINS; sky)
    v = filter(!isnan, g.albedo)
    println("   sky = :", sky, " -> median ", isempty(v) ? "none" : round(median(v); digits = 4),
            ", slope ", round(g.fit.slope_per_km; digits = 4), " per km")
end

# ---------------------------------------------------------------------------------------------
banner(7, "Other geometries — a point and a line")

# A cell centre from `a`, so it is certainly on glacier.
mid = argmax(collect(a[:n_valid_bsa]))
p = GI.Point(a[:longitude][mid], a[:latitude][mid])
ap = bare_ice_albedo(p)
println("point  -> ", length(ap[:albedo_bsa]), " cell, albedo ",
        round(ap[:albedo_bsa][1]; digits = 4), ", ", ap[:n_valid_bsa][1], " retrievals")

line = GI.LineString([(wrap_lon(lon) - 0.04, lat - 0.04), (wrap_lon(lon) + 0.04, lat + 0.04)])
al = bare_ice_albedo(line)
println("line   -> ", length(al[:albedo_bsa]), " cells it passes through, ",
        count(al[:in_product]), " glacierized")
println("polygon-> ", n, " cells (section 2)")
println("\nThe burn is inferred from the geometry trait; `shape=` overrides it.")

# ---------------------------------------------------------------------------------------------
if get(ENV, "SKIP_TILE", "0") == "1"
    banner(8, "Whole-tile sweep — SKIPPED (SKIP_TILE=1)")
    global sweep = nothing
else
    banner(8, "Whole-tile sweep (~25 s: one albedo query, then the DEM windows)")

    table = GeoDataFrames.read(joinpath(@__DIR__, "..", "data",
                                        "era5land_glacier_elevation_classes.parquet"))
    table[!, :longitude] = GI.x.(table.geometry)
    table[!, :latitude] = GI.y.(table.geometry)
    tiles = downscaling_tiles(table; tile_size = 2, buffer = 1, area_minimum = 1.0)
    # The tile containing the cell under test.
    idx = tile_index(lon, lat; tile_size = 2)
    match = filter(t -> t.index == idx, tiles)
    isempty(match) && error("no tile at index $(idx) holds a cell with >= 1 km2 of glacier")
    tile = only(match)

    # Bins from the table's own hyps_<lo>_<hi> column names, so albedo bins and area bins agree.
    global sweep = bare_ice_albedo_tile(tile, dem; bins = hypsometry_bin_edges(table))

    println("\nper_cell (one row per grid cell):")
    show(stdout, select(first(sort(sweep.per_cell, :glacier_area_km2; rev = true), 5),
                        [:longitude, :latitude, :glacier_area_km2, :n_modis_cells,
                         :albedo_median, :slope_per_km, :r2]); allrows = true, allcols = true)
    println("\n\nper_bin (one row per cell x elevation bin):")
    show(stdout, first(sweep.per_bin, 5); allrows = true, allcols = true)
    println("\n\nsweep.hyps[i] is the callable for row i of per_cell:")
    biggest = argmax(sweep.per_cell.n_modis_cells)
    show(stdout, MIME"text/plain"(), sweep.hyps[biggest])
    println()
end

# ---------------------------------------------------------------------------------------------
if get(ENV, "SKIP_PLOT", "0") == "1"
    banner(9, "Plot — SKIPPED (SKIP_PLOT=1)")
else
    banner(9, "Plot")
    # CairoMakie, not GLMakie: this writes a file, and the backend cannot be swapped after load.
    using CairoMakie
    CairoMakie.activate!()

    fig = Figure(size = (900, 400))
    ax = Axis(fig[1, 1]; xlabel = "elevation (m)", ylabel = "bare-ice albedo ($(f.sky))",
              title = "Cell ($(wrap_lon(lon)), $(lat)): f(z) and the cells behind it")
    hspan!(ax, 0.2, 0.6; color = (:seagreen, 0.12))
    scatter!(ax, zs, Float64.(collect(a[:albedo_bsa])); color = (:grey40, 0.35), markersize = 4,
             label = "MODIS cells")
    obs = findall(==(:observed), f.sources)
    fit = findall(==(:fit), f.sources)
    ctr = (f.edges[1:end - 1] .+ f.edges[2:end]) ./ 2
    !isempty(obs) && scatter!(ax, ctr[obs], f.albedo[obs]; color = :firebrick, markersize = 12,
                              label = "f(z), observed")
    !isempty(fit) && scatter!(ax, ctr[fit], f.albedo[fit]; color = :goldenrod, marker = :diamond,
                              markersize = 12, label = "f(z), fit-filled")
    if isfinite(f.fit.slope_per_km)
        zr = [f.fit.elevation_range...]
        lines!(ax, zr, f.fit.intercept .+ f.fit.slope_per_km / 1000 .* zr;
               color = :navy, linestyle = :dash, label = "linear fit")
    end
    xlims!(ax, minimum(finite) - 200, maximum(finite) + 200)
    axislegend(ax; position = :lt, framevisible = false)

    if !isnothing(sweep)
        ax2 = Axis(fig[1, 2]; xlabel = "elevation (m)", ylabel = "albedo",
                   title = "Whole tile, observed bins")
        hspan!(ax2, 0.2, 0.6; color = (:seagreen, 0.12))
        o = filter(r -> r.source == "observed" && !isnan(r.albedo), sweep.per_bin)
        scatter!(ax2, o.center, o.albedo; color = (:steelblue, 0.3), markersize = 3)
    end

    path = joinpath(tempdir(), "try_bare_ice_albedo.png")
    save(path, fig)
    println("wrote ", path)
end

# ---------------------------------------------------------------------------------------------
if isnothing(sweep)
    banner(10, "Per-tile profile and albedo_ice per class — SKIPPED (needs section 8)")
    global pooled = nothing
else
    banner(10, "The tile-level profile, and one albedo_ice per elevation class")

    using GEMB: initialize_parameters
    # Pooled over the tile's *buffered* cells — a 4x4 deg window at the 2/1 deg default — so
    # neighbouring tiles share half their window and the albedo does not step at a tile seam. This
    # is what the downscaling sweep stores and what GEMB reads.
    dem_cache = joinpath(CLIMATE_CACHE, "invariant", "copernicus_dem_30m")
    global pooled = derive_bare_ice_albedo(tile.buffered; bins = hypsometry_bin_edges(table),
                                           dem_cache_path = dem_cache, label = tile.name,
                                           verbose = false)
    show(stdout, MIME"text/plain"(), pooled)
    println("\n  (pooled over ", nrow(tile.buffered), " buffered cells against ",
            nrow(tile.core), " the tile owns)\n")

    mp = initialize_parameters()
    classes = hypsometry_intervals(tile.core)
    resolved = [(c.center, resolve_albedo_ice(pooled, c.center; default = mp.albedo_ice)...)
                for c in classes]

    println("resolve_albedo_ice over the tile's ", length(classes), " elevation classes:")
    for s in BARE_ICE_ALBEDO_SOURCES
        n = count(r -> r[3] === s, resolved)
        n > 0 && println("   ", rpad(s, 9), lpad(n, 4))
    end
    println("   every value inside GEMB's ", GEMB_ALBEDO_ICE_RANGE, ": ",
            all(GEMB_ALBEDO_ICE_RANGE[1] <= r[2] <= GEMB_ALBEDO_ICE_RANGE[2] for r in resolved))
    println("\n   class (m)   albedo_ice   source        default would have been ", mp.albedo_ice)
    for r in resolved[1:max(1, length(resolved) ÷ 14):end]
        println("   ", lpad(Int(r[1]), 8), "   ", lpad(round(r[2]; digits = 4), 10), "   ", r[3])
    end
    println("""
   This is the vector gemb_glacier_tile now runs on: one ModelParameters per band, built with
   `setproperties(mp, (; albedo_ice = ...))`, used for the spinup as well as the run.""")
end

# ---------------------------------------------------------------------------------------------
banner(11, "What to poke at")

println("""
Left in scope if you included this in a REPL:

  geom    the ERA5-Land cell polygon
  a       bare_ice_albedo(geom) — the DimStack of MODIS cells
  pts     bare_ice_albedo_points(a) — the cell centres as GI points
  window  the cropped Copernicus DEM
  e, zs   the extracted elevations, raw and as Float64
  f       the per-cell callable. f.fit, f.sources, bare_ice_albedo_table(f)
  sweep   the per-cell tile result: sweep.per_cell, sweep.per_bin, sweep.hyps, sweep.census
  pooled  the per-TILE profile the downscaling sweep stores and GEMB reads

Things worth trying — this cell holds ice from $(round(Int, minimum(finite))) to $(round(Int, maximum(finite))) m:

  f(z)                                           one elevation
  f.($(round(Int, minimum(finite))):100:$(round(Int, maximum(finite))))                        the whole profile
  bare_ice_albedo_source(f, z)                   :observed, :fit or :none
  bare_ice_albedo_hyps(a, e, 0:250:5000)         a coarser binning
  bare_ice_albedo_hyps(a, e; extrapolate = :hold) hold the end bins outward instead
  bare_ice_albedo(geom, median)                  reduce over cells instead, no elevation
  resolve_albedo_ice(pooled, 2000; default = 0.48)   what GEMB gets for a 2000 m class

Two things this deliberately does not do, so decide them where you use the result:

  * Nothing is clamped to GEMB's 0.2 <= albedo_ice <= 0.6. Above the ELA there is no bare ice,
    and the darkest-percentile statistic is measuring snow — a real surface, above 0.6.
  * A fit-filled bin can leave the product's own QC range $(POOLED_ICE_ALBEDO_RANGE), where no
    observation behind it was that dark or bright. `sweep.census.bins_fit_beyond_qc` counts them.
""")
