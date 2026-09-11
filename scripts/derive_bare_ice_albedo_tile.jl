# Derive observed bare-ice albedo as a function of elevation, for every glacier grid cell in one
# downscaling tile.
#
# For each ERA5-Land cell the tile owns, this collects the MODIS MCD43A3 500 m cells inside it
# (`bare_ice_albedo`), samples the Copernicus 30 m DEM at each of them, and reduces the pair to
# albedo binned on the same 100 m elevation classes the glacier hypsometry uses
# (`bare_ice_albedo_hyps`). The result per cell is a callable: `f(z)` is the best estimate of
# bare-ice albedo at elevation `z`, which is the form GEMB needs — one `albedo_ice` per elevation
# class it runs.
#
# Writes two tables and a diagnostic figure per tile:
#   bare_ice_albedo_cells_<tile>.parquet   one row per grid cell: coverage, median albedo, the fit
#   bare_ice_albedo_bins_<tile>.parquet    one row per (cell, elevation bin): albedo beside area
#   bare_ice_albedo_<tile>.png             the tile's albedo-elevation profile and two maps
#
# Needs network for the Copernicus DEM (GDAL /vsicurl), but **not** for the albedo: that is read
# from the pooled tables vendored in GEMB_ClimateForcing's `data/`, so no MODIS granule is
# downloaded and no Earthdata token is needed.
#
# Run:  julia --project=. -t 8 scripts/derive_bare_ice_albedo_tile.jl [tile_name]
#
# `-t` matters: the DEM windows are read concurrently and are the bulk of the runtime. Measured on
# N60_W142 (326 cells, the largest Alaska tile) at `-t 8`: ~25 s end to end.

using GEMB_GlacierSims
using GEMB_ClimateForcing
using DataFrames
using Statistics
import GeoDataFrames
import GeoParquet
import GeoInterface as GI
import Extents

# Headless before any backend loads: this writes a PNG, and switching backends needs a reload.
using CairoMakie
CairoMakie.activate!()

const CLIMATE_MODEL = :era5land

# The tiling. Matches `derive_downscaling_parameter_tiles.jl` so a cell belongs to the same tile in
# both sweeps and the two outputs join. No buffer is used here: unlike a lapse rate, a cell's albedo
# profile is derived from that cell's own MODIS cells and not regressed across a neighbourhood.
const TILE_SIZE = 2
const BUFFER = 1

# Skip cells holding less than this total glacier area (km²). A cell with a sliver of ice holds a
# handful of MODIS cells and cannot support a profile.
const AREA_MINIMUM = 1.0

# MODIS cells a 100 m bin needs before its own mean is used instead of the cell's linear fit.
const MIN_CELLS = 3

# How far (m) past the observed elevation range a bin may be fit-filled. 0 leaves bins outside the
# glacier's own elevation span unresolved, which is the honest answer: there is no ice there to have
# an albedo, and a linear fit extended kilometres past the data returns albedos outside 0-1.
const EXTRAPOLATE = 0.0

# Black-sky (directional-hemispherical) albedo, which is what a surface energy-balance model wants.
const SKY = :bsa

const DEFAULT_TILE = "N60_W142.nc"

# Persistent shared cache, off `tempdir()` so a reboot does not throw it away.
const CLIMATE_CACHE = get(ENV, "CLIMATE_CACHE", joinpath("/mnt/bylot-r3/data", string(CLIMATE_MODEL)))
const OUTPUT_DIR = joinpath(CLIMATE_CACHE, "bare_ice_albedo")

# Where the Copernicus DEM's 1° tiles are kept. Off `tempdir()` on purpose: `cache_tiles = true`
# refuses a path the OS may reap, and a global sweep should not fill the GEMB_ClimateForcing repo,
# which is where the default resolves. GLO-30 tiles are 19–40 MB each.
const DEM_CACHE = joinpath(CLIMATE_CACHE, "invariant", "copernicus_dem_30m")

tile_name = isempty(ARGS) ? DEFAULT_TILE : (endswith(ARGS[1], ".nc") ? ARGS[1] : ARGS[1] * ".nc")
stem = replace(tile_name, ".nc" => "")

# ---------------------------------------------------------------- the table and the tile

elevation_classes_file = joinpath(@__DIR__, "..", "data",
                                  "$(CLIMATE_MODEL)_glacier_elevation_classes.parquet")
isfile(elevation_classes_file) || error("""
    No glacier elevation-class table at $(elevation_classes_file). Build it with
    `src/era5_example.jl` — that takes hours, so do not delete the cached file.
    """)

@info "Reading glacier elevation-class table" file = elevation_classes_file
table = GeoDataFrames.read(elevation_classes_file)
# The cached table carries only the Point geometry; `downscaling_tiles` is keyed on these columns.
# Longitudes stay in the native 0-359.9°E convention, which is what the forcing loader uses.
table[!, :longitude] = GI.x.(table.geometry)
table[!, :latitude] = GI.y.(table.geometry)

# The same edges the glacier-area hypsometry is stored on, recovered from the `hyps_*` column names,
# so an albedo bin and an area bin are the same bin.
bins = hypsometry_bin_edges(table)

tiles = downscaling_tiles(table; tile_size = TILE_SIZE, buffer = BUFFER,
                          area_minimum = AREA_MINIMUM)
matching = filter(t -> t.name == tile_name, tiles)
isempty(matching) && error("""
    No tile named $(tile_name) holds a cell with at least $(AREA_MINIMUM) km² of glacier at
    tile_size = $(TILE_SIZE). Available Alaska tiles, largest first:
    $(join([t.name for t in sort(filter(t -> -170 <= t.index[1] < -130 && 54 <= t.index[2] < 70, tiles);
                                 by = t -> -sum(glacier_area_column(t.core)))][1:min(8, end)], ", "))
    """)
tile = only(matching)

@info "Tile selected" name = tile.name index = tile.index cells = nrow(tile.core) glacier_area_km2 =
    round(sum(glacier_area_column(tile.core)); digits = 1) bins = length(bins) - 1

# ------------------------------------------------------------------------------ derivation

tile_extent = Extents.Extent(X = (tile.bounds.lon_min - 0.1, tile.bounds.lon_max + 0.1),
                             Y = (tile.bounds.lat_min - 0.1, tile.bounds.lat_max + 0.1))
# `cache_tiles` keeps whole 1° DEM tiles, so a re-run or a neighbouring tile touches no network.
# Bounded only because an `extent` is given: on the global mosaic it would fetch every published
# tile. The path must be durable or the call throws rather than write tiles the OS will reap.
dem = climate_model_invariant(model = :copernicus_dem_30m, extent = tile_extent,
                              cache_path = DEM_CACHE, cache_tiles = true, verbose = false)

out = bare_ice_albedo_tile(tile, dem; bins, sky = SKY, min_cells = MIN_CELLS,
                           extrapolate = EXTRAPOLATE)

# The tile-level profile the downscaling sweep stores: pooled over the *buffered* cells, so
# neighbouring tiles overlap by half their window. This is what GEMB actually reads.
pooled = derive_bare_ice_albedo(tile.buffered; bins, sky = SKY, min_cells = MIN_CELLS,
                                dem_cache_path = DEM_CACHE, label = stem, verbose = false)

# --------------------------------------------------------------------------------- output

mkpath(OUTPUT_DIR)
cells_path = joinpath(OUTPUT_DIR, "bare_ice_albedo_cells_$(stem).parquet")
bins_path = joinpath(OUTPUT_DIR, "bare_ice_albedo_bins_$(stem).parquet")
GeoParquet.Parquet2.writefile(cells_path, out.per_cell)
GeoParquet.Parquet2.writefile(bins_path, out.per_bin)
@info "Wrote tables" cells = cells_path cell_rows = nrow(out.per_cell) bins = bins_path bin_rows =
    nrow(out.per_bin)

# --------------------------------------------------------------------------------- figure

per_bin = out.per_bin
per_cell = out.per_cell
observed = filter(r -> r.source == "observed" && !isnan(r.albedo), per_bin)

# The tile-wide profile: each bin's mean over cells, weighted by how many MODIS cells stand behind
# each cell's value, so a cell resting on three retrievals does not count as much as one on three
# hundred.
function weighted_profile(rows)
    prof = combine(groupby(rows, :center),
                   [:albedo, :n_cells] => ((a, w) -> sum(a .* w) / sum(w)) => :albedo,
                   :n_cells => sum => :n_cells,
                   :n_valid => sum => :n_valid,
                   :glacier_area_km2 => sum => :area)
    return sort!(prof, :center)
end
profile = weighted_profile(observed)

fig = Figure(size = (1250, 950))

ax1 = Axis(fig[1, 1]; xlabel = "elevation (m)", ylabel = "bare-ice albedo ($(SKY))",
           title = "$(stem): albedo vs elevation, $(nrow(observed)) observed (cell, bin) pairs")
# The range GEMB accepts for albedo_ice, for reference only.
# `GEMB_ALBEDO_ICE_RANGE` is the range GEMB asserts for `albedo_ice`, drawn for reference only:
# nothing here is clamped to it, since above the ELA the statistic is measuring snow.
hspan!(ax1, GEMB_ALBEDO_ICE_RANGE[1], GEMB_ALBEDO_ICE_RANGE[2]; color = (:seagreen, 0.12))
hlines!(ax1, collect(GEMB_ALBEDO_ICE_RANGE); color = (:seagreen, 0.5), linestyle = :dash)
if !isempty(observed)
    sc = scatter!(ax1, observed.center, observed.albedo;
                  color = log10.(observed.n_cells), colormap = :viridis, markersize = 4,
                  alpha = 0.5)
    Colorbar(fig[1, 2], sc; label = "log₁₀ MODIS cells in bin")
end
if nrow(profile) > 1
    lines!(ax1, profile.center, profile.albedo; color = :firebrick, linewidth = 3,
           label = "per-cell mean, weighted by MODIS cells")
end
# The pooled profile is the one GEMB reads, so it belongs on the same axes as the per-cell cloud it
# is meant to summarise.
let t = bare_ice_albedo_table(pooled), keep = findall(!isnan, t.albedo)
    isempty(keep) || lines!(ax1, t.center[keep], t.albedo[keep]; color = :navy, linewidth = 3,
                            linestyle = :dash, label = "pooled over the buffered tile (what GEMB uses)")
end
axislegend(ax1; position = :lt, framevisible = false)
text!(ax1, 0.98, 0.02; text = "green band = GEMB albedo_ice range (not applied)",
      space = :relative, align = (:right, :bottom), fontsize = 10, color = :seagreen)

ax2 = Axis(fig[2, 1]; xlabel = "elevation (m)", ylabel = "count in bin, tile-wide",
           yscale = log10,
           title = "How much observation each elevation rests on")
if nrow(profile) > 1
    # Lines, not bars: a bar is drawn from zero, which a log axis cannot place, so a barplot here
    # renders every bin at the axis floor regardless of its value.
    lines!(ax2, profile.center, max.(profile.n_valid, 1); color = :steelblue, linewidth = 2,
           label = "MCD43A3 retrievals")
    lines!(ax2, profile.center, max.(profile.n_cells, 1); color = :darkorange, linewidth = 2,
           label = "MODIS 463 m cells")
    lines!(ax2, profile.center, max.(profile.area, 1e-2); color = :grey35, linewidth = 2,
           linestyle = :dash, label = "glacier area (km²)")
    axislegend(ax2; position = :rt, framevisible = false)
end

# Maps. Longitudes are native 0-359.9°E in the table; wrapped here so the axis reads as Alaska.
lonw = wrap_lon.(per_cell.longitude)
resolved = findall(!isnan, per_cell.albedo_median)

ax3 = Axis(fig[1, 3]; xlabel = "longitude (°E)", ylabel = "latitude (°N)",
           title = "Median albedo per cell", aspect = DataAspect())
if !isempty(resolved)
    m3 = scatter!(ax3, lonw[resolved], per_cell.latitude[resolved];
                  color = per_cell.albedo_median[resolved], colormap = :inferno, markersize = 9)
    Colorbar(fig[1, 4], m3; label = "median albedo")
end

# A per-cell slope is only meaningful where the fit had elevation range to work with, so cells whose
# standard error swamps the slope are left out rather than drawn as noise.
usable = findall(i -> isfinite(per_cell.slope_per_km[i]) &&
                      isfinite(per_cell.slope_stderr[i]) &&
                      abs(per_cell.slope_per_km[i]) > 2 * per_cell.slope_stderr[i],
                 1:nrow(per_cell))
ax4 = Axis(fig[2, 3]; xlabel = "longitude (°E)", ylabel = "latitude (°N)",
           title = "Fitted d(albedo)/dz, $(length(usable)) of $(nrow(per_cell)) cells at >2σ",
           aspect = DataAspect())
if !isempty(usable)
    lim = maximum(abs, per_cell.slope_per_km[usable])
    m4 = scatter!(ax4, lonw[usable], per_cell.latitude[usable];
                  color = per_cell.slope_per_km[usable], colormap = :balance,
                  colorrange = (-lim, lim), markersize = 9)
    Colorbar(fig[2, 4], m4; label = "albedo per km")
end

png_path = joinpath(OUTPUT_DIR, "bare_ice_albedo_$(stem).png")
save(png_path, fig)
@info "Wrote figure" file = png_path

# -------------------------------------------------------------------------------- reporting

# Everything dropped is counted here. A census that silently omits a rejection reads as full
# coverage when it is not.
@info "Census" out.census...

if !isempty(observed)
    lo, hi = GEMB_ALBEDO_ICE_RANGE
    above = count(>(hi), observed.albedo)
    below = count(<(lo), observed.albedo)
    @info "Observed bins against GEMB's albedo_ice range" range = GEMB_ALBEDO_ICE_RANGE within =
        nrow(observed) - above - below above_ceiling = above below_floor = below
    fitted = filter(r -> r.source == "fit" && !isnan(r.albedo), per_bin)
    @info "Fit-filled bins" n = nrow(fitted) beyond_product_qc = out.census.bins_fit_beyond_qc qc_range =
        POOLED_ICE_ALBEDO_RANGE
    @info "Elevation span of the observed profile" lowest_bin = Int(minimum(observed.center)) highest_bin =
        Int(maximum(observed.center))
    slopes = filter(isfinite, per_cell.slope_per_km)
    isempty(slopes) || @info "Per-cell albedo-elevation slope (albedo per km)" median =
        round(median(slopes); digits = 4) q10 = round(quantile(slopes, 0.1); digits = 4) q90 =
        round(quantile(slopes, 0.9); digits = 4) n_negative = count(<(0), slopes) n = length(slopes)
    # Glacier area the profile actually speaks for, versus the area the tile holds. This is the
    # number that says whether the result is usable, not the cell count.
    area_observed = sum(observed.glacier_area_km2)
    area_total = sum(glacier_area_column(tile.core))
    @info "Glacier area covered by an observed bin" observed_km2 = round(area_observed; digits = 1) tile_km2 =
        round(area_total; digits = 1) fraction = round(area_observed / area_total; digits = 3)
end

# The pooled profile, and the `albedo_ice` each of the tile's elevation classes will actually run at.
using GEMB: initialize_parameters
mp = initialize_parameters()
classes = hypsometry_intervals(tile.core)
resolved = [(c.center, resolve_albedo_ice(pooled, c.center; default = mp.albedo_ice)...)
            for c in classes]
@info "Pooled tile profile" buffered_cells = nrow(tile.buffered) core_cells = nrow(tile.core) modis_cells =
    pooled.n_used bins_observed = count(==(:observed), pooled.sources) bins_held =
    count(==(:hold), pooled.sources) bins_fit = count(==(:fit), pooled.sources) bins_unresolved =
    count(==(:none), pooled.sources) slope_per_km = round(pooled.fit.slope_per_km; digits = 4)
@info "albedo_ice resolved onto the tile's elevation classes" n_classes = length(classes) default_would_be =
    mp.albedo_ice all_inside_gemb_range =
    all(GEMB_ALBEDO_ICE_RANGE[1] <= r[2] <= GEMB_ALBEDO_ICE_RANGE[2] for r in resolved) sources =
    Dict(String(s) => count(r -> r[3] === s, resolved) for s in BARE_ICE_ALBEDO_SOURCES
         if count(r -> r[3] === s, resolved) > 0)
let v = [r[2] for r in resolved]
    @info "albedo_ice across the classes" min = round(minimum(v); digits = 4) median =
        round(median(v); digits = 4) max = round(maximum(v); digits = 4)
end
