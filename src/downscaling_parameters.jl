# Regional downscaling parameters for a global glacier run: assembling a region, and turning its
# fitted parameters into per-elevation-interval forcing.
#
# Three things a global run needs that cannot be derived from a single climate grid cell, because
# each is a *relationship across* cells within a region:
#
#   1. the local on-glacier temperature decoupling factor `k`, from how the warm excess varies with
#      the ERA5-Land glacier mask fraction at matched elevation;
#   2. the local on-glacier temperature lapse rate, from how decoupled temperature varies with
#      elevation;
#   3. glacier-area weighted forcing for every elevation interval in the region, which is the input
#      to the downstream GEMB parameter sweep that solves for the precipitation and temperature bias
#      correction factors against satellite laser altimetry.
#
# **The two fits (1 and 2) live in GEMB_ClimateForcing**, in its own `downscaling_parameters.jl`:
# `derive_decoupling_factor`, `derive_lapse_rate`, and `decoupling_factor_at_elevation`. They are
# regressions on that package's own adjustment conventions — each one inverts a
# `climate_adjust_for_*` — and know nothing about glacier elevation-class tables. This file is what
# knows about tables: it selects a region's cells ([`grid_cells_in_region`](@ref)), streams their
# forcing into the cross-cell sums the fits consume (`_accumulate_cross_cell`), and applies what
# comes back (`_ElevationIntervalForcing`). See that file's header for the `acc` contract the two
# halves meet on.
#
# Terminology, used consistently below because the objects are different:
#   region             — the caller's arbitrary GeoInterface geometry. Any shape.
#   grid cell          — one row of the glacier elevation-class table, i.e. one 0.1° ERA5-Land cell.
#   elevation interval — one `hyps_<lo>_<hi>` elevation bin of glacier area.
#
# The fits *measure*: they report what the region's forcing supports at the forcing's own time
# resolution, and `NaN` where it supports nothing, substituting and clamping nothing. The interval
# aggregation (3) is the exception, and deliberately so: it *applies* the fits, and a single `NaN` in
# an applied series makes that interval's forcing fail `forcing_is_complete`, which a sweep skips as
# silently as it skips an ocean cell. So filling and clamping live here, at the one point of
# application, as explicit keywords rather than as a policy baked into the fits.
#
# The fits' constants (`_MIN_CELLS_DEFAULT`, `_LAPSE_RATE_LIMITS`, `_DECOUPLING_FACTOR_LIMITS`,
# `_FORCING_VARIABLES`, …) are imported from GEMB_ClimateForcing in `GEMB_GlacierSims.jl` rather
# than restated here — a second copy would stop being a guarantee the moment upstream moved.

"""
    RegionForcingUnavailable(n_cells)

Thrown by [`derive_downscaling_parameters`](@ref) when **no** grid cell of the region has usable
forcing — ERA5-Land is land-only, so a region of small coastal or island glaciers can have every
cell centre fall on water ([`forcing_is_complete`](@ref)).

A distinct type for the same reason [`ForcingUnavailable`](@ref) is one per cell: sweeping a global
table hits this legitimately, and a driver has to be able to tell "this region is unrunnable" from
"the driver is broken" without matching on an error message.
"""
struct RegionForcingUnavailable <: Exception
    n_cells::Int
end

Base.showerror(io::IO, e::RegionForcingUnavailable) = print(io,
    "RegionForcingUnavailable: none of the region's $(e.n_cells) grid cell(s) has usable " *
    "forcing (ERA5-Land is land-only; every cell center may have fallen on water)")

"""
    grid_cells_in_region(glacier_elevation_classes, roi_polygon; area_minimum = 0.0)

The grid cells of a glacier elevation-class table whose center falls within `roi_polygon`, as a
`SubDataFrame` view.

`roi_polygon` is any GeoInterface-compatible geometry — a `Polygon`, a `MultiPolygon`, a
`GeometryCollection`, or a vector of geometries (a cell is kept if it falls in any of them). Its
shape is not otherwise constrained; `GeometryOps.contains` does the test, so there is no
bespoke point-in-polygon code and no assumption that the region is a box.

Cell centers are stored in the table's native 0–359.9°E convention, so they are wrapped with
[`wrap_lon`](@ref) before the test — a region polygon is `(-180, 180]`. A region whose extent spans
more than 180° of longitude throws: crossing the antimeridian is not handled (the elevation-class
builder does not handle it either), and a silently wrong selection is worse than an error.

`area_minimum` screens cells by total glacier area (km²) with [`glacier_area_column`](@ref) before
the geometric test, since that is much the cheaper of the two.

Returns a view, not a copy: the table carries ~100 `hyps_*` columns over ~47,000 rows, and
materializing a second frame per region is pure waste. Row order is preserved, so a `chunk_id`-
sorted table stays sorted and a streaming pass over the result keeps its Zarr cache locality.
"""
function grid_cells_in_region(glacier_elevation_classes, roi_polygon; area_minimum::Real = 0.0)
    region = _validate_region(roi_polygon)
    bbox = _region_extent(region)

    # Longitude span of the region, from its own extent: a region wider than a hemisphere is
    # either antimeridian-crossing or a coordinate-convention mistake. Either way, refuse.
    lon_span = bbox.X[2] - bbox.X[1]
    lon_span <= 180.0 || throw(ArgumentError(
        "region spans $(round(lon_span, digits = 1))° of longitude; more than 180° means it " *
        "either crosses the antimeridian (not handled — split it into two regions) or is in a " *
        "different longitude convention than the (-180, 180] one expected here"))

    # The area screen runs first and as a column, so the geometric tests below are only asked about
    # rows that passed it. Done row-wise this is a 100-column sum per row and dominates the whole
    # selection on a global table; as a column it is a single pass (`glacier_area_column`).
    area = glacier_area_column(glacier_elevation_classes)

    # `area` is in the order of the frame it was built from, which is the frame being indexed here,
    # so the two line up whether that frame is a table or already a view.
    keep = trues(length(area))
    for (i, row) in enumerate(eachrow(glacier_elevation_classes))
        # Cheap screens first: area, then the region's bounding box, then the exact geometry.
        if area[i] < area_minimum
            keep[i] = false
            continue
        end
        lon_raw, lat_raw = _cell_lonlat(row.geometry)
        lon = wrap_lon(Float64(lon_raw))
        lat = Float64(lat_raw)
        if !(bbox.X[1] <= lon <= bbox.X[2] && bbox.Y[1] <= lat <= bbox.Y[2])
            keep[i] = false
            continue
        end
        point = GeoInterface.Point(lon, lat)
        keep[i] = any(g -> GO.contains(g, point), region)
    end

    return view(glacier_elevation_classes, keep, :)
end

# Normalize the region argument to a vector of geometries so the containment test is one `any` for
# every accepted shape. A DataFrame/table of geometries is accepted too, since `GeoDataFrames.read`
# of a region file returns one.
function _validate_region(roi_polygon)
    if GeoInterface.isgeometry(roi_polygon)
        return (roi_polygon,)
    elseif Tables.istable(roi_polygon) && hasproperty(roi_polygon, :geometry)
        geoms = getproperty(roi_polygon, :geometry)
        isempty(geoms) && throw(ArgumentError("region table has no rows"))
        return geoms
    elseif roi_polygon isa Union{AbstractVector,Tuple} && !isempty(roi_polygon) &&
           all(GeoInterface.isgeometry, roi_polygon)
        return roi_polygon
    end
    throw(ArgumentError(
        "roi_polygon must be a GeoInterface-compatible geometry, a vector of them, or a table " *
        "with a `geometry` column; got $(typeof(roi_polygon))"))
end

# Union extent of the region's geometries, as an `Extents.Extent` with X/Y bounds.
#
# `GeoInterface.extent` rather than `Extents.extent`: the latter only reads an extent a geometry
# already carries and returns `nothing` for the wrapper types a hand-built or parsed polygon uses,
# while the former computes one from the coordinates.
function _region_extent(region)
    exts = [GeoInterface.extent(g) for g in region]
    any(isnothing, exts) &&
        throw(ArgumentError("region contains a geometry with no computable extent"))
    ext = reduce(Extents.union, exts)
    haskey(ext, :X) && haskey(ext, :Y) ||
        throw(ArgumentError("region extent has no X/Y bounds; is it a valid 2D geometry?"))
    return ext
end

"""
    derive_downscaling_parameters(climate_model, time_range, glacier_elevation_classes, roi_polygon;
                              token, cache_path, kwargs...)

Derive the regional downscaling parameters a global glacier run needs, and the glacier-area weighted
per-elevation-interval forcing that the downstream bias-correction sweep runs on.

Returns `(; grid_cells, time, decoupling, lapse_rate, elevation_interval_forcing, provenance)`:

- `grid_cells`: the region's grid cells actually used (a view of `glacier_elevation_classes`, with
  cells whose forcing was unavailable dropped — see below).
- `time`: the forcing time axis, shared by everything below.
- `decoupling`, `lapse_rate`: `DimStack`s on that axis carrying the fits at the forcing's **native
  time resolution** plus their per-timestep diagnostics — see [`derive_decoupling_factor`](@ref) and
  [`derive_lapse_rate`](@ref). Both are pure measurements: `NaN` at every timestep the region's
  forcing could not support a fit, nothing substituted, nothing clamped.
- `elevation_interval_forcing`: a **lazy** iterator over the region's elevation intervals, ascending.
  This is the one thing here that *applies* the fits, so it is where filling and clamping live.
  Each element is `(; lo, hi, center, area, n_cells, forcing)` where `forcing` is a `DimStack`
  shaped exactly like `climate_forcing` output — same seven layers, same `Ti` axis — with
  `metadata["elevation"]` set to the interval center, so it satisfies the contract
  [`gemb_glacier_cell`](@ref) enforces and a `forcing_at_elevation(interval.forcing, 0)` is an
  identity.
- `provenance`: what was derived from what — cell counts, time range, the region's extent.

Per grid cell, per interval, forcing is adjusted in the order [`forcing_at_elevation`](@ref) uses
and `climate_adjust_for_glacier` requires: **lapse to the interval center first, then decouple**.
`k` multiplies an ambient temperature already at the glacier's elevation, so the reverse order is
wrong. Cells sharing an interval are then averaged weighted by each cell's glacier area in it.

**The fits report; they do not decide.** Both are `NaN` wherever the region's forcing carried no
information, and neither is clamped — a lapse rate of 200 K/km is real evidence that the region does
not constrain a slope at that timestep, and hiding it behind a clamp would be the one thing a
diagnostic must not do. What to do about a gap or a wide spread is a downstream judgement: the
returned `Ti` axis makes it a `groupby`, at whatever resolution the decision actually needs.

```julia
using Statistics
p = derive_downscaling_parameters(...)
monthly = map(g -> median(filter(isfinite, g.lapse_rate)), groupby(p.lapse_rate, Ti => month))
diurnal = map(g -> median(filter(isfinite, g.lapse_rate)), groupby(p.lapse_rate, Ti => hour))
```

Lapse rates vary strongly on diurnal and seasonal timescales — nocturnal valley inversions against
steep midday cooling — so a wide spread across timesteps is usually physics rather than fit error.
That distinction cannot be made here, which is exactly why this function does not try to.

**Screen the two series together, not separately.** `derive_lapse_rate` consumes `k` to bring the
cells to a common on-glacier state, so a `k` outside `(0, 1]` corrupts that timestep's slope as well:
at Wrangell every fitted slope above $(_LAPSE_RATE_LIMITS[2]) K/km came from an out-of-domain `k`, and
none from a valid one. The corruption is a smooth false gradient rather than scattered noise, so it is
invisible in the slope series alone and a robust estimator does not remove it. The minimum screen is
therefore a joint one:

```julia
k, Γ = p.decoupling.decoupling_factor, p.lapse_rate.lapse_rate
usable = @. isfinite(k) & (0 < k <= 1)          # what climate_adjust_for_glacier accepts
Γ_clean = [u && isfinite(g) ? g : NaN for (u, g) in zip(usable, Γ)]
```

Timesteps where `k` is `NaN` are safe to keep: no correction was applied to them, so their slope is
the ambient one. See [`derive_decoupling_factor`](@ref) and [`derive_lapse_rate`](@ref) for the
measurements behind this.

# Keywords
- `token`, `cache_path`: passed straight through to `climate_forcing`. `cache_path` is the existing
  Zarr chunk cache; nothing else is cached, so a region is re-derived on every call (but its second
  and later passes read from that warm cache rather than the network).
- `area_minimum = 0.0`: minimum total glacier area (km²) for a cell to be included.
- `elevation_interval_batch = 8`: how many intervals `elevation_interval_forcing` accumulates per
  pass over the region. Peak memory is that many forcing stacks; cost is
  `ceil(n_intervals / elevation_interval_batch)` passes over the warm cache. `0` means one pass
  holding every interval at once.
- `min_cells = $(_MIN_CELLS_DEFAULT)`: fewest usable grid cells the region needs before either fit is
  attempted; below it every timestep reports `NaN`. Note this is effectively a *per-region* gate,
  not a per-timestep one: forcing completeness is screened per cell, so the contributing cell count
  is the same at every timestep. `k` needs four cells regardless, since it has four parameters. See
  [`derive_decoupling_factor`](@ref) for what the number was chosen against.
- `decoupling_factor_fill = 1.0`, `lapse_rate_fill = $(_DEFAULT_LAPSE_RATE)`,
  `clamp_to_valid_domain = true`: how `elevation_interval_forcing` handles a timestep the fit could
  not measure, and only there — the returned fits are untouched by these. Forwarded to the iterator;
  see it for why filling is unavoidable at the point of application.
- `forcing_loader = climate_forcing`: the loader, called as
  `forcing_loader(climate_model, lat, lon; time_range, token, cache_path)`. Injectable so the
  derivation can be exercised without a CDS token.

Grid cells whose forcing is unavailable are skipped, not fatal: ERA5-Land is land-only, so a
coastal or island cell can land on water and come back all-`NaN` ([`forcing_is_complete`](@ref)).
This mirrors how a sweep treats [`ForcingUnavailable`](@ref).
"""
function derive_downscaling_parameters(climate_model::Symbol, time_range, glacier_elevation_classes,
                                   roi_polygon;
                                   token,
                                   cache_path,
                                   area_minimum::Real = 0.0,
                                   kwargs...)
    selected = grid_cells_in_region(glacier_elevation_classes, roi_polygon; area_minimum)
    nrow(selected) == 0 && throw(ArgumentError(
        "no grid cells in the region with at least $area_minimum km² of glacier area"))

    # The region's own extent, not the selected cells' bounding box: they differ whenever the region
    # is larger than the ice in it, and it is the region the caller asked about that belongs in the
    # provenance.
    return derive_downscaling_parameters(climate_model, time_range, selected;
                                         token, cache_path, area_minimum,
                                         region_extent = _region_extent(
                                             _validate_region(roi_polygon)),
                                         kwargs...)
end

"""
    derive_downscaling_parameters(climate_model, time_range, grid_cells; token, cache_path,
                                  region_extent = nothing, kwargs...)

Derive the downscaling parameters from an **already-selected** set of grid cells, skipping the
region test entirely.

This is the method to call when the selection was made some other way — most importantly by
[`downscaling_tiles`](@ref), whose arithmetic partition cannot be expressed as a polygon (a tile's
buffer may cross the antimeridian, which [`grid_cells_in_region`](@ref) refuses by design). The
four-argument `roi_polygon` method is a thin wrapper over this one, so both paths derive
identically.

`grid_cells` is any `AbstractDataFrame` of glacier elevation-class rows — typically a `SubDataFrame`
view — carrying `:latitude`, `:longitude` and `:glm`. No area screen is applied here; `area_minimum`
is accepted only so it can be recorded in the provenance of a selection that was already screened.

`region_extent` is recorded in the provenance as-is. When `nothing`, no extent is recorded — the
cells are the selection, and a bounding box computed from them would describe the ice rather than
the region, which is a different (and easily misread) fact.

See the four-argument method for everything else: the returned fields, the keywords, and the
warnings about screening the two fitted series together.
"""
function derive_downscaling_parameters(climate_model::Symbol, time_range,
                                   grid_cells::AbstractDataFrame;
                                   token,
                                   cache_path,
                                   region_extent = nothing,
                                   area_minimum::Real = 0.0,
                                   elevation_interval_batch::Int = 8,
                                   min_cells::Int = _MIN_CELLS_DEFAULT,
                                   decoupling_factor_fill::Real = 1.0,
                                   lapse_rate_fill::Real = _DEFAULT_LAPSE_RATE,
                                   clamp_to_valid_domain::Bool = true,
                                   forcing_loader = climate_forcing)
    elevation_interval_batch >= 0 || throw(ArgumentError(
        "elevation_interval_batch must be >= 0, got $elevation_interval_batch"))
    min_cells >= 2 ||
        throw(ArgumentError("min_cells must be >= 2 to fit a slope, got $min_cells"))

    selected = grid_cells
    nrow(selected) == 0 && throw(ArgumentError("no grid cells were given"))

    load(row) = forcing_loader(climate_model, row.latitude, row.longitude;
                               time_range, token, cache_path)

    @info "Deriving downscaling parameters" grid_cells=nrow(selected) time_range

    # One pass to accumulate the cross-cell regression sums for both fits, and to learn which cells
    # actually have forcing. Both fits need only per-timestep sums over cells, so neither the cell
    # forcing stacks nor their temperature vectors are retained.
    acc = _accumulate_cross_cell(selected, load)
    isempty(acc.used) && throw(RegionForcingUnavailable(nrow(selected)))

    grid_cells = @view selected[acc.used, :]

    decoupling = derive_decoupling_factor(acc; min_cells)
    # The lapse fit needs a `k` per timestep to bring every cell to a common on-glacier state before
    # taking the slope. It reads the raw series and treats a `NaN` timestep as `k = 1` locally — the
    # bit-exact no-op — rather than being handed a pre-filled one, so no substituted value is ever
    # written into what either fit reports.
    lapse_rate = derive_lapse_rate(acc, decoupling; min_cells)

    # The whole `decoupling` stack, not just its series: the interval pass re-evaluates `k` at each
    # interval's own center from the fitted coefficients, since `k` varies with elevation and the
    # intervals are the glacier rather than the reanalysis surface.
    interval_forcing = _ElevationIntervalForcing(grid_cells, load, acc.time, decoupling,
                                                 lapse_rate.lapse_rate,
                                                 elevation_interval_batch;
                                                 decoupling_factor_fill, lapse_rate_fill,
                                                 clamp_to_valid_domain)

    provenance = Dict{String,Any}(
        "climate_model" => string(climate_model),
        "time_range" => [first(acc.time), last(acc.time)],
        "n_grid_cells_in_region" => nrow(selected),
        "n_grid_cells_used" => length(acc.used),
        "n_elevation_intervals" => length(interval_forcing.intervals),
        "area_minimum" => Float64(area_minimum),
        "mean_elevation" => acc.mean_elevation,
        "adjustment_order" => "lapse_then_decouple",
        # How much of each fit is measurement rather than absence. The judgement about whether that
        # is enough is the caller's; this just reports the count so it can be made.
        "n_timesteps" => acc.n,
        "n_decoupling_factor_fitted" => count(isfinite, decoupling.decoupling_factor),
        "n_lapse_rate_fitted" => count(isfinite, lapse_rate.lapse_rate),
    )
    # Only when there is one: an absent extent is not the same fact as a bounding box of the ice.
    region_extent === nothing || (provenance["region_extent"] = region_extent)

    # `cell_elevations`/`cell_areas` line up with `grid_cells` row for row — they are the reanalysis
    # surface elevation and glacier area of each cell the regressions actually saw. Returned because
    # they are what the fits were taken *over*: `reference_elevation` alone is their area-weighted
    # mean, from which the spread that made the slope identifiable cannot be recovered.
    return (; grid_cells, time = acc.time, decoupling, lapse_rate,
            cell_elevations = acc.elevations, cell_areas = acc.areas,
            elevation_interval_forcing = interval_forcing, provenance)
end

# Per-timestep cross-cell sums for both fits, plus the per-cell metadata the interval pass needs.
#
# The regressions are of the warm excess `E = max(T - T_ref, 0)` and of `T` itself on elevation `z`
# and glacier fraction `glm`. Accumulating the normal-equation sums as cells stream by keeps this
# O(n_time) rather than O(n_cells * n_time) — a region of 200 cells over 76 hourly years would
# otherwise be ~100 GB of retained temperature.
function _accumulate_cross_cell(selected, load)
    n = nothing
    time = nothing
    used = Int[]
    elevations = Float64[]
    areas = Float64[]

    # Sums over cells, per timestep. `s1` counts finite cells; `z`/`g` are the regressors.
    S = nothing

    for (i, row) in enumerate(eachrow(selected))
        fd = try
            load(row)
        catch e
            e isa InterruptException && rethrow()
            # A broken *caller* is not a bad cell. A typo, a stale session, or a loader whose
            # signature has moved fails identically for every cell of every region, and skipping it
            # per cell turns that into "no cell had usable forcing" — which a sweep then records as
            # an unrunnable region, i.e. reports a code bug as a property of the data. Let those
            # through so the real error is what surfaces; the network and data failures this skip
            # exists for (a CDS timeout, a cell over water) are none of these types.
            is_caller_error(e) && rethrow()
            @warn "Forcing load failed for grid cell; skipping" cell=i exception=e
            continue
        end
        forcing_is_complete(fd) || continue

        meta = DimensionalData.metadata(fd)
        z = Float64(meta["elevation"])
        T = fd[:temperature_air]

        if S === nothing
            time = collect(dims(fd, Ti))
            n = length(time)
            # Cross-products of the design vector `x = [1, z, glm, glm*z]` (the interaction term is
            # required — see `derive_decoupling_factor`), against the warm excess `E` and the
            # temperature `T`. Only the upper triangle of the symmetric XᵀX is stored.
            S = (; s1 = zeros(Int, n),
                 z = zeros(n), g = zeros(n), w = zeros(n),
                 zz = zeros(n), zg = zeros(n), zw = zeros(n),
                 gg = zeros(n), gw = zeros(n), ww = zeros(n),
                 E = zeros(n), zE = zeros(n), gE = zeros(n), wE = zeros(n),
                 T = zeros(n), zT = zeros(n),
                 # Second moment of the warm excess, for the decoupling fit's `r2`.
                 EE = zeros(n))
        else
            length(T) == n || throw(ArgumentError(
                "grid cell $i has $(length(T)) forcing steps but the region's first cell has " *
                "$n; every cell must share one time axis"))
        end

        glm = _row_glm(row)
        area = glacier_area_total(row)

        # The interaction regressor, formed once per cell rather than per timestep.
        gz = glm * z

        @inbounds for t in 1:n
            Tt = T[t]
            Et = max(Tt - _DECOUPLING_REFERENCE_TEMPERATURE, 0.0)
            S.s1[t] += 1
            S.z[t]  += z;          S.g[t]  += glm;         S.w[t]  += gz
            S.zz[t] += z * z;      S.zg[t] += z * glm;     S.zw[t] += z * gz
            S.gg[t] += glm * glm;  S.gw[t] += glm * gz;    S.ww[t] += gz * gz
            S.E[t]  += Et;         S.zE[t] += z * Et
            S.gE[t] += glm * Et;   S.wE[t] += gz * Et
            S.T[t]  += Tt;         S.zT[t] += z * Tt
            S.EE[t] += Et * Et
        end

        push!(used, i)
        push!(elevations, z)
        push!(areas, area)
    end

    S === nothing && return (; time = DateTime[], n = 0, used, elevations, areas,
                             sums = nothing, mean_elevation = NaN)

    # Area-weighted mean elevation: the reference the fitted factor is reported *at*, since the
    # regression's intercept alone would be an extrapolation to sea level.
    total_area = sum(areas)
    mean_elevation = total_area > 0 ? sum(areas .* elevations) / total_area :
                     Statistics.mean(elevations)

    return (; time, n, used, elevations, areas, sums = S, mean_elevation)
end

"""
    _ElevationIntervalForcing(grid_cells, load, time, decoupling, lapse_rate,
                              elevation_interval_batch; decoupling_factor_fill,
                              lapse_rate_fill, clamp_to_valid_domain)

Lazy iterator over a region's elevation intervals, yielding glacier-area weighted forcing for each.

Each element is `(; lo, hi, center, area, n_cells, decoupling_factor, forcing)`. Intervals ascend in
elevation, and `area` sums over the iterator to the region's total glacier area, so nothing is
dropped.

**This is the one place a fit is applied, so it is the one place fill and clamp live.** Both fits
report `NaN` where the region's forcing supported nothing, and a single `NaN` in an applied series
propagates through the arithmetic into that interval's temperature — at which point the whole interval
fails [`forcing_is_complete`](@ref) and a sweep skips it as silently as it skips an ocean cell. So an
unmeasurable timestep is filled (`decoupling_factor_fill`, default `1.0`, the bit-exact no-op; and
`lapse_rate_fill`) and, with `clamp_to_valid_domain`, every applied value is brought into the domain
its consumer validates against — `k` into `(0, 1]`, the lapse rate into `$(_LAPSE_RATE_LIMITS)` K/km.
Without the clamp a fitted `k` of 3.2 warms the glacier surface instead of damping it, and the failure
surfaces as a units-validation stacktrace from inside `climate_adjust_for_glacier`, several frames
from its cause. Each interval's metadata records how many timesteps were filled and how many clamped,
so a downstream reader can tell a measured interval from a mostly-substituted one.

**`k` is re-evaluated at each interval's own center**, not taken once at the region's mean reanalysis
elevation — see [`decoupling_factor_at_elevation`](@ref). The fit carries a `glm × z` interaction, so
`k` is a function of elevation, and the intervals are the glacier: on real Alpine forcing 76% of the
glacier area sits above every contributing reanalysis surface, so evaluating at `z̄` would apply the
wrong factor to almost all of the ice. In the Ötztal that difference runs from `k = 0.94` at 2150 m to
0.78 at 3750 m, and it is a *gradient* in the forcing, which is precisely what the downstream sweep
reads as a mass-balance signal.

Laziness is what makes a large region tractable. All intervals at once is `n_intervals * 7 * n_time`
Float64s — ~2 GB for 50 intervals of 76 hourly years — so instead each `iterate` streams the region's
cells again and accumulates only the next `elevation_interval_batch` intervals, holding that many
interval stacks. The cost is `ceil(n_intervals / elevation_interval_batch)` passes over the (warm,
after the parameter fit) Zarr chunk cache rather than one. `elevation_interval_batch = 0` opts out and
does a single pass.
"""
struct _ElevationIntervalForcing{C,L,B}
    grid_cells::C
    load::L
    time::Vector{DateTime}
    lapse_rate::Vector{Float64}
    elevation_interval_batch::Int
    intervals::B
    n_lapse_rate_filled::Int
    n_lapse_rate_clamped::Int
end

function _ElevationIntervalForcing(grid_cells, load, time, decoupling, lapse_rate,
                                   elevation_interval_batch::Int;
                                   decoupling_factor_fill::Real = 1.0,
                                   lapse_rate_fill::Real = _DEFAULT_LAPSE_RATE,
                                   clamp_to_valid_domain::Bool = true)
    k_fill = Float64(decoupling_factor_fill)
    lr_fill = Float64(lapse_rate_fill)
    lo_k, hi_k = _DECOUPLING_FACTOR_LIMITS
    lo_lr, hi_lr = _LAPSE_RATE_LIMITS
    # The fills are applied, so they have to be applicable — an out-of-domain fill would be rejected
    # downstream no differently from an out-of-domain fit, but with nothing to inspect afterwards.
    lo_k < k_fill <= hi_k || throw(ArgumentError(
        "decoupling_factor_fill must lie in $(_DECOUPLING_FACTOR_LIMITS), the domain " *
        "`climate_adjust_for_glacier` accepts, got $decoupling_factor_fill"))
    lo_lr <= lr_fill <= hi_lr || throw(ArgumentError(
        "lapse_rate_fill must lie within $(_LAPSE_RATE_LIMITS) K/km, the range " *
        "`climate_adjust_for_elevation` validates against, got $lapse_rate_fill"))

    # The lapse series, made applicable once here rather than per interval — it does not vary with
    # elevation, so every interval uses the same vector.
    applied_lr, n_lr_filled, n_lr_clamped =
        _make_applicable(lapse_rate, lr_fill, _LAPSE_RATE_LIMITS;
                         open_lower = false, clamp_to_valid_domain)

    # Union of populated intervals across the region. The `hyps_*` column names are authoritative —
    # the table's `hypsometry_bin_edges` metadata does not survive a GeoParquet round-trip — so
    # `glacier_hypsometry` is the decoder to go through.
    # Which intervals are populated, not how much area is in them: the area each interval ends up
    # reporting is accumulated during iteration, over the cells whose forcing was actually usable,
    # and a second total summed here would disagree with it whenever a cell drops out.
    seen = Set{Tuple{Int,Int}}()
    for row in eachrow(grid_cells)
        for b in glacier_hypsometry(row; area_minimum = 0)
            push!(seen, (b.lo, b.hi))
        end
    end
    ordered = sort!(collect(seen))
    # The hypsometry range is passed as the validity interval, but every interval center is inside it
    # by construction — it is there so an interval can never be the thing that produces a `NaN`
    # factor.
    z_range = isempty(ordered) ? nothing :
              (Float64(first(ordered)[1]), Float64(last(ordered)[2]))

    # `k(z)` per interval, evaluated at the center and then made applicable. Eager (one series per
    # interval, cheap next to a forcing stack).
    intervals = map(ordered) do (lo, hi)
        center = (lo + hi) / 2
        raw_k = decoupling_factor_at_elevation(decoupling, center; elevation_range = z_range)
        applied, n_filled, n_clamped =
            _make_applicable(raw_k, k_fill, _DECOUPLING_FACTOR_LIMITS;
                             open_lower = true, clamp_to_valid_domain)
        (; lo, hi, center,
         decoupling_factor = applied,
         n_decoupling_factor_filled = n_filled,
         n_decoupling_factor_clamped = n_clamped,
         n_decoupling_factor_held = get(DimensionalData.metadata(raw_k), "n_held", 0))
    end

    return _ElevationIntervalForcing(grid_cells, load, collect(time), applied_lr,
                                     elevation_interval_batch, intervals,
                                     n_lr_filled, n_lr_clamped)
end

Base.length(ivf::_ElevationIntervalForcing) = length(ivf.intervals)
Base.eltype(::Type{<:_ElevationIntervalForcing}) = NamedTuple

function Base.iterate(ivf::_ElevationIntervalForcing, state = (1, nothing, 0))
    next, batch, batch_start = state
    next > length(ivf.intervals) && return nothing

    # Refill when the current batch is exhausted: one pass over the region's cells accumulating the
    # next `elevation_interval_batch` intervals.
    if batch === nothing || next >= batch_start + length(batch)
        stop = ivf.elevation_interval_batch == 0 ? length(ivf.intervals) :
               min(next + ivf.elevation_interval_batch - 1, length(ivf.intervals))
        batch = _accumulate_intervals(ivf, next:stop)
        batch_start = next
    end

    return batch[next - batch_start + 1], (next + 1, batch, batch_start)
end

# One pass over the region's grid cells, accumulating area-weighted forcing for `range`'s intervals.
function _accumulate_intervals(ivf::_ElevationIntervalForcing, range)
    n = length(ivf.time)
    intervals = ivf.intervals[range]
    # Running weighted sums per interval, one array per forcing layer, plus the weight totals.
    sums = [Dict(v => zeros(n) for v in _FORCING_VARIABLES) for _ in intervals]
    weights = zeros(length(intervals))
    counts = zeros(Int, length(intervals))
    template = nothing
    # Highest reanalysis surface contributing to each interval. Per interval, not per batch, so the
    # extrapolation this reports does not depend on how the batching happens to group them.
    z_max = fill(-Inf, length(intervals))

    for row in eachrow(ivf.grid_cells)
        # Which of this batch's intervals does this cell hold ice in, and how much?
        areas = _interval_areas(row, intervals)
        all(iszero, areas) && continue

        fd = try
            ivf.load(row)
        catch e
            e isa InterruptException && rethrow()
            # Same reasoning as in `_accumulate_cross_cell`: a broken caller is not a bad cell, and
            # swallowing it here would silently drop that cell's ice out of the interval weights.
            is_caller_error(e) && rethrow()
            @warn "Forcing load failed during elevation interval aggregation; skipping cell" exception=e
            continue
        end
        forcing_is_complete(fd) || continue
        template === nothing && (template = fd)

        z_cell = Float64(DimensionalData.metadata(fd)["elevation"])

        # How much of the correction this cell still needs, after the reanalysis has already applied
        # its own `glm`-worth of it — the same weighting `cell_decoupling_factor` applies, and the
        # same one `derive_lapse_rate` undoes to fit the slope.
        glm = _row_glm(row)

        for (i, interval) in enumerate(intervals)
            areas[i] > 0 || continue
            # `k` at *this interval's* elevation, weighted down by the cell's `glm`. Per interval
            # rather than per cell, because `k` varies with elevation and the interval center is
            # where the forcing is being delivered — only `glm` differs between cells.
            cell_factor = [_effective_decoupling_factor(k, glm) for k in interval.decoupling_factor]
            adjusted = _cell_forcing_at_interval(fd, interval.center - z_cell, cell_factor,
                                                 ivf.lapse_rate)
            w = areas[i]
            for v in _FORCING_VARIABLES
                s = sums[i][v]
                a = adjusted[v]
                @inbounds for t in 1:n
                    s[t] += w * a[t]
                end
            end
            weights[i] += w
            counts[i] += 1
            z_max[i] = max(z_max[i], z_cell)
        end
    end

    template === nothing && throw(ArgumentError(
        "no grid cell in the region yielded usable forcing during elevation interval aggregation"))

    # How far each interval is being lapse-extrapolated past the reanalysis surface it came from.
    # Glaciers systematically occupy the high ground *within* a 0.1° cell, so an interval well above
    # every contributing cell is the normal case rather than an error — on real Alpine forcing 76%
    # of the glacier area sat above every contributing reanalysis surface. But it is a genuine
    # extrapolation, and it is the dominant uncertainty in the forcing the downstream sweep then fits
    # its bias correction against, so it should not be silent. One line per batch.
    far = [(intervals[i].center, intervals[i].center - z_max[i])
           for i in eachindex(intervals)
           if isfinite(z_max[i]) && intervals[i].center - z_max[i] > 500]
    isempty(far) || @info("Elevation intervals extrapolated far above the reanalysis surface " *
                          "(expected for glaciers, but this forcing is a lapse extrapolation)",
                          n_elevation_intervals = length(far),
                          highest_interval = maximum(first, far),
                          max_extrapolation = round(maximum(last, far), digits = 1))

    return [_interval_stack(intervals[i], sums[i], weights[i], counts[i], template, ivf, z_max[i])
            for i in eachindex(intervals)]
end

# This cell's glacier area (km²) in each of `intervals`, by decoding its flat `hyps_*` columns.
# Indexed by `(lo, hi)` rather than scanned per interval: this runs once per cell per batch pass, so
# a linear scan makes it O(n_intervals * n_bins) on every pass over the region.
function _interval_areas(row, intervals)
    have = Dict((b.lo, b.hi) => b.area for b in glacier_hypsometry(row; area_minimum = 0))
    return [get(have, (interval.lo, interval.hi), 0.0) for interval in intervals]
end

# Finish one elevation interval: divide the weighted sums by the total weight and rebuild a stack
# shaped like `climate_forcing` output, so it drops straight into the existing sweep.
function _interval_stack(interval, sums, weight, n_cells, template, ivf, z_max)
    weight > 0 || throw(ArgumentError(
        "elevation interval $(interval.lo)-$(interval.hi) m accumulated zero area"))
    time_dim = dims(template, Ti)
    layers = NamedTuple(v => DimArray(sums[v] ./ weight, (time_dim,);
                                      metadata = DimensionalData.metadata(template[v]))
                        for v in _FORCING_VARIABLES)

    meta = merge(copy(DimensionalData.metadata(template)), Dict(
        # The interval center is this stack's reference elevation: it is already *at* the interval, so
        # a downstream `forcing_at_elevation(interval.forcing, 0)` is an identity and
        # `forcing_is_complete` finds a finite elevation where it looks for one.
        "elevation" => interval.center,
        "glacier_area" => weight,
        "n_grid_cells" => n_cells,
        "elevation_interval_lower" => Float64(interval.lo),
        "elevation_interval_upper" => Float64(interval.hi),
        # How far above the highest contributing reanalysis surface this interval sits. Positive means
        # the forcing here is a lapse extrapolation, which for glaciers is the norm rather than the
        # exception — they occupy the high ground within a 0.1° cell — and is the dominant
        # uncertainty in what the downstream sweep fits its bias correction against.
        "extrapolation_above_reanalysis" => interval.center - z_max,
        "temperature_lapse_rate" => Statistics.mean(ivf.lapse_rate),
        # `k` at *this* interval's center, not the region's mean — see
        # `decoupling_factor_at_elevation`. The three counts say how much of the applied series was
        # measured: `n_held` is timesteps carrying the ceiling value because the fit's ambient excess
        # runs out below this elevation, `n_filled` is timesteps the fit could not measure at all,
        # and `n_clamped` is fits that landed outside the domain their consumer accepts.
        "glacier_decoupling_factor_mean" => Statistics.mean(interval.decoupling_factor),
        "glacier_decoupling_factor_n_held" => interval.n_decoupling_factor_held,
        "glacier_decoupling_factor_n_filled" => interval.n_decoupling_factor_filled,
        "glacier_decoupling_factor_n_clamped" => interval.n_decoupling_factor_clamped,
        "temperature_lapse_rate_n_filled" => ivf.n_lapse_rate_filled,
        "temperature_lapse_rate_n_clamped" => ivf.n_lapse_rate_clamped,
        "adjustment_order" => "lapse_then_decouple",
        "temperature_air_mean" => Statistics.mean(layers.temperature_air),
        "precipitation_mean" => Statistics.mean(layers.precipitation),
        "wind_speed_mean" => Statistics.mean(layers.wind_speed)))
    # Cell-specific keys would be a lie on a regional average.
    for k in ("latitude", "longitude", "chunk_strategy", "delta_elevation", "decoupling_factor")
        delete!(meta, k)
    end

    return (; interval.lo, interval.hi, interval.center, area = weight, n_cells,
            decoupling_factor = interval.decoupling_factor,
            forcing = DimStack(layers; metadata = meta))
end
