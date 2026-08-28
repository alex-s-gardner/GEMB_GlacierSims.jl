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
# resolution, and `NaN` where it supports nothing, substituting and clamping nothing. Forcing a glacier
# needs a finite, in-domain value everywhere, because a single `NaN` in an applied series makes that
# band's forcing fail `forcing_is_complete` and a sweep skips it as silently as it skips an ocean cell.
# That crossing is `applied_downscaling.jl`'s job, not this file's: `resolve_downscaling` accepts or
# rejects each fit against its own diagnostics and labels the provenance of every substitute. The
# aggregation here consumes what it produced and neither fills nor clamps.
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
- `applied`: the fits resolved into applied parameters, with the provenance of every value — see
  [`resolve_downscaling`](@ref). This is the crossing from measurement to application, and the one
  place a substitution happens.
- `elevation_interval_forcing`: a **lazy** iterator over the region's elevation intervals, ascending,
  built on `applied`.
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
- `downscaling_basis = :fitted`, `lapse_rate_stderr_maximum`, `elevation_spread_minimum`,
  `lapse_rate_window`,
  `lapse_rate_prior`, `decoupling_factor_prior`: how the fits are turned into the applied parameters
  `elevation_interval_forcing` uses, and only there — the returned fits are untouched by these. All
  forwarded to [`resolve_downscaling`](@ref); see it for what each one accepts or rejects. `:fitted`
  is the default here because this method derives and applies over one window, so every timestep has
  its own fit to prefer; a run over a window wider than the fits needs `:climatology`.
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
                                   downscaling_basis::Symbol = :fitted,
                                   elevation_spread_minimum::Real =
                                       APPLIED_ELEVATION_SPREAD_MINIMUM,
                                   lapse_rate_stderr_maximum::Real =
                                       APPLIED_LAPSE_RATE_STDERR_MAXIMUM,
                                   lapse_rate_window = APPLIED_LAPSE_RATE_WINDOW,
                                   lapse_rate_prior = _DEFAULT_LAPSE_RATE,
                                   decoupling_factor_prior = nothing,
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
    # How well determined that slope is, from the same sums. Separate from the fit because upstream owns
    # the fit and this package owns the accumulator the extra moments live in.
    lapse_rate_uncertainty = derive_lapse_rate_uncertainty(acc, decoupling; min_cells)

    # The crossing from measured to applied. `resolve_downscaling` re-evaluates `k` at each band's own
    # center from the fitted coefficients — `k` varies with elevation, and the bands are the glacier
    # rather than the reanalysis surface — accepts or rejects each fit against its diagnostics, and
    # labels the provenance of whatever it substitutes. Nothing downstream of it fills or clamps.
    fit = (; time = acc.time, decoupling, lapse_rate)
    applied = resolve_downscaling(fit, hypsometry_intervals(grid_cells), acc.time;
                                  basis = downscaling_basis, min_cells,
                                  spread_minimum = elevation_spread_minimum,
                                  stderr_maximum = lapse_rate_stderr_maximum,
                                  lapse_rate_window, lapse_rate_prior, decoupling_factor_prior)
    interval_forcing = _ElevationIntervalForcing(grid_cells, load, applied,
                                                 elevation_interval_batch)

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
    # How the fits were turned into applied parameters, so a stored derivation records the policy it
    # was produced under rather than only its inputs.
    merge!(provenance, applied.settings)
    # Only when there is one: an absent extent is not the same fact as a bounding box of the ice.
    region_extent === nothing || (provenance["region_extent"] = region_extent)

    # `cell_elevations`/`cell_areas` line up with `grid_cells` row for row — they are the reanalysis
    # surface elevation and glacier area of each cell the regressions actually saw. Returned because
    # they are what the fits were taken *over*: `reference_elevation` alone is their area-weighted
    # mean, from which the spread that made the slope identifiable cannot be recovered.
    return (; grid_cells, time = acc.time, decoupling, lapse_rate, lapse_rate_uncertainty, applied,
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
                 EE = zeros(n),
                 # Second moments involving temperature, for the *lapse* fit's residual scatter —
                 # see `derive_lapse_rate_uncertainty`. Three are needed rather than one because the
                 # lapse fit regresses the decoupling-*corrected* temperature, and squaring
                 # `T + c(α + βz)(1 - glm)` needs the cross terms in `glm` and `glm*z` too.
                 #
                 # `TT` accumulates the square of the temperature **about the melting point**, not of
                 # the temperature itself. The residual sum of squares is translation-invariant, so
                 # this changes no result — but it is the difference between computing it and not.
                 # Absolute temperatures are ~270 K, so `ΣT²` runs to ~1e5 per cell while the variance
                 # about the mean is O(1): forming `ΣT² - (ΣT)²/m` from unshifted sums cancels away
                 # eleven digits and reports ~5e-6 K of scatter for a fit that is exact. Shifted, the
                 # magnitudes are O(20) and the floor drops with the square of that ratio.
                 TT = zeros(n), gT = zeros(n), wT = zeros(n))
        else
            length(T) == n || throw(ArgumentError(
                "grid cell $i has $(length(T)) forcing steps but the region's first cell has " *
                "$n; every cell must share one time axis"))
        end

        _accumulate_cell!(S, T, z, _row_glm(row), n)

        push!(used, i)
        push!(elevations, z)
        push!(areas, glacier_area_total(row))
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
    elevation_interval_forcing(grid_cells, applied; climate_model, time_range, token, cache_path,
                               elevation_interval_batch = 0, forcing_loader = climate_forcing)

Lazy iterator over the elevation bands of `applied`, yielding glacier-area weighted forcing for each.

Each element is `(; lo, hi, center, area, n_cells, decoupling_factor, forcing)`. Bands ascend in
elevation, and `area` sums over the iterator to the total glacier area of the cells whose forcing was
usable, so nothing is dropped between the cells and the bands.

`applied` is an [`AppliedDownscaling`](@ref): it supplies the time axis, the applied lapse rate, and
one applied `k` series per band, each already resolved onto that axis and inside the domain its
consumer validates. **Nothing is filled or clamped here** — that decision belongs to
[`resolve_downscaling`](@ref), which labels the provenance of every value it substitutes, and doing
it in two places would let a band's forcing and its stated provenance disagree.

`grid_cells` decides the *area*, and is deliberately a separate argument from the cells the parameters
were fitted from. A tile fits its parameters over a buffered neighbourhood so the cross-cell
regressions have enough elevation range to be identifiable, but it must aggregate area over its
**core** cells only, or the ice in the buffer is counted once for this tile and again for its
neighbour.

Per grid cell, per band, forcing is adjusted in the order `climate_adjust_for_glacier` requires:
**lapse to the band center first, then decouple.** `k` multiplies an ambient temperature already at the
glacier's elevation, so the reverse order is wrong, and the two do not commute — the decoupling damps
only the excess above a fixed melting point, and lapsing changes which timesteps are above it. Cells
sharing a band are then averaged weighted by each cell's glacier area in it.

`k` is applied at **each band's own center**, weighted down per donor cell by that cell's ERA5-Land
glacier fraction (`_effective_decoupling_factor`): the reanalysis has already damped the share of the
cell it runs as ice, and only the rest needs correcting.

Laziness is what makes a large region tractable. All bands at once is `n_bands * 7 * n_time` Float64s
— ~2 GB for 50 bands of 76 hourly years — so instead each `iterate` streams the cells again and
accumulates only the next `elevation_interval_batch` bands, holding that many forcing stacks. The cost
is `ceil(n_bands / elevation_interval_batch)` passes over the Zarr chunk cache rather than one.
`elevation_interval_batch = 0` opts out and does a single pass, which is the right choice once the
cache is warm.
"""
function elevation_interval_forcing(grid_cells, applied::AppliedDownscaling;
                                    climate_model::Symbol,
                                    time_range,
                                    token = nothing,
                                    cache_path = nothing,
                                    elevation_interval_batch::Int = 0,
                                    forcing_loader = climate_forcing)
    load(row) = forcing_loader(climate_model, row.latitude, row.longitude;
                               time_range, token, cache_path)
    return _ElevationIntervalForcing(grid_cells, load, applied, elevation_interval_batch)
end

struct _ElevationIntervalForcing{C,L,B}
    grid_cells::C
    load::L
    time::Vector{DateTime}
    lapse_rate::Vector{Float64}
    lapse_rate_source::Vector{Int8}
    elevation_interval_batch::Int
    intervals::B
end

function _ElevationIntervalForcing(grid_cells, load, applied::AppliedDownscaling,
                                   elevation_interval_batch::Int)
    elevation_interval_batch >= 0 || throw(ArgumentError(
        "elevation_interval_batch must be >= 0, got $elevation_interval_batch"))
    return _ElevationIntervalForcing(grid_cells, load, applied.time, applied.lapse_rate,
                                     applied.lapse_rate_source, elevation_interval_batch,
                                     applied.bands)
end

Base.eltype(::Type{<:_ElevationIntervalForcing}) = NamedTuple

# Deliberately not `HasLength`. `ivf.intervals` is every interval the hypsometry populates, but an
# interval whose only donor cells turn out to have no usable forcing accumulates no area and cannot be
# emitted — and which those are is not known until the forcing has been read. Declaring a length that
# the iteration then failed to produce would break `collect`, so the count is honest instead: it is
# `length(collect(...))`, and `ivf.intervals` is the upper bound.
Base.IteratorSize(::Type{<:_ElevationIntervalForcing}) = Base.SizeUnknown()

function Base.iterate(ivf::_ElevationIntervalForcing, state = (1, nothing, 0))
    next, batch, batch_start = state

    while next <= length(ivf.intervals)
        # Refill when the current batch is exhausted: one pass over the region's cells accumulating the
        # next `elevation_interval_batch` intervals.
        if batch === nothing || next >= batch_start + length(batch)
            stop = ivf.elevation_interval_batch == 0 ? length(ivf.intervals) :
                   min(next + ivf.elevation_interval_batch - 1, length(ivf.intervals))
            batch = _accumulate_intervals(ivf, next:stop)
            batch_start = next
        end

        element = batch[next - batch_start + 1]
        next += 1
        # `nothing` is an interval that accumulated no area; skip it rather than emitting a stack with
        # no forcing behind it.
        element === nothing || return element, (next, batch, batch_start)
    end

    return nothing
end

# One pass over the region's grid cells, accumulating area-weighted forcing for `range`'s intervals.
function _accumulate_intervals(ivf::_ElevationIntervalForcing, range)
    n = length(ivf.time)
    intervals = ivf.intervals[range]
    # Running weighted sums per interval, one array per forcing layer, plus the weight totals.
    sums = [Dict(v => zeros(n) for v in _FORCING_VARIABLES) for _ in intervals]
    weights = zeros(length(intervals))
    counts = zeros(Int, length(intervals))
    # Intervals no donor cell can legally be adjusted to, and why. Extrapolating a cell's forcing far
    # enough above its own surface eventually takes it outside the range `climate_forcing` validates —
    # in the Khumbu the top bands sit ~2.6 km above every contributing cell, and the re-derived downward
    # longwave there falls below the 50 W/m² floor. That is the validator working, so the interval is
    # refused rather than fudged; refusing the whole interval rather than the failing cell's share keeps
    # the average from silently re-weighting onto whichever donors happened to be close enough.
    unreachable = falses(length(intervals))
    reasons = Vector{String}(undef, length(intervals))
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

        # Threaded over intervals, not over cells. This inner loop is the whole cost of the pass —
        # `n_cells * n_intervals` lapse-and-decouple passes over the record, which for a 335-cell tile
        # of 60 bands is 20,000 of them — and each iteration writes only its own interval's slot, so
        # the accumulation order within an interval stays the cell order and the result is bit-for-bit
        # what the serial loop gives. Threading over cells instead would have several threads adding
        # into the same interval, where floating-point addition is not associative.
        Threads.@threads for i in eachindex(intervals)
            areas[i] > 0 && !unreachable[i] || continue
            interval = intervals[i]
            # `k` at *this interval's* elevation, weighted down by the cell's `glm`. Per interval
            # rather than per cell, because `k` varies with elevation and the interval center is
            # where the forcing is being delivered — only `glm` differs between cells.
            cell_factor = [_effective_decoupling_factor(k, glm) for k in interval.decoupling_factor]
            adjusted = try
                _cell_forcing_at_interval(fd, interval.center - z_cell, cell_factor,
                                          ivf.lapse_rate)
            catch e
                e isa InterruptException && rethrow()
                is_caller_error(e) && rethrow()
                # Caught inside the loop body: an exception escaping a `Threads.@threads` region
                # becomes a `TaskFailedException` that names the thread rather than the interval, and
                # would take down a whole tile over its highest few bands.
                unreachable[i] = true
                reasons[i] = sprint(showerror, e)
                continue
            end
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

    # An interval the hypsometry populates but whose donor cells all turned out to have no usable
    # forcing accumulates no area, and there is nothing to average. It is dropped rather than emitted,
    # and named rather than dropped silently: the area it carried is real ice that this tile's totals
    # will not account for, which is the same gap a skipped cell leaves and has to be visible the same
    # way. Common on the Antarctic and Aleutian coasts, where ERA5-Land is land-only and a whole
    # elevation band's cells can fall on water.
    dropped = [intervals[i].center for i in eachindex(intervals)
               if weights[i] <= 0 && !unreachable[i]]
    isempty(dropped) || @warn("Elevation intervals dropped: their donor cells carried no usable " *
                              "forcing, so the interval accumulated no glacier area",
                              n_dropped = length(dropped), centers = dropped)

    # Reported separately from the above, because the cause and the remedy are different: this is the
    # lapse extrapolation running out, not missing data, and it is the signal that the tile's hypsometry
    # reaches higher than its reanalysis cells can be stretched to.
    far = findall(unreachable)
    isempty(far) || @warn("Elevation intervals dropped: no donor cell could be adjusted to them " *
                          "without leaving the range `climate_forcing` validates. Expected at the " *
                          "top of a tile whose glaciers sit far above every reanalysis surface.",
                          n_dropped = length(far),
                          centers = [intervals[i].center for i in far],
                          first_reason = reasons[first(far)])

    return [(weights[i] > 0 && !unreachable[i]) ?
            _interval_stack(intervals[i], sums[i], weights[i], counts[i], template, ivf, z_max[i]) :
            nothing
            for i in eachindex(intervals)]
end

"""
    derive_lapse_rate_uncertainty(acc, decoupling; min_cells = $(_MIN_CELLS_DEFAULT))

How well determined each timestep's fitted lapse rate is, from the same cross-cell sums the fit itself
uses.

Returns a `DimStack` on the forcing's time axis:

- `lapse_rate_stderr` — the standard error of the fitted slope, K km-1.
- `lapse_rate_r2` — fraction of the cross-cell temperature variance the slope explains.
- `residual_sd` — the residual scatter about the fitted line, K.

This is the diagnostic [`derive_lapse_rate`](@ref) cannot supply and the one a trust threshold actually
wants, because it is the only one that **varies with time**. `elevation_spread` and the cell count are
fixed by which cells a tile has, so they are tile constants; the scatter is not. The standard error
factors into exactly those two parts,

    SE(slope) = residual_sd / sqrt(Szz)

— leverage the tile always has, times scatter it has at this instant. On a clear afternoon with clean
lapse structure the cells fall on a line tightly and the slope is well determined; during a frontal
passage or a patchy surface inversion they scatter and it is not, with the same cells throughout.

These are not iid measurement errors — the residuals are real spatial structure, different cells having
genuinely different microclimates. So this is not a sampling standard error in the textbook sense. That
makes it more apt for this decision rather than less: what it measures is how well *one* linear lapse
rate describes these cells at this timestep, which is the question behind accepting the fit.

The regression described is the same one `derive_lapse_rate` performs — on the decoupling-**corrected**
temperature `T' = T + (k-1)(1-glm)·E_ambient(z)`, under the same `fit_ok` screen, so the reported error
belongs to the reported slope. That is why three extra accumulator sums are needed and not one:
squaring `T'` brings in the cross terms in `glm` and `glm·z`. `NaN` wherever the slope itself is `NaN`,
and wherever there are too few cells to have any residual degrees of freedom.
"""
function derive_lapse_rate_uncertainty(acc, decoupling; min_cells::Int = _MIN_CELLS_DEFAULT)
    n = acc.n
    ti = Ti(acc.time)

    stderr = fill(NaN, n)
    r2 = fill(NaN, n)
    residual_sd = fill(NaN, n)

    S = acc.sums
    # A tile accumulated before the temperature second moments existed carries no `TT`, and asking it
    # for a lapse-rate uncertainty must report "not measured" rather than throw on a missing field.
    if S !== nothing && :TT in propertynames(S)
        k = decoupling.decoupling_factor
        cα, cβ = decoupling.coef_alpha, decoupling.coef_beta

        @inbounds for t in 1:n
            # Three cells give a slope but no residual degrees of freedom; `m - 2` must be positive.
            S.s1[t] >= max(min_cells, 3) || continue
            m = Float64(S.s1[t])

            Szz = S.zz[t] - S.z[t] * S.z[t] / m
            Szz > 0 || continue

            # The same screen `derive_lapse_rate` applies, so this error belongs to that slope: an
            # out-of-domain `k` makes the correction arbitrary, and both treat such a timestep as
            # `k = 1`, the exact no-op.
            fit_ok = isfinite(k[t]) && 0.0 < k[t] <= 1.0 && isfinite(cα[t]) && isfinite(cβ[t])
            c = fit_ok ? k[t] - 1.0 : 0.0
            α, β = fit_ok ? (cα[t], cβ[t]) : (0.0, 0.0)

            # Sums of the correction `u = (α + βz)(1 - glm)` and of its products with `z` and `T`,
            # all from the stored regressor moments.
            su   = α * (m - S.g[t])      + β * (S.z[t]  - S.w[t])
            szu  = α * (S.z[t] - S.w[t]) + β * (S.zz[t] - S.zw[t])
            suu  = α * α * (m - 2 * S.g[t] + S.gg[t]) +
                   2 * α * β * (S.z[t] - 2 * S.w[t] + S.gw[t]) +
                   β * β * (S.zz[t] - 2 * S.zw[t] + S.ww[t])

            # Everything below works in temperature **relative to the melting point**, because that is
            # the frame `TT` was accumulated in and the frame the cancellation is tolerable in. Every
            # quantity produced from here — slope, residual, R² — is translation-invariant, so the
            # shift is free. `S.T`, `S.gT` and `S.wT` are unshifted, and their shifts are exactly
            # `T_ref` times sums already in hand.
            r = _DECOUPLING_REFERENCE_TEMPERATURE
            sT_r  = S.T[t]  - m * r
            szT_r = S.zT[t] - S.z[t] * r
            sTu_r = α * ((S.T[t] - m * r) - (S.gT[t] - S.g[t] * r)) +
                    β * ((S.zT[t] - S.z[t] * r) - (S.wT[t] - S.w[t] * r))

            sT  = sT_r  + c * su
            szT = szT_r + c * szu
            sTT = S.TT[t] + 2 * c * sTu_r + c * c * suu

            SzT = szT - S.z[t] * sT / m
            STT = sTT - sT * sT / m
            STT > 0 || continue

            # Residual sum of squares of the corrected temperatures about the fitted line. Clamped at
            # zero: with a near-perfect fit it is a difference of two nearly equal large numbers and
            # can land a rounding step below.
            rss = max(STT - SzT * SzT / Szz, 0.0)

            residual_sd[t] = sqrt(rss / (m - 2))
            # K m-1 -> K km-1, matching `lapse_rate`'s unit. Unsigned: an error has no direction.
            stderr[t] = 1000.0 * residual_sd[t] / sqrt(Szz)
            r2[t] = clamp(1.0 - rss / STT, 0.0, 1.0)
        end
    end

    metadata = Dict{String,Any}("min_cells" => min_cells,
                                "n_fitted" => count(isfinite, stderr),
                                "n_timesteps" => n)
    return DimStack((lapse_rate_stderr = DimArray(stderr, ti),
                     lapse_rate_r2 = DimArray(r2, ti),
                     residual_sd = DimArray(residual_sd, ti)); metadata)
end

# Add one cell's contribution to the cross-cell sums, for every timestep.
#
# A separate function purely as a **function barrier**, and it must stay one: inlining it back costs a
# factor of 200 on this loop.
#
# In `_accumulate_cross_cell` the sums start as `nothing` — that sentinel is how the first cell's
# forcing establishes the time axis — so `S` is inferred there as `Union{Nothing,NamedTuple{...}}`, and
# a `S.z` in the loop body is then a *dynamic* property lookup: 17 of them, 17,521 timesteps, once per
# cell. On a real tile that put `Base.getproperty` at 52% of the accumulation's self time and allocated
# ~13 MiB per cell in boxes. Passed as an argument the type is concrete at the callee, every
# `getproperty` folds to a field load, and the loop allocates nothing at all.
function _accumulate_cell!(S, T, z, glm, n)
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
        # About the melting point, for conditioning — see the comment where `TT` is allocated. `gT`
        # and `wT` stay unshifted: their shift is exactly `T_ref` times `Σglm` and `Σ(glm z)`, both
        # already accumulated, so it can be applied on the way out without losing anything.
        Tr = Tt - _DECOUPLING_REFERENCE_TEMPERATURE
        S.TT[t] += Tr * Tr;    S.gT[t] += glm * Tt;    S.wT[t] += gz * Tt
    end
    return nothing
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

    # Where `k` actually changed this band's forcing. The decoupling correction scales
    # `max(T - T_ref, 0)`, so at a below-freezing timestep every value of `k` yields bit-identical
    # forcing — which makes an unqualified count of substituted timesteps a statement about the fit
    # rather than about the run. On St Elias `k` is unfitted at 71% of timesteps and almost all of
    # them are winter, where the substitution changes nothing at all. This mask is the honest
    # denominator, and it is available only here, once the band's own temperature exists.
    warm = [T > _DECOUPLING_REFERENCE_TEMPERATURE for T in layers.temperature_air]
    k_source = downscaling_source_counts(interval.decoupling_factor_source)
    k_source_warm = downscaling_source_counts(interval.decoupling_factor_source, warm)
    lapse_source = downscaling_source_counts(ivf.lapse_rate_source)

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
        # `decoupling_factor_at_elevation`.
        "glacier_decoupling_factor_mean" => Statistics.mean(interval.decoupling_factor),
        # How many of the *fits* at this elevation were usable, and how many of those were evaluated
        # at a held-down elevation because the tile's ambient warm excess runs out below it. A band
        # where the second approaches the first carries the ceiling value rather than a fit at its own
        # elevation. Both describe the fits, whatever basis was applied — distinct from the
        # `n_held`/`n_fitted` *source* counts below, which describe the applied series.
        "glacier_decoupling_factor_n_fit_held" => interval.n_fit_held,
        "glacier_decoupling_factor_n_fit_in_domain" => interval.n_fit_in_domain,
        "n_timesteps_above_freezing" => count(warm),
        "adjustment_order" => "lapse_then_decouple",
        "temperature_air_mean" => Statistics.mean(layers.temperature_air),
        "precipitation_mean" => Statistics.mean(layers.precipitation),
        "wind_speed_mean" => Statistics.mean(layers.wind_speed)))

    # The provenance of every applied value, by source, for `k` at this band and for the tile's lapse
    # rate. `*_above_freezing` is the same count over the timesteps where `k` mattered.
    for name in DOWNSCALING_SOURCES
        meta["glacier_decoupling_factor_n_$name"] = k_source[name]
        meta["glacier_decoupling_factor_n_$(name)_above_freezing"] = k_source_warm[name]
        meta["temperature_lapse_rate_n_$name"] = lapse_source[name]
    end

    # Cell-specific keys would be a lie on a regional average.
    for k in ("latitude", "longitude", "chunk_strategy", "delta_elevation", "decoupling_factor")
        delete!(meta, k)
    end

    return (; interval.lo, interval.hi, interval.center, area = weight, n_cells,
            decoupling_factor = interval.decoupling_factor,
            decoupling_factor_source = interval.decoupling_factor_source,
            forcing = DimStack(layers; metadata = meta))
end
