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

Per grid cell, per interval, forcing is brought to the ice-free state, lapsed to the interval center
with the ambient lapse rate, and only then decoupled with the interval's `k` — see
[`elevation_interval_forcing`](@ref). Cells sharing an interval are then averaged weighted by each
cell's glacier area in it.

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

**Screen the two series together, not separately.** `derive_lapse_rate` consumes the decoupling fit to
bring the cells to the ice-free state, so a `k` outside `(0, 1]` corrupts that timestep's slope as well:
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
- `donor_max_distance_km = $_DONOR_MAX_DISTANCE_KM_DEFAULT`: how far a cell with ice but no usable
  forcing may borrow a neighbour's, in the returned `elevation_interval_forcing` only — the fits
  themselves screen such cells out, since a borrowed series carries no information about the
  cross-cell gradient they measure. `0` disables it. See [`elevation_interval_forcing`](@ref).
- `min_cells = $(_MIN_CELLS_DEFAULT)`: fewest usable grid cells the region needs before either fit is
  attempted; below it every timestep reports `NaN`. Note this is effectively a *per-region* gate,
  not a per-timestep one: forcing completeness is screened per cell, so the contributing cell count
  is the same at every timestep. `k` needs four cells regardless, since it has four parameters. See
  [`derive_decoupling_factor`](@ref) for what the number was chosen against.
- `downscaling_basis = :fitted`, `lapse_rate_stderr_maximum`, `elevation_spread_minimum`,
  `lapse_rate_window`,
  `lapse_rate_prior`, `decoupling_factor_prior`, `glacier_correction`: how the fits are turned into the applied parameters
  `elevation_interval_forcing` uses, and only there — the returned fits are untouched by these. All
  forwarded to [`resolve_downscaling`](@ref); see it for what each one accepts or rejects. `:fitted`
  is the default here because this method derives and applies over one window, so every timestep has
  its own fit to prefer; a run over a window wider than the fits needs `:climatology`.
- `forcing_loader = climate_forcing`: the loader, called as
  `forcing_loader(climate_model, lat, lon; time_range, token, cache_path)`. Injectable so the
  derivation can be exercised without a CDS token.

Grid cells whose forcing is unavailable are skipped, not fatal: ERA5-Land is land-only, so a
coastal or island cell can land on water and come back all-`NaN` ([`forcing_is_complete`](@ref)).
This mirrors how a sweep treats [`ForcingUnavailable`](@ref). A loader that *throws* — a network or
cache failure — is not a property of the cell, and the exception propagates.
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
                                   bare_ice_albedo = nothing,
                                   donor_max_distance_km::Real = _DONOR_MAX_DISTANCE_KM_DEFAULT,
                                   glacier_fraction_form::Symbol = :fraction,
                                   glacier_correction::Symbol = DEFAULT_GLACIER_CORRECTION,
                                   forcing_loader = climate_forcing)
    glacier_fraction_form in GLACIER_FRACTION_FORMS || throw(ArgumentError(
        "glacier_fraction_form must be one of $(GLACIER_FRACTION_FORMS), got :$glacier_fraction_form"))
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
    acc = _accumulate_cross_cell(selected, load; glacier_fraction_form)
    isempty(acc.used) && throw(RegionForcingUnavailable(nrow(selected)))

    grid_cells = @view selected[acc.used, :]

    decoupling = derive_decoupling_factor(acc; min_cells)
    # The lapse fit needs the decoupling fit per timestep to bring every cell to the ice-free state
    # before taking the slope. It reads the raw series and treats a `NaN` timestep as `k = 1` locally — the
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
    fit = (; time = acc.time, decoupling, lapse_rate, bare_ice_albedo, glacier_fraction_form)
    applied = resolve_downscaling(fit, hypsometry_intervals(grid_cells), acc.time;
                                  basis = downscaling_basis, min_cells,
                                  spread_minimum = elevation_spread_minimum,
                                  stderr_maximum = lapse_rate_stderr_maximum,
                                  lapse_rate_window, lapse_rate_prior, decoupling_factor_prior,
                                  glacier_correction)
    # Donors are the region's own cells: a derivation is handed the neighbourhood it should draw on, so
    # there is no wider set to reach into the way a tiled sweep has its buffer.
    interval_forcing = _ElevationIntervalForcing(grid_cells, load, applied,
                                                 elevation_interval_batch, grid_cells,
                                                 Float64(donor_max_distance_km))

    provenance = Dict{String,Any}(
        "climate_model" => string(climate_model),
        "time_range" => [first(acc.time), last(acc.time)],
        "n_grid_cells_in_region" => nrow(selected),
        "n_grid_cells_used" => length(acc.used),
        "n_elevation_intervals" => length(interval_forcing.intervals),
        "area_minimum" => Float64(area_minimum),
        "mean_elevation" => acc.mean_elevation,
        "adjustment_order" => "ambient_lapse_decouple",
        "glacier_fraction_form" => string(glacier_fraction_form),
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
            glacier_fraction_form,
            cell_elevations = acc.elevations, cell_areas = acc.areas,
            elevation_interval_forcing = interval_forcing, bare_ice_albedo, provenance)
end

"""
    GLACIER_FRACTION_FORMS

How a cell's ERA5-Land glacier-mask fraction `glm` enters the decoupling model
`E = α + β·z + g·(γ + δ·z)` (`derive_decoupling_factor`), its lapse-rate correction and the donor
conversion: `:fraction` (`g = glm`) or `:indicator` (`g = 1` for `glm ≥`
[`GLACIER_INDICATOR_THRESHOLD`](@ref), else 0).
"""
const GLACIER_FRACTION_FORMS = (:fraction, :indicator)

"""
    GLACIER_INDICATOR_THRESHOLD

The `glm` at and above which `:indicator` counts a cell as glaciated. ERA5-Land assigns 10 m of snow
water equivalent to grid boxes more than half covered by ice (Muñoz-Sabater et al. 2021, p. 4351), so
such a cell is damped as a whole.
"""
const GLACIER_INDICATOR_THRESHOLD = 0.5

"""
    decoupling_glacier_fraction(glm, form) -> Float64

The glacier fraction `g` the decoupling model uses for a cell with glacier-mask fraction `glm`, under
the [`GLACIER_FRACTION_FORMS`](@ref) `form`.
"""
function decoupling_glacier_fraction(glm::Real, form::Symbol)
    form === :fraction && return Float64(glm)
    form === :indicator && return glm >= GLACIER_INDICATOR_THRESHOLD ? 1.0 : 0.0
    throw(ArgumentError("glacier fraction form must be one of $(GLACIER_FRACTION_FORMS), got :$form"))
end

# Per-timestep cross-cell sums for both fits, plus the per-cell metadata the interval pass needs.
#
# The regressions are of the warm excess `E = max(T - T_ref, 0)` and of `T` itself on elevation `z`
# and glacier fraction `glm`. Accumulating the normal-equation sums as cells stream by keeps this
# O(n_time) rather than O(n_cells * n_time) — a region of 200 cells over 76 hourly years would
# otherwise be ~100 GB of retained temperature.
function _accumulate_cross_cell(selected, load; glacier_fraction_form::Symbol = :fraction)
    n = nothing
    time = nothing
    used = Int[]
    elevations = Float64[]
    areas = Float64[]

    # Sums over cells, per timestep. `s1` counts finite cells; `z`/`g` are the regressors.
    S = nothing

    for (i, row) in enumerate(eachrow(selected))
        # A load that throws is not skipped: a CDS timeout or a broken cache would otherwise drop the
        # cell from the fits and the band areas without being counted anywhere. A cell over water
        # loads successfully as all-`NaN`, and that is what `forcing_is_complete` screens out.
        fd = load(row)
        forcing_is_complete(fd) || continue

        meta = DimensionalData.metadata(fd)
        z = Float64(meta["elevation"])
        T = fd[:temperature_air]

        if S === nothing
            time = collect(dims(fd, Ti))
            n = length(time)
            # Upstream's layout for the two fits, plus the moments `derive_lapse_rate_uncertainty`
            # needs to square the ice-free temperature: `TT` over every cell, `gT_warm` and `wT_warm`
            # (Σglm·T and Σglm·z·T) over the cells above melting, where the correction acts.
            #
            # `TT` accumulates the square of the temperature **about the melting point**. The residual
            # sum of squares is translation-invariant, so this changes no result, but absolute
            # temperatures are ~270 K: `ΣT²` runs to ~1e5 per cell against an O(1) variance, and
            # forming `ΣT² - (ΣT)²/m` from unshifted sums cancels eleven digits.
            S = merge(_downscaling_sums(n), (; TT = zeros(n), gT_warm = zeros(n), wT_warm = zeros(n)))
        else
            length(T) == n || throw(ArgumentError(
                "grid cell $i has $(length(T)) forcing steps but the region's first cell has " *
                "$n; every cell must share one time axis"))
        end

        glm = decoupling_glacier_fraction(_row_glm(row), glacier_fraction_form)
        _accumulate_downscaling_cell!(S, T, z, glm)
        _accumulate_lapse_moments!(S, T, z, glm)

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
                               elevation_interval_batch = 0, forcing_loader = climate_forcing,
                               donor_cells = nothing,
                               donor_max_distance_km = $_DONOR_MAX_DISTANCE_KM_DEFAULT)

Lazy iterator over the elevation bands of `applied`, yielding glacier-area weighted forcing for each.

Each element is
`(; lo, hi, center, area, n_cells, n_cells_substituted, area_substituted, area_unrecovered,
max_donor_distance_km, decoupling_factor, decoupling_factor_source, forcing)`. Bands ascend in
elevation, and `area` sums over the iterator to the glacier area whose forcing was accounted for,
whether from a cell's own forcing or a donor's. `area_unrecovered` is the remainder — ice this band
holds that no cell could supply forcing for.

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

ERA5-Land is defined on land only, so a cell the glacier table credits with ice can still return an
all-`NaN` stack — `climate_forcing` succeeds and the geopotential elevation resolves, but no variable
carries data. Such a cell borrows forcing from the nearest cell of `donor_cells` that has usable
forcing, within `donor_max_distance_km`; `donor_cells` defaults to `grid_cells`, and a tiled caller
should pass its buffered neighbourhood so a tile whose *every* core cell is masked still runs. The
substitution takes the donor's reference elevation and the donor's `glm`, because both describe the
borrowed series: `glm` is how much the reanalysis already damped *that* forcing, and the reference
elevation is the surface the lapse to the band centre starts from. Only the ice area is the masked
cell's own. `donor_max_distance_km = 0` disables the substitution, dropping a masked cell's area as
`area_unrecovered` instead.

The alternative to borrowing is to drop the masked cell, which silently removes real ice from the
band area and from every total derived from it. Over the global 2° tiling that comes to 2.3% of the
table's glacier area, concentrated on the Antarctic Peninsula, north Greenland and the Russian High
Arctic — a regional bias rather than noise, which is why the substitution is the default and why
whatever it cannot reach is counted rather than skipped.

Per grid cell, per band, forcing is adjusted in three steps (`_cell_forcing_at_interval`):

1. **to ambient** — the glacier correction. The reanalysis already runs a fraction `glm` of the
   cell as ice, and the decoupling fit models its warm excess as
   `α + β·z + glm·(γ + δ·z) = [1 - glm·(1 - k(z))]·E_ambient(z)`. The correction removes that
   damping at the cell's own reanalysis elevation (`GEMB_ClimateForcing._ambient_forcing`) in the
   form the resolution names (`glacier_correction` of [`resolve_downscaling`](@ref)): by default it
   subtracts the glacier term, `T - glm·(γ + δ·z)`, carrying the cell's residual from the fit
   unscaled; the multiplicative form divides the observed excess by `1 - glm·(1 - k(z))`, with the
   donor's `k` kept inside `$(_DONOR_DECOUPLING_FACTOR_LIMITS)`, so the excess is multiplied by at
   most `1/$(_DONOR_DECOUPLING_FACTOR_LIMITS[1])`. A step at or below melting is left as it is;
2. **lapse** — to the band center, with the applied ambient lapse rate;
3. **decouple** — with the band's own `k` at the band center, since the band is glacier.

A region whose forcing follows the mixing model the fits measure therefore gives the band temperature
`T_ref + k(z_band)·E_ambient(z_band)` from every cell, whatever its `glm`. The order matters: the
decoupling damps only the excess above a fixed melting point, and lapsing changes which timesteps are
above it. Cells sharing a band are then averaged weighted by each cell's glacier area in it — every
layer by plain area weight, except two that follow the shortwave. `solar_zenith_angle` is the angle
whose cosine is the shortwave-weighted mean cosine (90° at a step with no shortwave).
`cloud_optical_thickness` is the `τ` of the band's mean cloud transmissivity, the band's shortwave
over its clear-sky shortwave `Σ w·SW/t(τ)` with `t(τ) = 1/(1 + 0.13τ)`, capped at
`CLOUD_OPTICAL_THICKNESS_MAX`; it equals the shortwave-weighted mean `τ`, a band whose cells all
report one `τ` takes that `τ` exactly, and a step with no shortwave in any cell takes 0.

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
                                    forcing_loader = climate_forcing,
                                    donor_cells = nothing,
                                    donor_max_distance_km::Real = _DONOR_MAX_DISTANCE_KM_DEFAULT)
    load(row) = forcing_loader(climate_model, row.latitude, row.longitude;
                               time_range, token, cache_path)
    return _ElevationIntervalForcing(grid_cells, load, applied, elevation_interval_batch,
                                     donor_cells === nothing ? grid_cells : donor_cells,
                                     Float64(donor_max_distance_km))
end

struct _ElevationIntervalForcing{C,L,B,D,K}
    grid_cells::C
    load::L
    time::Vector{DateTime}
    lapse_rate::Vector{Float64}
    lapse_rate_source::Vector{Int8}
    elevation_interval_batch::Int
    intervals::B
    donor_cells::D
    donor_max_distance_km::Float64
    # `k` at a donor's reanalysis elevation, from `AppliedDownscaling.donor_decoupling_factor_at`,
    # and the series already evaluated, keyed by elevation. Cells share elevations only by
    # coincidence, but each cell is met once per batching pass, so the memo saves the re-evaluation
    # on every pass after the first.
    donor_decoupling_factor_at::K
    donor_decoupling_factor::Dict{Float64,Any}
    # Forcing for donors that have been used, and the coordinates already found unusable. Masked cells
    # cluster, so the second one in a tile usually resolves to a donor already here and costs no I/O;
    # the unusable set stops a rejected candidate from being loaded again by the next masked cell.
    # Keyed by coordinate rather than by row, so the two batching passes over the cells share it.
    donor_forcing::Dict{Tuple{Float64,Float64},Any}
    donor_unusable::Set{Tuple{Float64,Float64}}
    # How a cell's `glm` enters the donor conversion: the form the fits were derived under
    # (`decoupling_glacier_fraction`).
    glacier_fraction_form::Symbol
    # The factor on each band's wind speed (`downscaling_wind_scaling`), applied to the band average.
    wind_scaling::Float64
    # The slope-dependent wind correction (`downscaling_wind_correction` = "slope"): each interval's
    # 2 km surface slope keyed by its edges, and the correction's coefficients; `nothing` when the
    # band wind is corrected by `wind_scaling` alone.
    wind_slope::Union{Nothing,@NamedTuple{band_slope::Dict{Tuple{Int,Int},Float64},
                                          coefficients::@NamedTuple{m0::Float64, m1::Float64,
                                              theta_floor::Float64, u_min_a::Float64,
                                              u_min_b::Float64}}}
end

# The `wind_slope` field of `_ElevationIntervalForcing` from a resolution's settings.
function _wind_slope(applied::AppliedDownscaling)
    get(applied.settings, "downscaling_wind_correction", "scaling") == "slope" || return nothing
    c = applied.settings["downscaling_wind_correction_coefficients"]
    θ = applied.settings["downscaling_wind_correction_band_slope"]
    length(θ) == length(applied.bands) || throw(DimensionMismatch(
        "$(length(θ)) band slopes recorded for $(length(applied.bands)) bands"))
    return (; band_slope = Dict((b.lo, b.hi) => Float64(t) for (b, t) in zip(applied.bands, θ)),
            coefficients = (; m0 = Float64(c[1]), m1 = Float64(c[2]), theta_floor = Float64(c[3]),
                            u_min_a = Float64(c[4]), u_min_b = Float64(c[5])))
end

function _ElevationIntervalForcing(grid_cells, load, applied::AppliedDownscaling,
                                   elevation_interval_batch::Int, donor_cells,
                                   donor_max_distance_km::Float64)
    elevation_interval_batch >= 0 || throw(ArgumentError(
        "elevation_interval_batch must be >= 0, got $elevation_interval_batch"))
    donor_max_distance_km >= 0 || throw(ArgumentError(
        "donor_max_distance_km must be >= 0 (0 disables donor substitution), got " *
        string(donor_max_distance_km)))
    return _ElevationIntervalForcing(grid_cells, load, applied.time, applied.lapse_rate,
                                     applied.lapse_rate_source, elevation_interval_batch,
                                     applied.bands, donor_cells, donor_max_distance_km,
                                     applied.donor_decoupling_factor_at,
                                     Dict{Float64,Any}(),
                                     Dict{Tuple{Float64,Float64},Any}(),
                                     Set{Tuple{Float64,Float64}}(),
                                     Symbol(get(applied.settings, "downscaling_glacier_fraction_form",
                                                "fraction")),
                                     Float64(get(applied.settings, "downscaling_wind_scaling", 1.0)),
                                     _wind_slope(applied))
end

# The applied glacier correction per timestep at the reanalysis elevation `z` of a cell whose glacier
# fraction is `glm`, for bringing its forcing to the ice-free state: a `k` series or an
# `AdditiveGlacierCorrection`, as `donor_decoupling_factor_at` returns it. A cell with no glacier
# fraction is already ambient (`k = 1` leaves it bit for bit), which is the one case a resolution
# without `donor_decoupling_factor_at` can force.
function _donor_decoupling_factor(ivf::_ElevationIntervalForcing, z::Float64, glm::Float64)
    glm == 0 && return ones(length(ivf.time))
    ivf.donor_decoupling_factor_at === nothing && throw(ArgumentError(
        "this AppliedDownscaling carries no `donor_decoupling_factor_at`, so the forcing of a cell " *
        "with glacier fraction $glm cannot be brought to the ice-free state; build it with " *
        "`resolve_downscaling`"))
    return get!(() -> ivf.donor_decoupling_factor_at(z), ivf.donor_decoupling_factor, z)
end

# Forcing to use for `row`, which has ice but no usable forcing of its own: the nearest cell of
# `ivf.donor_cells` inside the distance cap whose forcing is complete. Returns
# `(; forcing, elevation, glm, distance_km)`, or `nothing` when the cap holds no usable cell.
#
# Candidates are probed in ascending distance so the first usable one is the nearest, and the scan
# stops there — the cost is one load per rejected candidate, which the memo pays only once per
# coordinate for the whole tile.
function _nearest_usable_donor(ivf::_ElevationIntervalForcing, row)
    ivf.donor_max_distance_km > 0 || return nothing
    lat, lon = Float64(row.latitude), Float64(row.longitude)

    rows = eachrow(ivf.donor_cells)
    candidates = Tuple{Float64,Int}[]
    for i in eachindex(rows)
        donor = rows[i]
        d = _haversine_km(lat, lon, donor.latitude, donor.longitude)
        d <= ivf.donor_max_distance_km && push!(candidates, (d, i))
    end
    sort!(candidates; by = first)

    for (d, i) in candidates
        donor = rows[i]
        key = (Float64(donor.latitude), Float64(donor.longitude))
        key in ivf.donor_unusable && continue
        fd = get(ivf.donor_forcing, key, nothing)
        if fd === nothing
            fd = ivf.load(donor)
            if !forcing_is_complete(fd)
                push!(ivf.donor_unusable, key)
                continue
            end
            ivf.donor_forcing[key] = fd
        end
        return (; forcing = fd,
                elevation = Float64(DimensionalData.metadata(fd)["elevation"]),
                glm = _row_glm(donor),
                distance_km = d)
    end
    return nothing
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
    # Running weighted sums per interval, one array per forcing layer plus the range of the cells'
    # cloud optical thickness, and the weight totals.
    sums = [merge(Dict(v => zeros(n) for v in _FORCING_VARIABLES),
                  Dict(_TAU_MIN => fill(Inf, n), _TAU_MAX => fill(-Inf, n)))
            for _ in intervals]
    weights = zeros(length(intervals))
    counts = zeros(Int, length(intervals))
    # Intervals no donor cell can legally be adjusted to, and why. Extrapolating a cell's forcing far
    # enough above its own surface eventually takes it outside the range `climate_forcing` validates —
    # in the Khumbu the top bands sit ~2.6 km above every contributing cell, and the re-derived downward
    # longwave there falls below the 50 W/m² floor. That is the validator working, so the interval is
    # refused rather than fudged; refusing the whole interval rather than the failing cell's share keeps
    # the average from silently re-weighting onto whichever donors happened to be close enough.
    # `Vector{Bool}`, not a `BitVector`: threads set flags of neighbouring intervals concurrently, and
    # a `BitVector` packs 64 of them into one word, so concurrent writes would lose flags.
    unreachable = fill(false, length(intervals))
    reasons = Vector{String}(undef, length(intervals))
    template = nothing
    # Highest reanalysis surface contributing to each interval. Per interval, not per batch, so the
    # extrapolation this reports does not depend on how the batching happens to group them.
    z_max = fill(-Inf, length(intervals))
    # What the donor substitution did, per interval: how many of its cells borrowed forcing, how much
    # ice that covered, the worst distance borrowed over, and the ice left with no donor at all.
    n_substituted = zeros(Int, length(intervals))
    substituted = zeros(length(intervals))
    donor_distance = zeros(length(intervals))
    unrecovered = zeros(length(intervals))

    for row in eachrow(ivf.grid_cells)
        # Which of this batch's intervals does this cell hold ice in, and how much?
        areas = _interval_areas(row, intervals)
        all(iszero, areas) && continue

        # A load that throws propagates, for the reason given in `_accumulate_cross_cell`.
        fd = ivf.load(row)

        # How much of this cell the reanalysis already runs as ice, which sets how far its forcing is
        # from the ice-free state the lapse starts from.
        glm = _row_glm(row)
        distance = 0.0
        substituting = false

        if !forcing_is_complete(fd)
            # This cell is outside ERA5-Land's mask: it carries ice but no forcing of its own. Its own
            # coordinate is a candidate donor whenever `donor_cells` includes `grid_cells`, so record
            # it as unusable first — otherwise the scan reloads the stack just read.
            push!(ivf.donor_unusable, (Float64(row.latitude), Float64(row.longitude)))
            donor = _nearest_usable_donor(ivf, row)
            if donor === nothing
                for i in eachindex(intervals)
                    unrecovered[i] += areas[i]
                end
                continue
            end
            # The donor supplies the series, its own reference elevation and its own `glm`; the masked
            # cell supplies only `areas`.
            fd = donor.forcing
            glm = donor.glm
            distance = donor.distance_km
            substituting = true
        end

        lookup(fd, Ti) == ivf.time || throw(DimensionMismatch(
            "forcing for the cell at ($(row.latitude), $(row.longitude)) covers " *
            "$(first(lookup(fd, Ti))) .. $(last(lookup(fd, Ti))) ($(length(lookup(fd, Ti))) steps), but " *
            "the applied downscaling is resolved on $(first(ivf.time)) .. $(last(ivf.time)) " *
            "($(length(ivf.time)) steps); resolve it on the forcing's time axis"))
        missing_layers = [v for v in _FORCING_VARIABLES if !haskey(fd, v)]
        isempty(missing_layers) || throw(ArgumentError(
            "forcing for the cell at ($(row.latitude), $(row.longitude)) lacks the layer(s) " *
            "$(join(missing_layers, ", ")); every band averages all of $(_FORCING_VARIABLES)"))

        template === nothing && (template = fd)
        # Whether substituted or not, this is the reference surface of the series in `fd`, which is what
        # the lapse to the interval centre must start from.
        z_cell = Float64(DimensionalData.metadata(fd)["elevation"])

        # The series brought to the ice-free state once per cell, with `k` at the cell's own
        # reanalysis elevation and the `glm` of the cell the series belongs to, so every band lapses
        # from the same ambient temperature.
        g = decoupling_glacier_fraction(glm, ivf.glacier_fraction_form)
        ambient = _ambient_forcing(fd, g, _donor_decoupling_factor(ivf, z_cell, g))

        # A function barrier: with substitution, `fd` may come from the donor memo, whose element type
        # is not concrete. Resolving it at this one call keeps the record-length inner loops specialized
        # on the stack's real type rather than dispatching per timestep.
        _accumulate_cell_intervals!(sums, weights, counts, z_max, unreachable, reasons, n_substituted,
                                    substituted, donor_distance, intervals, areas, ambient, z_cell,
                                    ivf, n, substituting, distance)
    end

    template === nothing && throw(ArgumentError(
        "no grid cell in the region yielded usable forcing during elevation interval aggregation" *
        (ivf.donor_max_distance_km > 0 ?
         ", and no donor within $(ivf.donor_max_distance_km) km had any either" :
         " (donor substitution is disabled)")))

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

    # What the substitution covered. Borrowed forcing is a spatial approximation on top of the lapse
    # extrapolation, so how much of a band rests on it belongs in the log next to how far the band was
    # extrapolated, and the worst distance is the number that says whether the donor is a neighbour or
    # a different climate.
    subs = findall(>(0), n_substituted)
    isempty(subs) || @info("Elevation intervals partly forced by a donor cell: these cells carry ice " *
                           "but fall outside the reanalysis land mask, so forcing came from the " *
                           "nearest cell that has it",
                           n_elevation_intervals = length(subs),
                           n_cells = sum(n_substituted),
                           substituted_area = round(sum(substituted[subs]), digits = 2),
                           max_donor_distance_km = round(maximum(donor_distance[subs]), digits = 1))

    # Ice no cell could supply forcing for, either its own or a donor's within the cap. Reported with
    # its area, because the area is the consequence: it is real ice that this tile's totals will not
    # account for. Raising `donor_max_distance_km` is what recovers it, so the cap is named.
    lost = findall(>(0), unrecovered)
    isempty(lost) || @warn("Elevation intervals hold ice with no usable forcing and no donor within " *
                           "the distance cap; that area is absent from this interval's average",
                           n_elevation_intervals = length(lost),
                           unrecovered_area = round(sum(unrecovered[lost]), digits = 2),
                           centers = [intervals[i].center for i in lost],
                           donor_max_distance_km = ivf.donor_max_distance_km)

    # An interval the hypsometry populates but whose cells all turned out to have no usable forcing
    # accumulates no area, and there is nothing to average. It is dropped rather than emitted, and
    # named rather than dropped silently. Common on the Antarctic and Aleutian coasts, where ERA5-Land
    # is land-only and a whole elevation band's cells can fall on water.
    dropped = [intervals[i].center for i in eachindex(intervals)
               if weights[i] <= 0 && !unreachable[i]]
    isempty(dropped) || @warn("Elevation intervals dropped: their cells carried no usable " *
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
            _interval_stack(intervals[i], sums[i], weights[i], counts[i], template, ivf, z_max[i],
                            n_substituted[i], substituted[i], donor_distance[i], unrecovered[i]) :
            nothing
            for i in eachindex(intervals)]
end

# One cell's contribution to every interval of the current batch. Split out of `_accumulate_intervals`
# as a function barrier: `ambient` may derive from the donor memo, whose element type is not concrete,
# and this is where the record-length loops live.
#
# `ambient` is the cell's forcing already at the ice-free state, so each interval only lapses it and
# decouples with the interval's own `k`; `_cell_forcing_at_interval` is handed a glacier fraction of
# zero and a donor `k` of 1, which make its own ambient step the identity.
#
# Threaded over intervals, not over cells. This inner loop is the whole cost of the pass —
# `n_cells * n_intervals` lapse-and-decouple passes over the record, which for a 335-cell tile of 60
# bands is 20,000 of them — and each iteration writes only its own interval's slot, so the accumulation
# order within an interval stays the cell order and the result is bit-for-bit what the serial loop
# gives. Threading over cells instead would have several threads adding into the same interval, where
# floating-point addition is not associative.
#
# Every layer is a glacier-area weighted mean except two. The `solar_zenith_angle` slot accumulates
# `w·SW·cos(θ)`, which `_interval_stack` divides by the `w·SW` the shortwave slot already holds. The
# `cloud_optical_thickness` slot accumulates the clear-sky shortwave `w·SW/t(τ)`, from which
# `_band_layer` forms the band transmissivity, and the `_TAU_MIN`/`_TAU_MAX` slots the range of the
# cells' `τ`.
function _accumulate_cell_intervals!(sums, weights, counts, z_max, unreachable, reasons,
                                     n_substituted, substituted, donor_distance,
                                     intervals, areas, ambient, z_cell, ivf, n,
                                     substituting::Bool, distance::Float64)
    already_ambient = ones(n)
    # The zenith angle and the cloud optical thickness are the cell's own at every interval (lapsing
    # and decoupling leave them unchanged), so the cosine and the clear-sky ratio `1/t(τ)` are taken
    # once per cell.
    cos_zenith = cosd.(parent(ambient[:solar_zenith_angle]))
    τ = parent(ambient[:cloud_optical_thickness])
    clear_sky_ratio = inv.(cloud_transmissivity.(τ))
    Threads.@threads for i in eachindex(intervals)
        areas[i] > 0 && !unreachable[i] || continue
        interval = intervals[i]
        adjusted = try
            _cell_forcing_at_interval(ambient, interval.center - z_cell, interval.decoupling_factor,
                                      ivf.lapse_rate, 0.0, already_ambient)
        catch e
            e isa InterruptException && rethrow()
            is_caller_error(e) && rethrow()
            # Caught inside the loop body: an exception escaping a `Threads.@threads` region becomes a
            # `TaskFailedException` that names the thread rather than the interval, and would take down
            # a whole tile over its highest few bands.
            unreachable[i] = true
            reasons[i] = sprint(showerror, e)
            continue
        end
        w = areas[i]
        sw = adjusted[:shortwave_downward]
        for v in _FORCING_VARIABLES
            # `adjusted[v]` is looked up by a runtime name, so its type is not known here; the
            # record-length loops run behind a function barrier, where it is.
            if v === :solar_zenith_angle
                _add_weighted_product!(sums[i][v], w, sw, cos_zenith)
            elseif v === :cloud_optical_thickness
                _add_weighted_product!(sums[i][v], w, sw, clear_sky_ratio)
                _extend_range!(sums[i][_TAU_MIN], sums[i][_TAU_MAX], τ)
            else
                _add_weighted!(sums[i][v], w, adjusted[v])
            end
        end
        weights[i] += w
        counts[i] += 1
        z_max[i] = max(z_max[i], z_cell)
        # Recorded here rather than after the call, so it counts exactly the intervals this cell
        # actually contributed to — an interval refused above must not be credited with a substitution.
        if substituting
            n_substituted[i] += 1
            substituted[i] += w
            donor_distance[i] = max(donor_distance[i], distance)
        end
    end
    return nothing
end

# `s .+= w .* a`, and `s .+= w .* sw .* c`, written as loops so the accumulation order is the
# same as a scalar sum's.
function _add_weighted!(s, w, a)
    for t in eachindex(s, a)
        s[t] += w * a[t]
    end
    return s
end

function _add_weighted_product!(s, w, sw, c)
    for t in eachindex(s, sw, c)
        s[t] += w * sw[t] * c[t]
    end
    return s
end

# Widen the per-timestep range `[lo, hi]` to include `x`.
function _extend_range!(lo, hi, x)
    for t in eachindex(lo, hi, x)
        lo[t] = min(lo[t], x[t])
        hi[t] = max(hi[t], x[t])
    end
    return lo, hi
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

The regression described is the same one `derive_lapse_rate` performs — on the ice-free temperature
`T' = T - glm·(γ + δ·z)` for a cell above melting and `T' = T` otherwise, under the same
`_corrects_to_ambient` screen, so the reported error belongs to the reported slope. Squaring `T'`
needs `Σglm·T` and `Σglm·z·T` over the cells above melting beside the regressor moments upstream
already keeps. `NaN`
wherever the slope itself is `NaN`, and wherever there are too few cells to have any residual degrees
of freedom.
"""
function derive_lapse_rate_uncertainty(acc, decoupling; min_cells::Int = _MIN_CELLS_DEFAULT)
    n = acc.n
    ti = Ti(acc.time)

    stderr = fill(NaN, n)
    r2 = fill(NaN, n)
    residual_sd = fill(NaN, n)

    S = acc.sums
    # A tile accumulated without the temperature second moments carries no `TT`, and asking it for a
    # lapse-rate uncertainty must report "not measured" rather than throw on a missing field.
    if S !== nothing && :TT in propertynames(S)
        k = decoupling.decoupling_factor
        cγ, cδ = decoupling.coef_gamma, decoupling.coef_delta
        r = _DECOUPLING_REFERENCE_TEMPERATURE

        for t in 1:n
            # Three cells give a slope but no residual degrees of freedom; `m - 2` must be positive.
            S.s1[t] >= max(min_cells, 3) || continue
            m = Float64(S.s1[t])

            Szz = S.zz[t] - S.z[t] * S.z[t] / m
            Szz > 0 || continue

            # The same screen `derive_lapse_rate` applies, so this error belongs to that slope: an
            # unmeasured or out-of-domain `k` leaves the temperatures uncorrected.
            γ, δ = _corrects_to_ambient(k[t], cγ[t], cδ[t]) ? (cγ[t], cδ[t]) : (0.0, 0.0)

            # Sums over the warm cells of the correction `u = glm·(γ + δz)`, of `z·u`, `u²`, and of
            # `u·(T - T_ref)`. Cells at or below melting have `u = 0`.
            su  = γ * S.g_warm[t]  + δ * S.w_warm[t]
            szu = γ * S.zg_warm[t] + δ * S.zw_warm[t]
            suu = γ * γ * S.gg_warm[t] + 2 * γ * δ * S.gw_warm[t] + δ * δ * S.ww_warm[t]
            suT = γ * (S.gT_warm[t] - r * S.g_warm[t]) + δ * (S.wT_warm[t] - r * S.w_warm[t])

            # Temperature relative to the melting point, the frame `TT` was accumulated in. Slope,
            # residual and R² are translation-invariant, so the shift changes none of them.
            sT  = (S.T[t] - m * r) - su
            szT = (S.zT[t] - S.z[t] * r) - szu
            sTT = S.TT[t] - 2 * suT + suu

            SzT = szT - S.z[t] * sT / m
            STT = sTT - sT * sT / m
            STT > 0 || continue

            # Residual sum of squares of the ice-free temperatures about the fitted line. Clamped at
            # zero: with a near-perfect fit it is a difference of two nearly equal numbers and can
            # land a rounding step below.
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

# Add one cell's temperature moments for `derive_lapse_rate_uncertainty` to `S`: `(T - T_ref)²` over
# every cell, and `glm·T`, `glm·z·T` over the cells above melting — the same warm set upstream's
# `_accumulate_downscaling_cell!` restricts the decoupling sums to.
#
# Takes `S` as an argument, which makes it a function barrier: in `_accumulate_cross_cell` the sums
# start as `nothing`, so there `S` is a `Union` and every `S.field` in a loop body would be a dynamic
# property lookup per timestep.
function _accumulate_lapse_moments!(S, T, z::Real, glm::Real)
    z, glm = Float64(z), Float64(glm)
    gz = glm * z
    for t in eachindex(T, S.TT)
        Tt = T[t]
        Tr = Tt - _DECOUPLING_REFERENCE_TEMPERATURE
        S.TT[t] += Tr * Tr
        Tr > 0 || continue
        S.gT_warm[t] += glm * Tt
        S.wT_warm[t] += gz * Tt
    end
    return S
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
function _interval_stack(interval, sums, weight, n_cells, template, ivf, z_max,
                         n_substituted, substituted, donor_distance, unrecovered)
    weight > 0 || throw(ArgumentError(
        "elevation interval $(interval.lo)-$(interval.hi) m accumulated zero area"))
    time_dim = dims(template, Ti)
    layers = NamedTuple(v => DimArray(_band_layer(v, sums, weight), (time_dim,);
                                      metadata = DimensionalData.metadata(template[v]))
                        for v in _FORCING_VARIABLES)
    # The wind correction acts on the band's wind alone, after the average; skipped at 1 so the band
    # is then the uncorrected one bit for bit.
    ivf.wind_scaling == 1 || (layers.wind_speed .*= ivf.wind_scaling)
    # The slope correction likewise acts on the band average, with the band's own slope.
    wind_slope = ivf.wind_slope === nothing ? nothing :
                 ivf.wind_slope.band_slope[(interval.lo, interval.hi)]
    if wind_slope !== nothing
        c = ivf.wind_slope.coefficients
        layers.wind_speed .= slope_wind_correction.(layers.wind_speed, wind_slope; coefficients = c)
    end

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
        # How much of this band rests on borrowed forcing, and from how far. A cell outside the
        # reanalysis land mask keeps its ice here by taking the nearest usable cell's series, so these
        # say how much of the average is a spatial approximation rather than the band's own cells.
        "n_grid_cells_substituted" => n_substituted,
        "substituted_area" => substituted,
        "max_donor_distance_km" => donor_distance,
        # Ice in this band that no cell could supply forcing for, its own or a donor's. Absent from
        # `glacier_area` and from every total derived from it, so it is on record here rather than left
        # as the difference between two numbers a reader has to think to compare.
        "unrecovered_area" => unrecovered,
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
        "adjustment_order" => "ambient_lapse_decouple",
        "temperature_air_mean" => Statistics.mean(layers.temperature_air),
        # kg m-2 yr-1, as GEMB reads it: the mean per step times the steps in a 365-day year, the
        # convention `climate_forcing` and `precipitation_adjust` use.
        "precipitation_mean" => Statistics.mean(layers.precipitation) * _steps_per_year(time_dim),
        "wind_speed_mean" => Statistics.mean(layers.wind_speed),
        "wind_speed_scaling" => ivf.wind_scaling))
    if wind_slope !== nothing
        meta["wind_speed_correction"] = "slope"
        meta["surface_slope_2km"] = wind_slope
        meta["wind_speed_slope_multiplier"] = slope_wind_multiplier(wind_slope; coefficients = ivf.wind_slope.coefficients)
        meta["wind_speed_slope_floor"] = slope_wind_floor(wind_slope; coefficients = ivf.wind_slope.coefficients)
    end

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
            n_cells_substituted = n_substituted, area_substituted = substituted,
            area_unrecovered = unrecovered, max_donor_distance_km = donor_distance,
            decoupling_factor = interval.decoupling_factor,
            decoupling_factor_source = interval.decoupling_factor_source,
            # Carried on the band rather than looked up again downstream: `_runnable_bands` and
            # `_bands_after` both reshape this list, so a band's index here is not its index in
            # `applied.bands`, and re-matching by elevation edges is how the two fall out of step.
            albedo_ice = interval.albedo_ice,
            albedo_ice_source = interval.albedo_ice_source,
            forcing = DimStack(layers; metadata = meta))
end

# Forcing steps in a 365-day year, from the step of a `Ti` axis.
function _steps_per_year(ti)
    t = lookup(ti)
    length(t) >= 2 || throw(ArgumentError("a forcing axis of $(length(t)) step(s) has no step length"))
    return Dates.value(Millisecond(Day(365))) / Dates.value(Millisecond(t[2] - t[1]))
end

# Keys of the per-interval sums holding the per-timestep range of the contributing cells'
# `cloud_optical_thickness`; see `_band_cloud_optical_thickness`.
const _TAU_MIN = :_cloud_optical_thickness_min
const _TAU_MAX = :_cloud_optical_thickness_max

# One layer of a band's forcing from its accumulated sums.
function _band_layer(v::Symbol, sums, weight)
    v === :solar_zenith_angle && return _band_zenith_angle(sums[v], sums[:shortwave_downward])
    v === :cloud_optical_thickness &&
        return _band_cloud_optical_thickness(sums[v], sums[:shortwave_downward], sums[_TAU_MIN],
                                             sums[_TAU_MAX])
    return sums[v] ./ weight
end

# A band's cloud optical thickness from the accumulated clear-sky shortwave `Σ w·SW/t(τ)`, the
# shortwave `Σ w·SW`, and the range of the cells' `τ`: the `τ` of the band's mean transmissivity,
#
#     t̄ = Σ w·SW / Σ w·SW/t(τ),    τ_band = τ(t̄),
#
# with `t(τ) = 1/(1 + 0.13τ)` and its inverse `τ(t)` (capped at `CLOUD_OPTICAL_THICKNESS_MAX`) the
# ClimateForcing pair `cloud_transmissivity` / `cloud_optical_thickness` (Lacis & Hansen 1974, the
# relation the cells' `τ` were inverted with). `SW/t(τ)` is each cell's clear-sky shortwave, so `t̄`
# is the band's shortwave over the band's clear-sky shortwave: it weights each cell's transmissivity by
# the shortwave it would receive under clear sky, and gives the band the cloud transmission of the
# shortwave it actually receives. Algebraically this is the shortwave-weighted mean of the cells'
# `τ`. A plain area mean of `τ` would instead let a dark cell — `τ` at the cap with almost no
# shortwave — set the band's cloud term.
#
# Where every contributing cell reports the same `τ`, that `τ` is the band's exactly: a weighted mean
# of equal values, which the sums would reproduce only to rounding. This makes a one-cell band carry
# its cell's `τ` bit for bit. Otherwise, a step whose clear-sky shortwave sums to zero (no cell
# receives shortwave) has no band transmissivity and takes `τ = 0`, the value
# `shortwave_forcing_layers` gives a step without sun.
function _band_cloud_optical_thickness(sum_clear, sum_sw, τ_min, τ_max)
    τ = similar(sum_sw)
    for t in eachindex(τ, sum_clear, sum_sw, τ_min, τ_max)
        if τ_min[t] == τ_max[t]
            τ[t] = τ_min[t]
        elseif sum_clear[t] > 0
            tbar = sum_sw[t] / sum_clear[t]
            # A weighted mean of transmissivities in (0, 1]; anything further above 1 than rounding
            # is a bug.
            0 < tbar <= 1 + 1e-12 || throw(ErrorException(
                "band cloud transmissivity is $(tbar) at step $t, outside (0, 1]"))
            τ[t] = cloud_optical_thickness(min(tbar, 1.0))
        else
            τ[t] = 0.0
        end
    end
    return τ
end

# A band's solar zenith angle (degrees) from the accumulated `Σ w·SW·cos θ` and `Σ w·SW`: the angle
# whose cosine is the shortwave-weighted mean cosine, so the band's direct-beam geometry is that of the
# shortwave it receives. 90° (sun at the horizon) at a step with no shortwave, where no angle is
# defined and GEMB applies none.
function _band_zenith_angle(sum_sw_cos, sum_sw)
    θ = similar(sum_sw)
    for t in eachindex(θ, sum_sw_cos, sum_sw)
        if sum_sw[t] > 0
            c = sum_sw_cos[t] / sum_sw[t]
            # A weighted mean of cosines lies in [-1, 1]; anything further out than rounding is a bug.
            abs(c) <= 1 + 1e-12 || throw(ErrorException(
                "shortwave-weighted mean cos(solar_zenith_angle) is $c at step $t, outside [-1, 1]"))
            θ[t] = acosd(clamp(c, -1.0, 1.0))
        else
            θ[t] = 90.0
        end
    end
    return θ
end
