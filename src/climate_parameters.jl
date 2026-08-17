# Regional climate parameters for a global glacier run.
#
# Three things a global run needs that cannot be derived from a single climate grid cell, because
# each is a *relationship across* cells within a region:
#
#   1. the local on-glacier temperature decoupling factor `k`, from how the warm excess varies with
#      the ERA5-Land glacier mask fraction at matched elevation;
#   2. the local on-glacier temperature lapse rate, from how decoupled temperature varies with
#      elevation;
#   3. glacier-area weighted forcing for every hypsometry band in the region, which is the input to
#      the downstream GEMB parameter sweep that solves for the precipitation and temperature bias
#      correction factors against satellite laser altimetry.
#
# Terminology, used consistently below because the objects are different:
#   region    — the caller's arbitrary GeoInterface geometry. Any shape.
#   grid cell — one row of the glacier elevation-class table, i.e. one 0.1° ERA5-Land cell.
#   band      — one `hyps_<lo>_<hi>` elevation bin.

# Melting point (K). The decoupling correction damps only the excess above it, matching
# `climate_adjust_for_glacier`'s `reference_temperature` default.
const _DECOUPLING_REFERENCE_TEMPERATURE = 273.15

# Fallback lapse rate (K/km) for a timestep with too few grid cells to fit one, matching the
# default every other call site in this package uses.
const _DEFAULT_LAPSE_RATE = 6.5

# Smallest ambient warm excess (K) that can support a decoupling fit. `k` is a ratio of fitted
# excesses, so a timestep where the region is barely above melting divides two small numbers and
# returns noise — on real Alpine forcing, winter timesteps with a ~0.1 K excess fit wildly and then
# pin at whatever clamp bound they overshoot, which reads as a confident seasonal signal but is not
# one. 0.5 K is well above the fit residual while still admitting every melt-season timestep, which
# is where the decoupling actually matters.
const _MIN_AMBIENT_EXCESS = 0.5

# Physically plausible lapse-rate bounds (K/km), mirroring `_LAPSE_RATE_MIN`/`_LAPSE_RATE_MAX` in
# GEMB_ClimateForcing's `elevation_adjustment.jl`. A thin-sample timestep can fit an absurd slope,
# and `climate_adjust_for_elevation` would then throw on it downstream — clamp here instead.
const _LAPSE_RATE_LIMITS = (-30.0, 25.0)

# The seven forcing layers `climate_forcing` returns. Named here rather than derived from a stack so
# the band accumulator can allocate before it has seen one.
const _FORCING_VARIABLES = (:temperature_air, :pressure_air, :vapor_pressure, :wind_speed,
                            :precipitation, :shortwave_downward, :longwave_downward)

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

`area_minimum` screens cells by total glacier area (km²) with [`glacier_area_total`](@ref) before
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

    function keep(row)
        # Cheap screens first: area, then the region's bounding box, then the exact geometry.
        glacier_area_total(row) >= area_minimum || return false
        lon = wrap_lon(Float64(GeoInterface.x(row.geometry)))
        lat = Float64(GeoInterface.y(row.geometry))
        (bbox.X[1] <= lon <= bbox.X[2] && bbox.Y[1] <= lat <= bbox.Y[2]) || return false
        point = GeoInterface.Point(lon, lat)
        return any(g -> GO.contains(g, point), region)
    end

    return filter(keep, glacier_elevation_classes; view = true)
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
    derive_climate_parameters(climate_model, time_range, glacier_elevation_classes, roi_polygon;
                              token, cache_path, kwargs...)

Derive the regional climate parameters a global glacier run needs, and the glacier-area weighted
per-band forcing that the downstream bias-correction sweep runs on.

Returns `(; grid_cells, time, decoupling, lapse_rate, band_forcing, provenance)`:

- `grid_cells`: the region's grid cells actually used (a view of `glacier_elevation_classes`, with
  cells whose forcing was unavailable dropped — see below).
- `time`: the forcing time axis, shared by everything below.
- `decoupling`, `lapse_rate`: see [`derive_decoupling_factor`](@ref) and
  [`derive_lapse_rate`](@ref). Each carries `native` (per timestep, the forcing's own resolution),
  `monthly` (a 12-element climatology), `scalar` (one pooled value), and per-timestep fit
  diagnostics.
- `band_forcing`: a **lazy** iterator over the region's hypsometry bands, ascending in elevation.
  Each element is `(; lo, hi, center, area, n_cells, forcing)` where `forcing` is a `DimStack`
  shaped exactly like `climate_forcing` output — same seven layers, same `Ti` axis — with
  `metadata["elevation"]` set to the band center, so it satisfies the contract
  [`gemb_glacier_cell`](@ref) enforces and a `forcing_at_elevation(band.forcing, 0)` is an
  identity.
- `provenance`: what was derived from what — cell counts, time range, the region's extent.

Per grid cell, per band, forcing is adjusted in the order
[`forcing_at_elevation`](@ref) uses and `climate_adjust_for_glacier` requires: **lapse to the band
center first, then decouple**. `k` multiplies an ambient temperature already at the glacier's
elevation, so the reverse order is wrong. Cells sharing a band are then averaged weighted by each
cell's glacier area in that band.

# Keywords
- `token`, `cache_path`: passed straight through to `climate_forcing`. `cache_path` is the existing
  Zarr chunk cache; nothing else is cached, so a region is re-derived on every call (but its second
  and later passes read from that warm cache rather than the network).
- `area_minimum = 0.0`: minimum total glacier area (km²) for a cell to be included.
- `band_batch = 8`: how many bands `band_forcing` accumulates per pass over the region. Peak memory
  is `band_batch` band stacks; cost is `ceil(n_bands / band_batch)` passes over the warm cache. `0`
  means one pass holding every band at once.
- `min_cells = 3`: fewest cells a timestep needs for a fit. Below it the timestep falls back to the
  no-op values (`k = 1`, the identity; `lapse_rate = $(_DEFAULT_LAPSE_RATE)`).
- `decoupling_factor_bounds = (0.2, 1.0)`: clamp on the fitted factor. The default is the range
  Shaw et al. (2025) clamp their published estimates to, and lies within the `(0, 1]` domain
  `climate_adjust_for_glacier` requires.
- `forcing_loader = climate_forcing`: the loader, called as
  `forcing_loader(climate_model, lat, lon; time_range, token, cache_path)`. Injectable so the
  derivation can be exercised without a CDS token.

Grid cells whose forcing is unavailable are skipped, not fatal: ERA5-Land is land-only, so a
coastal or island cell can land on water and come back all-`NaN` ([`forcing_is_complete`](@ref)).
This mirrors how a sweep treats [`ForcingUnavailable`](@ref).
"""
function derive_climate_parameters(climate_model::Symbol, time_range, glacier_elevation_classes,
                                   roi_polygon;
                                   token,
                                   cache_path,
                                   area_minimum::Real = 0.0,
                                   band_batch::Int = 8,
                                   min_cells::Int = 3,
                                   decoupling_factor_bounds = (0.2, 1.0),
                                   forcing_loader = climate_forcing)
    band_batch >= 0 || throw(ArgumentError("band_batch must be >= 0, got $band_batch"))
    min_cells >= 2 ||
        throw(ArgumentError("min_cells must be >= 2 to fit a slope, got $min_cells"))

    selected = grid_cells_in_region(glacier_elevation_classes, roi_polygon; area_minimum)
    nrow(selected) == 0 && throw(ArgumentError(
        "no grid cells in the region with at least $area_minimum km² of glacier area"))

    load(row) = forcing_loader(climate_model, row.latitude, row.longitude;
                               time_range, token, cache_path)

    @info "Deriving climate parameters" grid_cells=nrow(selected) time_range

    # One pass to accumulate the cross-cell regression sums for both fits, and to learn which cells
    # actually have forcing. Both fits need only per-timestep sums over cells, so neither the cell
    # forcing stacks nor their temperature vectors are retained.
    acc = _accumulate_cross_cell(selected, load)
    isempty(acc.used) && throw(ArgumentError(
        "no grid cell in the region has usable forcing (ERA5-Land is land-only; every cell " *
        "center may have fallen on water)"))

    grid_cells = @view selected[acc.used, :]

    decoupling = derive_decoupling_factor(acc; min_cells, bounds = decoupling_factor_bounds)
    lapse_rate = derive_lapse_rate(acc, decoupling.native, decoupling.ambient_excess; min_cells)

    band_forcing = _BandForcing(grid_cells, load, acc.time, decoupling.native, lapse_rate.native,
                                band_batch)

    provenance = Dict{String,Any}(
        "climate_model" => string(climate_model),
        "time_range" => [first(acc.time), last(acc.time)],
        "n_grid_cells_in_region" => nrow(selected),
        "n_grid_cells_used" => length(acc.used),
        "n_bands" => length(band_forcing.bands),
        "region_extent" => _region_extent(_validate_region(roi_polygon)),
        "area_minimum" => Float64(area_minimum),
        "mean_elevation" => acc.mean_elevation,
        "adjustment_order" => "lapse_then_decouple",
    )

    return (; grid_cells, time = acc.time, decoupling, lapse_rate, band_forcing, provenance)
end

# Per-timestep cross-cell sums for both fits, plus the per-cell metadata the band pass needs.
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
    glms = Float64[]
    areas = Float64[]

    # Sums over cells, per timestep. `s1` counts finite cells; `z`/`g` are the regressors.
    S = nothing

    for (i, row) in enumerate(eachrow(selected))
        fd = try
            load(row)
        catch e
            e isa InterruptException && rethrow()
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
                 # Second moments of the responses, for the fit-quality diagnostics.
                 EE = zeros(n), TT = zeros(n), TE = zeros(n))
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
            S.EE[t] += Et * Et;    S.TT[t] += Tt * Tt;     S.TE[t] += Tt * Et
        end

        push!(used, i)
        push!(elevations, z)
        push!(glms, glm)
        push!(areas, area)
    end

    S === nothing && return (; time = DateTime[], n = 0, used, elevations, glms, areas,
                             sums = nothing, mean_elevation = NaN)

    # Area-weighted mean elevation: the reference the fitted factor is reported *at*, since the
    # regression's intercept alone would be an extrapolation to sea level.
    total_area = sum(areas)
    mean_elevation = total_area > 0 ? sum(areas .* elevations) / total_area :
                     Statistics.mean(elevations)

    return (; time, n, used, elevations, glms, areas, sums = S, mean_elevation)
end

# `glm` for a row, as a finite fraction in [0, 1]. Same contract as `cell_decoupling_factor`: the
# column is required (a table missing it cannot support any of this), while `missing`/`NaN` are real
# data gaps that fall back to the uncorrected reanalysis assumption.
function _row_glm(row)
    hasproperty(row, :glm) || throw(ArgumentError(
        "glacier elevation-class table has no `:glm` column, which the decoupling fit regresses " *
        "against. A table built before the current invariant column naming carries `:glm_frac`; " *
        "run `scripts/migrate_invariant_colnames.jl` to rename it in place."))
    v = row.glm
    (v === missing || !isfinite(Float64(v))) && return 0.0
    return clamp(Float64(v), 0.0, 1.0)
end

"""
    derive_decoupling_factor(acc; min_cells = 3, bounds = (0.2, 1.0))

Fit the on-glacier temperature decoupling factor `k` per timestep from cross-cell sums, as
accumulated over a region's grid cells.

`climate_adjust_for_glacier` damps the warm excess above melting, `T′ = T + (k - 1)·max(T - T_ref, 0)`,
so `E′ = k·E`. The estimator inverts that using the one place the reanalysis already varies its own
degree of decoupling: `glm`, the fraction of each cell its land-surface scheme treats as ice.

**Which end of `glm` is decoupled matters, and it is the high end.** A cell with `glm = 1` is one
ERA5-Land already runs as ice, so its near-surface temperature is *already* damped and needs no
correction — which is precisely why [`cell_decoupling_factor`](@ref) weights the correction it
applies by `1 - glm`. A cell with `glm = 0` is run as ice-free ground and its temperature is
ambient, so it needs the full correction. Mixing the two, a cell's observed excess is

    E_i = [1 - glm_i·(1 - k)] · E_ambient(z_i)

falling from `E_ambient` at `glm = 0` to `k·E_ambient` at `glm = 1`. So `k` is read off as the ratio
of the fitted excess at the two ends of `glm`, and the warm excess must *decrease* with `glm` for the
fit to mean anything.

Note the *product*: the `glm` weighting scales an ambient excess that itself varies with elevation.
Taking `E_ambient` locally linear in `z`, `E_ambient ≈ a + b·z`, expands that product into

    E_i = [1 - glm_i(1 - k)]·(a + b·z_i)
        = α + β·z_i + γ·glm_i + δ·(glm_i·z_i)

with `α = a`, `β = b`, `γ = -(1-k)a`, `δ = -(1-k)b`. So the fit is on four terms, and the `glm × z`
interaction is not optional — dropping it biases `k` toward 1, because the `glm` slope then has to
absorb part of the elevation dependence of the very quantity it multiplies. Evaluating at the
region's area-weighted mean elevation `z̄`:

    k = [(α + γ) + (β + δ)·z̄] / (α + β·z̄)

i.e. the fitted excess of a fully-glaciated cell over that of an ice-free one at the same elevation.

Elevation is a control, not a nuisance: `glm` and elevation covary strongly (high cells are more
glaciated), so without it the `glm` slope is partly a topographic signal.

Returns `(; native, monthly, scalar, n_cells, r2, fitted, clamped, ambient_excess)`. `native` is per
timestep at the forcing's own resolution; `monthly` is the 12-element calendar-month climatology,
which is the form `climate_adjust_for_elevation` accepts directly; `scalar` is one value pooled over
the record. `fitted` marks the timesteps a fit was actually attempted on and `clamped` those whose
fit fell outside `bounds` and was pulled to an edge — **check `clamped` before trusting `monthly`**,
since a clamped value is indistinguishable from a confident one in `native` and a region with little
`glm` spread can pin most of its timesteps at a bound. A warning is emitted past 25%.
`ambient_excess` is the fitted `(α, β)` of the ice-free excess profile `E_ambient ≈ α + β·z` per
timestep, which [`derive_lapse_rate`](@ref) needs to undo the reanalysis's own partial decoupling
per cell.

A timestep gets the identity `1.0` — a bit-exact no-op — when it has too few cells, no spread in
`glm` or elevation, collinear regressors, or an ambient excess below `$(_MIN_AMBIENT_EXCESS)` K. That
last case is most of a glacier record, and the threshold is not merely a divide-by-zero guard: `k` is
a *ratio* of fitted excesses, so a region sitting near melting fits one from noise and then pins at a
clamp bound. On real Alpine forcing that produced a confident-looking winter `k` of 0.2 from a 0.1 K
excess. Four parameters need at least four cells, so the effective cell threshold is
`max(min_cells, 4)`.

`monthly` and `scalar` average only the timesteps that were actually fit, so a record dominated by
sub-freezing identity values does not drag them toward 1.
"""
function derive_decoupling_factor(acc; min_cells::Int = 3, bounds = (0.2, 1.0))
    lo, hi = Float64(bounds[1]), Float64(bounds[2])
    0 < lo <= hi <= 1 || throw(ArgumentError(
        "decoupling_factor_bounds must satisfy 0 < lo <= hi <= 1, got $bounds"))

    n = acc.n
    native = ones(Float64, n)
    r2 = fill(NaN, n)
    n_cells = zeros(Int, n)
    fitted = falses(n)
    clamped = falses(n)
    # The ice-free excess profile `E_ambient ≈ α + β·z`, per timestep. Zero means "no ambient excess
    # fitted", which makes `derive_lapse_rate`'s correction a no-op for that timestep — consistent
    # with the identity `k` it also gets.
    ambient_excess = (zeros(n), zeros(n))
    acc.sums === nothing && return (; native, monthly = fill(1.0, 12), scalar = 1.0, n_cells, r2,
                                    fitted, clamped, ambient_excess)

    S = acc.sums
    z̄ = acc.mean_elevation

    @inbounds for t in 1:n
        n_cells[t] = S.s1[t]
        # Four parameters need four cells before the system is even determined.
        S.s1[t] >= max(min_cells, 4) || continue

        # XᵀX for x = [1, z, glm, glm*z], upper triangle, and Xᵀy for y = E.
        A = (Float64(S.s1[t]), S.z[t], S.g[t],  S.w[t],
                               S.zz[t], S.zg[t], S.zw[t],
                                        S.gg[t], S.gw[t],
                                                 S.ww[t])
        coef = _solve_sym4(A, (S.E[t], S.zE[t], S.gE[t], S.wE[t]))
        coef === nothing && continue
        α, β, γ, δ = coef

        # The fitted excess at the reference elevation, at each end of `glm`: an ice-free cell
        # (`glm = 0`), which the reanalysis leaves ambient, and a fully-glaciated one (`glm = 1`),
        # which it has already decoupled.
        ambient = α + β * z̄
        decoupled = ambient + (γ + δ * z̄)
        # `k` is a ratio of fitted excesses, so it is only identifiable when the denominator is
        # meaningfully above zero. A region at or near melting carries no information about `k` —
        # and, worse, fits one anyway from noise. Keep the identity instead.
        ambient >= _MIN_AMBIENT_EXCESS || continue

        raw = decoupled / ambient
        native[t] = clamp(raw, lo, hi)
        clamped[t] = raw != native[t]
        r2[t] = _r2(S.s1[t], S.E[t], S.EE[t],
                    α * S.E[t] + β * S.zE[t] + γ * S.gE[t] + δ * S.wE[t])
        ambient_excess[1][t] = α
        ambient_excess[2][t] = β
        fitted[t] = true
    end

    monthly, scalar = _climatology(native, acc.time, fitted, 1.0)

    # A fit that mostly hits its bounds is not a fit. Worth saying out loud, because the clamped
    # values are indistinguishable from confident ones in `native` and will otherwise propagate
    # into the monthly climatology as a plausible-looking seasonal cycle.
    n_fit = count(fitted)
    n_clamped = count(clamped)
    if n_fit > 0 && n_clamped > n_fit ÷ 4
        @warn """
              Decoupling factor hit its clamp bounds on $(round(100 * n_clamped / n_fit, digits = 1))% \
              of fitted timesteps: the region may have too little spread in `glm` to identify `k`, or \
              too few cells. Treat `monthly` and `scalar` with suspicion and check `clamped`.""" n_fitted=n_fit n_clamped bounds=(lo, hi)
    end

    return (; native, monthly, scalar, n_cells, r2, fitted, clamped, ambient_excess)
end

"""
    derive_lapse_rate(acc, decoupling_factor, ambient_excess; min_cells = 3)

Fit the **on-glacier** temperature lapse rate (K/km, positive for cooling with height) per timestep
from cross-cell sums, after decoupling each cell's temperature with `decoupling_factor`.

Decoupling comes first because the lapse rate wanted here is the on-glacier one and not the ambient
one: `k` damps the warm excess, which compresses the near-surface temperature range over melting ice
and so flattens the slope against elevation. Fitting the raw reanalysis temperature would give the
ambient lapse rate, which is not what a glacier surface sees.

The correction applied to each cell is the *effective* factor, weighted by that cell's non-glacier
fraction exactly as [`cell_decoupling_factor`](@ref) weights it:

    T′_i = T_i + (k_eff_i - 1)·E_i,   k_eff_i = 1 - (1 - k)(1 - glm_i)
         = T_i + (k - 1)(1 - glm_i)·E_i

A cell ERA5-Land already treats as fully glaciated (`glm = 1`) therefore gets no correction, and an
ice-free one gets the full correction, so every cell is brought to a common on-glacier state before
the slope is taken. Applying one uniform `k` to all of them instead would over-damp the glaciated
cells — which are systematically the high ones — and so tilt the fitted slope.

`E_i` here is the **ambient** excess at cell `i`'s elevation — what an ice-free cell there would see
— not the cell's observed excess, which is already damped by its own `glm`. That is exactly the
ice-free end of the decoupling fit, `α + β·z`, so `ambient_excess` takes the `(α, β)` per-timestep
vectors [`derive_decoupling_factor`](@ref) returns as `ambient_excess` rather than re-deriving them.
The whole correction is then linear in the sums already accumulated: no second pass over the forcing.

Returns `(; native, monthly, scalar, n_cells, elevation_spread, fitted)`, shaped like
[`derive_decoupling_factor`](@ref)'s result except that the fit diagnostic is `elevation_spread` —
the standard deviation (m) of the contributing cells' reanalysis elevations, which is what actually
governs whether a slope is identifiable. Values are clamped to
$(_LAPSE_RATE_LIMITS) K/km, the same physical bounds `climate_adjust_for_elevation` validates
against, so a thin-sample timestep cannot fit a slope that later throws downstream. A timestep with
fewer than `min_cells` cells, or no spread in elevation, falls back to
$(_DEFAULT_LAPSE_RATE) K/km — the value every other call site in this package uses.
"""
function derive_lapse_rate(acc, decoupling_factor, ambient_excess; min_cells::Int = 3)
    n = acc.n
    ambient = ambient_excess
    native = fill(_DEFAULT_LAPSE_RATE, n)
    elevation_spread = fill(NaN, n)
    n_cells = zeros(Int, n)
    fitted = falses(n)
    acc.sums === nothing &&
        return (; native, monthly = fill(_DEFAULT_LAPSE_RATE, 12), scalar = _DEFAULT_LAPSE_RATE,
                n_cells, elevation_spread, fitted)

    S = acc.sums
    lo, hi = _LAPSE_RATE_LIMITS

    @inbounds for t in 1:n
        n_cells[t] = S.s1[t]
        S.s1[t] >= min_cells || continue

        m = Float64(S.s1[t])

        # Elevation spread. Zero means every cell in the region sits at the same reanalysis
        # elevation, so no slope is identifiable however many cells there are.
        Szz = S.zz[t] - S.z[t] * S.z[t] / m
        Szz > 0 || continue
        elevation_spread[t] = sqrt(Szz / m)

        # Bring every cell to the same fully-on-glacier state before taking the slope. The
        # remaining correction for cell `i` is what is left of `k` after the reanalysis has already
        # applied its own `glm`-worth of it:
        #
        #     T_glacier_i = T_i + (k - 1)(1 - glm_i)·E_ambient(z_i)
        #
        # `E_ambient` is the ambient excess — the excess an ice-free cell at that elevation would
        # see — which is *not* the observed excess, since the observed one is already damped by
        # `glm_i`. It is exactly the ice-free end of the decoupling fit, `α + β·z`, so the
        # coefficients that fit came out of are reused here rather than re-derived.
        c = decoupling_factor[t] - 1.0
        α, β = ambient[1][t], ambient[2][t]

        # Σ(1-glm)E_ambient and Σz(1-glm)E_ambient, from the stored regressor sums: Σz(1-glm) is
        # `z - w` and Σz²(1-glm) is `zz - zw`, `w` being the `glm·z` interaction regressor.
        s_corr  = α * (m - S.g[t]) + β * (S.z[t] - S.w[t])
        sz_corr = α * (S.z[t] - S.w[t]) + β * (S.zz[t] - S.zw[t])

        sT  = S.T[t]  + c * s_corr
        szT = S.zT[t] + c * sz_corr

        # Simple linear regression of the on-glacier temperature on elevation.
        SzT = szT - S.z[t] * sT / m

        # K/m -> K/km, sign flipped so positive means cooling with height (the convention
        # `climate_adjust_for_elevation` expects).
        native[t] = clamp(-1000.0 * (SzT / Szz), lo, hi)
        fitted[t] = true
    end

    monthly, scalar = _climatology(native, acc.time, fitted, _DEFAULT_LAPSE_RATE)
    return (; native, monthly, scalar, n_cells, elevation_spread, fitted)
end

"""
    _solve_sym4(A, b) -> NTuple{4,Float64} or nothing

Solve the symmetric positive-semidefinite 4x4 system `A x = b` by Cholesky, where `A` is the upper
triangle in row-major order — `(a11, a12, a13, a14, a22, a23, a24, a33, a34, a44)`.

These are normal equations, so `A = XᵀX` is symmetric and PSD by construction and Cholesky is both
the cheapest factorization and its own rank test: a non-positive pivot means the regressors are
collinear, which returns `nothing` rather than a garbage fit. That happens for real regions — every
cell at the same elevation, every cell with the same glacier fraction, or a region small enough that
`glm·z` is indistinguishable from `z`.

The pivot test is scale-relative. Elevation is in metres, so `Σz²` runs to 1e8 per cell and an
absolute tolerance would reject well-conditioned systems.
"""
function _solve_sym4(A::NTuple{10,Float64}, b::NTuple{4,Float64})
    a11, a12, a13, a14, a22, a23, a24, a33, a34, a44 = A
    # Diagonal scale, for the relative pivot test below.
    scale = max(a11, a22, a33, a44)
    (isfinite(scale) && scale > 0) || return nothing
    tol = 1e-12 * scale

    # Cholesky: A = LLᵀ. Each pivot is tested before its square root, so a rank-deficient system
    # exits here rather than producing a NaN that propagates into the fit.
    a11 > tol || return nothing
    l11 = sqrt(a11)
    l21 = a12 / l11
    l31 = a13 / l11
    l41 = a14 / l11

    d22 = a22 - l21 * l21
    d22 > tol || return nothing
    l22 = sqrt(d22)
    l32 = (a23 - l31 * l21) / l22
    l42 = (a24 - l41 * l21) / l22

    d33 = a33 - l31 * l31 - l32 * l32
    d33 > tol || return nothing
    l33 = sqrt(d33)
    l43 = (a34 - l41 * l31 - l42 * l32) / l33

    d44 = a44 - l41 * l41 - l42 * l42 - l43 * l43
    d44 > tol || return nothing
    l44 = sqrt(d44)

    # Forward substitution L y = b, then back substitution Lᵀ x = y.
    y1 = b[1] / l11
    y2 = (b[2] - l21 * y1) / l22
    y3 = (b[3] - l31 * y1 - l32 * y2) / l33
    y4 = (b[4] - l41 * y1 - l42 * y2 - l43 * y3) / l44

    x4 = y4 / l44
    x3 = (y3 - l43 * x4) / l33
    x2 = (y2 - l32 * x3 - l42 * x4) / l22
    x1 = (y1 - l21 * x2 - l31 * x3 - l41 * x4) / l11

    all(isfinite, (x1, x2, x3, x4)) || return nothing
    return (x1, x2, x3, x4)
end

"""
    _r2(m, sy, syy, explained)

Coefficient of determination from accumulated sums: `1 - SS_res / SS_tot`, where
`SS_res = SS_tot - SS_reg` and `SS_reg = β̂ᵀXᵀy - (Σy)²/m` is the explained sum of squares.

`NaN` when the response has no variance across cells — for the decoupling fit that is every
timestep with all cells at or below melting, where `E` is identically zero and there is nothing for
a fit to explain.
"""
function _r2(m::Real, sy::Real, syy::Real, explained::Real)
    m > 0 || return NaN
    ss_tot = syy - sy * sy / m
    ss_tot > 0 || return NaN
    ss_reg = explained - sy * sy / m
    return clamp(ss_reg / ss_tot, 0.0, 1.0)
end

# Calendar-month climatology and pooled scalar over the timesteps that were actually fit. Falling
# back to `default` for a month with no fitted timestep keeps the 12-element vector complete, which
# is what `climate_adjust_for_elevation`'s monthly path requires.
function _climatology(native, time, fitted, default)
    sums = zeros(12)
    counts = zeros(Int, 12)
    for t in eachindex(native)
        fitted[t] || continue
        m = month(time[t])
        sums[m] += native[t]
        counts[m] += 1
    end
    monthly = [counts[m] > 0 ? sums[m] / counts[m] : default for m in 1:12]
    total = sum(counts)
    scalar = total > 0 ? sum(sums) / total : default
    return monthly, scalar
end

"""
    _BandForcing(grid_cells, load, time, decoupling_factor, lapse_rate, band_batch)

Lazy iterator over a region's hypsometry bands, yielding glacier-area weighted forcing for each.

Each element is `(; lo, hi, center, area, n_cells, forcing)`. Bands ascend in elevation, and
`area` sums over the iterator to the region's total glacier area, so nothing is dropped.

Laziness is what makes a large region tractable. All bands at once is `n_bands * 7 * n_time`
Float64s — ~2 GB for 50 bands of 76 hourly years — so instead each `iterate` streams the region's
cells again and accumulates only the next `band_batch` bands, holding `band_batch` band stacks.
The cost is `ceil(n_bands / band_batch)` passes over the (warm, after the parameter fit) Zarr
chunk cache rather than one. `band_batch = 0` opts out and does a single pass.
"""
struct _BandForcing{C,L,B}
    grid_cells::C
    load::L
    time::Vector{DateTime}
    decoupling_factor::Vector{Float64}
    lapse_rate::Vector{Float64}
    band_batch::Int
    bands::B
end

function _BandForcing(grid_cells, load, time, decoupling_factor, lapse_rate, band_batch::Int)
    # Union of populated bands across the region. The `hyps_*` column names are authoritative —
    # the table's `hypsometry_bin_edges` metadata does not survive a GeoParquet round-trip — so
    # `glacier_hypsometry` is the decoder to go through.
    seen = Dict{Tuple{Int,Int},Float64}()
    for row in eachrow(grid_cells)
        for b in glacier_hypsometry(row; area_minimum = 0)
            seen[(b.lo, b.hi)] = get(seen, (b.lo, b.hi), 0.0) + b.area
        end
    end
    bands = [(; lo, hi, center = (lo + hi) / 2, area)
             for ((lo, hi), area) in sort!(collect(seen); by = first)]
    return _BandForcing(grid_cells, load, collect(time), collect(Float64, decoupling_factor),
                        collect(Float64, lapse_rate), band_batch, bands)
end

Base.length(bf::_BandForcing) = length(bf.bands)
Base.eltype(::Type{<:_BandForcing}) = NamedTuple
Base.IteratorSize(::Type{<:_BandForcing}) = Base.HasLength()

function Base.iterate(bf::_BandForcing, state = (1, nothing, 0))
    next, batch, batch_start = state
    next > length(bf.bands) && return nothing

    # Refill when the current batch is exhausted: one pass over the region's cells accumulating the
    # next `band_batch` bands.
    if batch === nothing || next >= batch_start + length(batch)
        stop = bf.band_batch == 0 ? length(bf.bands) :
               min(next + bf.band_batch - 1, length(bf.bands))
        batch = _accumulate_bands(bf, next:stop)
        batch_start = next
    end

    return batch[next - batch_start + 1], (next + 1, batch, batch_start)
end

# One pass over the region's grid cells, accumulating area-weighted forcing for `range`'s bands.
function _accumulate_bands(bf::_BandForcing, range)
    n = length(bf.time)
    bands = bf.bands[range]
    # Running weighted sums per band, one array per forcing layer, plus the weight totals.
    sums = [Dict(v => zeros(n) for v in _FORCING_VARIABLES) for _ in bands]
    weights = zeros(length(bands))
    counts = zeros(Int, length(bands))
    template = nothing
    # Highest reanalysis surface contributing to each band. Per band, not per batch, so the
    # extrapolation this reports does not depend on how `band_batch` happens to group the bands.
    z_max = fill(-Inf, length(bands))

    for row in eachrow(bf.grid_cells)
        # Which of this batch's bands does this cell hold ice in, and how much?
        areas = _band_areas(row, bands)
        all(iszero, areas) && continue

        fd = try
            bf.load(row)
        catch e
            e isa InterruptException && rethrow()
            @warn "Forcing load failed during band aggregation; skipping cell" exception=e
            continue
        end
        forcing_is_complete(fd) || continue
        template === nothing && (template = fd)

        z_cell = Float64(DimensionalData.metadata(fd)["elevation"])

        # The correction this cell still needs, after the reanalysis has already applied its own
        # `glm`-worth of it — the same weighting `cell_decoupling_factor` applies, and the same one
        # `derive_lapse_rate` undoes to fit the slope. Formed once per cell, not once per band.
        glm = _row_glm(row)
        cell_factor = [1 - (1 - k) * (1 - glm) for k in bf.decoupling_factor]

        for (i, band) in enumerate(bands)
            areas[i] > 0 || continue
            adjusted = _cell_forcing_at_band(fd, band.center - z_cell, cell_factor,
                                             bf.lapse_rate)
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
        "no grid cell in the region yielded usable forcing during band aggregation"))

    # How far each band is being lapse-extrapolated past the reanalysis surface it came from.
    # Glaciers systematically occupy the high ground *within* a 0.1° cell, so a band well above
    # every contributing cell is the normal case rather than an error — on real Alpine forcing 76%
    # of the glacier area sat above every contributing reanalysis surface. But it is a genuine
    # extrapolation, and it is the dominant uncertainty in the forcing the downstream sweep then
    # fits its bias correction against, so it should not be silent. One line per batch, not per band.
    far = [(bands[i].center, bands[i].center - z_max[i])
           for i in eachindex(bands) if isfinite(z_max[i]) && bands[i].center - z_max[i] > 500]
    isempty(far) || @info("Bands extrapolated far above the reanalysis surface (expected for " *
                          "glaciers, but this forcing is a lapse extrapolation)",
                          n_bands = length(far),
                          highest_band = maximum(first, far),
                          max_extrapolation = round(maximum(last, far), digits = 1))

    return [_band_stack(bands[i], sums[i], weights[i], counts[i], template, bf, z_max[i])
            for i in eachindex(bands)]
end

# This cell's glacier area (km²) in each of `bands`, by decoding its flat `hyps_*` columns.
function _band_areas(row, bands)
    areas = zeros(length(bands))
    have = glacier_hypsometry(row; area_minimum = 0)
    for (i, band) in enumerate(bands)
        j = findfirst(b -> b.lo == band.lo && b.hi == band.hi, have)
        j === nothing || (areas[i] = have[j].area)
    end
    return areas
end

# One cell's forcing at a band center: lapse first, then decouple.
#
# That order is not incidental. `k` multiplies an *ambient* temperature at the glacier's elevation,
# so the elevation adjustment has to run first — `climate_adjust_for_glacier` says so explicitly,
# and `forcing_at_elevation` (the path the sweep itself uses) does the same. The two do not commute,
# since the decoupling damps only the excess above a fixed melting point and lapsing changes which
# timesteps are above it.
function _cell_forcing_at_band(fd, delta_elevation, decoupling_factor, lapse_rate)
    lapsed = climate_adjust_for_elevation(fd, delta_elevation; lapse_rate)
    return _decouple_per_timestep(lapsed, decoupling_factor)
end

# Per-timestep decoupling. `climate_adjust_for_glacier` takes a scalar `k`, but the derived factor
# varies with the forcing's own time resolution, so apply the same increment directly:
#
#     T′  = T + (k - 1) * max(T - T_ref, 0)
#     LW′ = longwave re-derived at T′, vapor pressure unchanged
#
# written as an increment so `k = 1` is bit-exact — the property that makes a sub-freezing or
# unfitted timestep a true no-op. Skips the whole pass when every `k` is 1.
function _decouple_per_timestep(stack, decoupling_factor)
    all(==(1.0), decoupling_factor) && return stack

    T = stack[:temperature_air]
    e = stack[:vapor_pressure]
    LW = stack[:longwave_downward]

    T′ = similar(T)
    @inbounds for t in eachindex(T)
        k = decoupling_factor[t]
        T′[t] = T[t] + (k - 1) * max(T[t] - _DECOUPLING_REFERENCE_TEMPERATURE, 0.0)
    end
    LW′ = GEMB_ClimateForcing._adjust_longwave(LW, e, T, e, T′)

    meta = merge(copy(DimensionalData.metadata(stack)), Dict(
        "glacier_decoupling_factor_mean" => Statistics.mean(decoupling_factor),
        "temperature_air_mean" => Statistics.mean(T′)))
    return DimensionalData.rebuild(stack;
                   data = merge(NamedTuple(stack),
                                (temperature_air = parent(T′), longwave_downward = parent(LW′))),
                   metadata = meta)
end

# Finish one band: divide the weighted sums by the total weight and rebuild a stack shaped like
# `climate_forcing` output, so it drops straight into the existing sweep.
function _band_stack(band, sums, weight, n_cells, template, bf, z_max)
    weight > 0 || throw(ArgumentError("band $(band.lo)-$(band.hi) m accumulated zero area"))
    time_dim = dims(template, Ti)
    layers = NamedTuple(v => DimArray(sums[v] ./ weight, (time_dim,);
                                      metadata = DimensionalData.metadata(template[v]))
                        for v in _FORCING_VARIABLES)

    meta = merge(copy(DimensionalData.metadata(template)), Dict(
        # The band center is this stack's reference elevation: it is already *at* the band, so a
        # downstream `forcing_at_elevation(band.forcing, 0)` is an identity and
        # `forcing_is_complete` finds a finite elevation where it looks for one.
        "elevation" => band.center,
        "glacier_area" => weight,
        "n_grid_cells" => n_cells,
        "band_lower" => Float64(band.lo),
        "band_upper" => Float64(band.hi),
        # How far above the highest contributing reanalysis surface this band sits. Positive means
        # the forcing here is a lapse extrapolation, which for glaciers is the norm rather than the
        # exception — they occupy the high ground within a 0.1° cell — and is the dominant
        # uncertainty in what the downstream sweep fits its bias correction against.
        "extrapolation_above_reanalysis" => band.center - z_max,
        "temperature_lapse_rate" => Statistics.mean(bf.lapse_rate),
        "glacier_decoupling_factor" => Statistics.mean(bf.decoupling_factor),
        "adjustment_order" => "lapse_then_decouple",
        "temperature_air_mean" => Statistics.mean(layers.temperature_air),
        "precipitation_mean" => Statistics.mean(layers.precipitation),
        "wind_speed_mean" => Statistics.mean(layers.wind_speed)))
    # Cell-specific keys would be a lie on a regional average.
    for k in ("latitude", "longitude", "chunk_strategy", "delta_elevation", "decoupling_factor")
        delete!(meta, k)
    end

    return (; band.lo, band.hi, band.center, area = weight, n_cells,
            forcing = DimStack(layers; metadata = meta))
end
