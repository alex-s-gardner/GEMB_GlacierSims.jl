# Slope-dependent correction of the reanalysis 10 m wind speed on glacier bands.
#
# ERA5-Land's 10 m wind is too weak over sloping ice and about right over flat ice: against on-ice
# weather stations (station wind taken to 10 m with a neutral log profile) the ratio of station to
# ERA5-Land wind grows with the local surface slope, and above about 4 m s-1 it does not depend on
# the wind speed. The correction is therefore a multiplier that is a function of slope alone, with
# no regional term, and an optional floor for near-calm hours:
#
#     U = max(m(θ) · U_ERA5, U_min(θ)),   m(θ) = exp(m0) · max(θ, θ_floor)^m1,
#     U_min(θ) = max(a + b · log(max(θ, θ_floor)), 0),
#
# with θ the surface slope in degrees, measured as a least-squares plane through the Copernicus
# GLO-30 DEM over a 2 km × 2 km window. A band's θ is the glacier-area weighted mean of that slope
# over the band's glacier pixels, carried in the elevation-class table as `slope2km_<lo>_<hi>`
# columns beside the `hyps_<lo>_<hi>` areas.

"""
    SLOPE_WIND_COEFFICIENTS

The coefficients of the slope-dependent wind correction ([`slope_wind_correction`](@ref)):
`m0 = 0.398`, `m1 = 0.168`, `theta_floor = 0.1` (degrees), `u_min_a = 0.0`, `u_min_b = 0.0`
(m s-1). They are a least-squares fit of log(station / ERA5-Land mean wind) on log slope over 112
on-ice stations of the Greenland and Antarctic ice sheets, ice caps and glaciers (two alpine
valley-glacier stations excluded), which gives m = 1.0 at 0.1°, 1.5 at 1° and 2.2 at 10°. The floor
is zero because a positive one moves the corrected low quantiles further from the stations'.
"""
const SLOPE_WIND_COEFFICIENTS = (; m0 = 0.398, m1 = 0.168, theta_floor = 0.1, u_min_a = 0.0, u_min_b = 0.0)

"""
    BAND_SLOPE_COLUMN_PREFIX

The prefix of the elevation-class table columns that carry each band's 2 km surface slope
(degrees): `slope2km_<lo>_<hi>`, with the edges of the matching `hyps_<lo>_<hi>` column.
"""
const BAND_SLOPE_COLUMN_PREFIX = "slope2km"

"""
    WIND_CORRECTIONS

The wind corrections [`resolve_downscaling`](@ref) applies to the band wind: `:scaling`, the scalar
`wind_scaling` (1 by default), and `:slope`, [`slope_wind_correction`](@ref) with each band's slope.
"""
const WIND_CORRECTIONS = (:scaling, :slope)

"""
    slope_wind_multiplier(θ; coefficients = SLOPE_WIND_COEFFICIENTS) -> Float64

The wind multiplier m(θ) = exp(m0) · max(θ, θ_floor)^m1 at a surface slope of `θ` degrees.
"""
function slope_wind_multiplier(θ::Real; coefficients = SLOPE_WIND_COEFFICIENTS)
    (isfinite(θ) && θ >= 0) || throw(ArgumentError("surface slope must be finite and ≥ 0°, got $θ"))
    c = coefficients
    return exp(c.m0) * max(Float64(θ), c.theta_floor)^c.m1
end

"""
    slope_wind_floor(θ; coefficients = SLOPE_WIND_COEFFICIENTS) -> Float64

The calm floor U_min(θ) = max(a + b · log(max(θ, θ_floor)), 0) in m s-1.
"""
function slope_wind_floor(θ::Real; coefficients = SLOPE_WIND_COEFFICIENTS)
    c = coefficients
    return max(c.u_min_a + c.u_min_b * log(max(Float64(θ), c.theta_floor)), 0.0)
end

"""
    slope_wind_correction(u, θ; coefficients = SLOPE_WIND_COEFFICIENTS) -> Float64

The corrected wind speed max(m(θ) · `u`, U_min(θ)) for a reanalysis wind speed `u` (m s-1) on a
surface of slope `θ` degrees ([`slope_wind_multiplier`](@ref), [`slope_wind_floor`](@ref)).
"""
slope_wind_correction(u::Real, θ::Real; coefficients = SLOPE_WIND_COEFFICIENTS) =
    max(slope_wind_multiplier(θ; coefficients) * u, slope_wind_floor(θ; coefficients))

# Throw unless `c` holds every coefficient of the correction, finite, with a positive slope floor.
function _check_slope_wind_coefficients(c)
    expected = (:m0, :m1, :theta_floor, :u_min_a, :u_min_b)
    Set(propertynames(c)) == Set(expected) || throw(ArgumentError(
        "slope wind coefficients must be $(expected), got $(propertynames(c))"))
    all(k -> isfinite(getproperty(c, k)), expected) ||
        throw(ArgumentError("slope wind coefficients must be finite, got $c"))
    c.theta_floor > 0 || throw(ArgumentError("theta_floor must be > 0°, got $(c.theta_floor)"))
    return (; m0 = Float64(c.m0), m1 = Float64(c.m1), theta_floor = Float64(c.theta_floor),
            u_min_a = Float64(c.u_min_a), u_min_b = Float64(c.u_min_b))
end

"""
    wind_correction_rule(spec; n_bands = nothing) -> NamedTuple

The validated form of a `wind_correction` keyword of [`resolve_downscaling`](@ref):
`(; rule = :scaling)`, or `(; rule = :slope, band_slope, coefficients)` with one finite slope ≥ 0
(degrees) per band and `coefficients` as [`SLOPE_WIND_COEFFICIENTS`](@ref), which is the default
when the field is omitted. A field the rule does not use throws.
"""
function wind_correction_rule(spec; n_bands = nothing)
    hasproperty(spec, :rule) || throw(ArgumentError(
        "wind_correction needs a `rule`, one of $(WIND_CORRECTIONS); got $spec"))
    rule = spec.rule
    rule in WIND_CORRECTIONS || throw(ArgumentError(
        "wind_correction rule must be one of $(WIND_CORRECTIONS), got $(repr(rule))"))
    fields = Set(propertynames(spec))
    allowed = rule === :slope ? Set((:rule, :band_slope, :coefficients)) : Set((:rule,))
    issubset(fields, allowed) && (rule === :scaling || :band_slope in fields) || throw(ArgumentError(
        "wind_correction rule :$rule takes the fields $(sort!(collect(allowed))), got " *
        "$(sort!(collect(fields)))"))
    rule === :scaling && return (; rule)
    band_slope = collect(Float64, spec.band_slope)
    n_bands === nothing || length(band_slope) == n_bands || throw(ArgumentError(
        "wind_correction :slope gives $(length(band_slope)) band slopes for $n_bands bands"))
    bad = findfirst(θ -> !(isfinite(θ) && θ >= 0), band_slope)
    bad === nothing || throw(ArgumentError(
        "wind_correction :slope band slope must be finite and ≥ 0°, got $(band_slope[bad]) for band $bad"))
    coefficients = _check_slope_wind_coefficients(get(spec, :coefficients, SLOPE_WIND_COEFFICIENTS))
    return (; rule, band_slope, coefficients)
end

"""
    tile_interval_slopes(cells, intervals; prefix = BAND_SLOPE_COLUMN_PREFIX) -> Vector{Float64}

The surface slope (degrees) of each of `intervals` over `cells`: the mean of the cells'
`<prefix>_<lo>_<hi>` slopes weighted by their `hyps_<lo>_<hi>` glacier area, over the cells whose
slope is finite. Throws when an interval holds no ice in `cells`, has no slope column, or has no
cell with ice and a finite slope.
"""
function tile_interval_slopes(cells, intervals; prefix::AbstractString = BAND_SLOPE_COLUMN_PREFIX)
    cols = [Symbol(prefix * "_" * lpad(iv.lo, 5, '0') * "_" * lpad(iv.hi, 5, '0')) for iv in intervals]
    hyps = [Symbol("hyps_" * lpad(iv.lo, 5, '0') * "_" * lpad(iv.hi, 5, '0')) for iv in intervals]
    have = Set(propertynames(cells))
    missing_cols = [c for c in cols if !(c in have)]
    isempty(missing_cols) || throw(ArgumentError(
        "the elevation-class table has no band slope column $(first(missing_cols)) " *
        "($(length(missing_cols)) missing); build the table with $(prefix)_<lo>_<hi> columns"))
    slope = fill(NaN, length(intervals))
    for i in eachindex(intervals, cols, hyps)
        hyps[i] in have || throw(ArgumentError("the elevation-class table has no column $(hyps[i])"))
        a = cells[!, hyps[i]]; θ = cells[!, cols[i]]
        w = 0.0; s = 0.0; ice = 0.0
        for k in eachindex(a, θ)
            a[k] > 0 || continue
            ice += a[k]
            t = θ[k]
            (t === missing || !isfinite(t)) && continue
            w += a[k]; s += a[k] * t
        end
        ice > 0 || throw(ArgumentError(
            "band $(intervals[i].lo)-$(intervals[i].hi) m holds no ice in these cells"))
        w > 0 || throw(ArgumentError(
            "no cell with ice in band $(intervals[i].lo)-$(intervals[i].hi) m has a finite $(cols[i])"))
        slope[i] = s / w
    end
    return slope
end
