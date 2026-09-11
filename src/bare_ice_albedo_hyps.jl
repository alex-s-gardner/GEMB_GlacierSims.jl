"""
Bare-ice albedo reduced to a function of elevation.

[`bare_ice_albedo`](@ref) returns one albedo per MCD43A3 cell with no elevation attached.
[`bare_ice_albedo_hyps`](@ref) pairs those cells with an elevation each, bins them, and returns a
**callable**: `f(z)` is the best estimate of bare-ice albedo at elevation `z`.

The bins are the unit a surface model consumes — GEMB runs one simulation per elevation class and
needs one `albedo_ice` for it — so `f` is piecewise constant at bin resolution and every value it
returns is traceable to a bin.
"""

# One albedo-versus-elevation regression. Named once so the struct field, the constructor and the
# degenerate-case fallback cannot spell the same type differently.
#
# `slope_per_km` is the plain derivative: **positive means albedo rises with height**, which on a
# glacier is the ablation-to-accumulation transition. This is the opposite convention to
# `derive_lapse_rate`, whose sign is flipped so that positive means cooling.
const AlbedoElevationFit = @NamedTuple{slope_per_km::Float64, intercept::Float64,
                                      slope_stderr::Float64, r2::Float64, n::Int,
                                      elevation_range::Tuple{Float64,Float64}}

# The fit that says "no fit was identifiable". Every field NaN rather than zero: a zero slope is a
# real, defensible result and must not be confused with the absence of one.
const _NO_ALBEDO_FIT = AlbedoElevationFit((NaN, NaN, NaN, NaN, 0, (NaN, NaN)))

"""
    BareIceAlbedoHyps

Bare-ice albedo binned by elevation, callable as `f(z)`.

Fields, all indexed by bin (`bin i` spans `[edges[i], edges[i+1])`):

- `edges`: bin edges (m).
- `albedo`: the best estimate — the observed mean where the bin holds at least `min_cells` cells,
  the fit evaluated at the bin centre where it does not, `NaN` where neither resolves.
- `observed`: the observed mean regardless of `min_cells`, `NaN` in an empty bin. Carried so a
  fit-filled bin can be checked against the handful of cells that were in it.
- `n_cells`: MCD43A3 cells in the bin.
- `n_valid`: MCD43A3 retrievals behind those cells, summed.
- `sources`: `:observed`, `:hold`, `:fit` or `:none`, per bin.
- `fit`: the [`AlbedoElevationFit`](@ref) over the individual cells.
- `sky`: `:bsa` or `:wsa`, which albedo this was built from.
- `n_input`, `n_used`, `n_no_albedo`, `n_no_elevation`: cells supplied, cells that survived, and
  the two reasons for dropping one.

See [`bare_ice_albedo_hyps`](@ref) to build one, and [`bare_ice_albedo_source`](@ref) /
[`bare_ice_albedo_table`](@ref) to inspect it.
"""
struct BareIceAlbedoHyps
    edges::Vector{Float64}
    albedo::Vector{Float64}
    observed::Vector{Float64}
    n_cells::Vector{Int}
    n_valid::Vector{Int}
    sources::Vector{Symbol}
    fit::AlbedoElevationFit
    sky::Symbol
    n_input::Int
    n_used::Int
    n_no_albedo::Int
    n_no_elevation::Int
end

# ------------------------------------------------------------------- normalising the inputs

"""
    _bia_hyps_albedo(a, sky) -> (albedo, n_valid, in_product)

The three per-cell vectors the reduction needs, from either a [`bare_ice_albedo`](@ref) stack or a
plain albedo vector.

A plain vector carries no quality companions, so `n_valid` is zero and every cell is treated as
in-product: the caller who passed raw numbers has already decided what to keep.
"""
function _bia_hyps_albedo(a, sky::Symbol)
    if a isa AbstractVector
        n = length(a)
        return (a, zeros(Int, n), trues(n))
    end
    key = Symbol("albedo_", sky)
    haskey(a, key) || throw(ArgumentError(
        "no :$(key) layer in the albedo input. Pass the DimStack `bare_ice_albedo` returns " *
        "(its layers are $(join(keys(a), ", "))), or a plain vector of albedo values."))
    nkey = Symbol("n_valid_", sky)
    n_valid = haskey(a, nkey) ? Int.(parent(a[nkey])) : zeros(Int, length(a[key]))
    in_product = haskey(a, :in_product) ? collect(parent(a[:in_product])) :
                 trues(length(a[key]))
    return (parent(a[key]), n_valid, in_product)
end

"""
    _bia_hyps_elevation(e, n) -> Vector{Float64}

`n` elevations as plain `Float64`, with anything unusable — `missing`, `nothing`, `NaN`, `Inf` —
mapped to `NaN`.

Accepts `Rasters.extract`'s output directly, whose elements are `NamedTuple`s
(`(; geometry, var"")` or `(; var"")` with `geometry = false`). The first field not named
`:geometry` is the sampled value, which holds whatever the raster's own layer name is — the
Copernicus DEM's is empty, so it cannot be requested by name.
"""
function _bia_hyps_elevation(e, n::Integer)
    length(e) == n || throw(ArgumentError(
        "got $(length(e)) elevations for $(n) albedo cells; they must be one-to-one and in the " *
        "same order. `Rasters.extract(dem, a.points)` preserves that order."))
    out = Vector{Float64}(undef, n)
    for (i, raw) in enumerate(e)
        out[i] = _bia_hyps_scalar(raw)
    end
    return out
end

_bia_hyps_scalar(::Missing) = NaN
_bia_hyps_scalar(::Nothing) = NaN
_bia_hyps_scalar(x::Real) = isfinite(x) ? Float64(x) : NaN
function _bia_hyps_scalar(x::NamedTuple)
    for k in keys(x)
        k === :geometry && continue
        return _bia_hyps_scalar(x[k])
    end
    throw(ArgumentError(
        "an elevation entry carries only a :geometry field, so it holds no sampled value. " *
        "Extract from a raster that covers the points."))
end

# ------------------------------------------------------------------------------- the fit

"""
    _bia_albedo_fit(elevation, albedo) -> AlbedoElevationFit

Ordinary least squares of `albedo` on `elevation`, over the cells themselves.

Two passes (means, then centred sums) rather than one over the raw power sums: elevation runs to
1e4 m so `Σz²` reaches 1e8 per cell, and the cancellation in `Σz² - (Σz)²/n` costs significant
digits exactly where the spread is small — which is the case the fit is least able to afford.

Returns [`_NO_ALBEDO_FIT`](@ref) when fewer than three cells survive or the elevation spread is
zero, so a caller can never mistake "no slope was identifiable" for "the slope is zero". The
spread test is scale-relative because elevation is in metres.

`r2` is `NaN` when the albedo is constant: the total sum of squares is then zero and the
proportion of it explained is undefined, though the slope (zero) and its zero standard error are
both exact.
"""
function _bia_albedo_fit(elevation::AbstractVector{Float64}, albedo::AbstractVector{Float64})
    n = length(elevation)
    n >= 3 || return _NO_ALBEDO_FIT

    z̄ = mean(elevation)
    ā = mean(albedo)
    Szz = 0.0
    Sza = 0.0
    Saa = 0.0
    for i in eachindex(elevation, albedo)
        dz = elevation[i] - z̄
        da = albedo[i] - ā
        Szz += dz * dz
        Sza += dz * da
        Saa += da * da
    end
    # Relative tolerance: `Szz` scales with n and with z², so an absolute floor would reject a
    # well-conditioned fit over a low-lying region.
    Szz > 1e-12 * n * (1 + z̄ * z̄) || return _NO_ALBEDO_FIT

    slope = Sza / Szz                      # albedo per metre
    intercept = ā - slope * z̄              # albedo at 0 m
    ss_res = max(0.0, Saa - slope * Sza)   # clamped: the subtraction can go slightly negative
    stderr = sqrt(ss_res / (n - 2) / Szz)
    r2 = Saa > 0 ? 1 - ss_res / Saa : NaN
    return AlbedoElevationFit((1000 * slope, intercept, 1000 * stderr, r2, n,
                              extrema(elevation)))
end

# ---------------------------------------------------------------------------- construction

"""
    bare_ice_albedo_hyps(a, elevation, bins = 0:100:10000; sky = :bsa, min_cells = 3)
        -> BareIceAlbedoHyps

Reduce per-cell bare-ice albedo to a function of elevation.

`f = bare_ice_albedo_hyps(...)` is callable: `f(z)` returns the best estimate of bare-ice albedo at
elevation `z` in metres, and `f.(zs)` broadcasts.

# Arguments
- `a`: the `DimStack` [`bare_ice_albedo`](@ref) returns, or a plain vector of albedo values. From a
  stack, cells with `in_product == false` are dropped — those are off the RGI 7.0 outlines, so
  they are not glacier at all.
- `elevation`: one elevation (m) per cell of `a`, in the same order. Takes
  `Rasters.extract(dem, a.points)` output directly, or any vector of reals, `missing`s included.
- `bins`: monotonically increasing bin edges (m); bin `i` spans `[bins[i], bins[i+1])`. The
  default matches the 100 m glacier hypsometry the elevation-class tables are built on, so albedo
  bins and area bins line up.

# Keywords
- `sky = :bsa`: `:bsa` (black-sky, directional-hemispherical) or `:wsa` (white-sky,
  bihemispherical). Black-sky is the default because that is what a surface energy-balance model
  wants.
- `min_cells = 3`: cells a bin needs before its own mean is used rather than the fit.
- `extrapolate = 0`: what to do about bins outside the sampled elevation range.
  - `:hold` — freeze the linear fit at the edges of the sampled range: a bin below the lowest
    observed bin takes `fit` evaluated at the lowest observed *elevation*, above the highest
    takes it at the highest. **Bounded**, because the fit is never extended past the data, and
    **not set by a thin end bin**, because the fit is informed by every observation. This is what
    a surface model should be handed. Needs an identifiable fit; without one such bins are
    `:none`.
  - a distance in metres — how far (m) beyond the observed elevation range a bin centre may lie and
  still be fit-filled. At the default a bin outside that range is `NaN`, because the fit is
  linear and unbounded: extending it kilometres past the highest observed cell returns albedos
  outside 0–1, and a negative best estimate is worse than none. Raise it to reach a little past
  the observed span; `fit.elevation_range` is what it is measured from.
- `n_valid = nothing`: retrieval counts per cell, overriding whatever `a` carries. A stack
  supplies its own `:n_valid_*`; pass this when `a` is a plain albedo vector, or the reported
  per-bin counts are zero and there is no way to tell a bin resting on three retrievals from one
  resting on three thousand.

# What `f(z)` returns

| bin holding `z`                              | `f(z)`                   | `sources[bin]` |
|:---------------------------------------------|:-------------------------|:---------------|
| at least `min_cells` cells                   | that bin's observed mean | `:observed`    |
| fewer, outside the span, `extrapolate = :hold` | fit at the sampled edge | `:hold`     |
| fewer, fit resolved, within `extrapolate` m | fit at the bin centre    | `:fit`         |
| fewer, and either of those fails             | `NaN`                    | `:none`        |
| `z` outside `bins`                           | `NaN`                    | —              |

Piecewise constant, not interpolated: a returned value is always some bin's mean or the fit at
some bin's centre, so it can be audited against the cells behind it.
[`bare_ice_albedo_source`](@ref) reports which case applied.

A cell's glacier spans a limited elevation range, and outside it there is no ice to have an
albedo — so `NaN` there is the answer, not a gap to be filled.

!!! warning "Above the ELA this is snow, and it is not clamped"
    The underlying statistic is the darkest 5 % of a cell's retrievals. Below the ELA that is bare
    ice; above it there is no bare ice to see and the darkest retrievals are still snow, so the
    profile rises past the `0.2 ≤ albedo_ice ≤ 0.6` range GEMB asserts. Nothing here clips it and
    no ELA is estimated: that is a real measurement of a real surface, and clipping it would hide
    where the statistic stopped describing ice. Decide the ceiling at the point of use.

# Example
```julia
using GEMB_ClimateForcing, Rasters
import GeoInterface as GI

poly = GI.Polygon([GI.LinearRing([(-141.05, 60.45), (-140.95, 60.45),
                                  (-140.95, 60.55), (-141.05, 60.55), (-141.05, 60.45)])])
a = bare_ice_albedo(poly)

dem = climate_model_invariant(model = :copernicus_dem_30m)
# Crop before extracting: sampling the lazy global mosaic point by point costs an HTTP read each.
window = read(view(dem, X = (-141.06 .. -140.94), Y = (60.44 .. 60.56)))
e = extract(window, bare_ice_albedo_points(a); geometry = false)

f = bare_ice_albedo_hyps(a, e)
f(1150)                                  # best estimate at 1150 m
bare_ice_albedo_source(f, 1150)          # :observed or :fit
f.fit.slope_per_km                       # albedo change per km of elevation
bare_ice_albedo_table(f)                 # the whole profile, per bin
```

See also [`bare_ice_albedo`](@ref) for the per-cell product this reduces.
"""
function bare_ice_albedo_hyps(a, elevation, bins = 0:100:10000;
                              sky::Symbol = :bsa, min_cells::Integer = 3,
                              extrapolate::Union{Real,Symbol} = 0,
                              n_valid::Union{Nothing,AbstractVector{<:Integer}} = nothing)
    sky in (:bsa, :wsa) || throw(ArgumentError(
        "sky must be :bsa (black-sky) or :wsa (white-sky), got :$(sky)"))
    min_cells >= 1 || throw(ArgumentError("min_cells must be >= 1, got $(min_cells)"))
    if extrapolate isa Symbol
        extrapolate === :hold || throw(ArgumentError(
            "extrapolate as a symbol must be :hold (freeze the fit at the sampled range's edges); " *
            "got :$(extrapolate). A number is a distance in metres over which the linear fit may " *
            "be extended instead."))
    else
        (extrapolate >= 0 && isfinite(extrapolate)) || throw(ArgumentError(
            "extrapolate must be :hold, or a finite non-negative distance in metres; got " *
            "$(extrapolate)"))
    end

    edges = collect(Float64, bins)
    length(edges) >= 2 || throw(ArgumentError(
        "bins needs at least two edges to make one bin, got $(length(edges))"))
    all(i -> edges[i] < edges[i + 1], 1:length(edges) - 1) || throw(ArgumentError(
        "bins must increase strictly; got edges $(edges[1]) … $(edges[end]) that do not. " *
        "`searchsortedlast` locates a value's bin, which needs a sorted, non-degenerate edge list."))

    albedo, counts, in_product = _bia_hyps_albedo(a, sky)
    n_input = length(albedo)
    if !isnothing(n_valid)
        length(n_valid) == n_input || throw(ArgumentError(
            "got $(length(n_valid)) retrieval counts for $(n_input) albedo cells; they must be " *
            "one-to-one and in the same order"))
        counts = n_valid
    end
    z = _bia_hyps_elevation(elevation, n_input)

    n_bins = length(edges) - 1
    sum_albedo = zeros(Float64, n_bins)
    sum_valid = zeros(Int, n_bins)
    n_cells = zeros(Int, n_bins)

    # The surviving cells, kept so the fit sees the cells rather than the bin means: fitting the
    # means would reweight the regression by how the bins happen to be populated.
    fit_z = Float64[]
    fit_a = Float64[]
    n_no_albedo = 0
    n_no_elevation = 0

    for i in 1:n_input
        alb = albedo[i]
        if !in_product[i] || ismissing(alb) || !isfinite(alb)
            n_no_albedo += 1
            continue
        end
        isnan(z[i]) && (n_no_elevation += 1; continue)
        push!(fit_z, z[i])
        push!(fit_a, Float64(alb))
        # `_bin_index` returns 0 outside the binning. The top edge is exclusive, so a value sitting
        # exactly on it belongs to no bin — it still informs the fit, being a real observation, just
        # outside the binning.
        b = _bin_index(z[i], edges, n_bins)
        b == 0 && continue
        sum_albedo[b] += Float64(alb)
        sum_valid[b] += counts[i]
        n_cells[b] += 1
    end

    fit = _bia_albedo_fit(fit_z, fit_a)
    slope_per_m = fit.slope_per_km / 1000

    observed = [n_cells[b] > 0 ? sum_albedo[b] / n_cells[b] : NaN for b in 1:n_bins]
    best = Vector{Float64}(undef, n_bins)
    sources = Vector{Symbol}(undef, n_bins)
    # Which bins carry their own mean. `:hold` freezes the fit at the outermost of these.
    observed_bins = [b for b in 1:n_bins if n_cells[b] >= min_cells]
    first_obs = isempty(observed_bins) ? 0 : first(observed_bins)
    last_obs = isempty(observed_bins) ? 0 : last(observed_bins)
    hold = extrapolate === :hold
    reach = hold ? 0.0 : Float64(extrapolate)
    # With a numeric `extrapolate`, a bin may be fit-filled only if its centre is within that
    # distance of the elevations that informed the fit. The fit is linear and unbounded, so filling
    # far outside that span returns albedos beyond 0-1, and a negative "best estimate" is worse than
    # no estimate. Both bounds are `NaN` when no fit was identifiable, which makes the comparison
    # false and lands on `:none` with no special case.
    zlo, zhi = fit.elevation_range
    for b in 1:n_bins
        centre = (edges[b] + edges[b + 1]) / 2
        if n_cells[b] >= min_cells
            best[b] = observed[b]
            sources[b] = :observed
        elseif hold && first_obs != 0 && isfinite(slope_per_m) && (b < first_obs || b > last_obs)
            # Outside the sampled span, hold the fit **evaluated at the nearest observed elevation**
            # — not that end bin's own mean, which rests on whatever few cells happened to fall in
            # it, and not the fit extended past the data, which is unbounded. The fit is informed by
            # every observation, and freezing it at the edge of the sampled range is what keeps the
            # value inside the range the observations actually cover.
            best[b] = fit.intercept + slope_per_m * (b < first_obs ? zlo : zhi)
            sources[b] = :hold
        elseif isfinite(slope_per_m) && zlo - reach <= centre <= zhi + reach
            best[b] = fit.intercept + slope_per_m * centre
            sources[b] = :fit
        else
            best[b] = NaN
            sources[b] = :none
        end
    end

    return BareIceAlbedoHyps(edges, best, observed, n_cells, sum_valid, sources, fit, sky,
                             n_input, length(fit_z), n_no_albedo, n_no_elevation)
end

# ------------------------------------------------------------------------------ evaluation

# The bin of `f` holding `z`, or 0 when `z` is outside the binning or not finite. The one place a
# profile's bin is located: the call operator, `bare_ice_albedo_source` and `resolve_albedo_ice` all
# go through it, so none of them can disagree about which bin a value is in.
@inline _bia_hyps_bin(f::BareIceAlbedoHyps, z::Real) =
    isfinite(z) ? _bin_index(z, f.edges, length(f.albedo)) : 0

function (f::BareIceAlbedoHyps)(z::Real)
    b = _bia_hyps_bin(f, z)
    return b == 0 ? NaN : f.albedo[b]
end

"""
    bare_ice_albedo_source(f, z) -> Symbol

Where `f(z)` came from: `:observed` (that bin's own cells), `:hold` (the fit frozen at the edge of
the sampled elevation range, because `z` is outside it and `extrapolate = :hold`), `:fit` (the
regression), or `:none` (none of those resolved, so `f(z)` is `NaN`).

`z` outside the bin edges also gives `:none`.
"""
function bare_ice_albedo_source(f::BareIceAlbedoHyps, z::Real)
    b = _bia_hyps_bin(f, z)
    return b == 0 ? :none : f.sources[b]
end

"""
    bare_ice_albedo_table(f) -> NamedTuple of vectors

`f`'s profile as columns — `lo`, `hi`, `center` (m), `albedo`, `observed`, `n_cells`, `n_valid`,
`source` — one row per bin.

A `NamedTuple` of equal-length vectors is a Tables.jl-compatible column table, so it can be
written or plotted without reaching into the struct's fields.
"""
function bare_ice_albedo_table(f::BareIceAlbedoHyps)
    n = length(f.albedo)
    lo = f.edges[1:n]
    hi = f.edges[2:n + 1]
    return (; lo, hi, center = (lo .+ hi) ./ 2, albedo = f.albedo, observed = f.observed,
            n_cells = f.n_cells, n_valid = f.n_valid, source = f.sources)
end

function Base.show(io::IO, ::MIME"text/plain", f::BareIceAlbedoHyps)
    n_bins = length(f.albedo)
    n_obs = count(==(:observed), f.sources)
    n_hold = count(==(:hold), f.sources)
    n_fit = count(==(:fit), f.sources)
    resolved = findall(isfinite, f.albedo)
    println(io, "BareIceAlbedoHyps(sky = :", f.sky, ", ", n_bins, " bins of ",
            round(f.edges[2] - f.edges[1]; digits = 1), " m)")
    println(io, "  cells:  ", f.n_used, " of ", f.n_input, " used (",
            f.n_no_albedo, " no albedo, ", f.n_no_elevation, " no elevation)")
    println(io, "  bins:   ", n_obs, " observed, ", n_hold, " held, ", n_fit, " fit-filled, ",
            n_bins - n_obs - n_hold - n_fit, " unresolved")
    if isempty(resolved)
        println(io, "  albedo: none resolved")
    else
        println(io, "  albedo: ", round(minimum(f.albedo[resolved]); digits = 3), " … ",
                round(maximum(f.albedo[resolved]); digits = 3), " over ",
                round(f.edges[first(resolved)]; digits = 0), " … ",
                round(f.edges[last(resolved) + 1]; digits = 0), " m")
    end
    if f.fit.n == 0
        print(io, "  fit:    none identifiable")
    else
        print(io, "  fit:    ", round(f.fit.slope_per_km; digits = 4), " ± ",
              round(f.fit.slope_stderr; digits = 4), " per km, r2 = ",
              round(f.fit.r2; digits = 3), ", n = ", f.fit.n, ", observed over ",
              round(f.fit.elevation_range[1]; digits = 0), " … ",
              round(f.fit.elevation_range[2]; digits = 0), " m")
    end
end
