# Turning fitted downscaling parameters into applied ones.
#
# `derive_decoupling_factor` and `derive_lapse_rate` *measure*: they report what a tile's forcing
# supports and `NaN` where it supports nothing, substituting and clamping nothing. Forcing a glacier
# needs the opposite — a finite, in-domain value at every timestep, because a single `NaN` propagates
# into a band's temperature and the whole band then fails `forcing_is_complete` and is skipped as
# silently as an ocean cell. This file is the crossing between the two, and it is the one place a
# substitution happens.
#
# What it replaces is a fill-and-clamp: one constant for every unmeasured timestep, and every
# out-of-domain fit dragged to the nearest bound. That loses the distinction the numbers actually turn
# on. A tile like N60_W142 fits `k` at 29% of timesteps, and the other 71% are not failures — `k` is
# identifiable only when there is a warm ambient excess to damp, and outside the melt season there is
# none. Reporting those as "filled" understates how much of the forcing was measured; reporting a
# clamped `k` of 3.2 as an applied 1.0 overstates it.
#
# So every applied value carries a source label, and nothing is clamped: a fit is accepted only if it
# is already inside the domain its consumer validates, and every substitute is in-domain by
# construction. An applied value outside the domain is therefore a bug in this file, and is asserted
# rather than repaired.
#
# The other thing this crossing does is decouple the fit window from the run window. The fits are a
# pooled cross-cell regression, so they cannot be appended to and re-deriving them over a longer record
# costs a full pass over every cell's forcing. Reducing the accepted fits to a month-by-hour median
# field makes them applicable to any window, and keeps the two structures that are physical rather than
# noise: the melt-season lapse-rate minimum and the nocturnal inversion.

"""
    DOWNSCALING_SOURCES

Where an applied downscaling parameter came from, in decreasing directness. The source of every
applied value is recorded per timestep, so how much of a band's forcing was measured is a count rather
than an assumption.

- `:fitted` — this timestep's own fit, accepted as measured.
- `:held` — this timestep's fit, evaluated at the highest elevation where the tile's ambient warm
  excess still supports it rather than at the band centre. Only `k` can be held. It is a fit, not a
  substitute, and it is preferred to one because reverting to the identity above the ceiling would put
  a step in the vertical temperature profile — and a gradient in the forcing is exactly what the
  downstream sweep reads as a mass-balance signal.
- `:climatology` — the median of accepted fits for this month and hour of day.
- `:prior` — an external estimate independent of this tile's forcing: a published `k`
  ([`decoupling_factor_prior`](@ref)) or a regional lapse-rate cycle.
- `:ambient` — no source at all; the identity. For `k` that is 1.0, which leaves the forcing ambient.
"""
const DOWNSCALING_SOURCES = (:fitted, :held, :climatology, :prior, :ambient)

const _SOURCE_FITTED = Int8(1)
const _SOURCE_HELD = Int8(2)
const _SOURCE_CLIMATOLOGY = Int8(3)
const _SOURCE_PRIOR = Int8(4)
const _SOURCE_AMBIENT = Int8(5)

# How narrow a physically credible lapse rate is, against the much wider range
# `climate_adjust_for_elevation` merely validates against ($(_LAPSE_RATE_LIMITS) K/km). The validator's
# job is to catch a units mistake; this one's is to catch a fit that is arithmetically fine and
# physically absent. The upper bound is above the dry adiabatic rate (9.8 K/km) because a fit is a
# regression across cells at one instant, not a sounding, and clears it legitimately on clear summer
# afternoons; the lower bound admits the nocturnal and polar inversions that are real (the Antarctic
# Peninsula tiles fit -3 K/km at the 5th percentile) while rejecting the runaway slopes an
# out-of-domain `k` produces.
const APPLIED_LAPSE_RATE_WINDOW = (-5.0, 12.0)

# Fewest metres of elevation spread among the fitted cells before a slope is believed. `derive_lapse_rate`
# reports `elevation_spread` precisely because it matters far more than the cell count: eight cells
# spanning 40 m constrain a slope no better than two do.
const APPLIED_ELEVATION_SPREAD_MINIMUM = 300.0

"""
    hypsometry_intervals(grid_cells) -> Vector{@NamedTuple{lo::Int, hi::Int, center::Float64}}

The union of populated elevation intervals across a set of glacier elevation-class rows, ascending.

These are the bands a tile is forced on: an interval appears if *any* cell holds glacier area in it,
and the decoded `hyps_<lo>_<hi>` column names are authoritative — the table's `hypsometry_bin_edges`
metadata does not survive a GeoParquet round trip, so [`glacier_hypsometry`](@ref) is the decoder to go
through.

Which intervals are populated, not how much area is in them. The area an interval ends up carrying is
accumulated over the cells whose forcing was actually usable, and a total summed here would disagree
with it whenever a cell drops out.
"""
function hypsometry_intervals(grid_cells)
    seen = Set{Tuple{Int,Int}}()
    for row in eachrow(grid_cells)
        for b in glacier_hypsometry(row; area_minimum = 0)
            push!(seen, (b.lo, b.hi))
        end
    end
    return [(; lo, hi, center = (lo + hi) / 2) for (lo, hi) in sort!(collect(seen))]
end

"""
    AppliedDownscaling

Downscaling parameters resolved onto a run's time axis, ready to apply, with the provenance of every
value.

- `time`: the run time axis these are defined on. Not necessarily the fit's — see
  [`resolve_downscaling`](@ref).
- `lapse_rate`: one applied lapse rate (K/km) per timestep, inside $(_LAPSE_RATE_LIMITS). The lapse
  rate is a property of the tile's air column rather than of a band, so there is one series.
- `lapse_rate_source`: the [`DOWNSCALING_SOURCES`](@ref) code behind each of those.
- `bands`: one entry per elevation interval, ascending, each
  `(; lo, hi, center, decoupling_factor, decoupling_factor_source, n_fit_in_domain, n_fit_held)`.
  `decoupling_factor` is `k` at that band's centre per timestep, inside `(0, 1]`. The two counts
  describe the *fit* series the band was resolved from, not the applied one.
- `basis`: `:climatology` or `:fitted`, as requested.
- `settings`: the thresholds and priors this resolution used, for the record.

Use [`downscaling_source_counts`](@ref) to summarize a source vector, and pass it the band's
above-freezing mask to get the count that matters — `k` is arbitrary at a timestep with no warm excess
for it to act on, so an unqualified "71% substituted" is not the honest number.
"""
struct AppliedDownscaling{B}
    time::Vector{DateTime}
    lapse_rate::Vector{Float64}
    lapse_rate_source::Vector{Int8}
    bands::B
    basis::Symbol
    settings::Dict{String,Any}
end

Base.show(io::IO, a::AppliedDownscaling) = print(io,
    "AppliedDownscaling(", length(a.time), " timesteps, ", length(a.bands), " bands, basis=",
    a.basis, ")")

"""
    downscaling_source_counts(source [, mask]) -> Dict{Symbol,Int}

How many entries of a [`DOWNSCALING_SOURCES`](@ref) code vector came from each source, as a dict keyed
by source name with every source present (zero where unused), so two reports are comparable.

`mask` restricts the count to the timesteps where the parameter actually changed the forcing. For `k`
that is where the band was above freezing: the decoupling correction scales
`max(T - 273.15, 0)`, so at a below-freezing timestep every value of `k` produces bit-identical
forcing and counting those as substitutions describes the fit rather than the run.
"""
function downscaling_source_counts(source)
    counts = Dict{Symbol,Int}(name => 0 for name in DOWNSCALING_SOURCES)
    for code in source
        counts[DOWNSCALING_SOURCES[code]] += 1
    end
    return counts
end

function downscaling_source_counts(source, mask)
    axes(source) == axes(mask) || throw(DimensionMismatch(
        "source $(axes(source)) and mask $(axes(mask)) must share indices"))
    counts = Dict{Symbol,Int}(name => 0 for name in DOWNSCALING_SOURCES)
    for i in eachindex(source, mask)
        mask[i] && (counts[DOWNSCALING_SOURCES[source[i]]] += 1)
    end
    return counts
end

"""
    decoupling_factor_prior(grid_cells; max_distance = 10.0) -> Float64 or nothing

A tile's prior on-glacier temperature decoupling factor `k`, as the glacier-area weighted mean of the
published Shaw et al. (2025) factor over its cells, or `nothing` when none of them has one.

This is the `:prior` tier of [`resolve_downscaling`](@ref): an estimate that does not depend on the
tile's own forcing, for the timesteps that forcing cannot constrain. The **published** factor is
averaged, not the `glm`-weighted one [`cell_decoupling_factor`](@ref) returns, because the `glm`
weighting is a property of the donor cell and is applied per cell where the forcing is assembled
(`_effective_decoupling_factor`). Weighting it in twice would damp the correction toward the identity.

`nothing` is the normal outcome in RGI regions 05 (Greenland periphery) and 19 (Antarctic and
Subantarctic), which are absent from the published table entirely.
"""
function decoupling_factor_prior(grid_cells; max_distance::Real = 10.0)
    total = 0.0
    weighted = 0.0
    for row in eachrow(grid_cells)
        found = cell_decoupling_factor(row; max_distance)
        found === nothing && continue
        k = Float64(found.decoupling_factor_published)
        isfinite(k) && 0 < k <= 1 || continue
        # Cells with no ice cannot weight an average, but they can still carry a lookup; fall back to
        # counting them equally rather than dropping the only evidence the tile has.
        w = max(glacier_area_total(row), 0.0)
        w = w > 0 ? w : 1.0
        total += w
        weighted += w * k
    end
    total > 0 || return nothing
    return weighted / total
end

"""
    resolve_downscaling(fit, intervals, run_time; kwargs...) -> AppliedDownscaling

Resolve a tile's raw fitted downscaling parameters into applied ones on `run_time`, with a source label
for every value.

`fit` is either what [`derive_downscaling_parameters`](@ref) returns or what
[`read_downscaling_tile`](@ref) reads back — both carry `time` plus the `decoupling` and `lapse_rate`
`DimStack`s, so a stored tile and a freshly derived one resolve identically. `intervals` comes from
[`hypsometry_intervals`](@ref). `run_time` is the time axis of the forcing the run will actually use,
which need not be the fit's.

# The lapse rate

A timestep's own fit is accepted when all of these hold, and used as measured:

1. it is finite and its cell count reaches `min_cells`;
2. `elevation_spread` reaches `spread_minimum` — the diagnostic that says whether the cells span enough
   elevation to constrain a slope at all;
3. it lies inside `lapse_rate_window`, a physical range rather than the wider one
   `climate_adjust_for_elevation` validates against;
4. **the same timestep's `k` is not out of domain.** `derive_lapse_rate` consumes `k` to bring the cells
   to a common on-glacier state, so a `k` outside `(0, 1]` corrupts that timestep's slope through a
   correction proportional to `k - 1`. Because that correction carries a `1 - glm` weight and `glm`
   rises with elevation, the damage is a smooth false gradient rather than scatter: at Wrangell it took
   a 7.0 K/km slope to 46.1, and ordinary least squares, Theil–Sen and Huber all returned 46.1. No
   robust estimator removes it, so the only defence is to drop the timestep.

   A `NaN` `k` is *not* a rejection. `derive_lapse_rate` treats it as `k = 1` locally, the bit-exact
   no-op, so that timestep's slope is the ambient one and is sound. This is why the stored `fit_usable`
   flag is not the screen used here: it is 1 only where `k` is in domain, so screening on it alone
   would also discard every sound cold-season slope — on N60_W142 that is the difference between
   keeping 17,159 fits and keeping 4,777.

Anything else falls to the median of accepted fits for its month and hour of day, then for its month,
then over the whole record, and only failing all three to `lapse_rate_prior`.

# The decoupling factor

`k` is re-evaluated at **each band's own centre** with [`decoupling_factor_at_elevation`](@ref), not
taken once at the tile's mean reanalysis elevation: the fit carries a `glm × z` interaction, so `k` is a
function of elevation, and the bands are the glacier, which sits systematically above the reanalysis
surface. A fit at a band is accepted when it is finite and inside `(0, 1]`, and is labelled `:held`
rather than `:fitted` where the tile's ambient warm excess ran out below the band and the evaluation
elevation was held down to where it did not.

Everything else falls to the month-and-hour median of accepted in-domain fits at that band, then the
month median, then the whole-record median, then `decoupling_factor_prior`, and finally to 1.0.
Out-of-domain fits are excluded from those medians rather than clamped into them: a fitted `k` of 3.2
warms the glacier surface instead of damping it, and the median of the fits that *are* in domain is a
better estimate of that month than the bound it would be dragged to.

# Basis

`basis = :climatology` (the default) applies the month-by-hour medians throughout, so a fit window
shorter than the run window is not an obstacle. This is the normal case: the fits are a pooled
cross-cell regression and cannot be appended to, so a decades-long run over a two-year fit window would
otherwise have to re-derive every tile at full record length. Nothing is labelled `:fitted` under this
basis — every value is a climatology by construction — and the label then distinguishes a populated
month-hour cell from one that fell through to a prior.

`basis = :fitted` uses each timestep's own fit where it is accepted, with the same climatology as its
fallback. It requires `run_time` to equal the fit's time axis, and throws if it does not, rather than
silently reverting to the climatology.

# Keywords
- `basis = :climatology`: as above.
- `min_cells = $(_MIN_CELLS_DEFAULT)`, `spread_minimum = $APPLIED_ELEVATION_SPREAD_MINIMUM`,
  `lapse_rate_window = $APPLIED_LAPSE_RATE_WINDOW`: the lapse-rate acceptance tests.
- `lapse_rate_prior = $(_DEFAULT_LAPSE_RATE)`: a scalar or a 12-element monthly cycle (January first),
  used where no fit and no climatology exists. `GREENLAND_LAPSE_RATE`, `ARCTIC_LAPSE_RATE` and
  `ANTARCTICA_LAPSE_RATE` from `GEMB_ClimateForcing` are the published cycles to pass here when the
  tile's region is known; no region is inferred, because the elevation-class table does not carry one.
- `decoupling_factor_prior = nothing`: the prior `k`, typically from
  [`decoupling_factor_prior`](@ref). `nothing` means an unresolvable timestep is left ambient.

Every applied value is inside the domain its consumer validates, by construction rather than by
clamping, and that is asserted before returning. A failure there is a bug here, not bad data.
"""
function resolve_downscaling(fit, intervals, run_time;
                             basis::Symbol = :climatology,
                             min_cells::Int = _MIN_CELLS_DEFAULT,
                             spread_minimum::Real = APPLIED_ELEVATION_SPREAD_MINIMUM,
                             lapse_rate_window = APPLIED_LAPSE_RATE_WINDOW,
                             lapse_rate_prior = _DEFAULT_LAPSE_RATE,
                             decoupling_factor_prior = nothing)
    basis in (:climatology, :fitted) ||
        throw(ArgumentError("basis must be :climatology or :fitted, got :$basis"))
    run_time = _plain_times(run_time)
    isempty(run_time) && throw(ArgumentError("run_time is empty; there is nothing to resolve onto"))

    lo_w, hi_w = Float64(lapse_rate_window[1]), Float64(lapse_rate_window[2])
    lo_w < hi_w || throw(ArgumentError(
        "lapse_rate_window must be (low, high) with low < high, got $lapse_rate_window"))
    _within(lo_w, hi_w, _LAPSE_RATE_LIMITS) || throw(ArgumentError(
        "lapse_rate_window $lapse_rate_window must sit inside $(_LAPSE_RATE_LIMITS) K/km, the range " *
        "`climate_adjust_for_elevation` validates against — a window wider than the validator's " *
        "would accept a fit that is then rejected at the point of use"))
    spread_minimum >= 0 ||
        throw(ArgumentError("spread_minimum must be >= 0 m, got $spread_minimum"))

    lapse_prior = _monthly_prior(lapse_rate_prior, "lapse_rate_prior", _LAPSE_RATE_LIMITS;
                                 open_lower = false)
    # `k`'s domain is open at zero, so a prior of exactly 0 is as inapplicable as one of 1.5 — it would
    # be rejected by `climate_adjust_for_glacier` at the point of use.
    k_prior = _scalar_prior(decoupling_factor_prior, "decoupling_factor_prior",
                            _DECOUPLING_FACTOR_LIMITS; open_lower = true)

    fit_time = _plain_times(fit.time)
    if basis === :fitted && fit_time != run_time
        throw(ArgumentError(
            "basis = :fitted needs the fit and the run on one time axis, but the fit covers " *
            _time_span(fit_time) * " and the run covers " * _time_span(run_time) * ". Re-derive " *
            "the fits over the run window, or use basis = :climatology, which reduces the fits to " *
            "a month-by-hour median field and so applies to any window."))
    end

    k_reference = collect(Float64, fit.decoupling.decoupling_factor)
    lapse = collect(Float64, fit.lapse_rate.lapse_rate)
    lapse_cells = collect(Int, fit.lapse_rate.n_cells)
    spread = collect(Float64, fit.lapse_rate.elevation_spread)

    # The lapse-rate acceptance screen, including the joint one against `k`.
    accepted = [isfinite(lapse[t]) && lapse_cells[t] >= min_cells &&
                isfinite(spread[t]) && spread[t] >= spread_minimum &&
                lo_w <= lapse[t] <= hi_w && _lapse_uncorrupted(k_reference[t])
                for t in eachindex(lapse)]

    lapse_rate, lapse_rate_source =
        _resolve_series(fit_time, lapse, accepted, run_time, basis, lapse_prior)
    _assert_applicable(lapse_rate, _LAPSE_RATE_LIMITS, "lapse rate"; open_lower = false)

    # The hypsometry span, as the validity range `k` is evaluated within. Every band centre is inside it
    # by construction — the bands *are* the hypsometry — so it is there to guarantee that a band can
    # never itself be the thing that produces a `NaN` factor.
    z_range = isempty(intervals) ? nothing :
              (Float64(first(intervals).lo), Float64(last(intervals).hi))

    alpha = collect(Float64, fit.decoupling.coef_alpha)
    beta = collect(Float64, fit.decoupling.coef_beta)

    bands = map(intervals) do interval
        raw = decoupling_factor_at_elevation(fit.decoupling, interval.center;
                                             elevation_range = z_range)
        kz = collect(Float64, raw)
        in_domain = [isfinite(x) && _DECOUPLING_FACTOR_LIMITS[1] < x <= _DECOUPLING_FACTOR_LIMITS[2]
                     for x in kz]
        # Which timesteps were evaluated at a held-down elevation rather than at the band centre.
        # `decoupling_factor_at_elevation` reports only how many, and the label needs to know which, so
        # the same condition is re-formed here from the coefficients it holds: the ambient warm excess
        # `alpha + beta*z` is what its denominator thresholds.
        held = [isfinite(alpha[t]) && alpha[t] + beta[t] * interval.center < _MIN_AMBIENT_EXCESS
                for t in eachindex(kz)]

        factor, source = _resolve_series(fit_time, kz, in_domain, run_time, basis, k_prior;
                                         held, ambient = 1.0)
        _assert_applicable(factor, _DECOUPLING_FACTOR_LIMITS,
                           "decoupling factor at $(interval.center) m"; open_lower = true)

        (; interval.lo, interval.hi, interval.center,
         decoupling_factor = factor,
         decoupling_factor_source = source,
         n_fit_in_domain = count(in_domain),
         n_fit_held = count(i -> in_domain[i] && held[i], eachindex(in_domain)))
    end

    settings = Dict{String,Any}(
        "downscaling_basis" => string(basis),
        "downscaling_min_cells" => min_cells,
        "downscaling_elevation_spread_minimum" => Float64(spread_minimum),
        "downscaling_lapse_rate_window" => [lo_w, hi_w],
        "downscaling_lapse_rate_prior" => lapse_prior,
        "downscaling_decoupling_factor_prior" =>
            k_prior === nothing ? "none" : Float64(k_prior),
        "downscaling_fit_time_coverage_start" => isempty(fit_time) ? "none" : string(first(fit_time)),
        "downscaling_fit_time_coverage_end" => isempty(fit_time) ? "none" : string(last(fit_time)),
        "downscaling_n_fit_timesteps" => length(fit_time),
        "downscaling_n_lapse_rate_accepted" => count(accepted),
    )

    return AppliedDownscaling(run_time, lapse_rate, lapse_rate_source, bands, basis, settings)
end

# Whether a timestep's `k` leaves that timestep's lapse-rate fit uncorrupted. `NaN` does: the fit
# treats it as `k = 1`, the exact no-op, so the slope is the ambient one. A finite value outside
# `(0, 1]` does not — see `resolve_downscaling`'s docstring.
_lapse_uncorrupted(k) = !isfinite(k) ||
    (_DECOUPLING_FACTOR_LIMITS[1] < k <= _DECOUPLING_FACTOR_LIMITS[2])

_within(lo, hi, (limit_lo, limit_hi)) = limit_lo <= lo && hi <= limit_hi

# A time axis as a phrase for an error message, with the empty case named rather than indexed.
_time_span(t) = isempty(t) ? "no timesteps" :
                "$(first(t)) .. $(last(t)) ($(length(t)) steps)"

# A time axis as a plain, 1-based `Vector{DateTime}`.
#
# Filled element by element rather than with `collect` or a comprehension, both of which go through
# `similar` and so preserve a `DimArray` or `Ti` container. That container is the problem: everything
# resolved against this axis is a plain vector, and `DimArray` axes compare unequal to theirs. The
# 1-based counter is the intent here, not an assumption about the input's own indices.
function _plain_times(t)
    out = Vector{DateTime}(undef, length(t))
    for (i, x) in enumerate(t)
        out[i] = x
    end
    return out
end

# Resolve one raw fitted series onto `run_time`, returning the applied values and their source codes.
#
# Shared by the lapse rate and by each band's `k`, because the hierarchy is the same for both and a
# second copy would let the two drift: the whole point of the labels is that they mean the same thing
# in both reports.
function _resolve_series(fit_time, values, accepted, run_time, basis, prior;
                         held = nothing, ambient = nothing)
    climatology = _cyclic_median(fit_time, values, accepted)

    n = length(run_time)
    # Under `:fitted` the run index *is* the fit index, which `resolve_downscaling` established by
    # comparing the two axes. Stated here as well because it is the one place the two are indexed by
    # the same variable, and an off-by-one there would pair a timestep with another timestep's fit.
    if basis === :fitted && length(accepted) != n
        error("basis = :fitted with $(length(accepted)) fitted timesteps against $n run " *
              "timesteps; the two axes were supposed to be equal")
    end
    applied = Vector{Float64}(undef, n)
    source = Vector{Int8}(undef, n)

    for i in eachindex(run_time, applied, source)
        t = run_time[i]

        # A timestep's own fit, only when the caller asked for that basis and the fit was accepted.
        if basis === :fitted && accepted[i]
            applied[i] = values[i]
            source[i] = (held !== nothing && held[i]) ? _SOURCE_HELD : _SOURCE_FITTED
            continue
        end

        v = _cyclic_lookup(climatology, t)
        if isfinite(v)
            applied[i] = v
            source[i] = _SOURCE_CLIMATOLOGY
            continue
        end

        p = _prior_at(prior, t)
        if p !== nothing
            applied[i] = p
            source[i] = _SOURCE_PRIOR
            continue
        end

        ambient === nothing && error(
            "no fit, no climatology and no prior for $t, and this parameter has no identity value " *
            "to fall back to. A lapse-rate prior is always available, so reaching this is a bug.")
        applied[i] = ambient
        source[i] = _SOURCE_AMBIENT
    end

    return applied, source
end

# Medians of the accepted entries of a series, by month and hour of day, with the coarser reductions
# each cell falls back to.
#
# Month *and* hour because both structures are physical rather than fit noise, and a single constant
# per tile would have to stand in for both: melt-season lapse rates over ice are shallower than winter
# ones, and where nocturnal inversions form they flatten or reverse the slope entirely. The median
# rather than the mean because the tail of these fits is one-sided and drags a mean badly — against a
# synthetic truth of 0.930 the mean returns 0.875 and the median 0.926.
struct _CyclicMedian
    month_hour::Matrix{Float64}      # 12 x 24, NaN where no accepted fit landed
    month::Vector{Float64}           # 12
    overall::Float64
end

function _cyclic_median(times, values, accepted)
    axes(times) == axes(values) == axes(accepted) || throw(DimensionMismatch(
        "times $(axes(times)), values $(axes(values)) and accepted $(axes(accepted)) must share " *
        "indices"))

    by_month_hour = [Float64[] for _ in 1:12, _ in 0:23]
    by_month = [Float64[] for _ in 1:12]
    all_values = Float64[]

    for i in eachindex(times, values, accepted)
        accepted[i] || continue
        v = values[i]
        m, h = month(times[i]), hour(times[i])
        push!(by_month_hour[m, h + 1], v)
        push!(by_month[m], v)
        push!(all_values, v)
    end

    return _CyclicMedian(_median_or_nan.(by_month_hour), _median_or_nan.(by_month),
                         _median_or_nan(all_values))
end

_median_or_nan(v) = isempty(v) ? NaN : Statistics.median(v)

# The finest populated reduction for this instant, or `NaN` when the series had no accepted fit at all.
function _cyclic_lookup(c::_CyclicMedian, t::DateTime)
    v = c.month_hour[month(t), hour(t) + 1]
    isfinite(v) && return v
    v = c.month[month(t)]
    isfinite(v) && return v
    return c.overall
end

# A prior is either absent, a scalar, or a monthly cycle indexed by calendar month.
_prior_at(prior::Nothing, ::DateTime) = nothing
_prior_at(prior::Real, ::DateTime) = Float64(prior)
_prior_at(prior::AbstractVector, t::DateTime) = Float64(prior[month(t)])

# Validate a prior that may be a scalar or a 12-element monthly cycle. Priors are *applied*, so an
# out-of-domain one is rejected here rather than at the point of use, where it would surface as a
# units-validation stacktrace several frames from its cause.
function _monthly_prior(prior, name, limits; open_lower::Bool)
    if prior isa Real
        _check_prior(Float64(prior), name, limits; open_lower)
        return Float64(prior)
    end
    length(prior) == 12 || throw(ArgumentError(
        "$name must be a scalar or a 12-element monthly cycle ordered January to December, got " *
        "length $(length(prior))"))
    values = collect(Float64, prior)
    for v in values
        _check_prior(v, name, limits; open_lower)
    end
    return values
end

function _scalar_prior(prior, name, limits; open_lower::Bool)
    prior === nothing && return nothing
    _check_prior(Float64(prior), name, limits; open_lower)
    return Float64(prior)
end

function _check_prior(v, name, limits; open_lower::Bool)
    lo, hi = limits
    ok = isfinite(v) && (open_lower ? lo < v : lo <= v) && v <= hi
    ok || throw(ArgumentError(
        "$name is $v, outside $(open_lower ? "($lo, $hi]" : "[$lo, $hi]") — a prior is applied, so " *
        "it has to be applicable"))
    return nothing
end

# Every applied value must already lie in the domain its consumer validates. Asserted rather than
# clamped: a fit is accepted only when it is in domain, and every substitute is in domain by
# construction, so a violation here is a bug in the resolution above and not something to repair
# silently. `k`'s domain is open at zero, hence the two comparisons.
function _assert_applicable(values, limits, what; open_lower::Bool)
    lo, hi = limits
    for (i, v) in pairs(values)
        ok = isfinite(v) && (open_lower ? lo < v : lo <= v) && v <= hi
        ok || error("resolved $what is $v at index $i, outside the applicable domain $limits; " *
                    "this is a bug in `resolve_downscaling`, not a property of the fits")
    end
    return nothing
end
