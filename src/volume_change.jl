# Surface height change and volume change from a GEMB run.
#
# GEMB is an Eulerian window on the firn: it holds the column at a fixed depth and reports the basal
# flux that does so. Height change is therefore not an output layer — it is that flux, negated
# (`GEMB`'s `trim_bottom!`):
#
#     -cumsum(ice_flux) = (SMB - ΔW) / density_ice + Δ(firn_air_content) - cumsum(strain_thinning)
#
# which is the standard altimetry decomposition of elevation change into a mass term and a compaction
# term. `SMB = precipitation + evaporation_condensation + blowing_snow - runoff`; `ΔW` is the change in
# column liquid storage, which matters on any melting band because water in a pore adds mass without
# adding thickness. Refreezing is internal and does not appear.
#
# Height change is **not** a proxy for SMB and the two can carry opposite signs: at an accumulating
# synthetic site GEMB reports SMB of +0.135 m ice/yr against an elevation trend of -0.157 m/yr, because
# compaction lowers the surface faster than accumulation raises it. So mass totals are assembled from
# the flux layers and volume totals from the height change, never one from the other.

# m² per km². Band areas are stored in km² (the elevation-class table's unit) and the mass fluxes in
# kg m-2, so every aggregation crosses this once.
const _M2_PER_KM2 = 1e6

# kg per Gt.
const _KG_PER_GT = 1e12

# km³ per (km² × m). A metre of ice over a square kilometre is a thousandth of a cubic kilometre —
# the conversion the existing 2° geotile products fold into their `mie2cubickm` factor.
const _KM3_PER_KM2_M = 1e-3

"""
    surface_height_change(output) -> Vector{Float64}

Surface elevation change (m) over a `gemb` run, against a datum fixed in the ice at the **start of the
run**.

This is `-cumsum(output[:ice_flux])`. `ice_flux` is the basal flux the fixed-depth column performs, as
an interval sum, so its cumulative sum is the cumulative flux exactly and its negation is the surface
elevation change that flux compensates for. Element `i` is therefore the change from the start of the
run to output time `i`, and no element is zero — the first already carries the first output interval.

Subtract element 1 for an anomaly referenced to the first output time, which is what
[`height_change_components`](@ref) does and what an altimetry comparison wants.

GEMB carries no ice dynamics unless `horizontal_strain_rate` is set, so with the default this is the
surface-mass-balance-plus-firn component of elevation change and excludes flux divergence, basal melt
and bedrock motion.
"""
function surface_height_change(output)
    return -cumsum(_height_layer(output, :ice_flux))
end

"""
    height_change_components(output) -> (; total, mass, water, firn, strain, residual)

Surface elevation change (m) over a `gemb` run, split into the terms that produce it. Every series is
an anomaly referenced to the **first output time**, so all six are zero at element 1 and

    total == mass + water + firn + strain + residual

- `total`: `-cumsum(ice_flux)`, re-referenced. The quantity an altimeter measures.
- `mass`: cumulative surface mass balance divided by `density_ice`.
- `water`: minus the change in column liquid storage, divided by `density_ice`. Water held in pores
  adds mass without adding thickness, so it belongs to the budget but not to the height.
- `firn`: change in firn air content, i.e. the compaction term.
- `strain`: minus the cumulative `strain_thinning`, zero unless `horizontal_strain_rate` is set.
- `residual`: what the other four fail to explain. It is the numerical check on the run, and should sit
  at the rounding floor — a growing residual means the column's base is no longer at ice density (see
  [`column_reaches_ice_density`](@ref)) and the height series is under-counting compaction.

Referencing to the first output time is what makes the closure exact rather than approximate. The
alternative — comparing raw `-cumsum(ice_flux)` against `firn_air_content .- firn_air_content[1]` —
compares a quantity accumulated from the run start against one differenced from the first output time,
which leaves a half-interval offset that has to be tolerated instead of checked.

Firn air content is recomputed here from the instantaneous `dz`/`density` profile rather than read from
the `firn_air_content` layer, which GEMB reports as an interval **mean** while `ice_flux` is an interval
**sum**. Mixing the two reductions is the other half of that same offset.
"""
function height_change_components(output)
    density_ice = _output_density_ice(output)

    flux = _height_layer(output, :ice_flux)
    dz = _profile_layer(output, :dz)
    density = _profile_layer(output, :density)
    water = _profile_layer(output, :water)

    axes(dz) == axes(density) == axes(water) || throw(DimensionMismatch(
        "the dz, density and water profiles must share indices: $(axes(dz)), $(axes(density)), " *
        "$(axes(water))"))
    axes(dz, 2) == axes(flux, 1) || throw(DimensionMismatch(
        "the profile time axis $(axes(dz, 2)) does not match the ice_flux axis $(axes(flux, 1))"))

    # Instantaneous whole-column firn air content and liquid storage, per output time. `Z` is a bare
    # cell index and the profile is top-justified with every row populated, so the whole column is the
    # whole slice.
    fac = [firn_air_content(view(dz, :, i), view(density, :, i), density_ice) for i in axes(dz, 2)]
    stored = [sum(view(water, :, i)) for i in axes(water, 2)]

    # `blowing_snow` and `strain_thinning` are zero at GEMB's defaults, so a run configured without
    # them contributes nothing here rather than needing a different formula.
    smb = _height_layer(output, :precipitation) .+
          _height_layer(output, :evaporation_condensation) .+
          _optional_height_layer(output, :blowing_snow, flux) .-
          _height_layer(output, :runoff)

    total = _anomaly(-cumsum(flux))
    mass = _anomaly(cumsum(smb)) ./ density_ice
    liquid = -_anomaly(stored) ./ density_ice
    firn = _anomaly(fac)
    strain = -_anomaly(cumsum(_optional_height_layer(output, :strain_thinning, flux)))
    residual = total .- mass .- liquid .- firn .- strain

    return (; total, mass, water = liquid, firn, strain, residual)
end

# A series re-referenced to its first element, which is the convention every component of
# `height_change_components` is reported on.
_anomaly(v) = v .- first(v)

"""
    column_reaches_ice_density(output; tolerance = 1.0) -> Bool

Whether the deepest cell of the firn column has reached `density_ice` (within `tolerance` kg m-3) at
the end of the run.

[`surface_height_change`](@ref) removes mass at the bottom cell's own density, so its datum is a column
whose material below the model base is already ice and compacts no further. That is the standard
altimetry assumption, and `initialize_profile` derives a column deep enough to satisfy it — but
`ModelParameters.column_depth_max` is a ceiling on that derivation, and a column clipped by it, or one
that densifies more slowly than the steady-state guess, leaves the base short of ice density. The height
series then under-counts compaction, silently and progressively.

Check this before trusting a height series; a `false` means deepen the column
(`column_depth_max`) rather than that the run failed.
"""
function column_reaches_ice_density(output; tolerance::Real = 1.0)
    tolerance >= 0 || throw(ArgumentError("tolerance must be >= 0, got $tolerance"))
    density = _profile_layer(output, :density)
    base = density[last(axes(density, 1)), last(axes(density, 2))]
    return base >= _output_density_ice(output) - tolerance
end

"""
    tile_volume_change(dh, band_areas) -> Vector{Float64}

Glacier volume change (km³ of ice equivalent) from per-band surface height change.

`dh` is indexed `(time, band)` in metres — one column per elevation band, as
[`surface_height_change`](@ref) returns for each — and `band_areas` is that band's glacier area in km².
A band contributes `area * dh / 1000`, since a metre over a square kilometre is a thousandth of a cubic
kilometre.

This is the model-side counterpart of the altimetry `dv`, and shares its unit. Volume, not mass:
`dh` carries the firn compaction term, so multiplying it by an ice density would overstate the mass
change by whatever the column's air content did. Use [`tile_mass_total`](@ref) for mass.
"""
tile_volume_change(dh, band_areas) = _band_weighted_total(dh, band_areas, _KM3_PER_KM2_M)

"""
    tile_mass_total(flux, band_areas) -> Vector{Float64}

Glacier mass total (Gt) from a per-band GEMB mass flux.

`flux` is indexed `(time, band)` in kg m-2 — any of GEMB's interval-sum mass layers, or a budget
assembled from them — and `band_areas` is that band's glacier area in km².

Assembled from the flux layers rather than from a height change, because height carries the firn
compaction term and mass does not.
"""
tile_mass_total(flux, band_areas) = _band_weighted_total(flux, band_areas, _M2_PER_KM2 / _KG_PER_GT)

"""
    reference_discharge_rate(dv_mass, time; reference_years = 5) -> Float64

A stand-in ice discharge rate (km³ of ice equivalent per year), from the assumption that the glacier was
in balance over the first `reference_years` of the record.

Under that assumption whatever left as ice discharge matched what arrived as surface mass balance, so
the rate is the cumulative SMB volume at the end of the reference period divided by its length.

**This is a placeholder for making a figure readable, not a discharge model.** Real discharge is not
constant, is not equal to early-record SMB, and is derived elsewhere. What it buys is the one thing a
plot of modelled volume change needs: GEMB carries no ice dynamics, so nothing balances the mean SMB
and the raw series walks off at the accumulation rate — a tile can gain 6 km³/yr indefinitely. Removing
a constant rate leaves the variability about the reference state, which is the part that can be read
against an altimetric `dv` at all.

`reference_years` longer than the record throws rather than quietly using what is there: a rate divided
by the wrong interval is off by exactly that factor, and silently returning it would put a wrong slope
on the figure the correction exists to make readable.
"""
function reference_discharge_rate(dv_mass, time; reference_years::Real = 5)
    reference_years > 0 ||
        throw(ArgumentError("reference_years must be positive, got $reference_years"))
    axes(dv_mass, 1) == axes(time, 1) || throw(DimensionMismatch(
        "dv_mass $(axes(dv_mass, 1)) and time $(axes(time, 1)) must share a time axis"))
    length(time) >= 2 || throw(ArgumentError("need at least two output times"))

    span = _decimal_years(last(time) - first(time))
    span >= reference_years || throw(ArgumentError(
        "the record spans $(round(span, digits = 2)) yr but a $(reference_years) yr reference period " *
        "was asked for. Shorten `reference_years`, or run a longer record: dividing the reference " *
        "period's mass gain by the wrong interval scales the rate by exactly that error."))

    # Last output time at or before the end of the reference period, and the elapsed time to it — not
    # the nominal `reference_years`, since the output grid will not land exactly on it.
    cutoff = first(time) + Millisecond(round(Int, reference_years * 365.25 * 86_400_000))
    i = something(findlast(<=(cutoff), time), firstindex(time))
    elapsed = _decimal_years(time[i] - first(time))
    elapsed > 0 || throw(ArgumentError(
        "the reference period covers no elapsed time; the output interval is coarser than " *
        "$reference_years yr"))

    # `dv_mass` is cumulative from the record start, so its value at `i` *is* the reference period's
    # gain — no differencing needed.
    return dv_mass[i] / elapsed
end

"""
    discharge_corrected_volume_change(dv, dv_mass, time; reference_years = 5) -> Vector{Float64}

Modelled volume change (km³ of ice equivalent) with a constant reference discharge removed, so it can be
plotted against an altimetric `dv`.

`dv - rate * elapsed`, with `rate` from [`reference_discharge_rate`](@ref). Zero at the first output
time, and near zero on average across the reference period — after which it shows how the tile departs
from the state it was assumed to be in balance with.

See [`reference_discharge_rate`](@ref) for why this is a visualization aid rather than a mass budget.
Nothing downstream of a figure should consume it: the tile's own `dv` and the separately derived
discharge are what a real comparison uses.
"""
function discharge_corrected_volume_change(dv, dv_mass, time; reference_years::Real = 5)
    rate = reference_discharge_rate(dv_mass, time; reference_years)
    elapsed = [_decimal_years(t - first(time)) for t in time]
    return dv .- rate .* elapsed
end

# A `Period` as decimal years, on the 365.25-day year the surrounding rate conversions use.
_decimal_years(d) = Millisecond(d).value / (365.25 * 86_400_000)

"""
    mie2cubickm(band_areas) -> Float64

The factor converting a tile-mean thickness in metres of ice equivalent to a volume in km³:
`sum(band_areas) / 1000`.

Named and valued as in the existing 2° geotile products, so a per-tile scalar written here means the
same thing there.
"""
mie2cubickm(band_areas) = sum(band_areas) * _KM3_PER_KM2_M

# Area-weighted sum over bands of a `(time, band)` quantity, scaled into the target unit. The band
# loop is outermost so each band's column is walked once in order, and the accumulation order is fixed
# by the band index rather than by iteration order, so the result does not depend on how the caller
# assembled the matrix.
function _band_weighted_total(values, band_areas, scale)
    axes(values, 2) == axes(band_areas, 1) || throw(DimensionMismatch(
        "the band axis of `values` $(axes(values, 2)) does not match `band_areas` " *
        "$(axes(band_areas, 1))"))
    total = zeros(Float64, axes(values, 1))
    for b in axes(values, 2)
        w = Float64(band_areas[b]) * scale
        w == 0 && continue
        for t in axes(values, 1)
            total[t] += w * values[t, b]
        end
    end
    return total
end

# `density_ice` off a `gemb` output stack. Read from the run rather than assumed, because every
# conversion between mass and thickness here has to use the value the run itself used.
function _output_density_ice(output)
    meta = DimensionalData.metadata(output)
    meta isa DimensionalData.NoMetadata && throw(ArgumentError(
        "this output stack carries no metadata, so `density_ice` cannot be read from it; the mass " *
        "and thickness terms are not interconvertible without it"))
    haskey(meta, "density_ice") || throw(ArgumentError(
        "this output stack has no `density_ice` metadata, which `gemb` attaches to every run"))
    rho = Float64(meta["density_ice"])
    isfinite(rho) && rho > 0 ||
        throw(ArgumentError("`density_ice` metadata is $rho; expected a positive density"))
    return rho
end

# A monolevel (`Ti`) output layer as a plain array, with the missing-layer case named.
function _height_layer(output, name::Symbol)
    haskey(output, name) || throw(ArgumentError(
        "this run carries no `$name` output layer, which the height budget needs. `ice_flux` and " *
        "the mass fluxes are present at every `output_frequency` except `:last`."))
    return parent(output[name])
end

# A layer that is legitimately absent when the process it reports is switched off, as a zero series of
# the right length. `blowing_snow` and `strain_thinning` are both zero at GEMB's defaults.
_optional_height_layer(output, name::Symbol, template) =
    haskey(output, name) ? parent(output[name]) : zeros(Float64, axes(template, 1))

# A profile (`Z` x `Ti`) output layer as a plain array.
function _profile_layer(output, name::Symbol)
    haskey(output, name) || throw(ArgumentError(
        "this run carries no `$name` profile layer, which the height budget needs to recompute firn " *
        "air content and liquid storage at each output time"))
    layer = parent(output[name])
    ndims(layer) == 2 || throw(ArgumentError(
        "`$name` has $(ndims(layer)) dimension(s); the height budget expects a (Z, Ti) profile"))
    return layer
end
