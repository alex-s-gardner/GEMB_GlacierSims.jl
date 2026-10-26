"""
GEMB at a firn-core site, for testing densification physics against a measured density profile.

A site run forces one GEMB column with the reanalysis grid point nearest the core, taken to the
site's elevation by the production downscaling, and with precipitation scaled so the run's mean
snowfall equals the accumulation measured at the core. The downscaling is the one a glacier band
of the site's tile gets ([`site_downscaling`](@ref), [`site_forcing`](@ref)): the lapse rate fitted
across the glacierized cells of the tile's buffered neighbourhood, the glacier correction of the
site's own cell, and the decoupling factor at the site elevation. Matching accumulation takes the forcing's accumulation error out of the
comparison: the Arthern-family rates scale with accumulation and the Crocus rate with the overburden
it builds, so an accumulation error in the forcing would otherwise be fitted as a densification
error. The run therefore tests the compaction physics, not the forcing.

The column is spun up on a climatology of the forcing record, then run from the start of the record
to the drilling date, and the profile on that date is returned for comparison with the core.
"""

# Column depth below the core bottom [m]. The column's lower boundary is held at fixed temperature
# and density, so the core must lie clear of it.
const FIRN_SITE_DEPTH_MARGIN = 10.0

"""
    FirnSite(; record_id, latitude, longitude, elevation, date, accumulation, depth)

A firn-core site: `latitude`, `longitude` [degrees], site `elevation` [m], drilling `date`, the
measured mean `accumulation` [kg m⁻² yr⁻¹] that a site run's snowfall is matched to, and the core's
`depth` [m], which the modeled column must reach.
"""
Base.@kwdef struct FirnSite
    record_id::String
    latitude::Float64
    longitude::Float64
    elevation::Float64
    date::DateTime
    accumulation::Float64
    depth::Float64
    function FirnSite(record_id, latitude, longitude, elevation, date, accumulation, depth)
        -90 <= latitude <= 90 || throw(ArgumentError("latitude must be in [-90, 90], got $latitude"))
        isfinite(longitude) || throw(ArgumentError("longitude must be finite, got $longitude"))
        isfinite(elevation) || throw(ArgumentError(
            "site $record_id has no elevation; a site run lapses the forcing to it"))
        accumulation > 0 || throw(ArgumentError(
            "site $record_id needs a positive measured accumulation to match, got $accumulation"))
        depth > 0 || throw(ArgumentError("site $record_id: core depth must be positive, got $depth"))
        return new(record_id, latitude, longitude, elevation, date, accumulation, depth)
    end
end

"""
    site_snowfall_scaling(forcing, accumulation, mp) -> Float64

The factor on total precipitation that makes the mean annual snowfall of `forcing` equal
`accumulation` [kg m⁻² yr⁻¹]. Snowfall is the precipitation falling at or below
`mp.rain_temperature_threshold`, the partition `gemb` applies, so the factor is exact for the run
that uses it: scaling total precipitation scales snowfall by the same factor.

Throws when the forcing has no snowfall, since no scaling can then supply the accumulation.
"""
function site_snowfall_scaling(forcing, accumulation::Real, mp::ModelParameters)
    T = parent(forcing[:temperature_air])
    P = parent(forcing[:precipitation])
    times = lookup(forcing, Ti)
    years = Dates.value(last(times) - first(times) + _forcing_step(times)) /
            (1000 * 86400 * 365.25)
    snow = sum(P[i] for i in eachindex(P, T) if T[i] <= mp.rain_temperature_threshold; init = 0.0)
    snow > 0 || throw(ArgumentError(
        "the forcing has no snowfall at or below $(mp.rain_temperature_threshold) K, so no " *
        "precipitation scaling can supply an accumulation of $accumulation kg m-2 yr-1"))
    return Float64(accumulation) / (snow / years)
end

"""
    site_downscaling(fit, site::FirnSite, run_time; kwargs...) -> AppliedDownscaling

The downscaling of a tile, `fit` from [`read_downscaling_tile`](@ref), resolved for one band centred
on `site.elevation`, on the time axis `run_time` of the site's forcing. The basis is `:fitted` when
`run_time` is the fit's own axis and `:climatology` otherwise, as in the tile sweep. Keywords are
passed to [`resolve_downscaling`](@ref); pass the tile's `decoupling_factor_prior` and the physics
member's `band_decoupling` and `glacier_correction`, as the tile sweep does.
"""
function site_downscaling(fit, site::FirnSite, run_time; kwargs...)
    z = site.elevation
    interval = (; lo = floor(Int, z), hi = floor(Int, z) + 1, center = z)
    run_time = collect(DateTime, run_time)
    basis = collect(DateTime, fit.time) == run_time ? :fitted : :climatology
    return resolve_downscaling(fit, [interval], run_time; basis, kwargs...)
end

"""
    site_forcing(forcing, applied::AppliedDownscaling, glacier_fraction) -> DimStack

`forcing`, the raw reanalysis stack of the site's grid point, adjusted as
[`elevation_interval_forcing`](@ref) adjusts each cell for a band: brought to the ice-free state with
the cell's `glacier_fraction` (its `glm`, in the fit's glacier-fraction form) and the glacier
correction at the cell's reanalysis elevation, lapsed to the band centre at `applied.lapse_rate`, and
decoupled with the band's `k`. `applied` holds one band, the site, from [`site_downscaling`](@ref), on
the forcing's time axis. Wind is not rescaled: a physics member with a wind factor is refused.
"""
function site_forcing(forcing, applied::AppliedDownscaling, glacier_fraction::Real)
    band = only(applied.bands)
    lookup(forcing, Ti) == applied.time || throw(DimensionMismatch(
        "the downscaling is resolved on $(length(applied.time)) timesteps but the forcing has " *
        "$(length(lookup(forcing, Ti))); resolve it on the forcing's time axis"))
    get(applied.settings, "downscaling_wind_scaling", 1.0) == 1 &&
        get(applied.settings, "downscaling_wind_correction", "scaling") == "scaling" ||
        throw(ArgumentError("a site run applies no wind correction; resolve the downscaling with " *
                            "wind_scaling = 1 and the :scaling wind rule"))
    n = length(applied.time)
    z_cell = Float64(DimensionalData.metadata(forcing)["elevation"])
    form = Symbol(get(applied.settings, "downscaling_glacier_fraction_form", "fraction"))
    g = decoupling_glacier_fraction(glacier_fraction, form)
    correction = g == 0 ? ones(n) : applied.donor_decoupling_factor_at(z_cell)
    ambient = _ambient_forcing(forcing, g, correction)
    return _cell_forcing_at_interval(ambient, band.center - z_cell, band.decoupling_factor,
                                     applied.lapse_rate, 0.0, ones(n))
end

"""
    firn_site_run(site::FirnSite, forcing, mp; kwargs...) -> NamedTuple

Run GEMB at `site` on `forcing`, the raw reanalysis `DimStack` at the site's grid point (its
`metadata["elevation"]` is the grid point's height), and return the column on the drilling date.

With `downscaling` (from [`site_downscaling`](@ref), on the forcing's time axis) the forcing is taken
to the site by [`site_forcing`](@ref) with the cell's `glacier_fraction`. Without it, for a site whose
tile has no glacierized cells to fit, it is lapsed from the grid point to `site.elevation` at the
scalar `lapse_rate` [K km⁻¹] and left ambient. The record is then cut to one month past the drilling
date, its precipitation scaled by [`site_snowfall_scaling`](@ref) over that span, and its impurity
metadata set from `mp`. The
column is at least `depth_minimum` deep (default `site.depth` plus
$(FIRN_SITE_DEPTH_MARGIN) m), so the core lies inside it even where the column densifies more slowly
than the steady state GEMB sizes it by. It is spun up on the `:representative` climatology of
`spinup_window` (whole calendar years; default the first 30 of the record), to a firn-air-content
drift below `convergence_drift_fac` m yr⁻¹ over `drift_window` cycles or `simulation_years_maximum`
simulated years, then run from the start of the forcing to `site.date`.

Returns `(; site, profile, time, precipitation_scaling, snowfall, spinup)`: `profile` is the column on
the last output time at or before `site.date` (`dz` [m], `density` [kg m⁻³], `temperature` [K],
`water` [kg m⁻²], `age` [d], surface first); `time` is that output time; `snowfall` is the run's mean
annual snowfall, equal to `site.accumulation`; `spinup` holds the spinup provenance (`cycles`,
`converged`).

Throws `ArgumentError` when the drilling date lies outside the forcing, and `ForcingUnavailable` when
the forcing is incomplete.
"""
function firn_site_run(site::FirnSite, forcing, mp::ModelParameters;
                       downscaling = nothing,
                       glacier_fraction::Real = 0.0,
                       lapse_rate = _DEFAULT_LAPSE_RATE,
                       spinup_window = nothing,
                       simulation_years_maximum::Real = 1000,
                       convergence_drift_fac = SPINUP_DRIFT_FAC,
                       drift_window::Int = SPINUP_DRIFT_WINDOW,
                       depth_minimum::Real = site.depth + FIRN_SITE_DEPTH_MARGIN)
    forcing_is_complete(forcing) || throw(ForcingUnavailable(
        "forcing for site $(site.record_id) is incomplete (all-NaN or no reference elevation)"))
    times = lookup(forcing, Ti)
    first(times) < site.date <= last(times) || throw(ArgumentError(
        "site $(site.record_id) was drilled on $(site.date), outside the forcing " *
        "$(first(times)) .. $(last(times))"))

    z_forcing = Float64(DimensionalData.metadata(forcing)["elevation"])
    downscaled = downscaling === nothing ?
                 climate_adjust_for_elevation(forcing, site.elevation - z_forcing; lapse_rate) :
                 site_forcing(forcing, downscaling, glacier_fraction)
    # The profile is read at the last monthly output at or before the drilling date, so the record
    # runs one month past it, and the accumulation is matched over that same span.
    lapsed = downscaled[Ti(first(times) .. min(last(times), site.date + Month(1)))]
    scaling = site_snowfall_scaling(lapsed, site.accumulation, mp)
    adjusted = _with_impurity_metadata(precipitation_adjust(lapsed, scaling), mp)

    window = something(spinup_window, _default_spinup_window(adjusted))
    cf_spinup = _spinup_climatology(initialize_forcing(adjusted), window, mp)
    initial = initialize_profile(mp, cf_spinup; depth_minimum)
    spun = gemb_spinup(initial, cf_spinup, mp; simulation_years_maximum, drift_window,
                       convergence_delta_density = nothing,
                       convergence_drift_density = _spinup_drift_tolerance(
                           initial, cf_spinup, mp, convergence_drift_fac))

    mp_run = ConstructionBase.setproperties(mp, (; output_frequency = :monthly))
    output = gemb(spun, initialize_forcing(adjusted), mp_run)
    out_times = collect(dims(output, Ti))
    i = findlast(<=(site.date), out_times)
    i === nothing && throw(ArgumentError(
        "site $(site.record_id): no monthly output at or before $(site.date) (first output " *
        "$(first(out_times)))"))

    column(v) = collect(Float64, output[v][:, i])
    dz = column(:dz)
    keep = .!isnan.(dz) .& (dz .> 0)
    profile = (; dz = dz[keep], density = column(:density)[keep],
               temperature = column(:temperature)[keep], water = column(:water)[keep],
               age = column(:age)[keep])

    meta = DimensionalData.metadata(spun)
    spinup = (; cycles = meta.spinup_cycles, converged = meta.spinup_converged)
    snowfall = scaling * _mean_annual_snowfall(lapsed, mp)
    return (; site, profile, time = out_times[i], precipitation_scaling = scaling, snowfall, spinup)
end

# Mean annual snowfall [kg m-2 yr-1] of `forcing`, partitioned as `site_snowfall_scaling` does.
_mean_annual_snowfall(forcing, mp) = 1.0 / site_snowfall_scaling(forcing, 1.0, mp)

"""
    firn_profile_on_grid(profile, depths) -> Vector{Float64}

Density [kg m⁻³] of a modeled `profile` (from [`firn_site_run`](@ref)) at each of `depths` [m below
the surface], as the density of the cell containing that depth. `NaN` below the column.
"""
function firn_profile_on_grid(profile, depths)
    bottom = cumsum(profile.dz)
    return map(depths) do d
        d < 0 && throw(ArgumentError("depths must be non-negative, got $d"))
        k = searchsortedfirst(bottom, d)
        k > length(bottom) ? NaN : profile.density[k]
    end
end
