"""
GEMB at a firn-core site, for testing densification physics against a measured density profile.

A site run forces one GEMB column with the reanalysis grid point nearest the core, lapsed to the
site's elevation, and with precipitation scaled so the run's mean snowfall equals the accumulation
measured at the core. Matching accumulation takes the forcing's accumulation error out of the
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
    firn_site_run(site::FirnSite, forcing, mp; kwargs...) -> NamedTuple

Run GEMB at `site` on `forcing`, the raw reanalysis `DimStack` at the site's grid point (its
`metadata["elevation"]` is the grid point's height), and return the column on the drilling date.

The forcing is lapsed from the grid point to `site.elevation` at `lapse_rate` [K km⁻¹], its
precipitation scaled by [`site_snowfall_scaling`](@ref), and its impurity metadata set from `mp`. The
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
    lapsed = climate_adjust_for_elevation(forcing, site.elevation - z_forcing; lapse_rate)
    scaling = site_snowfall_scaling(lapsed, site.accumulation, mp)
    adjusted = _with_impurity_metadata(precipitation_adjust(lapsed, scaling), mp)

    window = something(spinup_window, _default_spinup_window(adjusted))
    cf_spinup = _spinup_climatology(initialize_forcing(adjusted), window, mp)
    initial = initialize_profile(mp, cf_spinup; depth_minimum)
    spun = gemb_spinup(initial, cf_spinup, mp; simulation_years_maximum, drift_window,
                       convergence_delta_density = nothing,
                       convergence_drift_density = _spinup_drift_tolerance(
                           initial, cf_spinup, mp, convergence_drift_fac))

    # Monthly output is enough to find the column on the drilling date; the profile is read at the
    # last output time at or before it, so the run stops one month after it.
    run_forcing = adjusted[Ti(first(times) .. min(last(times), site.date + Month(1)))]
    mp_run = ConstructionBase.setproperties(mp, (; output_frequency = :monthly))
    output = gemb(spun, initialize_forcing(run_forcing), mp_run)
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
