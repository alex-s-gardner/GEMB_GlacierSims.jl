"""
GEMB driver for the elevation bands of a 2° tile.

Where [`gemb_glacier_cell`](@ref) runs one reanalysis grid cell's own hypsometry on that cell's forcing,
this runs a whole tile's hypsometry on glacier-area weighted forcing pooled across the tile's cells. The
two differ in what a band's climate is: per cell it is that cell's forcing lapsed to the bin center with
one scalar `k`; per tile it is the area-weighted mean over every cell holding ice in the band, each
lapsed from its own reanalysis surface with a time-varying rate and decoupled with a `k` evaluated at
that band ([`elevation_interval_forcing`](@ref)).

The band forcing arrives already at the band and already decoupled, so nothing here adjusts elevation or
applies `k` again — only the prescribed temperature offset and precipitation scaling, which are the
perturbation grid the downstream fit against altimetry searches.

Output is surface height change per band and volume change for the tile, not just mass fluxes, because
that is what an altimeter measures. See [`surface_height_change`](@ref) for the identity that produces it
and why it is not interchangeable with surface mass balance.
"""

# The GEMB mass fluxes carried through to per-tile totals. `blowing_snow` joins `CELL_MASS_VARIABLES`
# here because the height budget accounts for it (`height_change_components`) and a tile total that
# omitted it would not reconcile with the tile's own dh.
const TILE_MASS_VARIABLES = (CELL_MASS_VARIABLES..., :blowing_snow)

# The per-band series a tile run keeps, over and above the mass fluxes: surface height change and its
# decomposition. `total` is the quantity to compare against altimetry; the rest say which term produced
# it, and `residual` is the numerical check that they add up.
const TILE_HEIGHT_VARIABLES = (:dh, :dh_mass, :dh_water, :dh_firn, :dh_residual,
                               :firn_air_content)

"""
    GlacierTileRun

Result of [`gemb_glacier_tile`](@ref) for one 2° tile.

- `bands`: the elevation bands run, ascending, each `(; lo, hi, center, area, n_cells)` with `area` in
  km². `sum(area)` is the total glacier area of the tile's cells whose forcing was usable.
- `delta_temperatures`, `precipitation_scalings`: the perturbation axes.
- `time`: the output time axis, shared by every series.
- `bands_series`: `Symbol => Array{Float64,4}` indexed `(time, band, delta_temperature,
  precipitation_scaling)`. Keys are [`TILE_MASS_VARIABLES`](@ref) in kg m-2 and
  [`TILE_HEIGHT_VARIABLES`](@ref) in m. Band-resolved because that is the resolution the altimetry
  `dh` is binned at, so a comparison needs no further reduction.
- `totals`: `Symbol => Array{Float64,3}` indexed `(time, delta_temperature,
  precipitation_scaling)`, area-weighted over bands. `:dv` is volume change in km³ of ice equivalent
  and `:fac` firn air volume in km³; every mass key is in Gt. Volume comes from the height change and
  mass from the fluxes — never one from the other, since height carries the firn compaction term.
- `profiles`: `Array{Union{Nothing,DimStack},3}` indexed `(band, delta_temperature,
  precipitation_scaling)`, the final firn column of each run, for restart.
- `run_provenance`: `String => Array{Float64,3}` indexed `(band, delta_temperature,
  precipitation_scaling)`, the numeric `spinup_*`/`climatology_*` outcome of each run. Every column
  spins up on its own, so these differ across the grid by more than an order of magnitude and no scalar
  stands for the tile. `NaN` where a run produced no output or did not record the key. Booleans are
  stored as `0.0`/`1.0`.
- `provenance`: the tile-level aggregates, the `spinup_*`/`climatology_*` keys that are constant across
  runs, and `spinup_converged` reduced with `all` — plus the downscaling settings and per-band parameter
  source counts.
- `parameters`: the settings that define how the tile was run, checked against on restart.
"""
struct GlacierTileRun
    index::Tuple{Int,Int}
    name::String
    geotile_id::String
    bounds::NamedTuple{(:lon_min, :lon_max, :lat_min, :lat_max),NTuple{4,Float64}}
    n_cells_core::Int
    n_cells_used::Int
    bands::Vector{@NamedTuple{lo::Int, hi::Int, center::Float64, area::Float64, n_cells::Int}}
    delta_temperatures::Vector{Float64}
    precipitation_scalings::Vector{Float64}
    time::Vector{DateTime}
    bands_series::Dict{Symbol,Array{Float64,4}}
    totals::Dict{Symbol,Array{Float64,3}}
    profiles::Array{Union{Nothing,DimStack},3}
    band_provenance::Vector{Dict{String,Any}}
    run_provenance::Dict{String,Array{Float64,3}}
    parameters::Dict{String,Any}
    provenance::Dict{String,Any}
end

Base.show(io::IO, r::GlacierTileRun) = print(io,
    "GlacierTileRun(", r.name, ", ", length(r.bands), " bands, ",
    length(r.delta_temperatures), "x", length(r.precipitation_scalings), " perturbations, ",
    length(r.time), " output steps)")

"""
    geotile_id(bounds) -> String
    geotile_id(index, tile_size) -> String

The tile identifier the 2° geotile products use: `"lat[+60+62]lon[-142-140]"`.

Latitude is zero-padded to two digits and longitude to three, each carrying an explicit sign. Written
alongside this package's own [`tile_output_name`](@ref) form so a tile's output joins against those
products without either side parsing the other's filenames.
"""
geotile_id(bounds) = _geotile_id_bounds(bounds.lon_min, bounds.lon_max,
                                       bounds.lat_min, bounds.lat_max)

geotile_id(index, tile_size::Real) =
    _geotile_id_bounds(index[1], index[1] + tile_size, index[2], index[2] + tile_size)

_geotile_id_bounds(lon_min, lon_max, lat_min, lat_max) =
    "lat[" * _signed_degrees(lat_min, 2) * _signed_degrees(lat_max, 2) *
    "]lon[" * _signed_degrees(lon_min, 3) * _signed_degrees(lon_max, 3) * "]"

# `%+03d`-style: an explicit sign then a zero-padded magnitude. The sign is always written, including
# for zero, because that is what the geotile ids on disk do (`lat[+00+02]`).
_signed_degrees(x, digits::Int) =
    (x < 0 ? "-" : "+") * lpad(abs(round(Int, x)), digits, '0')

"""
    gemb_glacier_tile(tile, applied, band_forcing, mp; kwargs...) -> GlacierTileRun

Run GEMB for every elevation band of one tile, over the outer product of `delta_temperatures` and
`precipitation_scalings`, and aggregate to per-tile volume and mass change.

`tile` is an entry from [`downscaling_tiles`](@ref); `applied` the [`AppliedDownscaling`](@ref) its
forcing was built with, carried through for provenance; `band_forcing` an iterable of bands as
[`elevation_interval_forcing`](@ref) yields — each `(; lo, hi, center, area, n_cells, forcing)` with
`metadata["elevation"]` equal to the band center.

Per (band, delta, scaling) the forcing chain is

    cf = initialize_forcing(precipitation_adjust(temperature_adjust(band.forcing, delta), scaling))

and nothing else. The band forcing has already been lapsed to the band center and decoupled with the
band's own `k`, so re-applying either would double-count: lapsing again would move the forcing off the
band it is defined at, and `climate_adjust_for_glacier` would damp the warm excess a second time. Both
adjustments act on the raw band stack and are exact identities at `delta = 0` / `scaling = 1`, so the
unperturbed run is bit-for-bit the plain band forcing.

`precipitation_adjust` scales **total** precipitation, and GEMB partitions phase downstream against
`mp.rain_temperature_threshold`, so a scaling above 1 raises rain as well as snow. Where the scaling
stands in for snow redistribution and avalanching that is an overstatement, and it is the reason the
published 2° products exclude rain from their aggregates.

# Keywords
- `delta_temperatures = [0.0]`, `precipitation_scalings = [1.0]`: the perturbation grid.
- `spinup_window`: `(start, stop)` the repeating spinup cycle is built from, by
  `SPINUP_CLIMATOLOGY_METHOD` (`:representative`, a block of `SPINUP_CLIMATOLOGY_N_YEARS` real years).
  Defaults to the first 30 complete years of the band forcing, but a sweep should fix it: tying it to the
  record means extending the record changes the spinup, and an appended segment then does not continue
  the record it is appended to.
- `simulation_years_maximum`: ceiling on the spinup as a span of simulated years, passed to
  `gemb_spinup`, which divides by the measured cycle length to get its own cycle ceiling.
- `convergence_drift_fac = $(SPINUP_DRIFT_FAC)`, `drift_window = $(SPINUP_DRIFT_WINDOW)`: spinup exits
  when the least-squares slope of firn air content against cycle, over the trailing `drift_window`
  cycles, falls below `convergence_drift_fac` metres per *year* (scaled to the cycle internally). Equilibrium is the absence of a trend,
  which a step between consecutive cycles does not measure: a step test passes a column still drifting
  steadily, and fails a settled column whose jitter exceeds it. Converted per band into the mean-density
  drift `gemb_spinup` tests, exactly, because the column depth is pinned
  ([`convergence_density_from_fac`](@ref)). Note `gemb_spinup` cannot judge a slope until it has
  `drift_window` samples and will not exit by abstention, so that is also the minimum cycle count. The
  default is sized for mountain-glacier firn; a dry ice-sheet plateau relaxes more slowly and changes by
  centimetres per year, and needs 1e-4 or tighter with `simulation_years_maximum` raised to match.
- `threaded = true`: run the (band × delta × scaling) simulations on all available threads. Each is
  independent, and the area-weighted reduction happens afterwards in a fixed index order, so the
  threaded result is identical to the serial one including bit-for-bit totals.
- `donor_max_distance_km = nothing`: the cap `band_forcing` was built under, recorded in the run
  parameters. It changes nothing here — the substitution happens in
  [`elevation_interval_forcing`](@ref), and these bands no longer know which cells borrowed — but two
  runs under different caps rest on different forcing and must not compare as the same experiment.
- `restart = nothing`: the value [`read_glacier_tile_restart`](@ref) returns, or `nothing`. Each run
  inherits its saved column and the spinup is skipped; the forcing is trimmed to the steps after the
  saved time, so pass only the new forcing rather than refetching the record. Throws
  [`ForcingUpToDate`](@ref) when fewer than `$(MIN_FORCING_STEPS)` new steps remain, and
  `RestartParameterMismatch` when the saved run parameters disagree with this call's.
- `force_restart = false`: append even across a run-parameter disagreement, warning instead of throwing.
  The record before the seam then does not reflect the new parameters.
- `on_output = nothing`: called as `on_output(output; band, delta, pscale)` with each simulation's full
  output stack, for inspection. Called from inside the task, so with `threaded = true` it runs
  concurrently and in an unspecified order.

A band whose run produces no output is dropped from the aggregation with a warning rather than
poisoning the tile's totals; a tile where no band produced output throws.
"""
function gemb_glacier_tile(tile, applied::AppliedDownscaling, band_forcing, mp::ModelParameters;
                           delta_temperatures = [0.0],
                           precipitation_scalings = [1.0],
                           spinup_window = nothing,
                           simulation_years_maximum::Real = 1000,
                           convergence_drift_fac = SPINUP_DRIFT_FAC,
                           drift_window::Int = SPINUP_DRIFT_WINDOW,
                           threaded::Bool = true,
                           donor_max_distance_km = nothing,
                           restart = nothing,
                           force_restart::Bool = false,
                           on_output = nothing)
    delta_temperatures = collect(Float64, delta_temperatures)
    precipitation_scalings = collect(Float64, precipitation_scalings)
    isempty(delta_temperatures) && throw(ArgumentError("delta_temperatures is empty"))
    isempty(precipitation_scalings) && throw(ArgumentError("precipitation_scalings is empty"))

    # Materialized because every band is visited once per perturbation and the iterator re-streams the
    # tile's cells on each pass. Peak memory is the whole tile's band forcing, which is why the caller
    # controls `elevation_interval_batch`: a full-record sweep should build this in batches and run
    # them, rather than holding sixty bands of hourly forcing at once.
    bands = collect(band_forcing)
    isempty(bands) && throw(ArgumentError(
        "this tile has no populated elevation bands, so there is nothing to run"))

    for band in bands
        forcing_is_complete(band.forcing) || throw(ForcingUnavailable(
            "band $(band.lo)-$(band.hi) m carries incomplete forcing (all-NaN or no reference " *
            "elevation); a NaN in an applied downscaling parameter is the usual cause"))
    end

    bands, dropped = _runnable_bands(bands, delta_temperatures, precipitation_scalings)
    isempty(bands) && throw(ForcingUnavailable(
        "no band of this tile could be perturbed into forcing GEMB accepts; the highest bands are " *
        "lapse extrapolations far above every contributing reanalysis surface, and a cold " *
        "perturbation of one can fall outside the range `climate_forcing`'s validator allows"))

    n_band = length(bands)
    n_dt = length(delta_temperatures)
    n_ps = length(precipitation_scalings)

    window = spinup_window === nothing ?
             _default_spinup_window(first(bands).forcing) : spinup_window
    parameters = tile_run_parameters(mp, applied; spinup_window = window, simulation_years_maximum,
                                     convergence_drift_fac, drift_window, donor_max_distance_km,
                                     climatology_method = SPINUP_CLIMATOLOGY_METHOD,
                                     climatology_n_years = SPINUP_CLIMATOLOGY_N_YEARS)

    # Resuming: keep only forcing newer than the saved state, and let each run inherit its saved column
    # instead of spinning up again. The saved time is the last *output* time, and every forcing step up to
    # and including it has already been consumed.
    #
    # A continuation is expected to have fetched only the new steps, so the spinup window will usually not
    # be present in this forcing at all. That is fine while every run has a saved profile, and
    # `_spinup_climatology` throws `SpinupWindowUnavailable` naming the window if one does not — which is
    # the honest outcome, since the alternative is spinning one band up on a different climate than the
    # rest of the file and saying nothing.
    if restart !== nothing
        _validate_tile_restart(restart, bands, delta_temperatures, precipitation_scalings)
        # Checked before the timestep subset, so a parameter change is reported as such rather than
        # masked by a `ForcingUpToDate` from a record that was never extended.
        _validate_restart_parameters(restart, parameters; force_restart, what = "tile")
        bands = _bands_after(bands, restart.time)
        n_new = length(dims(first(bands).forcing, Ti))
        n_new >= MIN_FORCING_STEPS || throw(ForcingUpToDate(restart.time, n_new))
        @info "Resuming tile from saved profiles" restart_time=restart.time new_steps=n_new bands=n_band
    end

    profiles = Array{Union{Nothing,DimStack}}(nothing, n_band, n_dt, n_ps)

    # One flat task list rather than nested threaded loops: bands per tile range from a handful to
    # ~60 while perturbations are a fixed few, so threading either loop alone would leave threads
    # idle on the tiles whose loop is shorter than the thread count.
    tasks = vec([(i_band, i_dt, i_ps) for i_ps in 1:n_ps, i_dt in 1:n_dt, i_band in 1:n_band])
    results = Vector{Union{Nothing,NamedTuple}}(nothing, length(tasks))

    run_one = function (t)
        i_band, i_dt, i_ps = tasks[t]
        band = bands[i_band]
        delta = delta_temperatures[i_dt]
        pscale = precipitation_scalings[i_ps]

        # No elevation or glacier adjustment: the band forcing is already at the band center and
        # already decoupled. Rebuilt per task rather than shared across perturbations, so concurrent
        # tasks never hand each other the same object.
        adjusted = precipitation_adjust(temperature_adjust(band.forcing, delta), pscale)
        cf = initialize_forcing(adjusted)

        # The band's own bare-ice albedo, carried on the band by `resolve_downscaling`. Built per
        # task, so concurrent tasks never share it, and used for the spinup as well as the run: a
        # column spun up at one ice albedo and then run at another starts from a state the run's own
        # physics would not have produced.
        mp_band = _with_albedo_ice(mp, band.albedo_ice)

        # A saved column is the point of a restart: it carries the spun-up state forward, so the spinup
        # is skipped rather than repeated over a window this forcing may not even contain.
        profile = restart === nothing ? nothing :
                  get(restart.profiles, (i_band, i_dt, i_ps), nothing)

        if profile === nothing
            restart === nothing ||
                @warn "No saved profile for this run; spinning up over the tile's spinup window" band=band.center delta pscale window
            cf_spinup = _spinup_climatology(cf, window, mp_band)
            # The depth is fixed by the initial profile, so the firn-air tolerance can only be converted
            # into the density one `gemb_spinup` tests once that profile exists.
            initial = initialize_profile(mp_band, cf_spinup)
            # The FAC trend is the *only* convergence criterion. `convergence_delta_density` is passed
            # explicitly as `nothing` rather than left out, so that the single-criterion design is visible
            # here and not mistaken for an omission: a step between consecutive cycles both passes a column
            # that is still drifting and fails a settled one whose jitter exceeds it.
            profile = gemb_spinup(initial, cf_spinup, mp_band; simulation_years_maximum, drift_window,
                                  convergence_delta_density = nothing,
                                  convergence_drift_density = _spinup_drift_tolerance(
                                      initial, cf_spinup, mp_band, convergence_drift_fac))
        end
        output = gemb(profile, cf, mp_band)

        if length(dims(output, Ti)) == 0
            @warn "GEMB produced no output for this run; dropping band" band=band.center delta pscale
            return nothing
        end

        on_output === nothing || on_output(output; band, delta, pscale)

        # Extracted here so `output` goes out of scope with the task. Held for every task at once it
        # would be n_band*n_dt*n_ps profile stacks of (Z x Ti), which for a 60-band tile over decades
        # is far more than the machine has.
        components = height_change_components(output)
        series = Dict{Symbol,Vector{Float64}}()
        for v in TILE_MASS_VARIABLES
            series[v] = haskey(output, v) ? Vector{Float64}(output[v]) :
                        zeros(Float64, length(dims(output, Ti)))
        end
        series[:dh] = surface_height_change(output)
        series[:dh_mass] = components.mass
        series[:dh_water] = components.water
        series[:dh_firn] = components.firn
        series[:dh_residual] = components.residual
        series[:firn_air_content] = Vector{Float64}(output[:firn_air_content])

        return (; i_band, i_dt, i_ps,
                time = collect(dims(output, Ti)),
                series,
                base_at_ice_density = column_reaches_ice_density(output),
                profile = gemb_profile(output),
                provenance = _stack_provenance(output))
    end

    if threaded && Threads.nthreads() > 1 && length(tasks) > 1
        Threads.@threads for t in eachindex(tasks)
            results[t] = run_one(t)
        end
    else
        for t in eachindex(tasks)
            results[t] = run_one(t)
        end
    end

    # Serial reduction in task order, so the totals do not depend on completion order and the threaded
    # run reproduces the serial one exactly. Floating-point addition is not associative, so an
    # accumulate-as-they-land reduction would not.
    out_time = DateTime[]
    bands_series = Dict{Symbol,Array{Float64,4}}()
    provenance = Dict{String,Any}()
    run_provenance = Dict{String,Array{Float64,3}}()
    shallow_base = Int[]

    for res in results
        res === nothing && continue
        if isempty(out_time)
            out_time = res.time
            for v in (TILE_MASS_VARIABLES..., TILE_HEIGHT_VARIABLES...)
                bands_series[v] = zeros(length(out_time), n_band, n_dt, n_ps)
            end
        elseif res.time != out_time
            throw(ErrorException(
                "output time axis differs between bands of the same tile (band " *
                "$(bands[res.i_band].center) m); cannot aggregate"))
        end
        _collect_run_provenance!(run_provenance, provenance, res, parameters, (n_band, n_dt, n_ps))
        res.base_at_ice_density || push!(shallow_base, res.i_band)
        for (v, values) in res.series
            @views bands_series[v][:, res.i_band, res.i_dt, res.i_ps] .= values
        end
        profiles[res.i_band, res.i_dt, res.i_ps] = res.profile
    end

    isempty(out_time) && throw(ErrorException("no band of this tile produced any output"))

    # The datum for a height series is a column whose base is already ice. Where it is not, the series
    # under-counts compaction — silently and progressively — so the bands it happened on are named
    # rather than left to be inferred from a drifting residual.
    isempty(shallow_base) ||
        @warn("Some bands ended with their deepest cell below ice density, so their height change " *
              "under-counts compaction; deepen the column with `column_depth_max`",
              bands = [bands[i].center for i in unique(shallow_base)])

    band_areas = [Float64(b.area) for b in bands]
    totals = _tile_totals(bands_series, band_areas)

    band_provenance = [_band_provenance(band) for band in bands]
    merge!(provenance, applied.settings)
    provenance["n_bands"] = n_band
    provenance["glacier_area_km2"] = sum(band_areas)
    provenance["mie2cubickm"] = mie2cubickm(band_areas)
    provenance["max_dh_residual"] = maximum(abs, bands_series[:dh_residual])
    # How much of this tile's forcing was borrowed from a neighbouring cell, because the cell holding
    # the ice falls outside the reanalysis land mask, and from how far away the furthest loan came.
    # `unrecovered` is ice no cell could supply forcing for at all: it is *not* in `glacier_area_km2`,
    # so a total computed from this tile understates its hypsometry by that much.
    provenance["n_cells_substituted"] = sum(Int(b.n_cells_substituted) for b in bands)
    provenance["substituted_area_km2"] = sum(Float64(b.area_substituted) for b in bands)
    provenance["unrecovered_area_km2"] = sum(Float64(b.area_unrecovered) for b in bands)
    provenance["max_donor_distance_km"] = maximum(Float64(b.max_donor_distance_km) for b in bands)
    # Ice this tile holds but did not model, so the gap between the tile's totals and its hypsometry is
    # a recorded number rather than something a reader has to notice. Zero on a tile whose every band
    # was runnable, which is the common case.
    provenance["n_bands_dropped"] = length(dropped)
    provenance["dropped_area_km2"] = isempty(dropped) ? 0.0 : sum(d.area for d in dropped)
    # One tile-level convergence verdict, and the only spinup outcome reduced to a scalar: a tile is
    # settled when every one of its columns is, so this is `all` and not a sample. The per-run detail is
    # in `run_provenance["spinup_converged"]`, which is where a reader finds *which* column failed. A
    # `NaN` slot is a run that produced no output rather than one that failed to settle — those are
    # counted by `n_bands_dropped` — so it does not make the tile unconverged.
    if haskey(run_provenance, "spinup_converged")
        flags = run_provenance["spinup_converged"]
        provenance["spinup_converged"] = all(f -> isnan(f) || f == 1.0, flags)
    end

    return GlacierTileRun(
        tile.index, tile.name, geotile_id(tile.bounds), tile.bounds,
        nrow(tile.core), length(applied.bands),
        [(; b.lo, b.hi, b.center, area = Float64(b.area), n_cells = Int(b.n_cells)) for b in bands],
        delta_temperatures, precipitation_scalings, out_time,
        bands_series, totals, profiles, band_provenance, run_provenance, parameters, provenance,
    )
end

# Sort one run's `spinup_*`/`climatology_*` provenance into the per-run arrays and the tile-level
# constants.
#
# Three cases, in this order. A key that is a *setting* — listed in `parameters`, and so compared by
# `tile_run_disposition` — is already a global and is left alone; `gemb` echoes several of those back on
# its output, so they arrive here mixed in with the outcomes. What remains and is numeric is a per-run
# outcome: every column spins up independently, so it gets a slot at its own `(band, delta_temperature,
# precipitation_scaling)`. What remains and is not numeric describes the climatology the whole tile was
# built on, so it stays a scalar — and disagreeing across runs would mean flattening something that is
# genuinely per-run, which throws rather than keeping whichever value landed last.
function _collect_run_provenance!(run_provenance, provenance, res, parameters, dims::NTuple{3,Int})
    for (key, value) in res.provenance
        haskey(parameters, key) && continue
        # `Bool <: Real`, so this covers the convergence flags; `nothing` is not `Real` and falls through
        # to the scalar branch, where the writer omits it as it does today.
        if value isa Real
            slots = get!(() -> fill(NaN, dims), run_provenance, key)
            slots[res.i_band, res.i_dt, res.i_ps] = Float64(value)
            # A key some runs report as a number and others as `nothing` belongs to the per-run array
            # alone, or it would also stand as a scalar with one run's value.
            delete!(provenance, key)
        elseif haskey(run_provenance, key)
            continue                      # numeric for some other run; the array owns it
        elseif haskey(provenance, key)
            provenance[key] == value || throw(ErrorException(
                "provenance key $key differs between runs of the same tile ($(provenance[key]) vs " *
                "$value) but is not numeric, so it cannot be stored per run; make it numeric or " *
                "stop varying it"))
        else
            provenance[key] = value
        end
    end
    return nothing
end

# Which bands can be perturbed into forcing GEMB will accept, and which cannot.
#
# The high bands of a tile are lapse extrapolations well above every contributing reanalysis surface —
# on Khumbu a quarter of the glacier area is, and the top bands by ~2.6 km. Extrapolated that far the
# forcing can leave the range `climate_forcing`'s validator allows: at 8750 m a -1 K perturbation drops
# the re-derived downward longwave to 48.8 W/m² against a floor of 50. That is the validator doing its
# job, not a bug, and it is checked here rather than left to fail inside a task — where it would take
# the whole tile down with it and lose the 98% of the area that is perfectly runnable.
#
# A band is kept only if **every** requested perturbation works. Keeping it for some and not others
# would leave the perturbation grid covering different areas at different corners, so a `dv` at
# -1 K would not be comparable to one at 0 K — and comparing those corners is the entire purpose of
# the grid.
#
# Only the adjustment chain is run here, not a spinup, so this costs a pass over each band's record per
# perturbation against thousands of column-years of simulation.
function _runnable_bands(bands, delta_temperatures, precipitation_scalings)
    keep = similar(bands, 0)
    dropped = NamedTuple[]

    for band in bands
        failure = nothing
        for pscale in precipitation_scalings, delta in delta_temperatures
            try
                initialize_forcing(precipitation_adjust(temperature_adjust(band.forcing, delta),
                                                        pscale))
            catch e
                e isa InterruptException && rethrow()
                is_caller_error(e) && rethrow()
                failure = (; delta, pscale, message = sprint(showerror, e))
                break
            end
        end
        if failure === nothing
            push!(keep, band)
        else
            push!(dropped, (; band.lo, band.hi, band.center, band.area, failure...))
        end
    end

    if !isempty(dropped)
        area = sum(d.area for d in dropped)
        total = sum(b.area for b in bands)
        @warn("Elevation bands dropped: their forcing could not be perturbed into the range " *
              "`climate_forcing` validates. Expected at the top of a heavily extrapolated tile; the " *
              "tile's totals cover the remaining area only.",
              n_dropped = length(dropped),
              centers = [d.center for d in dropped],
              dropped_area_km2 = round(area, digits = 2),
              dropped_area_fraction = round(area / total, digits = 4),
              first_reason = first(dropped).message)
    end
    return keep, dropped
end

# Per-tile totals from the per-band series: volume from the height change, mass from the fluxes.
#
# The two are not interchangeable. `dh` carries the firn compaction term, so `dh * density_ice` is not
# the mass change; and mass alone cannot produce a height, since a column can lose air without losing
# mass. Keeping the conversions in one place is what stops a caller reaching for the wrong one.
function _tile_totals(bands_series, band_areas)
    totals = Dict{Symbol,Array{Float64,3}}()
    n_t, _, n_dt, n_ps = size(bands_series[:dh])

    # Volume-type totals (km³ of ice equivalent) from the metre-valued series. Already cumulative for
    # the three `dh` terms; `fac` is the exception and is deliberately not accumulated — it is the
    # column's absolute firn air volume, a state rather than a flux, so a running sum of it would mean
    # nothing.
    for (name, source) in ((:dv, :dh), (:dv_mass, :dh_mass), (:dv_firn, :dh_firn),
                           (:fac, :firn_air_content))
        out = zeros(n_t, n_dt, n_ps)
        for i_ps in 1:n_ps, i_dt in 1:n_dt
            @views out[:, i_dt, i_ps] .=
                tile_volume_change(bands_series[source][:, :, i_dt, i_ps], band_areas)
        end
        totals[name] = out
    end

    # Mass-type totals (Gt) from the kg m-2 fluxes, plus the column budget assembled from them. The
    # budget uses the same formula `gemb_glacier_cell` records in `CELL_MASS_CHANGE_FORMULA`, with
    # `blowing_snow` added because a tile run carries it: `rain` is a fraction of `precipitation` and
    # must not be added again, and `refreeze` moves no mass across the column boundary.
    #
    # **Accumulated along time**, unlike the per-band series they are weighted from. GEMB reports a
    # mass flux as an interval sum, so the per-band series holds one output period's worth; the volume
    # totals beside these are cumulative by construction (`dh` is `-cumsum(ice_flux)`), and a `totals`
    # dict mixing the two would invite exactly the comparison that cannot be made — a record-long
    # volume change against a single month's melt. Cumulative is also the form the altimetry `dv`/`dm`
    # these are compared against take.
    for v in TILE_MASS_VARIABLES
        out = zeros(n_t, n_dt, n_ps)
        for i_ps in 1:n_ps, i_dt in 1:n_dt
            @views out[:, i_dt, i_ps] .=
                cumsum(tile_mass_total(bands_series[v][:, :, i_dt, i_ps], band_areas))
        end
        totals[v] = out
    end
    totals[:dm] = totals[:precipitation] .- totals[:runoff] .+
                  totals[:evaporation_condensation] .+ totals[:blowing_snow]

    return totals
end

# What a band's forcing recorded about the parameters applied to it, as the attributes a reader needs to
# judge it without re-deriving. The keys are the band forcing's own, so the writer and the aggregation
# cannot drift about their names.
function _band_provenance(band)
    meta = DimensionalData.metadata(band.forcing)
    keep = ("extrapolation_above_reanalysis", "temperature_lapse_rate",
            "glacier_decoupling_factor_mean", "glacier_decoupling_factor_n_fit_held",
            "glacier_decoupling_factor_n_fit_in_domain", "n_timesteps_above_freezing",
            "n_grid_cells_substituted", "substituted_area", "max_donor_distance_km",
            "unrecovered_area")
    prov = Dict{String,Any}(k => meta[k] for k in keep if haskey(meta, k))
    for source in DOWNSCALING_SOURCES
        for key in ("glacier_decoupling_factor_n_$source",
                    "glacier_decoupling_factor_n_$(source)_above_freezing",
                    "temperature_lapse_rate_n_$source")
            haskey(meta, key) && (prov[key] = meta[key])
        end
    end
    return prov
end

"""
    tile_run_parameters(mp::ModelParameters, applied; spinup_window) -> Dict{String,Any}

The settings that define *how* a tile was run, as netCDF global attributes.

These are exactly the values that must be identical for a continuation to be a continuation rather than
a different experiment spliced onto an old record, so they are what a restart is checked against. Every
`ModelParameters` field is included, prefixed `model_`, except those in `GEMB.DERIVED_PARAMETERS` —
the fields `gemb` computes from the forcing and overwrites whatever it is handed.

The downscaling settings from `applied` are included for the same reason: two runs of one tile that
resolved their lapse rate differently are different experiments, and a file that recorded only the
model parameters could not tell them apart.
"""
function tile_run_parameters(mp::ModelParameters, applied::AppliedDownscaling; spinup_window,
                             simulation_years_maximum = nothing, convergence_drift_fac = nothing,
                             drift_window = nothing, donor_max_distance_km = nothing,
                             climatology_method = nothing, climatology_n_years = nothing)
    params = Dict{String,Any}("spinup_window_start" => string(spinup_window[1]),
                              "spinup_window_stop" => string(spinup_window[2]))
    # How the columns were settled. Two runs that spun up to different tolerances, or under different
    # ceilings, are different experiments — and with a hard ceiling "converged" and "ran out of cycles"
    # are the same outcome unless the ceiling is on record.
    simulation_years_maximum === nothing ||
        (params["spinup_simulation_years_maximum"] = Float64(simulation_years_maximum))
    convergence_drift_fac === nothing ||
        (params["spinup_convergence_drift_fac"] = Float64(convergence_drift_fac))
    drift_window === nothing || (params["spinup_drift_window"] = drift_window)
    # How the cycle was built from that window. `:average` and `:representative` produce columns
    # equilibrated to different climates — averaging removes most of the melt — so this is a setting and
    # not provenance, and a run that changes it must not compare as current against one that did not.
    climatology_method === nothing ||
        (params["spinup_climatology_method"] = string(climatology_method))
    climatology_n_years === nothing ||
        (params["spinup_climatology_n_years"] = Int(climatology_n_years))
    # How far a masked cell was allowed to borrow forcing (see `elevation_interval_forcing`). Two runs
    # under different caps rest on different forcing for the same cells, so they are not the same
    # experiment even though every model parameter matches.
    donor_max_distance_km === nothing ||
        (params["forcing_donor_max_distance_km"] = Float64(donor_max_distance_km))
    merge!(params, applied.settings)
    for field in propertynames(mp)
        field in GEMB.DERIVED_PARAMETERS && continue
        params["model_" * string(field)] = getproperty(mp, field)
    end
    # The bare-ice albedo each band was actually run at, which `model_albedo_ice` cannot express —
    # that field is now only the fallback for a band the observations did not resolve. Recorded as
    # the whole vector so re-deriving the albedo product invalidates a restart, as it must: the same
    # forcing under a different `albedo_ice` per band is a different experiment.
    #
    # Only when the resolution actually had a profile. Without one every band runs at `mp.albedo_ice`
    # exactly as before, and a tile written then must still compare as current.
    if get(applied.settings, "downscaling_bare_ice_albedo_available", false) === true
        _record_bare_ice_albedo!(params, [(b.albedo_ice, b.albedo_ice_source)
                                          for b in applied.bands])
    end
    return params
end
