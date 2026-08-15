"""
Batch GEMB driver for glacier grid cells.

For one grid cell of the glacier elevation-class table this runs GEMB once per
(temperature delta, precipitation scaling, hypsometry bin) combination, then area-weights the
per-bin mass fluxes (kg m-2) by the glacier area in each bin (km²) to give per-cell mass
totals (kg). The final firn profile of every run is kept so the record can be extended when
new forcing arrives without repeating the spinup.
"""

# Mass-flux output layers that are area-weighted and summed to per-cell totals. All are
# `kg m-2` interval sums in GEMB output (`GEMB_CF_ATTRIBUTES`), so they aggregate identically.
const CELL_MASS_VARIABLES = (:melt, :runoff, :refreeze, :evaporation_condensation,
                            :precipitation, :rain)

# Column-mass budget assembled from the aggregated fluxes. `rain` is the liquid *fraction of*
# `precipitation` (GEMB classifies each precipitation step as all-snow or all-rain against
# `mp.rain_temperature_threshold`, `GEMB.jl/src/calculate_accumulation.jl`), so it must not be
# added again; `refreeze` is an internal phase change that moves no mass across the column
# boundary; `evaporation_condensation` is positive for mass gain.
const CELL_MASS_CHANGE_FORMULA = "precipitation - runoff + evaporation_condensation"

# The 7 layers that constitute a complete GEMB restart state (`gemb` reads exactly these,
# `GEMB.jl/src/gemb_driver.jl`).
const PROFILE_VARIABLES = (:dz, :temperature, :density, :water, :grain_radius,
                          :grain_dendricity, :grain_sphericity)

# Everything carried as a per-cell total: the area-weighted fluxes plus the budget assembled
# from them. Keys of `GlacierCellRun.totals` and the mass variables in the NetCDF output.
const CELL_TOTAL_VARIABLES = (CELL_MASS_VARIABLES..., :mass_change)

# `gemb` derives its timestep from the first two forcing times
# (`GEMB.jl/src/initialize_forcing.jl`), so a single step is as unrunnable as none at all.
const MIN_FORCING_STEPS = 2

"""
    run_parameters(mp::ModelParameters; coverage, lapse_rate) -> Dict{String,Any}

The settings that define *how* a cell is run, as NetCDF global attributes.

These are exactly the values that must be identical for a continuation to be a continuation
rather than a different experiment spliced onto an old record, so they are what
[`gemb_glacier_cell`](@ref) checks a restart against. Every `ModelParameters` field is included,
prefixed `model_`, except `dt_divisors`, which `gemb` computes from the forcing timestep and
overwrites whatever it is handed (`GEMB.jl/src/gemb_driver.jl` excludes it for the same reason).

Deliberately *excluded* are the things expected to differ between an original run and its
continuation: the output time axis, the `spinup_*`/`climatology_*` provenance (a resumed run
performs no spinup at all, and a longer forcing record shifts the climatology window), `history`,
and the `institution`/`references` labels.
"""
function run_parameters(mp::ModelParameters; coverage::Real, lapse_rate::Real)
    params = Dict{String,Any}("hypsometry_coverage" => Float64(coverage),
                              "temperature_lapse_rate" => Float64(lapse_rate))
    for field in propertynames(mp)
        field === :dt_divisors && continue
        params["model_" * string(field)] = getproperty(mp, field)
    end
    return params
end

"""
    GlacierCellRun

Result of [`gemb_glacier_cell`](@ref) for one glacier grid cell: area-weighted per-cell mass
totals over the full perturbation grid, plus the per-run final firn profiles for restart.

- `totals`: `Symbol => Array{Float64,3}` indexed `(time, delta_temperature,
  precipitation_scaling)`, in **kg** (mass, not mass per unit area). Keys are
  `CELL_MASS_VARIABLES` plus `:mass_change`.
- `profiles`: `Array{Union{Nothing,DimStack},3}` indexed `(bin, delta_temperature,
  precipitation_scaling)`; `nothing` for a bin whose run produced no output.
- `weights`: glacier area (km²) attributed to each modeled bin, including the area of
  unmodeled bins reassigned to their nearest modeled bin. `sum(weights)` is the cell's total
  glacier area, since the reassignment drops none of it.
- `parameters`: the settings that define how the cell was run ([`run_parameters`](@ref)),
  written as global attributes and checked against on restart. Includes `hypsometry_coverage`
  and `temperature_lapse_rate`.
- `provenance`: the `spinup_*`/`climatology_*` keys `gemb` attached to the output. Recorded but
  *not* checked on restart — a resumed run performs no spinup, so these are expected to differ.
"""
struct GlacierCellRun
    latitude::Float64
    longitude::Float64
    forcing_elevation::Float64
    chunk_id::Union{Int,Missing}
    glacier_frac::Union{Float64,Missing}
    delta_temperatures::Vector{Float64}
    precipitation_scalings::Vector{Float64}
    bins::Vector{@NamedTuple{lo::Int, hi::Int, center::Float64, area::Float64}}
    weights::Vector{Float64}
    time::Vector{DateTime}
    totals::Dict{Symbol,Array{Float64,3}}
    profiles::Array{Union{Nothing,DimStack},3}
    parameters::Dict{String,Any}
    provenance::Dict{String,Any}
end

"""
    glacier_area_total(row) -> Float64

Total glacier area (km²) in a glacier elevation-class `row`, summed over every hypsometry bin.

This is the cheap way to screen cells for a minimum area: it sums the flat `hyps_*` columns
directly, skipping the bin decoding, sorting and nearest-bin reassignment that
[`glacier_hypsometry_coverage`](@ref) does. Screening a global table is ~47,000 calls, so the
difference is worth having.
"""
glacier_area_total(row) =
    sum(Float64(row[c]) for c in _hyps_colnames_in(row); init = 0.0)

# The `hyps_<lo>_<hi>` columns of a row, in whatever order the table carries them. Order does not
# matter for a sum, so this skips parsing the edges out of the names.
_hyps_colnames_in(row) = filter(c -> startswith(string(c), "hyps_"), propertynames(row))

"""
    glacier_hypsometry_coverage(row; coverage = 0.95)

Select the hypsometry bins of a glacier elevation-class `row` that together hold at least
`coverage` of the cell's glacier area, and distribute the remaining (unmodeled) area onto them.

Returns `(; modeled, weights, total_area)`:

- `modeled`: the selected bins as returned by [`glacier_hypsometry`](@ref), sorted ascending
  by elevation. Bins are chosen **largest-area first**, so the selected set is generally not a
  contiguous elevation range.
- `weights`: glacier area (km²) attributed to each modeled bin — its own area plus the area of
  every unmodeled bin whose nearest modeled bin center it is (ties resolved to the lower
  elevation). `sum(weights) == total_area`, so no glacier area is dropped.
- `total_area`: total glacier area (km²) in the cell across all bins.

`coverage = 1.0` models every populated bin, leaving `weights == area`.
"""
function glacier_hypsometry_coverage(row; coverage::Real = 0.95)
    0 < coverage <= 1 || throw(ArgumentError("coverage must be in (0, 1], got $coverage"))

    # `area_minimum = 0` is the right threshold here: `glacier_hypsometry` filters
    # `area > area_minimum` strictly, so empty bins are already dropped and every returned bin
    # carries real ice.
    bins = glacier_hypsometry(row; area_minimum = 0)
    isempty(bins) && return (; modeled = bins, weights = Float64[], total_area = 0.0)

    total_area = sum(b.area for b in bins)

    # Largest-area first until the cumulative area reaches the coverage target.
    order = sortperm(bins; by = b -> -b.area)
    target = coverage * total_area
    cumulative = 0.0
    selected = Int[]
    for i in order
        push!(selected, i)
        cumulative += bins[i].area
        cumulative >= target && break
    end

    sort!(selected)                      # `bins` is elevation-sorted, so this restores that order
    modeled = bins[selected]
    weights = [b.area for b in modeled]

    # Reassign each unmodeled bin's area to the nearest modeled bin by center elevation.
    # `findmin` returns the first minimum, and `modeled` ascends in elevation, so an exactly
    # equidistant bin goes to the lower-elevation neighbour.
    unmodeled = setdiff(1:length(bins), selected)
    for i in unmodeled
        _, nearest = findmin(m -> abs(m.center - bins[i].center), modeled)
        weights[nearest] += bins[i].area
    end

    return (; modeled, weights, total_area)
end

"""
    ForcingUpToDate(restart_time, new_steps)

Thrown by [`gemb_glacier_cell`](@ref) when a cell's saved output already spans the available
forcing, so there is nothing to extend. This is a normal outcome of re-running a sweep before
new forcing is published — distinct from a failure, so a driver can skip such cells quietly.

Fewer than `MIN_FORCING_STEPS` new steps is already "up to date".
"""
struct ForcingUpToDate <: Exception
    restart_time::DateTime
    new_steps::Int
end

Base.showerror(io::IO, e::ForcingUpToDate) = print(io,
    "ForcingUpToDate: only $(e.new_steps) forcing step(s) follow the saved output time " *
    "$(e.restart_time); at least $MIN_FORCING_STEPS are needed to extend the record")

"""
    ForcingUnavailable(message)

Thrown by [`gemb_glacier_cell`](@ref) when a cell has no usable forcing at all — see
[`forcing_is_complete`](@ref). Like [`ForcingUpToDate`](@ref) this is a normal outcome of
sweeping a global table rather than a failure, so it is a distinct type that a driver can skip
quietly instead of logging as an error.
"""
struct ForcingUnavailable <: Exception
    message::String
end

Base.showerror(io::IO, e::ForcingUnavailable) = print(io, "ForcingUnavailable: ", e.message)

"""
    forcing_is_complete(forcing_data) -> Bool

Whether a `climate_forcing` stack actually carries data for its cell.

ERA5-Land is defined on land only, so a glacier grid cell whose reanalysis grid point falls on
water — common for coastal and island glaciers, e.g. the Aleutians — returns a stack that is
all-`NaN` (and a `NaN` reference elevation). GEMB's `initialize_forcing` catches this as an
unrelated-looking assertion (`temperature_air values unrealistic`), so test for it up front and
skip such cells.
"""
function forcing_is_complete(forcing_data)
    meta = DimensionalData.metadata(forcing_data)
    elevation = get(meta, "elevation", NaN)
    (elevation isa Real && isfinite(elevation)) || return false
    return all(k -> all(isfinite, forcing_data[k]), keys(forcing_data))
end

"""
    gemb_glacier_cell(row, forcing_data, mp; kwargs...) -> GlacierCellRun

Run GEMB for one glacier grid cell over the full outer product of `delta_temperatures` and
`precipitation_scalings`, for every hypsometry bin covering at least `coverage` of the cell's
glacier area, and area-weight the mass fluxes into per-cell totals (kg).

`row` is one element of `eachrow(glacier_elevation_classes)`; `forcing_data` is the raw
`DimStack` from `climate_forcing` for this cell (its metadata supplies the reference surface
elevation the per-bin lapse adjustment raises from).

Per (delta, scaling, bin) the forcing chain is

    adjusted = precipitation_adjust(temperature_adjust(forcing_data, delta), scaling)
    cf       = forcing_at_elevation(adjusted, bin.center - forcing_elevation; lapse_rate)

Each adjustment starts from the original `forcing_data`, so deltas never compound across bins
or perturbations.

# Keywords
- `delta_temperatures`: prescribed air temperature offsets (K).
- `precipitation_scalings`: prescribed precipitation multipliers (dimensionless).
- `coverage = 0.95`: fraction of cell glacier area the modeled bins must cover.
- `lapse_rate = 6.5`: temperature lapse rate (K/km) for the per-bin elevation adjustment.
- `spinup_window`: `(start, stop)` DateTime range averaged into the repeating climatological
  year used for spinup. Defaults to the first 30 complete years of the forcing.
- `max_iterations`, `convergence_delta_density`: passed to `gemb_spinup`.
- `restart`: the value returned by [`read_glacier_cell_restart`](@ref), or `nothing`. When
  given, each run resumes from its saved profile over forcing newer than the saved time and
  the spinup is skipped. The restart's stored run parameters ([`run_parameters`](@ref)) must
  match this run's, or [`RestartParameterMismatch`](@ref) is thrown.
- `force_restart = false`: append even when the restart's run parameters disagree with this
  run's. The seam then separates two differently-configured experiments, and the file's stored
  parameters are overwritten with the new ones.
"""
function gemb_glacier_cell(row, forcing_data, mp::ModelParameters;
                           delta_temperatures = [0.0],
                           precipitation_scalings = [1.0],
                           coverage::Real = 0.95,
                           lapse_rate::Real = 6.5,
                           spinup_window = nothing,
                           max_iterations::Int = 1000,
                           convergence_delta_density = 0.01,
                           restart = nothing,
                           force_restart::Bool = false)

    delta_temperatures = collect(Float64, delta_temperatures)
    precipitation_scalings = collect(Float64, precipitation_scalings)
    parameters = run_parameters(mp; coverage, lapse_rate)

    cov = glacier_hypsometry_coverage(row; coverage)
    isempty(cov.modeled) &&
        throw(ArgumentError("cell has no populated hypsometry bins"))

    # A cell whose reanalysis grid point is on water carries no forcing at all; catch that here
    # rather than letting GEMB's range assertions report it as a units problem.
    forcing_is_complete(forcing_data) ||
        throw(ForcingUnavailable("forcing for this cell is incomplete (all-NaN or no reference " *
                                 "elevation); the reanalysis grid point is most likely over water"))

    meta = DimensionalData.metadata(forcing_data)
    forcing_elevation = Float64(meta["elevation"])

    # Resuming: keep only forcing newer than the saved state. The saved time is the last
    # *output* time, and every forcing step up to and including it has been consumed.
    if restart !== nothing
        _validate_restart(restart, cov.modeled, delta_temperatures, precipitation_scalings)
        # Check *before* the timestep subset, so a parameter change is reported as such rather
        # than masked by a `ForcingUpToDate` from an unextended forcing record.
        _validate_restart_parameters(restart, parameters; force_restart)
        forcing_data = forcing_data[Ti(Where(t -> t > restart.time))]
        n_new = length(dims(forcing_data, Ti))
        n_new >= MIN_FORCING_STEPS || throw(ForcingUpToDate(restart.time, n_new))
        @info "Resuming from saved profile" restart_time=restart.time new_steps=n_new
    end

    if spinup_window === nothing
        forcing_times = collect(dims(forcing_data, Ti))
        spinup_start = DateTime(year(first(forcing_times)), 1, 1)
        spinup_window = (spinup_start, DateTime(year(spinup_start) + 29, 12, 31))
    end

    n_bin = length(cov.modeled)
    n_dt = length(delta_temperatures)
    n_ps = length(precipitation_scalings)

    profiles = Array{Union{Nothing,DimStack}}(nothing, n_bin, n_dt, n_ps)
    totals = Dict{Symbol,Array{Float64,3}}()
    out_time = DateTime[]
    provenance = Dict{String,Any}()

    for (i_ps, pscale) in enumerate(precipitation_scalings),
        (i_dt, delta) in enumerate(delta_temperatures)

        # Both adjustments act on the raw forcing DimStack and are exact identities at
        # delta = 0 / scaling = 1, so the unperturbed run is bit-for-bit the plain forcing.
        adjusted = precipitation_adjust(temperature_adjust(forcing_data, delta), pscale)

        for (i_bin, bin) in enumerate(cov.modeled)
            cf = forcing_at_elevation(adjusted, bin.center - forcing_elevation; lapse_rate)

            profile = restart === nothing ? nothing :
                      get(restart.profiles, (i_bin, i_dt, i_ps), nothing)

            if profile === nothing
                restart === nothing || @warn "No saved profile for this run; spinning up over the truncated forcing" bin=bin.center delta pscale
                cf_spinup = forcing_climatology(cf, spinup_window)
                profile = gemb_spinup(initialize_profile(mp, cf_spinup), cf_spinup, mp;
                                      max_iterations, convergence_delta_density)
            end

            output = gemb(profile, cf, mp)

            # `gemb` only warns when the forcing is shorter than one output period; that
            # leaves an empty time axis, which would otherwise silently contribute nothing.
            n_out = length(dims(output, Ti))
            if n_out == 0
                @warn "GEMB produced no output for this run; skipping bin" bin=bin.center delta pscale
                continue
            end

            if isempty(out_time)
                out_time = collect(dims(output, Ti))
                for v in CELL_TOTAL_VARIABLES
                    totals[v] = zeros(n_out, n_dt, n_ps)
                end
                merge!(provenance, _stack_provenance(output))
            elseif collect(dims(output, Ti)) != out_time
                throw(ErrorException("output time axis differs between bins of the same " *
                                     "cell (bin $(bin.center)); cannot aggregate"))
            end

            # km² -> m², so flux [kg m-2] * area [m2] = mass [kg].
            weight = cov.weights[i_bin] * 1e6
            for v in CELL_MASS_VARIABLES
                @views totals[v][:, i_dt, i_ps] .+= output[v] .* weight
            end
            @views totals[:mass_change][:, i_dt, i_ps] .+=
                (output[:precipitation] .- output[:runoff] .+
                 output[:evaporation_condensation]) .* weight

            # Keep only the restart state; the full output (Z x Ti, ~200 layers x decades of
            # steps) is far too large to hold for every bin of every perturbation.
            profiles[i_bin, i_dt, i_ps] = gemb_profile(output)
        end
    end

    isempty(out_time) && throw(ErrorException("no bin of this cell produced any output"))

    lon, lat = _cell_lonlat(row.geometry)

    return GlacierCellRun(
        Float64(lat), Float64(wrap_lon(lon)), forcing_elevation,
        hasproperty(row, :chunk_id) ? Int(row.chunk_id) : missing,
        hasproperty(row, :glacier_frac) ? Float64(row.glacier_frac) : missing,
        delta_temperatures, precipitation_scalings,
        cov.modeled, cov.weights,
        out_time, totals, profiles, parameters, provenance,
    )
end

# Spinup/climatology provenance carried on the output stack by `gemb` (`_profile_provenance`).
# Values can be `nothing`, `Bool` or `DateTime`; the NetCDF writer encodes them.
function _stack_provenance(output)
    meta = DimensionalData.metadata(output)
    prov = Dict{String,Any}()
    meta isa DimensionalData.NoMetadata && return prov
    for (k, v) in pairs(meta)
        key = string(k)
        (startswith(key, "spinup") || startswith(key, "climatology")) || continue
        prov[key] = v
    end
    return prov
end

# An existing record must describe the same run grid it is being extended into, or the saved
# profiles and mass slabs would be silently paired with the wrong bin or perturbation. The grid
# is structural, so `force_restart` cannot waive it: a mismatch means the arrays do not even
# line up. Shared by the restart path and the NetCDF appender, which read the same three axes
# from different places.
const RUN_GRID_AXES = ("bin_center", "delta_temperature", "precipitation_scaling")

function _assert_run_grid_matches(source, saved, requested)
    for (name, s, r) in zip(RUN_GRID_AXES, saved, requested)
        s == r || throw(ArgumentError("$name in $source ($s) does not match this run ($r); " *
                                      "rebuild the cell from scratch"))
    end
    return nothing
end

_validate_restart(restart, modeled, delta_temperatures, precipitation_scalings) =
    _assert_run_grid_matches("the saved restart state",
                             (restart.bin_centers, restart.delta_temperatures,
                              restart.precipitation_scalings),
                             ([b.center for b in modeled], delta_temperatures,
                              precipitation_scalings))

"""
    RestartParameterMismatch(differences)

Thrown by [`gemb_glacier_cell`](@ref) when the run parameters of an existing cell file disagree
with those of the continuation being appended to it. `differences` maps each disagreeing
attribute to `(saved, requested)`.

Appending across such a change would splice two different experiments into one record, so it is
refused by default. Pass `force_restart = true` to proceed anyway; the file's stored parameters
are then overwritten with the new ones, and the record before the seam no longer reflects them.
"""
struct RestartParameterMismatch <: Exception
    differences::Dict{String,Tuple{Any,Any}}
end

function Base.showerror(io::IO, e::RestartParameterMismatch)
    print(io, "RestartParameterMismatch: the run parameters of the existing cell file do not " *
              "match this run:")
    for key in sort!(collect(keys(e.differences)))
        saved, requested = e.differences[key]
        print(io, "\n  ", key, ": saved ", repr(saved), " vs requested ", repr(requested))
    end
    print(io, "\nAppending would splice two different experiments into one record. Rebuild the " *
              "cell from scratch (delete the file), or pass `force_restart = true` to append " *
              "anyway and overwrite the file's stored parameters.")
end

# Compare the stored run parameters against this run's. Only keys present in both are compared:
# a file written by an older version simply carries fewer of them, and that is a reason to note
# the gap, not to refuse the append.
function _validate_restart_parameters(restart, parameters; force_restart::Bool)
    saved = restart.parameters
    if isempty(saved)
        @warn "The existing cell file stores no run parameters; cannot verify that this " *
              "continuation matches it"
        return nothing
    end

    differences = Dict{String,Tuple{Any,Any}}()
    for (key, requested) in parameters
        haskey(saved, key) || continue
        # Values round-trip through NetCDF attributes, which hold no `Symbol` or `Bool`, so
        # compare the encoded forms the file actually stores.
        _encode_attribute(saved[key]) == _encode_attribute(requested) && continue
        differences[key] = (saved[key], requested)
    end

    missing_keys = setdiff(keys(parameters), keys(saved))
    isempty(missing_keys) ||
        @warn "The existing cell file does not store every run parameter; these could not be verified" unverified=sort!(collect(missing_keys))

    isempty(differences) && return nothing

    if force_restart
        @warn "Appending across a run-parameter change because force_restart = true; the " *
              "record before the seam does not reflect the new parameters" differences
        return nothing
    end
    throw(RestartParameterMismatch(differences))
end
