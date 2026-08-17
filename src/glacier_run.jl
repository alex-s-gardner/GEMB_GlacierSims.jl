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

# The layers that constitute a complete GEMB restart state (`gemb` reads exactly these,
# `GEMB.jl/src/gemb_driver.jl`). Every one is required: `gemb` indexes them by name off the
# profile, so a missing layer is a `FieldError` on the first timestep, not a silent default.
# `age` is part of the state even though no physics reads it — it is the column's only clock,
# and dropping it from the round-trip would restart every continuation's age at whatever the
# saved column happened to hold.
const PROFILE_VARIABLES = (:dz, :temperature, :density, :water, :grain_radius,
                          :grain_dendricity, :grain_sphericity, :age)

# Everything carried as a per-cell total: the area-weighted fluxes plus the budget assembled
# from them. Keys of `GlacierCellRun.totals` and the mass variables in the NetCDF output.
const CELL_TOTAL_VARIABLES = (CELL_MASS_VARIABLES..., :mass_change)

# `gemb` derives its timestep from the first two forcing times
# (`GEMB.jl/src/initialize_forcing.jl`), so a single step is as unrunnable as none at all.
const MIN_FORCING_STEPS = 2

"""
    run_parameters(mp::ModelParameters; coverage, lapse_rate, decoupling_factor = nothing)
        -> Dict{String,Any}

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
function run_parameters(mp::ModelParameters; coverage::Real, lapse_rate::Real,
                        decoupling_factor = nothing)
    params = Dict{String,Any}("hypsometry_coverage" => Float64(coverage),
                              "temperature_lapse_rate" => Float64(lapse_rate),
                              # 1.0 (the identity) rather than an absent key when no correction
                              # was applied, so switching the correction on or off is visible as
                              # a parameter change on restart rather than an unverifiable gap.
                              "glacier_decoupling_factor" =>
                                  decoupling_factor === nothing ? 1.0 : Float64(decoupling_factor))
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
- `decoupling_factor`: the effective on-glacier air temperature decoupling factor `k` applied to
  every run of this cell ([`cell_decoupling_factor`](@ref)), or `nothing` when the forcing was
  left ambient. Like `forcing_elevation` this is a property of the cell rather than of the
  perturbation grid, so it is carried as a scalar and written as one.
- `parameters`: the settings that define how the cell was run ([`run_parameters`](@ref)),
  written as global attributes and checked against on restart. Includes `hypsometry_coverage`,
  `temperature_lapse_rate` and `glacier_decoupling_factor`.
- `provenance`: the `spinup_*`/`climatology_*` keys `gemb` attached to the output. Recorded but
  *not* checked on restart — a resumed run performs no spinup, so these are expected to differ.
"""
struct GlacierCellRun
    latitude::Float64
    longitude::Float64
    forcing_elevation::Float64
    decoupling_factor::Union{Float64,Nothing}
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
    cell_decoupling_factor(row; max_distance = 10.0)

Effective on-glacier air temperature decoupling factor `k` for a glacier grid cell, or `nothing`
when the cell has no published `k`.

`k` comes from the Shaw et al. (2025) per-glacier table via `GEMB_ClimateForcing.glacier_decoupling`,
looked up by nearest glacier centroid to the cell center (within `max_distance` km). It corrects an
*ambient* (off-glacier) temperature to on-glacier conditions, so it only applies to the part of the
reanalysis cell that is not already glacier: ERA5-Land's `:glm` glacier mask says what fraction of
the cell its land-surface scheme already treats as ice, and that fraction needs no correction.
The published `k` is therefore weighted by the non-glacier fraction `1 - glm`,

    k_eff = 1 - (1 - k) * (1 - glm)

which is the full correction where the cell carries no glacier (`glm = 0`) and the identity where
the cell is entirely glacier (`glm = 1`). A cell with no `:glm` column, or a `missing` value, is
treated as `glm = 0` — the uncorrected reanalysis assumption, hence the full correction.

Returns `(; decoupling_factor, decoupling_factor_published, glm, rgi_id, distance)`, or `nothing`
when the table has no glacier within `max_distance` — RGI regions 05 (Greenland periphery) and 19
(Antarctic) are absent from it entirely, and those cells are then run on ambient forcing rather
than failing.

The `:glm` column is required. A table without it cannot weight the correction at all, and
defaulting to the unweighted published factor would apply a silently different correction to every
cell in the sweep — so this throws instead. A table predating the current invariant column naming
(`glm_frac`) is migrated in place by `scripts/migrate_invariant_colnames.jl`.
"""
function cell_decoupling_factor(row; max_distance::Real = 10.0)
    hasproperty(row, :glm) || throw(ArgumentError(
        "glacier elevation-class row has no `:glm` column, so the decoupling factor cannot be " *
        "weighted by the cell's non-glacier fraction. A table built before the current invariant " *
        "column naming carries `:glm_frac`; run `scripts/migrate_invariant_colnames.jl` to " *
        "rename it in place."))

    lon, lat = _cell_lonlat(row.geometry)

    # Positional lookups error rather than return a sentinel when nothing is close enough (or the
    # RGI region is uncovered). No `k` is a run-on-ambient-forcing outcome, not a failure.
    found = try
        glacier_decoupling(Float64(lat), wrap_lon(Float64(lon)); max_distance)
    catch e
        e isa InterruptException && rethrow()
        return nothing
    end

    # `missing`/`NaN` are real data gaps (an invariant field undefined for the cell), not schema
    # drift, so they fall back to the uncorrected reanalysis assumption rather than throwing.
    glm_raw = row.glm
    glm = (glm_raw === missing || !isfinite(Float64(glm_raw))) ? 0.0 :
          clamp(Float64(glm_raw), 0.0, 1.0)

    # Weighted toward the identity as the cell becomes more glaciated; exact at both ends.
    return (; decoupling_factor = 1 - (1 - found.k) * (1 - glm),
            decoupling_factor_published = found.k, glm,
            rgi_id = found.rgi_id, distance = found.distance)
end

"""
    decoupling_factor_label(k; digits = 3) -> String

Compact label for a decoupling factor, for a plot header or a log line: `"k=0.774"`, or `""` when
`k` is `nothing`.

Empty rather than `"k=1.0"` for no correction: a factor printed on every figure of an uncorrected
sweep is noise, and its *absence* is what distinguishes ambient forcing from a cell the correction
happened to leave alone.
"""
decoupling_factor_label(k::Nothing; digits::Int = 3) = ""
decoupling_factor_label(k::Real; digits::Int = 3) = "k=$(round(k; digits))"

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
    cf       = forcing_at_elevation(adjusted, bin.center - forcing_elevation;
                                    lapse_rate, decoupling_factor)

Each adjustment starts from the original `forcing_data`, so deltas never compound across bins
or perturbations.

# Keywords
- `delta_temperatures`: prescribed air temperature offsets (K).
- `precipitation_scalings`: prescribed precipitation multipliers (dimensionless).
- `coverage = 0.95`: fraction of cell glacier area the modeled bins must cover.
- `lapse_rate = 6.5`: temperature lapse rate (K/km) for the per-bin elevation adjustment.
- `glacier_decoupling = true`: correct ambient air temperature to on-glacier conditions with the
  Shaw et al. (2025) factor `k`, weighted by the cell's non-glacier fraction `1 - glm`
  ([`cell_decoupling_factor`](@ref)). Applied after the elevation adjustment, as that correction
  requires. A cell with no `k` in the table (RGI regions 05 and 19 are not covered) is run on
  ambient forcing with no correction at all. Set `false` to skip the lookup entirely; pass a
  `Real` to prescribe `k` directly, bypassing both the table and the `glm` weighting.
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
- `threaded = true`: run the (bin x delta x scaling) simulations on all available threads.
  Each simulation is independent — `gemb`/`gemb_spinup` copy their state out of `profile` and
  read `mp` (an immutable struct) without mutating any shared object — so the parallel result is
  identical to the serial one, including bit-for-bit totals: the area-weighted sum is reduced in
  a fixed index order after the tasks finish, not accumulated as they land. Set `false` to run
  serially (useful when the caller is already saturating the machine by running many cells in
  parallel, as grouping cells by `chunk_id` does). Thread count comes from `julia -t N`.
- `on_output = nothing`: called as `on_output(output; bin, delta, pscale, decoupling_factor)` with
  each simulation's full output `DimStack`, for inspection (e.g. `gemb_plot_output`) or diagnostics.
  The stack is otherwise dropped as soon as its flux vectors are extracted, so this is the only
  place it can be reached; the callback must not retain it, or the memory the per-task extraction
  avoids is held after all. Called from inside the task, so with `threaded = true` it runs
  concurrently on an unspecified thread and in an unspecified order — make it thread-safe, or
  pass `threaded = false` alongside it (plotting backends generally are not).
"""
function gemb_glacier_cell(row, forcing_data, mp::ModelParameters;
                           delta_temperatures = [0.0],
                           precipitation_scalings = [1.0],
                           coverage::Real = 0.95,
                           lapse_rate::Real = 6.5,
                           glacier_decoupling = true,
                           spinup_window = nothing,
                           max_iterations::Int = 1000,
                           convergence_delta_density = 0.01,
                           restart = nothing,
                           force_restart::Bool = false,
                           threaded::Bool = true,
                           on_output = nothing)

    delta_temperatures = collect(Float64, delta_temperatures)
    precipitation_scalings = collect(Float64, precipitation_scalings)

    # One lookup per cell, not per (bin x delta x scaling): `k` is a property of the glacier, not
    # of the perturbation or the elevation band. `nothing` here means every run stays ambient.
    decoupling_factor = resolve_decoupling_factor(row, glacier_decoupling)

    parameters = run_parameters(mp; coverage, lapse_rate, decoupling_factor)

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

    # The (bin x delta x scaling) simulations are mutually independent, so they are flattened
    # into one task list and run on all threads. Flattening rather than threading the outer loop
    # matters: bins per cell range from 1 to ~38 while perturbations are fixed at a handful, so
    # threading either loop alone would leave threads idle on cells whose loop is shorter than
    # the thread count. One flat list of n_bin*n_dt*n_ps tasks always has work for everyone.
    tasks = [(i_bin, i_dt, i_ps) for i_ps in 1:n_ps, i_dt in 1:n_dt, i_bin in 1:n_bin]
    tasks = vec(tasks)

    # Each task writes only its own slot; nothing is accumulated in place. The area-weighted sum
    # happens afterwards in a fixed order, so the totals do not depend on completion order and
    # the threaded run reproduces the serial one exactly (floating-point addition is not
    # associative, so an accumulate-as-they-land reduction would not).
    #
    # A task keeps only what the reduction needs — the per-variable flux vectors (Ti), the time
    # axis, and the restart profile — never the full output stack. Serially that stack was
    # transient, but held for every task at once it would be ~n_bin*n_dt*n_ps x (Z x Ti), which
    # for a 38-bin cell is hundreds of ~200-layer x decades-of-steps arrays live simultaneously.
    results = Vector{Union{Nothing,NamedTuple}}(nothing, length(tasks))

    run_one = function (t)
        i_bin, i_dt, i_ps = tasks[t]
        bin    = cov.modeled[i_bin]
        delta  = delta_temperatures[i_dt]
        pscale = precipitation_scalings[i_ps]

        # Both adjustments act on the raw forcing DimStack and are exact identities at
        # delta = 0 / scaling = 1, so the unperturbed run is bit-for-bit the plain forcing.
        # Rebuilt per task rather than hoisted per perturbation: the adjustment is cheap next
        # to a spinup, and sharing one `adjusted` across bins would hand the same object to
        # concurrent tasks.
        adjusted = precipitation_adjust(temperature_adjust(forcing_data, delta), pscale)
        cf = forcing_at_elevation(adjusted, bin.center - forcing_elevation;
                                  lapse_rate, decoupling_factor)

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
        if length(dims(output, Ti)) == 0
            @warn "GEMB produced no output for this run; skipping bin" bin=bin.center delta pscale
            return nothing
        end

        # The caller's only window onto the full stack: below this point only the flux vectors
        # and the restart profile survive. `decoupling_factor` is constant across the sweep but
        # passed anyway, so a hook can label a figure without reaching back for the cell.
        on_output === nothing ||
            on_output(output; bin, delta, pscale, decoupling_factor)

        # Extract the flux vectors and the restart profile here so `output` goes out of scope
        # with the task rather than being retained until the reduction.
        fluxes = Dict{Symbol,Vector{Float64}}(v => Vector{Float64}(output[v])
                                             for v in CELL_MASS_VARIABLES)
        return (; i_bin, i_dt, i_ps,
                time = collect(dims(output, Ti)),
                fluxes,
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

    # Serial reduction in task order: allocate the totals from the first run that produced
    # output, check every other run against that time axis, then area-weight.
    for res in results
        res === nothing && continue
        if isempty(out_time)
            out_time = res.time
            for v in CELL_TOTAL_VARIABLES
                totals[v] = zeros(length(out_time), n_dt, n_ps)
            end
            merge!(provenance, res.provenance)
        elseif res.time != out_time
            throw(ErrorException("output time axis differs between bins of the same " *
                                 "cell (bin $(cov.modeled[res.i_bin].center)); cannot aggregate"))
        end

        # km² -> m², so flux [kg m-2] * area [m2] = mass [kg].
        weight = cov.weights[res.i_bin] * 1e6
        for v in CELL_MASS_VARIABLES
            @views totals[v][:, res.i_dt, res.i_ps] .+= res.fluxes[v] .* weight
        end
        @views totals[:mass_change][:, res.i_dt, res.i_ps] .+=
            (res.fluxes[:precipitation] .- res.fluxes[:runoff] .+
             res.fluxes[:evaporation_condensation]) .* weight

        # Keep only the restart state; the full output (Z x Ti, ~200 layers x decades of
        # steps) is far too large to hold for every bin of every perturbation.
        profiles[res.i_bin, res.i_dt, res.i_ps] = res.profile
    end

    isempty(out_time) && throw(ErrorException("no bin of this cell produced any output"))

    lon, lat = _cell_lonlat(row.geometry)

    return GlacierCellRun(
        Float64(lat), Float64(wrap_lon(lon)), forcing_elevation, decoupling_factor,
        hasproperty(row, :chunk_id) ? Int(row.chunk_id) : missing,
        hasproperty(row, :glacier_frac) ? Float64(row.glacier_frac) : missing,
        delta_temperatures, precipitation_scalings,
        cov.modeled, cov.weights,
        out_time, totals, profiles, parameters, provenance,
    )
end

"""
    resolve_decoupling_factor(row, glacier_decoupling) -> Float64 or nothing

Resolve the `glacier_decoupling` keyword of [`gemb_glacier_cell`](@ref) to the `k` handed to every
run of this cell, or `nothing` for ambient forcing. `false` skips the lookup; a `Real` is taken as
`k` verbatim (no `glm` weighting — the caller has prescribed the effective factor); `true` looks
`k` up and weights it by the cell's non-glacier fraction ([`cell_decoupling_factor`](@ref)). A cell
absent from the table gets no correction at all, which is why this is not an error.

Public because it is idempotent — passing its result back as `glacier_decoupling` resolves to
itself — so a driver can resolve `k` once per cell, build [`run_parameters`](@ref) from it to check
an existing file *before* fetching any forcing, and then hand the same value to the run. Calling it
per cell rather than per sweep matters: `k` is looked up by position, so it is a property of the
cell.
"""
resolve_decoupling_factor(row, glacier_decoupling::Bool) =
    glacier_decoupling ? _lookup_decoupling_factor(row) : nothing
resolve_decoupling_factor(row, glacier_decoupling::Real) = Float64(glacier_decoupling)
resolve_decoupling_factor(row, glacier_decoupling::Nothing) = nothing

function _lookup_decoupling_factor(row)
    found = cell_decoupling_factor(row)
    if found === nothing
        @info "No Shaw et al. (2025) decoupling factor for this cell; running on ambient forcing"
        return nothing
    end
    @info "Applying glacier decoupling weighted by the non-glacier fraction" k=found.decoupling_factor k_published=found.decoupling_factor_published glm=found.glm rgi_id=found.rgi_id match_distance_km=round(found.distance, digits=2)
    # An entirely glaciated cell weights k to exactly 1.0, which is the identity — skip the
    # adjustment rather than paying for a no-op pass over the whole forcing record.
    return found.decoupling_factor >= 1 ? nothing : found.decoupling_factor
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

"""
    run_parameter_differences(saved, requested) -> Dict{String,Tuple{Any,Any}}

Where a file's stored run parameters disagree with a requested run's, as
`key => (saved, requested)`. `saved` is what [`read_glacier_cell_parameters`](@ref) (or the
`parameters` field of [`read_glacier_cell_restart`](@ref)) returned; `requested` is a
[`run_parameters`](@ref) dict.

Only keys present in both are compared: a file written by an older version of `run_parameters`
simply carries fewer of them, and that is a reason to note the gap rather than to refuse an append.
Values are compared in the NetCDF-encoded form the file stores, since attributes hold no `Symbol`
or `Bool`.

This is what [`gemb_glacier_cell`](@ref) checks a restart against, exposed so a driver can make the
same comparison up front — before paying for a forcing download — and get the same verdict.
"""
function run_parameter_differences(saved, requested)
    differences = Dict{String,Tuple{Any,Any}}()
    for (key, value) in requested
        haskey(saved, key) || continue
        _encode_attribute(saved[key]) == _encode_attribute(value) && continue
        differences[key] = (saved[key], value)
    end
    return differences
end

# Compare the stored run parameters against this run's, and refuse the append on a disagreement
# unless the caller has forced it.
function _validate_restart_parameters(restart, parameters; force_restart::Bool)
    saved = restart.parameters
    if isempty(saved)
        @warn "The existing cell file stores no run parameters; cannot verify that this " *
              "continuation matches it"
        return nothing
    end

    differences = run_parameter_differences(saved, parameters)

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
