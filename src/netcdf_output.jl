"""
CF-compliant NetCDF output for glacier-cell GEMB runs.

GEMB.jl ships a complete CF *attribute* table (`GEMB_CF_ATTRIBUTES`, `cf_attributes`) but no
writer, so the attributes here are taken from that table rather than restated — with `units`
deliberately overridden from `kg m-2` to `kg` for the area-weighted per-cell totals.

One file per glacier grid cell. Every mass variable carries the perturbation grid as
dimensions, so a whole sensitivity sweep for a cell lives in one file:

    melt(time, delta_temperature, precipitation_scaling)                         [kg]
    restart/density(layer, bin, delta_temperature, precipitation_scaling)        [kg m-3]

The `restart` group holds the final firn profile of every run, which is the complete state
`gemb` needs to extend the record when new forcing arrives.
"""

# Time encoding. A fixed epoch keeps the encoded values stable across appends.
const NC_TIME_UNITS = "days since 1950-01-01 00:00:00"
const NC_CALENDAR = "proleptic_gregorian"

# Encoded explicitly rather than handing NCDatasets `DateTime` values so the on-disk numbers are
# identical whichever path (create or append) wrote them. The conversion itself is NCDatasets'
# own CF time codec, so it agrees with what a reader applies.
_nc_encode_time(t::DateTime) = NCDatasets.CFTime.timeencode([t], NC_TIME_UNITS, NC_CALENDAR)[1]
_nc_decode_time(x::Real) = NCDatasets.CFTime.timedecode(x, NC_TIME_UNITS, NC_CALENDAR)
# NCDatasets applies CF decoding on read, so a `time` variable carrying `units`/`calendar`
# already comes back as `DateTime`. Accept either form so callers need not know which.
_nc_decode_time(t::DateTime) = t

const NC_FILL = NaN

"""
    write_glacier_cell_netcdf(path, run::GlacierCellRun; kwargs...)

Write a [`GlacierCellRun`](@ref) to a CF-1.11 NetCDF file at `path`, overwriting any existing
file. The `time` dimension is unlimited so [`append_glacier_cell_netcdf`](@ref) can extend it.

`institution` and `references` are written as global attributes when given.
"""
function write_glacier_cell_netcdf(path::AbstractString, run::GlacierCellRun;
                                   institution = nothing, references = nothing)
    mkpath(dirname(abspath(path)))
    n_layer = _max_profile_length(run.profiles)

    NCDatasets.NCDataset(path, "c") do ds
        NCDatasets.defDim(ds, "time", Inf)          # unlimited, for append
        NCDatasets.defDim(ds, "delta_temperature", length(run.delta_temperatures))
        NCDatasets.defDim(ds, "precipitation_scaling", length(run.precipitation_scalings))
        NCDatasets.defDim(ds, "bin", length(run.bins))
        NCDatasets.defDim(ds, "layer", n_layer)

        _write_globals!(ds, run; institution, references)
        _write_coordinates!(ds, run, n_layer)
        _write_mass_variables!(ds, run)
        _write_restart_group!(ds, run, n_layer)
    end

    return path
end

"""
    append_glacier_cell_netcdf(path, run::GlacierCellRun)

Append a continuation run to an existing cell file: extend the unlimited `time` dimension with
the new mass totals and overwrite the restart group with the new final profiles.

The new record must start strictly after the file's last time; the run grid (bins, temperature
deltas, precipitation scalings) must match the file's.
"""
function append_glacier_cell_netcdf(path::AbstractString, run::GlacierCellRun)
    NCDatasets.NCDataset(path, "a") do ds
        n_existing = length(ds["time"])
        last_time = _nc_decode_time(ds["time"][n_existing])
        first(run.time) > last_time ||
            throw(ArgumentError("new record starts at $(first(run.time)), which is not after " *
                                "the file's last time $(last_time); would duplicate or " *
                                "overlap existing output"))

        _assert_grid_matches(ds, run)

        # `gemb_glacier_cell` has already refused a parameter change unless the caller passed
        # `force_restart`, so by here the run's parameters are what the file should advertise.
        # Rewriting them keeps the file honest in the forced case; in the normal case they are
        # identical and this is a no-op.
        _write_run_parameters!(ds, run)
        ds.attrib["history"] = get(ds.attrib, "history", "") *
            "\n$(Dates.format(now(), "yyyy-mm-ddTHH:MM:SS")): extended through " *
            "$(Dates.format(last(run.time), "yyyy-mm-ddTHH:MM:SS")) by " *
            "GEMB_GlacierSims.append_glacier_cell_netcdf"

        n_new = length(run.time)
        idx = (n_existing + 1):(n_existing + n_new)
        ds["time"][idx] = _nc_encode_time.(run.time)
        for v in CELL_TOTAL_VARIABLES
            ds[string(v)][idx, :, :] = run.totals[v]
        end

        # The restart group is state, not a time series: overwrite it in place. Its `layer`
        # dimension is fixed at creation, so a continuation whose columns grew past it cannot
        # be stored — that means the grid changed and the cell needs a fresh file.
        n_layer = ds.dim["layer"]
        deepest = _max_profile_length(run.profiles)
        deepest <= n_layer ||
            throw(ArgumentError("continuation profiles are deeper ($deepest layers) than the " *
                                "file's layer dimension ($n_layer); rewrite the cell from " *
                                "scratch"))
        _fill_restart!(ds.group["restart"], run, n_layer)
    end

    return path
end

"""
    read_glacier_cell_status(path) -> NamedTuple or nothing

What an existing cell file already covers, without reading the firn state. Returns `nothing` if
the file does not exist, so a driver can fall through to a cold start.

Returns `(; time, bin_centers, delta_temperatures, precipitation_scalings, parameters)` — the same
fields as [`read_glacier_cell_restart`](@ref) minus `profiles`. `time` is the last output time in
the file, i.e. the point a continuation must start after.

This exists to be cheap: the restart group is a `(layer x bin x delta x scaling)` slab per profile
variable, which for a 38-bin cell is tens of MB, and a driver deciding whether a cell needs any
work at all does not need it. Reading only the coordinates and attributes makes the up-front check
— do the run parameters still match, and is there any new forcing to fetch — cost a file open.
"""
function read_glacier_cell_status(path::AbstractString)
    isfile(path) || return nothing

    NCDatasets.NCDataset(path, "r") do ds
        n_time = length(ds["time"])
        n_time == 0 && return nothing        # created but never filled; treat as no record
        return (; time = _nc_decode_time(ds["time"][n_time]),
                bin_centers = collect(Float64, ds["bin_center"][:]),
                delta_temperatures = collect(Float64, ds["delta_temperature"][:]),
                precipitation_scalings = collect(Float64, ds["precipitation_scaling"][:]),
                parameters = _read_run_parameters(ds))
    end
end

"""
    read_glacier_cell_restart(path) -> NamedTuple or nothing

Read the restart state written by [`write_glacier_cell_netcdf`](@ref). Returns `nothing` if the
file does not exist, so a driver can fall through to a cold start.

Returns `(; time, profiles, bin_centers, delta_temperatures, precipitation_scalings, parameters)`
where `profiles` maps `(i_bin, i_delta, i_scaling)` to the profile `DimStack` that `gemb`
accepts (the layers in [`PROFILE_VARIABLES`](@ref)), truncated to that run's real column length. `time` is the last output time in the
file, i.e. the point the continuation must start after. `parameters` is the file's stored run
configuration ([`run_parameters`](@ref)), which `gemb_glacier_cell` checks the continuation
against.
"""
function read_glacier_cell_restart(path::AbstractString)
    isfile(path) || return nothing

    NCDatasets.NCDataset(path, "r") do ds
        haskey(ds.group, "restart") || return nothing
        g = ds.group["restart"]

        bin_centers = collect(Float64, ds["bin_center"][:])
        deltas = collect(Float64, ds["delta_temperature"][:])
        scalings = collect(Float64, ds["precipitation_scaling"][:])
        n_time = length(ds["time"])
        time = _nc_decode_time(ds["time"][n_time])

        valid = g["valid_layers"][:, :, :]
        profiles = Dict{Tuple{Int,Int,Int},DimStack}()

        # Files written before a state layer was added to `PROFILE_VARIABLES` are missing it
        # here. Reading on would hand `gemb` an incomplete profile and fail deep in the first
        # timestep with a bare `FieldError`, so name the gap and the remedy instead.
        missing_vars = [v for v in PROFILE_VARIABLES if !haskey(g, string(v))]
        isempty(missing_vars) ||
            throw(ArgumentError("the restart group in $path is missing " *
                                join(missing_vars, ", ") * "; it predates those layers " *
                                "becoming part of the saved state and cannot be continued. " *
                                "Delete the file to rebuild the cell from scratch."))

        # One hyperslab per variable rather than one per (variable, bin, delta, scaling): the
        # runs are all padded to the same `layer` dimension, so the whole array reads at once
        # and is then sliced in memory. A full perturbation grid is otherwise hundreds of reads.
        columns = Dict(v => g[string(v)][:, :, :, :] for v in PROFILE_VARIABLES)
        layer_metadata = cf_layer_index_attributes()

        for i_ps in eachindex(scalings), i_dt in eachindex(deltas), i_bin in eachindex(bin_centers)
            n = valid[i_bin, i_dt, i_ps]
            (ismissing(n) || n <= 0) && continue     # bin that produced no output
            n = Int(n)
            zdim = Z(1:n; metadata = layer_metadata)
            layers = NamedTuple(
                v => DimArray(collect(Float64, @view columns[v][1:n, i_bin, i_dt, i_ps]), (zdim,))
                for v in PROFILE_VARIABLES)
            profiles[(i_bin, i_dt, i_ps)] =
                DimStack(layers; layermetadata = cf_layermetadata(layers; time_axis = false))
        end

        return (; time, profiles, bin_centers,
                delta_temperatures = deltas, precipitation_scalings = scalings,
                parameters = _read_run_parameters(ds))
    end
end

# ---------------------------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------------------------

_max_profile_length(profiles) =
    maximum(p -> p === nothing ? 0 : length(p[:dz]), profiles; init = 0)

# Copy an attribute table onto a variable, optionally dropping keys the write does not want.
function _set_attributes!(v, attributes; skip = ())
    for (k, val) in attributes
        k in skip || (v.attrib[k] = val)
    end
    return v
end

function _write_globals!(ds, run::GlacierCellRun; institution, references)
    for (k, v) in GEMB_CF_GLOBAL_ATTRIBUTES
        ds.attrib[k] = v
    end
    # GEMB.jl deliberately omits `history` to keep its output deterministic; a file on disk
    # should record when it was made, so it is added here instead.
    ds.attrib["history"] = "$(Dates.format(now(), "yyyy-mm-ddTHH:MM:SS")): created by " *
                           "GEMB_GlacierSims.write_glacier_cell_netcdf"
    ds.attrib["title"] = "GEMB glacier grid-cell surface mass balance and firn state"
    ds.attrib["featureType"] = "timeSeries"
    institution === nothing || (ds.attrib["institution"] = institution)
    references === nothing || (ds.attrib["references"] = references)

    ds.attrib["hypsometry_bin_selection"] =
        "bins chosen largest-area-first until cumulative glacier area reached " *
        "hypsometry_coverage of the cell total; the area of unselected bins was added to " *
        "the nearest selected bin by center elevation"
    ds.attrib["temperature_lapse_rate_units"] = "K km-1"
    ismissing(run.chunk_id) || (ds.attrib["chunk_id"] = run.chunk_id)
    ismissing(run.glacier_frac) || (ds.attrib["glacier_fraction"] = run.glacier_frac)

    _write_run_parameters!(ds, run)

    # NetCDF attributes hold no `nothing`, `Bool` or `DateTime`, so encode those.
    for (k, v) in run.provenance
        enc = _encode_attribute(v)
        enc === nothing || (ds.attrib[k] = enc)
    end
    return nothing
end

# The run parameters, plus the roster naming them. Reading the roster back beats recomputing the
# key set, so a file written by a different version of `run_parameters` is still read correctly.
function _write_run_parameters!(ds, run::GlacierCellRun)
    names = sort!(collect(keys(run.parameters)))
    for k in names
        enc = _encode_attribute(run.parameters[k])
        enc === nothing || (ds.attrib[k] = enc)
    end
    ds.attrib["run_parameters"] = join(names, " ")
    ds.attrib["run_parameters_comment"] =
        "The settings that define how this cell was run; a continuation must match them or " *
        "gemb_glacier_cell refuses to append without force_restart = true. Excludes what is " *
        "expected to differ between a run and its continuation: the spinup_*/climatology_* " *
        "provenance (a resumed run performs no spinup) and history."
    return nothing
end

"""
    read_glacier_cell_parameters(path) -> Dict{String,Any} or nothing

Read the run parameters stored by [`write_glacier_cell_netcdf`](@ref), as named by the file's
`run_parameters` attribute. Returns `nothing` if the file does not exist, and an empty `Dict` if
it stores no parameter roster (a file written before they were recorded).

Values come back in their NetCDF-encoded form — `Symbol`s and `Bool`s as strings — which is the
form [`gemb_glacier_cell`](@ref) compares against.
"""
function read_glacier_cell_parameters(path::AbstractString)
    isfile(path) || return nothing
    NCDatasets.NCDataset(path, "r") do ds
        _read_run_parameters(ds)
    end
end

function _read_run_parameters(ds)
    haskey(ds.attrib, "run_parameters") || return Dict{String,Any}()
    names = split(ds.attrib["run_parameters"])
    return Dict{String,Any}(k => ds.attrib[k] for k in names if haskey(ds.attrib, k))
end

_encode_attribute(::Nothing) = nothing
_encode_attribute(v::Bool) = v ? "true" : "false"
_encode_attribute(v::DateTime) = Dates.format(v, "yyyy-mm-ddTHH:MM:SS")
_encode_attribute(v::Union{AbstractString,Real}) = v
_encode_attribute(v) = string(v)

function _write_coordinates!(ds, run::GlacierCellRun, n_layer::Int)
    t = NCDatasets.defVar(ds, "time", Float64, ("time",))
    _set_attributes!(t, cf_time_attributes())
    t.attrib["units"] = NC_TIME_UNITS
    t.attrib["calendar"] = NC_CALENDAR
    t.attrib["long_name"] = "end of the output accumulation interval"
    # `time` is unlimited and therefore starts at length 0, so `t[:]` would write nothing —
    # the range has to be given explicitly. Same for every variable on this dimension.
    t[1:length(run.time)] = _nc_encode_time.(run.time)

    dt = NCDatasets.defVar(ds, "delta_temperature", Float64, ("delta_temperature",))
    dt.attrib["units"] = "K"
    dt.attrib["long_name"] = "prescribed uniform air temperature offset"
    dt.attrib["comment"] = "Applied to the forcing before the elevation lapse adjustment; " *
                           "vapour pressure (at constant relative humidity) and downward " *
                           "longwave are adjusted consistently."
    dt[:] = run.delta_temperatures

    ps = NCDatasets.defVar(ds, "precipitation_scaling", Float64, ("precipitation_scaling",))
    ps.attrib["units"] = "1"
    ps.attrib["long_name"] = "prescribed precipitation multiplier"
    ps[:] = run.precipitation_scalings

    for (name, long_name, values) in (
        ("bin_center", "hypsometry bin center elevation", [b.center for b in run.bins]),
        ("bin_lower", "hypsometry bin lower edge elevation", Float64[b.lo for b in run.bins]),
        ("bin_upper", "hypsometry bin upper edge elevation", Float64[b.hi for b in run.bins]),
    )
        v = NCDatasets.defVar(ds, name, Float64, ("bin",))
        v.attrib["units"] = "m"
        v.attrib["standard_name"] = "height_above_reference_ellipsoid"
        v.attrib["long_name"] = long_name
        v.attrib["positive"] = "up"
        v[:] = values
    end

    for (name, long_name, values, comment) in (
        ("bin_area", "glacier area within the hypsometry bin",
         [b.area for b in run.bins], ""),
        ("bin_weight", "glacier area attributed to the hypsometry bin", run.weights,
         "bin_area plus the area of unmodeled bins assigned to this bin as their nearest " *
         "modeled bin center. Sums to glacier_area_total, and is the weight used for the " *
         "per-cell mass totals."),
    )
        v = NCDatasets.defVar(ds, name, Float64, ("bin",))
        v.attrib["units"] = "km2"
        v.attrib["long_name"] = long_name
        isempty(comment) || (v.attrib["comment"] = comment)
        v[:] = values
    end

    # GEMB's own attributes for this dimension: an index, deliberately carrying no
    # `standard_name` and no `positive`, so it is not presented as a physical height.
    lay = NCDatasets.defVar(ds, "layer", Int32, ("layer",))
    _set_attributes!(lay, cf_layer_index_attributes())
    lay[:] = Int32.(1:n_layer)

    for (name, units, long_name, value, standard_name) in (
        ("latitude", "degrees_north", "cell center latitude", run.latitude, "latitude"),
        ("longitude", "degrees_east", "cell center longitude", run.longitude, "longitude"),
        ("forcing_elevation", "m", "reanalysis cell surface elevation",
         run.forcing_elevation, "surface_altitude"),
        ("glacier_area_total", "km2", "total glacier area in the cell", sum(run.weights), ""),
    )
        v = NCDatasets.defVar(ds, name, Float64, ())
        v.attrib["units"] = units
        v.attrib["long_name"] = long_name
        isempty(standard_name) || (v.attrib["standard_name"] = standard_name)
        v[] = value
    end

    # A property of the cell, not of the perturbation grid, so it is a scalar alongside
    # `forcing_elevation`. `NaN` (the fill) when the forcing was left ambient — a cell with no
    # published k must be distinguishable from one corrected by exactly 1.0.
    dk = NCDatasets.defVar(ds, "glacier_decoupling_factor", Float64, (); fillvalue = NC_FILL)
    dk.attrib["units"] = "1"
    dk.attrib["long_name"] = "on-glacier air temperature decoupling factor"
    dk.attrib["comment"] =
        "Effective Shaw et al. (2025) factor k applied to this cell's forcing, weighted by the " *
        "non-glacier fraction as 1 - (1 - k)*(1 - glm) and applied after the elevation " *
        "adjustment. Absent (fill) when no correction was applied — either it was disabled or " *
        "the cell has no published k (RGI regions 05 and 19 are not covered by the dataset). " *
        "The global attribute of the same name records the same setting as the identity 1.0 " *
        "rather than a fill, because it is compared numerically on restart and a fill would " *
        "never equal itself."
    dk[] = run.decoupling_factor === nothing ? NC_FILL : run.decoupling_factor
    return nothing
end

function _write_mass_variables!(ds, run::GlacierCellRun)
    dimnames = ("time", "delta_temperature", "precipitation_scaling")
    for name in CELL_TOTAL_VARIABLES
        v = NCDatasets.defVar(ds, string(name), Float64, dimnames; fillvalue = NC_FILL)

        if name === :mass_change
            v.attrib["units"] = "kg"
            v.attrib["long_name"] = "glacier mass change in the cell"
            v.attrib["comment"] =
                "Column mass budget, $(CELL_MASS_CHANGE_FORMULA), area-weighted by " *
                "bin_weight and summed over the modeled hypsometry bins. `rain` is the " *
                "liquid fraction of `precipitation` and so is not added again; `refreeze` " *
                "is an internal phase change that moves no mass across the column boundary; " *
                "`evaporation_condensation` is positive for mass gain."
        else
            # Carry GEMB's own CF attributes, but the area weighting changes the quantity from
            # a per-area flux to a mass, so `units` must not be reused as-is. `standard_name`
            # is dropped for the same reason: the CF names are amount-per-area quantities.
            cf = cf_attributes(name)
            _set_attributes!(v, cf; skip = ("units", "standard_name"))
            v.attrib["units"] = "kg"
            v.attrib["comment"] =
                get(cf, "comment", "") * (haskey(cf, "comment") ? " " : "") *
                "Per-bin GEMB output (kg m-2) multiplied by bin_weight and summed over the " *
                "modeled hypsometry bins, giving a mass for the cell's glacier area."
        end

        v.attrib["cell_methods"] = "time: sum area: sum"
        v.attrib["coordinates"] = "latitude longitude"
        v[1:length(run.time), :, :] = run.totals[name]
    end
    return nothing
end

function _write_restart_group!(ds, run::GlacierCellRun, n_layer::Int)
    g = NCDatasets.defGroup(ds, "restart")
    g.attrib["comment"] =
        "Firn column state at the last output time of each run — the complete state `gemb` " *
        "needs to extend the record. Columns shorter than the layer dimension are padded " *
        "with the fill value; valid_layers gives each column's real length."
    g.attrib["restart_fidelity"] =
        "A resumed run is close to but not identical to a continuous one: GEMB re-zeros its " *
        "per-step evaporation/condensation and surface-melt accumulators on restart, so the " *
        "trajectory diverges slightly at the seam (order 0.1% in cumulative melt and runoff). " *
        "Separately, extending the forcing record changes the climatology the spinup uses, so " *
        "a cell rebuilt from scratch over a longer record will not reproduce an appended one " *
        "even before the seam."

    dimnames = ("layer", "bin", "delta_temperature", "precipitation_scaling")
    for name in PROFILE_VARIABLES
        # Float64 with no packing or scaling: `gemb` pins the column depth and cell count from
        # the restored `dz`, and `_assert_grid_feasible` errors on a perturbed grid, so these
        # must round-trip exactly.
        v = NCDatasets.defVar(g, string(name), Float64, dimnames; fillvalue = NC_FILL)
        _set_attributes!(v, cf_attributes(name; time_axis = false))
    end

    # No fill value: 0 is a meaningful value here (a bin with no saved state), and declaring it
    # as the fill would make those entries read back as `missing`.
    vl = NCDatasets.defVar(g, "valid_layers", Int32,
                           ("bin", "delta_temperature", "precipitation_scaling"))
    vl.attrib["units"] = "1"
    vl.attrib["long_name"] = "number of real layers in the saved column"
    vl.attrib["comment"] = "Zero for a bin that produced no output and therefore has no " *
                           "saved state."

    rt = NCDatasets.defVar(g, "restart_time", Float64,
                           ("bin", "delta_temperature", "precipitation_scaling");
                           fillvalue = NC_FILL)
    rt.attrib["units"] = NC_TIME_UNITS
    rt.attrib["calendar"] = NC_CALENDAR
    rt.attrib["long_name"] = "time of the saved column state"

    _fill_restart!(g, run, n_layer)
    return nothing
end

function _fill_restart!(g, run::GlacierCellRun, n_layer::Int)
    n_bin, n_dt, n_ps = size(run.profiles)
    data = Dict(v => fill(NC_FILL, n_layer, n_bin, n_dt, n_ps) for v in PROFILE_VARIABLES)
    valid = zeros(Int32, n_bin, n_dt, n_ps)
    rtime = fill(NC_FILL, n_bin, n_dt, n_ps)
    t_end = _nc_encode_time(last(run.time))

    for i_ps in 1:n_ps, i_dt in 1:n_dt, i_bin in 1:n_bin
        p = run.profiles[i_bin, i_dt, i_ps]
        p === nothing && continue
        n = length(p[:dz])
        valid[i_bin, i_dt, i_ps] = n
        rtime[i_bin, i_dt, i_ps] = t_end
        for v in PROFILE_VARIABLES
            data[v][1:n, i_bin, i_dt, i_ps] = p[v]
        end
    end

    for v in PROFILE_VARIABLES
        g[string(v)][:, :, :, :] = data[v]
    end
    g["valid_layers"][:, :, :] = valid
    g["restart_time"][:, :, :] = rtime
    return nothing
end

_assert_grid_matches(ds, run::GlacierCellRun) =
    _assert_run_grid_matches("the existing file",
                             Tuple(collect(Float64, ds[name][:]) for name in RUN_GRID_AXES),
                             _run_grid(run.bins, run.delta_temperatures,
                                       run.precipitation_scalings))
