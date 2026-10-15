"""
CF-compliant netCDF output for tile GEMB runs.

One file per 2° tile. Every series carries the perturbation grid as dimensions, so a whole
(temperature × precipitation) sweep for a tile lives in one file:

    dh(time, band, delta_temperature, precipitation_scaling)                  [m]
    dv(time, delta_temperature, precipitation_scaling)                        [km3]
    restart/density(layer, band, delta_temperature, precipitation_scaling)    [kg m-3]

Two resolutions, deliberately both: the band-resolved height change is what the altimetry `dh` is
binned at, and the tile-integrated volume and mass are what a regional total is assembled from.
Storing only the second would make a per-band comparison impossible; storing only the first would
leave every reader to redo the area weighting, and the conversions between metres, km³ and Gt are
exactly where a reader gets it wrong (see [`tile_volume_change`](@ref)).

The geotile identifier is written alongside this package's own tile index so the file joins against
the 2° geotile products without either side parsing the other's filenames.
"""

# The tile totals written to file, with their units and how each was assembled. Volume comes from the
# height change and mass from the fluxes; the pairing is recorded here so the writer cannot label one
# with the other's unit.
const TILE_TOTAL_UNITS = Dict{Symbol,String}(
    :dv => "km3", :dv_mass => "km3", :dv_firn => "km3", :fac => "km3",
    :dm => "Gt",
    (v => "Gt" for v in TILE_MASS_VARIABLES)...,
)

"""
    write_glacier_tile_netcdf(path, run::GlacierTileRun; kwargs...) -> path

Write a [`GlacierTileRun`](@ref) to a CF-1.11 netCDF-4 file at `path`, replacing any existing file.
The `time` dimension is unlimited so [`append_glacier_tile_netcdf`](@ref) can extend it.

The file is written to `path * ".part"` and renamed onto `path` only once it is whole, with its
`file_complete` attribute set, so a file at `path` is never a partial one: a writer that dies leaves the
previous file, or none, in place.

`precision` sets the element type of the band-resolved series, which are the bulk of the file — a
60-band tile over the full record at 8 perturbations each way is the dominant cost. The tile totals and
the restart group are always Float64: the totals are what a regional sum is built from, and the restart
profiles must round-trip exactly or `gemb` cannot resume from them.

`institution` and `references` are written as global attributes when given.
"""
function write_glacier_tile_netcdf(path::AbstractString, run::GlacierTileRun;
                                  precision::Type = Float32,
                                  deflatelevel::Int = 4,
                                  institution = nothing, references = nothing)
    mkpath(dirname(abspath(path)))
    n_layer = _max_profile_length(run.profiles)

    return _write_atomically(path) do tmp
        NCDatasets.NCDataset(tmp, "c") do ds
            NCDatasets.defDim(ds, "time", Inf)          # unlimited, for append
            NCDatasets.defDim(ds, "band", length(run.bands))
            NCDatasets.defDim(ds, "delta_temperature", length(run.delta_temperatures))
            NCDatasets.defDim(ds, "precipitation_scaling", length(run.precipitation_scalings))
            NCDatasets.defDim(ds, "layer", n_layer)

            _write_tile_run_globals!(ds, run; institution, references)
            _write_tile_run_coordinates!(ds, run, n_layer)
            _write_tile_run_provenance!(ds, run)
            _write_tile_band_series!(ds, run, precision, deflatelevel)
            _write_tile_totals!(ds, run, deflatelevel)
            _write_tile_restart_group!(ds, run, n_layer)
        end
    end
end

# The global attribute that marks a tile file as whole. Set last, on the temporary file, before it is
# renamed into place.
const TILE_FILE_COMPLETE = "file_complete"

# Build the file at `path` by `f(tmp)` on `tmp = path * ".part"` (a copy of `copy_from` when given),
# mark it complete and rename it onto `path`. The rename is atomic on POSIX within one filesystem, so a
# reader sees the old file or the new one, never a partial one; on failure the temporary file is
# removed and `path` is left as it was.
function _write_atomically(f, path::AbstractString; copy_from = nothing)
    tmp = path * ".part"
    rm(tmp; force = true)
    try
        copy_from === nothing || cp(copy_from, tmp)
        f(tmp)
        NCDatasets.NCDataset(tmp, "a") do ds
            ds.attrib[TILE_FILE_COMPLETE] = "true"
        end
    catch
        rm(tmp; force = true)
        rethrow()
    end
    Base.Filesystem.rename(tmp, path)
    return path
end

"""
    append_glacier_tile_netcdf(path, run::GlacierTileRun) -> path

Append a continuation run to an existing tile file: extend the unlimited `time` dimension with the new
series and overwrite the restart group with the new final profiles.

The new record must start strictly after the file's last time, and the run grid (bands, temperature
deltas, precipitation scalings) must match the file's — a mismatch means the arrays would not line up,
so it is refused rather than reconciled.

The extended file is built on a copy at `path * ".part"` and renamed onto `path` once whole, so an
append that fails or is interrupted leaves the file as it was.

A continuation's cumulative series start from its own seam, so each is offset by the file's last
stored value of the same series: the band-resolved `dh`, `dh_mass`, `dh_water`, `dh_firn` and
`dh_residual`, and every `total_*` except the state `total_fac`. A run slot with no stored value at the
seam has no datum to continue from and is written as fill. The per-band forcing summaries become
record-long: a time mean (`band_temperature_lapse_rate`, `band_glacier_decoupling_factor_mean`) is
weighted by the forcing steps of each segment, and a per-step count (`band_*_n_<source>`,
`band_n_timesteps_above_freezing`) is summed. `max_dh_residual` becomes the larger of the two.
"""
function append_glacier_tile_netcdf(path::AbstractString, run::GlacierTileRun)
    isfile(path) || throw(ArgumentError("no tile file to append to at $path"))
    return _write_atomically(tmp -> _append_glacier_tile!(tmp, run), path; copy_from = path)
end

function _append_glacier_tile!(path, run::GlacierTileRun)
    NCDatasets.NCDataset(path, "a") do ds
        n_existing = length(ds["time"])
        last_time = _nc_decode_time(ds["time"][n_existing])
        first(run.time) > last_time ||
            throw(ArgumentError("new record starts at $(first(run.time)), which is not after the " *
                                "file's last time $(last_time); would duplicate or overlap " *
                                "existing output"))

        _assert_tile_grid_matches(ds, run)

        ds.attrib["history"] = get(ds.attrib, "history", "") *
            "\n$(Dates.format(now(), "yyyy-mm-ddTHH:MM:SS")): extended through " *
            "$(Dates.format(last(run.time), "yyyy-mm-ddTHH:MM:SS")) by " *
            "GEMB_GlacierSims.append_glacier_tile_netcdf"

        n_new = length(run.time)
        idx = (n_existing + 1):(n_existing + n_new)
        ds["time"][idx] = _nc_encode_time.(run.time)
        for v in keys(run.bands_series)
            name = string(v)
            series = run.bands_series[v]
            v in TILE_CUMULATIVE_HEIGHT_VARIABLES &&
                (series = _continue_from_seam(series, ds[name][n_existing, :, :, :]))
            ds[name][idx, :, :, :] = series
        end
        for v in keys(run.totals)
            name = _tile_total_variable(v)
            total = run.totals[v]
            v === :fac || (total = _continue_from_seam(total, ds[name][n_existing, :, :]))
            ds[name][idx, :, :] = total
        end
        _append_band_forcing_summaries!(ds, run)
        if haskey(run.provenance, "max_dh_residual")
            ds.attrib["max_dh_residual"] = max(Float64(get(ds.attrib, "max_dh_residual", 0.0)),
                                               Float64(run.provenance["max_dh_residual"]))
        end

        # The spinup provenance is left as written, both the per-run variables and the reduced
        # `spinup_converged`. A continuation resumes from the saved columns and runs no spinup of its
        # own, so the history already in the file is the history of these columns; rewriting it from
        # the continuation would only copy it back.
        #
        # The restart group is state, not a time series: overwrite it in place. Its `layer` dimension
        # is fixed at creation, so a continuation whose columns grew past it cannot be stored — that
        # means the grid changed and the tile needs a fresh file.
        n_layer = ds.dim["layer"]
        deepest = _max_profile_length(run.profiles)
        deepest <= n_layer ||
            throw(ArgumentError("continuation profiles are deeper ($deepest layers) than the file's " *
                                "layer dimension ($n_layer); rewrite the tile from scratch"))
        _fill_tile_restart!(ds.group["restart"], run, n_layer)
    end

    return path
end

# The band-resolved height series that accumulate from the start of the run, and so continue across a
# seam by adding the stored last value. `firn_air_content` is a state and the mass fluxes are per
# output interval; neither is offset.
const TILE_CUMULATIVE_HEIGHT_VARIABLES = (:dh, :dh_mass, :dh_water, :dh_firn, :dh_residual)

# A continuation's cumulative series, offset by the stored value at the seam (fill where there is none)
# so it continues the record. `seam` has the series' shape without its leading time axis.
_continue_from_seam(series, seam) =
    series .+ reshape(coalesce.(seam, NaN), 1, size(seam)...)

# Per-band forcing summaries that are time means over the forcing steps, and those that count steps.
const BAND_STEP_MEAN_KEYS = ("temperature_lapse_rate", "glacier_decoupling_factor_mean")
const BAND_LAPSE_SOURCE_KEYS = Tuple("temperature_lapse_rate_n_$s" for s in DOWNSCALING_SOURCES)
const BAND_STEP_COUNT_KEYS = ("n_timesteps_above_freezing", BAND_LAPSE_SOURCE_KEYS...,
    ("glacier_decoupling_factor_n_$s" for s in DOWNSCALING_SOURCES)...,
    ("glacier_decoupling_factor_n_$(s)_above_freezing" for s in DOWNSCALING_SOURCES)...)

# Per-band forcing summaries over the whole record: time means weighted by each segment's forcing steps
# and per-step counts summed. A band's forcing steps are the sum of its lapse-rate source counts, since
# every step has exactly one source. Anything else in the band provenance describes the band or the fit
# and is the same for every segment.
function _append_band_forcing_summaries!(ds, run::GlacierTileRun)
    isempty(run.band_provenance) && return nothing
    present = filter(in(keys(first(run.band_provenance))), (BAND_STEP_MEAN_KEYS..., BAND_STEP_COUNT_KEYS...))
    stored = Dict(key => begin
                      haskey(ds, "band_" * key) || throw(ArgumentError(
                          "the tile file has no band_$key, which the continuation reports; the file " *
                          "predates that summary and its record-long value cannot be formed"))
                      collect(Float64, coalesce.(ds["band_" * key][:], NaN))
                  end for key in present)
    new = Dict(key => [Float64(p[key]) for p in run.band_provenance] for key in present)
    sources = filter(in(present), BAND_LAPSE_SOURCE_KEYS)
    isempty(sources) && throw(ArgumentError(
        "the continuation's band provenance carries no temperature_lapse_rate_n_* counts, so its " *
        "forcing steps per band are unknown and the record-long band means cannot be formed"))
    n_old = sum(stored[k] for k in sources)
    n_new = sum(new[k] for k in sources)
    for key in present
        ds["band_" * key][:] = key in BAND_STEP_MEAN_KEYS ?
            (stored[key] .* n_old .+ new[key] .* n_new) ./ (n_old .+ n_new) : stored[key] .+ new[key]
    end
    return nothing
end

"""
    read_glacier_tile_status(path) -> NamedTuple or nothing

What an existing tile file already covers, read from its coordinates and attributes only — **no series
and no firn state**. `nothing` when the file does not exist, and `(; complete = false, reason)` when it
cannot be read or does not carry the `file_complete` mark the writers set last, which is what a partial
or truncated file looks like.

Otherwise returns `(; complete = true, time, n_timesteps, band_centers, band_edges, delta_temperatures,
precipitation_scalings, geotile_id, parameters, dropped_band_centers, hypsometry_area_km2,
lost_area_km2)`. `time` is the last
output time in the file, i.e. the point a continuation must start after. `band_edges` holds each
stored band's `(lo, hi)`. `dropped_band_centers` lists the intervals holding ice that the run did not
model (empty when it dropped none), and the two areas are the run's lost-ice accounting in km²
(`missing` in a file that does not record them; see [`gemb_glacier_tile`](@ref)).

This exists to be cheap, for the same reason [`read_downscaling_tile_status`](@ref) does: a sweep over
819 tiles decides "is this tile already done" before touching forcing, and that decision is what makes
a re-run cost a file open per tile instead of a forcing pass.
"""
function read_glacier_tile_status(path::AbstractString)
    isfile(path) || return nothing
    return try
        _read_glacier_tile_status(path)
    catch e
        e isa InterruptException && rethrow()
        (; complete = false, reason = "unreadable: " * sprint(showerror, e))
    end
end

function _read_glacier_tile_status(path)
    return NCDatasets.NCDataset(path, "r") do ds
        get(ds.attrib, TILE_FILE_COMPLETE, "") == "true" ||
            return (; complete = false, reason = "no $(TILE_FILE_COMPLETE) mark")
        n = length(ds["time"])
        area(key) = haskey(ds.attrib, key) ? Float64(ds.attrib[key]) : missing
        (; complete = true,
         time = n == 0 ? nothing : _nc_decode_time(ds["time"][n]),
         n_timesteps = n,
         band_centers = collect(Float64, ds["band_center"][:]),
         band_edges = collect(zip(collect(Float64, ds["band_lower"][:]),
                                  collect(Float64, ds["band_upper"][:]))),
         delta_temperatures = collect(Float64, ds["delta_temperature"][:]),
         precipitation_scalings = collect(Float64, ds["precipitation_scaling"][:]),
         geotile_id = get(ds.attrib, "geotile_id", ""),
         parameters = _read_run_parameters(ds),
         dropped_band_centers = _attribute_vector(ds, "dropped_band_centers"),
         hypsometry_area_km2 = area("hypsometry_area_km2"),
         lost_area_km2 = area("lost_area_km2"))
    end
end

"""
    tile_run_disposition(status, requested, run_time, intervals; delta_temperatures,
                         precipitation_scalings, output_frequency,
                         lost_area_fraction_maximum = LOST_AREA_FRACTION_MAXIMUM) -> Symbol

What an existing tile or native-cell file is worth to a request, from its
[`read_glacier_tile_status`](@ref) `status` alone, so the decision costs no forcing pass:

- `:current`: it already covers the record; nothing to do.
- `:extendable`: the same experiment, but the record has grown since. Resume from its firn state and
  append rather than repeating a spinup and a whole transient to add a few months.
- `:rebuild`: a different experiment, or nothing usable.

`requested` is the [`tile_run_parameters`](@ref) of the run about to happen, `run_time` its forcing
time axis, and `intervals` the elevation intervals (each with a `center`) the request's hypsometry
holds ice in. A partial or unreadable file (`status.complete` false) is rebuilt. The file is the same
experiment when its perturbation grid is the requested one, its run
parameters agree with `requested`, and every interval holding ice is accounted for: a band the file
stores, or a dropped interval it records (`dropped_band_centers`). A file whose bands are a subset of
the intervals is therefore current when the rest were dropped and recorded as such. A file that does
not record its lost-ice accounting, or whose loss exceeds `lost_area_fraction_maximum` of its ice, is
rebuilt.

The stored time is the last *output* sample while `run_time` ends at the last *forcing* step, and a
coarser output grid never reaches the forcing's final step, so a record extended by less than one
output period ([`output_period_bound`](@ref)) counts as covered: it spans no further output interval.
"""
function tile_run_disposition(status, requested, run_time, intervals; delta_temperatures,
                              precipitation_scalings, output_frequency::Symbol,
                              lost_area_fraction_maximum::Real = LOST_AREA_FRACTION_MAXIMUM)
    status === nothing && return :rebuild
    status.complete || return :rebuild
    (status.n_timesteps == 0 || status.time === nothing) && return :rebuild
    # The run grid: a band set or perturbation grid other than this request's means the stored arrays
    # describe something else, and no seam could join them.
    centers = Set(Float64(iv.center) for iv in intervals)
    stored = Set(status.band_centers)
    issubset(stored, centers) || return :rebuild
    Base.union(stored, status.dropped_band_centers) == centers || return :rebuild
    status.delta_temperatures == collect(Float64, delta_temperatures) || return :rebuild
    status.precipitation_scalings == collect(Float64, precipitation_scalings) || return :rebuild
    # The lost-ice accounting, which the rule on it needs.
    (status.hypsometry_area_km2 === missing || status.lost_area_km2 === missing) && return :rebuild
    status.lost_area_km2 <= lost_area_fraction_maximum * status.hypsometry_area_km2 ||
        return :rebuild
    # The settings: a changed model parameter or downscaling policy makes it a different experiment,
    # which must not be spliced onto an old record.
    isempty(status.parameters) && return :rebuild
    isempty(run_parameter_differences(status.parameters, requested)) || return :rebuild
    period = output_period_bound(output_frequency)
    covered = period === nothing ? status.time >= last(run_time) :
              status.time + period >= last(run_time)
    return covered ? :current : :extendable
end

"""
    output_period_bound(output_frequency::Symbol) -> Period or nothing

The longest span one GEMB output period covers: `Day(31)`, `Day(7)` or `Day(1)` for `:monthly`,
`:weekly` and `:daily`, and `nothing` for `:all`, where every output step is a forcing step. An upper
bound, because understating it would declare a complete file short and run it again. Throws for any
other frequency, which would otherwise be compared exactly against a forcing step its output grid never
reaches, so every such file would read as short.
"""
function output_period_bound(output_frequency::Symbol)
    output_frequency === :monthly && return Day(31)
    output_frequency === :weekly && return Day(7)
    output_frequency === :daily && return Day(1)
    output_frequency === :all && return nothing
    throw(ArgumentError("no output period is known for output_frequency = :$output_frequency; add " *
                        "one before using it for a tile run, or every file reads as short"))
end

# A numeric vector attribute, or an empty vector when it is absent. NetCDF hands a one-element vector
# attribute back as a scalar.
_attribute_vector(ds, key) =
    haskey(ds.attrib, key) ? collect(Float64, vcat(ds.attrib[key])) : Float64[]

"""
    read_glacier_tile_restart(path) -> NamedTuple or nothing

Read the firn state and run grid back out of a tile file, in the form [`gemb_glacier_tile`](@ref) would
resume from. `nothing` when the file does not exist.

Returns `(; time, band_centers, delta_temperatures, precipitation_scalings, profiles, parameters,
provenance)`, where `profiles` maps `(i_band, i_dt, i_ps)` to a `DimStack` carrying
[`PROFILE_VARIABLES`](@ref). A run slot that produced no output has no entry.

Each profile's metadata is that column's own spinup provenance over the tile-level constants, since
`gemb` copies it onto the output a continuation produces. `provenance` is the tile-level part alone,
which is what a slot with no profile has to fall back on.
"""
function read_glacier_tile_restart(path::AbstractString)
    isfile(path) || return nothing
    return NCDatasets.NCDataset(path, "r") do ds
        haskey(ds.group, "restart") || return nothing
        g = ds.group["restart"]

        band_centers = collect(Float64, ds["band_center"][:])
        deltas = collect(Float64, ds["delta_temperature"][:])
        scalings = collect(Float64, ds["precipitation_scaling"][:])
        n_time = length(ds["time"])

        # A file written before a state layer joined `PROFILE_VARIABLES` is missing it here. Reading
        # on would hand `gemb` an incomplete profile and fail deep in the first timestep, so name the
        # gap and the remedy instead.
        missing_vars = [v for v in PROFILE_VARIABLES if !haskey(g, string(v))]
        isempty(missing_vars) ||
            throw(ArgumentError("the restart group in $path is missing " *
                                join(missing_vars, ", ") * "; it predates those layers becoming " *
                                "part of the saved state and cannot be continued. Delete the file " *
                                "to rebuild the tile from scratch."))

        valid = g["valid_layers"][:, :, :]
        # One hyperslab per variable rather than one per (variable, band, delta, scaling): every run
        # is padded to the same `layer` dimension, so the whole array reads at once and is sliced in
        # memory. A 60-band tile over a full perturbation grid is otherwise thousands of reads.
        columns = Dict(v => g[string(v)][:, :, :, :] for v in PROFILE_VARIABLES)
        layer_metadata = cf_layer_index_attributes()
        provenance = _read_provenance(ds)
        run_provenance = _read_run_provenance(ds)
        isempty(run_provenance) &&
            @warn("This tile file stores spinup provenance only as tile-wide attributes, which hold " *
                  "one column's history — the lowest band at the first perturbation. Every restored " *
                  "profile therefore carries that same history, and an appended record will repeat " *
                  "it. Delete the file to rebuild it with per-run provenance.", path)

        profiles = Dict{Tuple{Int,Int,Int},DimStack}()
        for i_ps in eachindex(scalings), i_dt in eachindex(deltas), i_band in eachindex(band_centers)
            n = valid[i_band, i_dt, i_ps]
            (ismissing(n) || n <= 0) && continue     # band that produced no output
            n = Int(n)
            zdim = Z(1:n; metadata = layer_metadata)
            cols = NamedTuple(
                v => DimArray(collect(Float64, @view columns[v][1:n, i_band, i_dt, i_ps]), (zdim,))
                for v in PROFILE_VARIABLES)
            # This column's own spinup history, over the file-level constants. `gemb` copies a profile's
            # metadata onto the output it starts, so handing every column the same history would make an
            # appended record claim one column's spinup for all of them.
            profiles[(i_band, i_dt, i_ps)] =
                DimStack(cols; layermetadata = cf_layermetadata(cols; time_axis = false),
                         metadata = _profile_metadata(provenance, run_provenance,
                                                     i_band, i_dt, i_ps))
        end

        (; time = n_time == 0 ? nothing : _nc_decode_time(ds["time"][n_time]),
         band_centers, delta_temperatures = deltas, precipitation_scalings = scalings,
         profiles, parameters = _read_run_parameters(ds), provenance)
    end
end

# The per-run spinup provenance variables, or an empty dict for a file written before they existed.
#
# A provenance key is a variable here and an attribute in an older file, so the two are told apart by
# where the value lives rather than by a version stamp. `spinup_converged` is both — the attribute being
# the `all` reduction — so a name is only taken as per-run when a variable actually carries it.
function _read_run_provenance(ds)
    out = Dict{String,Array{Float64,3}}()
    for name in keys(ds)
        _is_provenance_key(name) || continue
        var = ds[name]
        ndims(var) == 3 || continue
        out[name] = collect(Float64, coalesce.(var[:, :, :], NaN))
    end
    return out
end

# One column's provenance: its own per-run values over the file-level constants.
#
# Falls back to the file-level provenance alone when the file has no per-run variables. Such a file
# records one column's spinup history for the whole tile, so every profile restored from it carries the
# same values — wrong for all but one column, and the reason the variables exist. Warned about once per
# read rather than per profile.
function _profile_metadata(provenance, run_provenance, i_band, i_dt, i_ps)
    isempty(run_provenance) && return provenance
    meta = copy(provenance)
    for (key, slots) in run_provenance
        value = slots[i_band, i_dt, i_ps]
        isnan(value) && continue
        meta[key] = _run_provenance_is_flag(key) ? _encode_attribute(value == 1.0) : value
    end
    return meta
end

function _write_tile_run_globals!(ds, run::GlacierTileRun; institution, references)
    _set_attributes!(ds, GEMB_CF_GLOBAL_ATTRIBUTES)
    ds.attrib["history"] = "$(Dates.format(now(), "yyyy-mm-ddTHH:MM:SS")): created by " *
                           "GEMB_GlacierSims.write_glacier_tile_netcdf"
    ds.attrib["title"] = "GEMB glacier tile surface mass balance, height change and firn state"
    ds.attrib["featureType"] = "timeSeries"
    institution === nothing || (ds.attrib["institution"] = institution)
    references === nothing || (ds.attrib["references"] = references)

    # Both identifiers. `tile_name` is this package's own form and names the file; `geotile_id` is the
    # form the 2° altimetry products key on, so a join needs no filename parsing on either side.
    ds.attrib["tile_name"] = run.name
    ds.attrib["geotile_id"] = run.geotile_id
    ds.attrib["tile_index_lon"] = run.index[1]
    ds.attrib["tile_index_lat"] = run.index[2]
    ds.attrib["tile_lon_min"] = run.bounds.lon_min
    ds.attrib["tile_lon_max"] = run.bounds.lon_max
    ds.attrib["tile_lat_min"] = run.bounds.lat_min
    ds.attrib["tile_lat_max"] = run.bounds.lat_max
    ds.attrib["longitude_convention"] = "(-180, 180]"
    ds.attrib["n_cells_core"] = run.n_cells_core
    ds.attrib["n_cells_used"] = run.n_cells_used
    ds.attrib["n_bands"] = length(run.bands)

    ds.attrib["height_change_identity"] =
        "dh = -cumsum(ice_flux) = (SMB - dWater)/density_ice + d(firn_air_content) - " *
        "cumsum(strain_thinning). GEMB pins the column depth and reports the basal flux that does " *
        "so, and the negation of that flux is surface elevation change against a datum fixed in " *
        "the ice at the start of the run. dh is NOT a proxy for surface mass balance and can carry " *
        "the opposite sign: compaction lowers the surface while accumulation raises it."
    ds.attrib["reference_frame_comment"] =
        "dh, its decomposition terms dh_mass, dh_water, dh_firn and dh_residual, and the volume " *
        "totals dv/dv_mass/dv_firn are all cumulative from the START OF THE RUN, against a datum " *
        "fixed in the ice at that time; no reference state has been removed, so adding an " *
        "independently derived discharge to dv gives total volume change. The identity holds at " *
        "every output time: dh = dh_mass + dh_water + dh_firn + dh_residual."
    ds.attrib["volume_and_mass_comment"] =
        "Volume totals are the area-weighted height change and mass totals the area-weighted " *
        "fluxes. They are not interconvertible: dh carries the firn compaction term, so " *
        "dh * density_ice overstates the mass change by whatever the column's air content did."
    ds.attrib["precipitation_scaling_comment"] =
        "Scales TOTAL precipitation before GEMB partitions phase against " *
        "model_rain_temperature_threshold, so a scaling above 1 raises rain as well as snow. " *
        "Where the scaling stands in for snow redistribution and avalanching that is an " *
        "overstatement of the rain flux."

    _write_tile_run_parameters!(ds, run)

    # The tile-level aggregates and the climatology descriptors that are constant across runs. The
    # per-run spinup outcome is not here — `_collect_run_provenance!` routed it to the variables
    # `_write_tile_run_provenance!` writes, because one column's history is not the tile's.
    #
    # `spinup_converged` is deliberately both an attribute and a variable: the attribute is the `all`
    # reduction over every run, the variable says which run. NetCDF keeps attributes and variables in
    # separate namespaces, so this is legal and the shared name is the point.
    #
    # NetCDF attributes hold no `nothing`, `Bool` or `DateTime`, so encode those.
    for (k, v) in run.provenance
        enc = _encode_attribute(v)
        enc === nothing || (ds.attrib[k] = enc)
    end
    return nothing
end

# The run parameters, plus the roster naming them — the same convention `_write_run_parameters!` uses
# for a cell run, so both file kinds are read back by `_read_run_parameters`.
function _write_tile_run_parameters!(ds, run::GlacierTileRun)
    names = sort!(collect(keys(run.parameters)))
    for k in names
        enc = _encode_attribute(run.parameters[k])
        enc === nothing || (ds.attrib[k] = enc)
    end
    ds.attrib["run_parameters"] = join(names, " ")
    ds.attrib["run_parameters_comment"] =
        "The settings that define how this tile was run: the GEMB model parameters, the spinup " *
        "window, and the downscaling resolution policy. A continuation must match them, or it " *
        "splices two different experiments into one record. Excludes what is not a setting: " *
        "history, and the spinup_*/climatology_* provenance."
    return nothing
end

function _write_tile_run_coordinates!(ds, run::GlacierTileRun, n_layer::Int)
    t = NCDatasets.defVar(ds, "time", Float64, ("time",); chunksizes = _time_chunks(length(run.time)))
    _set_attributes!(t, cf_time_attributes())
    t.attrib["units"] = NC_TIME_UNITS
    t.attrib["calendar"] = NC_CALENDAR
    t.attrib["long_name"] = "end of the output accumulation interval"
    # `time` is unlimited and therefore starts at length 0, so `t[:]` would write nothing — the range
    # has to be given explicitly. Same for every variable on this dimension.
    t[1:length(run.time)] = _nc_encode_time.(run.time)

    dt = NCDatasets.defVar(ds, "delta_temperature", Float64, ("delta_temperature",))
    dt.attrib["units"] = "K"
    dt.attrib["long_name"] = "prescribed uniform air temperature offset"
    dt.attrib["comment"] =
        "Applied to the band forcing, which has already been lapsed to the band center and " *
        "decoupled. Vapour pressure (at constant relative humidity) and downward longwave are " *
        "adjusted consistently; pressure, wind and shortwave are not."
    dt[:] = run.delta_temperatures

    ps = NCDatasets.defVar(ds, "precipitation_scaling", Float64, ("precipitation_scaling",))
    ps.attrib["units"] = "1"
    ps.attrib["long_name"] = "prescribed precipitation multiplier"
    ps[:] = run.precipitation_scalings

    for (name, long_name, values) in (
        ("band_center", "elevation band center", [b.center for b in run.bands]),
        ("band_lower", "elevation band lower edge", Float64[b.lo for b in run.bands]),
        ("band_upper", "elevation band upper edge", Float64[b.hi for b in run.bands]),
    )
        v = NCDatasets.defVar(ds, name, Float64, ("band",))
        v.attrib["units"] = "m"
        _set_elevation_datum!(v)
        v.attrib["long_name"] = long_name
        v.attrib["positive"] = "up"
        v[:] = values
    end

    ba = NCDatasets.defVar(ds, "band_area", Float64, ("band",))
    ba.attrib["units"] = "km2"
    ba.attrib["long_name"] = "glacier area within the elevation band"
    ba.attrib["comment"] =
        "Sums over the bands to the total glacier area of the tile's core cells whose forcing was " *
        "usable, so no ice is double-counted with a neighbouring tile: the parameters are fitted " *
        "over a buffered neighbourhood but the area is the core only."
    ba[:] = [b.area for b in run.bands]

    bn = NCDatasets.defVar(ds, "band_n_cells", Int32, ("band",))
    bn.attrib["units"] = "1"
    bn.attrib["long_name"] = "grid cells contributing ice to the elevation band"
    bn[:] = Int32[b.n_cells for b in run.bands]

    # The per-band parameter provenance, as variables rather than attributes: it varies with the band,
    # and `k` in particular is resolved at each band's own centre. Units come from the key name, so a
    # new provenance key reaches the file correctly labelled without a change here — see
    # `_band_provenance_units`.
    keys_seen = Set{String}()
    for p in run.band_provenance, k in keys(p)
        push!(keys_seen, k)
    end
    for key in sort!(collect(keys_seen))
        values = [Float64(get(p, key, NaN)) for p in run.band_provenance]
        v = NCDatasets.defVar(ds, "band_" * key, Float64, ("band",); fillvalue = NC_FILL)
        v.attrib["units"] = _band_provenance_units(key)
        v.attrib["long_name"] = replace(key, "_" => " ")
        v[:] = values
    end
    haskey(ds, "band_bare_ice_albedo_source") &&
        (ds["band_bare_ice_albedo_source"].attrib["comment"] =
            "Index into applied_bare_ice_albedo_source_meanings: how this band's albedo_ice was " *
            "resolved from the observed profile. band_bare_ice_albedo is the value itself, and it " *
            "is the one aligned with this file's bands — applied_bare_ice_albedo spans every band " *
            "the downscaling resolved, including any this run could not force.")
    haskey(ds, "band_extrapolation_above_reanalysis") &&
        (ds["band_extrapolation_above_reanalysis"].attrib["comment"] =
            "Signed distance from this band's center to the highest reanalysis surface that fed " *
            "it. Positive means the band's forcing is a lapse extrapolation above every " *
            "contributing cell, which for glaciers is the norm rather than an error — they occupy " *
            "the high ground within a 0.1 degree cell — and is the dominant uncertainty in this " *
            "forcing, so a fit against altimetry should weight a band by it.")
    haskey(ds, "band_max_donor_distance_km") &&
        (ds["band_max_donor_distance_km"].attrib["comment"] =
            "Furthest a cell of this band borrowed its forcing, for cells that carry ice but fall " *
            "outside the reanalysis land mask and so have none of their own. Zero where every cell " *
            "used its own forcing. How much ice rests on a loan is band_substituted_area, and ice " *
            "no donor could reach at all is band_unrecovered_area, which is excluded from " *
            "band_glacier_area.")

    # GEMB's own attributes for this dimension: an index, deliberately carrying no `standard_name`
    # and no `positive`, so it is not presented as a physical height.
    lay = NCDatasets.defVar(ds, "layer", Int32, ("layer",))
    _set_attributes!(lay, cf_layer_index_attributes())
    lay[:] = Int32.(1:n_layer)

    for (name, units, long_name) in (
        ("glacier_area_total", "km2", "total glacier area in the tile's core cells"),
        ("mie2cubickm", "km3", "volume of one metre of ice equivalent over the tile's glacier area"),
    )
        v = NCDatasets.defVar(ds, name, Float64, ())
        v.attrib["units"] = units
        v.attrib["long_name"] = long_name
        v[] = name == "mie2cubickm" ? mie2cubickm([b.area for b in run.bands]) :
              sum(b.area for b in run.bands)
    end
    return nothing
end

# Units for a `band_*` provenance variable, from its name. Keyed on the name rather than a table so a
# new provenance key in `_band_provenance` reaches the file labelled without a second edit here; the
# fallback is "1", which is right for the counts that most of these keys are.
function _band_provenance_units(key::AbstractString)
    # `_n_` marks a count *of* lapse-rate values, not a lapse rate.
    occursin("lapse_rate", key) && !occursin("_n_", key) && return "K km-1"
    startswith(key, "extrapolation") && return "m"
    endswith(key, "_area") && return "km2"
    endswith(key, "_km") && return "km"
    return "1"
end

# The spinup outcome of each run, as variables over the full run grid.
#
# Every column of a tile spins up on its own, and they do not agree: over one tile the final SMB of the
# spinup cycle spans more than an order of magnitude and changes sign between the ablation and
# accumulation bands. So this is per-run data, not a tile attribute. `spinup_converged` is additionally
# reduced with `all` onto the globals, because "did every column settle" is a question worth being able
# to answer without reading an array; nothing else is reduced.
#
# Booleans are stored as 0/1 with `flag_values`, so one fill convention (`NC_FILL`, NaN) covers a run
# that produced no output for both the flags and the rates.
function _write_tile_run_provenance!(ds, run::GlacierTileRun)
    dimnames = ("band", "delta_temperature", "precipitation_scaling")
    for key in sort!(collect(keys(run.run_provenance)))
        values = run.run_provenance[key]
        v = NCDatasets.defVar(ds, key, Float64, dimnames; fillvalue = NC_FILL)
        v.attrib["units"] = _run_provenance_units(key)
        v.attrib["long_name"] = replace(key, "_" => " ")
        v.attrib["comment"] =
            "Per-run spinup provenance: one value for each (band, temperature offset, precipitation " *
            "scaling), because each is a separately spun-up column. NaN where the run produced no " *
            "output. Do not read one element as a property of the tile."
        if _run_provenance_is_flag(key)
            v.attrib["flag_values"] = [0.0, 1.0]
            v.attrib["flag_meanings"] = key == "spinup_converged" ? "not_converged converged" :
                                        "not_performed performed"
        end
        v[:, :, :] = values
    end
    return nothing
end

_run_provenance_is_flag(key::AbstractString) = key in ("spinup_converged", "spinup_performed")

# Units for a per-run provenance variable, from its name — same convention as
# `_band_provenance_units`, so a new key `gemb` starts reporting reaches the file labelled without a
# second edit here. The fallback is "1", right for the cycle and iteration counts and the flags.
function _run_provenance_units(key::AbstractString)
    endswith(key, "_smb_rate") && return "m yr-1"
    # A drift is the trend across cycles and a delta the step between two, so they differ by a time
    # unit. Both exist for density and for firn air content, which is why this comes before the plain
    # density and fac cases.
    endswith(key, "_drift_density") && return "kg m-3 yr-1"
    endswith(key, "_drift_fac") && return "m yr-1"
    endswith(key, "_delta_density") && return "kg m-3"
    endswith(key, "_delta_fac") && return "m"
    occursin("_density", key) && return "kg m-3"
    # `years_per_cycle` and `years_simulated`, but not `steps_per_year`, which is a count.
    occursin("_years", key) && return "yr"
    return "1"
end

# Chunk shape for a variable on the unlimited `time` axis: `TIME_CHUNK` steps by the whole of every
# other axis. Left to the library, an unlimited axis is chunked one element at a time, and the per-chunk
# index then outweighs the data many times over — a three-band cell file reaches 8 MB for 0.6 MB of
# series. A chunk of a few hundred steps keeps an append writing whole chunks and a read of one band's
# record touching a handful of them.
const TIME_CHUNK = 512
_time_chunks(n_time::Int, rest::Int...) = (clamp(n_time, 1, TIME_CHUNK), rest...)

function _write_tile_band_series!(ds, run::GlacierTileRun, precision::Type, deflatelevel::Int)
    dimnames = ("time", "band", "delta_temperature", "precipitation_scaling")
    deflate = deflatelevel > 0 ? deflatelevel : nothing
    chunks = _time_chunks(length(run.time), length(run.bands), length(run.delta_temperatures),
                          length(run.precipitation_scalings))

    for name in TILE_MASS_VARIABLES
        v = NCDatasets.defVar(ds, string(name), precision, dimnames;
                              deflatelevel = deflate, fillvalue = precision(NC_FILL),
                              chunksizes = chunks)
        _set_attributes!(v, cf_attributes(name))
        v.attrib["comment"] = get(cf_attributes(name), "comment", "") *
            (haskey(cf_attributes(name), "comment") ? " " : "") *
            "Per-band GEMB output for one output interval, not area-weighted and not cumulative. " *
            "The total_ variable of the same name is the area-weighted sum in Gt, accumulated from " *
            "the start of the record."
        v[1:length(run.time), :, :, :] = run.bands_series[name]
    end

    for (name, long_name, comment) in (
        (:dh, "surface elevation change",
         "-cumsum(ice_flux), against a datum fixed in the ice at the start of the run. This is " *
         "the quantity an altimeter measures, at the band resolution the altimetry dh is binned " *
         "at. Element 1 already carries the first output interval."),
        (:dh_mass, "surface elevation change from surface mass balance",
         "Cumulative SMB divided by density_ice, from the start of the run."),
        (:dh_water, "surface elevation change from column liquid storage",
         "Minus the change in stored water since the start of the run, divided by density_ice. " *
         "Water in a pore adds mass without adding thickness, so it belongs to the mass budget " *
         "but not to the height."),
        (:dh_firn, "surface elevation change from firn compaction",
         "Change in firn air content since the start of the run."),
        (:dh_residual, "closure residual of the height change decomposition",
         "dh minus the sum of the mass, water and firn terms. Should sit at the rounding floor; a " *
         "growing residual means the column's base is no longer at ice density and the height " *
         "series under-counts compaction."),
        (:firn_air_content, "firn air content", ""),
    )
        v = NCDatasets.defVar(ds, string(name), precision, dimnames;
                              deflatelevel = deflate, fillvalue = precision(NC_FILL),
                              chunksizes = chunks)
        v.attrib["units"] = "m"
        v.attrib["long_name"] = long_name
        isempty(comment) || (v.attrib["comment"] = comment)
        v.attrib["cell_methods"] = name === :firn_air_content ? "time: mean" : "time: point"
        v[1:length(run.time), :, :, :] = run.bands_series[name]
    end
    return nothing
end

function _write_tile_totals!(ds, run::GlacierTileRun, deflatelevel::Int)
    dimnames = ("time", "delta_temperature", "precipitation_scaling")
    deflate = deflatelevel > 0 ? deflatelevel : nothing
    chunks = _time_chunks(length(run.time), length(run.delta_temperatures),
                          length(run.precipitation_scalings))

    # Float64 whatever `precision` says: these are what a regional total is summed from, and a tile
    # sum is the point at which a truncation stops being local.
    for name in sort!(collect(keys(run.totals)))
        v = NCDatasets.defVar(ds, _tile_total_variable(name), Float64, dimnames;
                              deflatelevel = deflate, fillvalue = NC_FILL, chunksizes = chunks)
        v.attrib["units"] = get(TILE_TOTAL_UNITS, name, "1")
        v.attrib["long_name"] = _tile_total_long_name(name)
        # `fac` is the column's absolute firn air volume — a state. Everything else here accumulates
        # from the start of the record, so the two carry different `cell_methods`.
        v.attrib["cell_methods"] = name === :fac ? "area: sum time: mean" : "area: sum time: sum"
        v[1:length(run.time), :, :] = run.totals[name]
    end

    # **Every total except `total_fac` is cumulative from the start of the record**, which is what makes
    # them comparable to each other and to the altimetry dv/dm. The per-band series they are weighted
    # from are not: GEMB reports a mass flux as an interval sum, so `melt` holds one output period and
    # `total_melt` holds the record so far. Stated on the group because getting it wrong compares a
    # record-long volume change against a single month's melt.
    ds.attrib["total_accumulation_comment"] =
        "Every total_* variable except total_fac is CUMULATIVE from the start of the run. The " *
        "per-band series of the same name are per output interval, as GEMB reports them. total_fac " *
        "is the column's absolute firn air volume, a state, and is neither cumulative nor a flux."

    ds["total_dv"].attrib["comment"] =
        "Cumulative glacier volume change in km3 of ice equivalent: the band-resolved dh weighted " *
        "by band_area. Directly comparable to the altimetry dv of the matching geotile, once both " *
        "are referenced to a common epoch. Carries no ice dynamics — GEMB has none — so the " *
        "difference from a measured dv is discharge plus bedrock motion plus basal melt."
    ds["total_dm"].attrib["comment"] =
        "Cumulative column mass budget in Gt: precipitation - runoff + evaporation_condensation + " *
        "blowing_snow, area-weighted over bands. `rain` is the liquid fraction of " *
        "`precipitation` and so is not added again; `refreeze` is an internal phase change that " *
        "moves no mass across the column boundary; `evaporation_condensation` is positive for " *
        "mass gain."
    return nothing
end

# Tile totals are prefixed, because several share a name with the per-band series they were weighted
# from (`melt` is kg m-2 per band, `total_melt` is Gt for the tile) and one netCDF group cannot hold
# both under one name. Defined once so the writer and the appender cannot disagree about the prefix.
_tile_total_variable(name::Symbol) = "total_" * string(name)

_tile_total_long_name(name::Symbol) =
    name === :dv ? "glacier volume change" :
    name === :dv_mass ? "glacier volume change from surface mass balance" :
    name === :dv_firn ? "glacier volume change from firn compaction" :
    name === :fac ? "firn air volume" :
    name === :dm ? "glacier mass change" :
    "tile total " * replace(string(name), "_" => " ")

function _write_tile_restart_group!(ds, run::GlacierTileRun, n_layer::Int)
    g = NCDatasets.defGroup(ds, "restart")
    g.attrib["comment"] =
        "Firn column state at the last output time of each run — the complete state `gemb` needs " *
        "to extend the record. Columns shorter than the layer dimension are padded with the fill " *
        "value; valid_layers gives each column's real length."
    g.attrib["restart_fidelity"] =
        "A resumed run is close to but not identical to a continuous one: GEMB re-zeros its " *
        "per-step evaporation/condensation and surface-melt accumulators on restart, so the " *
        "trajectory diverges slightly at the seam. A run resumed from here measures its height " *
        "series against the column at the seam; append_glacier_tile_netcdf offsets them by the " *
        "last stored values, so the file holds one series from the start of the record."

    dimnames = ("layer", "band", "delta_temperature", "precipitation_scaling")
    for name in PROFILE_VARIABLES
        # Float64 with no packing or scaling: `gemb` pins the column depth and cell count from the
        # restored `dz`, so these must round-trip exactly.
        v = NCDatasets.defVar(g, string(name), Float64, dimnames; fillvalue = NC_FILL)
        _set_attributes!(v, cf_attributes(name; time_axis = false))
    end

    # No fill value: 0 is a meaningful value here (a band with no saved state), and declaring it as
    # the fill would make those entries read back as `missing`.
    vl = NCDatasets.defVar(g, "valid_layers", Int32,
                           ("band", "delta_temperature", "precipitation_scaling"))
    vl.attrib["units"] = "1"
    vl.attrib["long_name"] = "number of real layers in the saved column"
    vl.attrib["comment"] = "Zero for a band that produced no output and therefore has no saved state."

    rt = NCDatasets.defVar(g, "restart_time", Float64,
                           ("band", "delta_temperature", "precipitation_scaling");
                           fillvalue = NC_FILL)
    rt.attrib["units"] = NC_TIME_UNITS
    rt.attrib["calendar"] = NC_CALENDAR
    rt.attrib["long_name"] = "time of the saved column state"

    _fill_tile_restart!(g, run, n_layer)
    return nothing
end

function _fill_tile_restart!(g, run::GlacierTileRun, n_layer::Int)
    n_band, n_dt, n_ps = size(run.profiles)
    data = Dict(v => fill(NC_FILL, n_layer, n_band, n_dt, n_ps) for v in PROFILE_VARIABLES)
    valid = zeros(Int32, n_band, n_dt, n_ps)
    rtime = fill(NC_FILL, n_band, n_dt, n_ps)
    t_end = _nc_encode_time(last(run.time))

    for i_ps in 1:n_ps, i_dt in 1:n_dt, i_band in 1:n_band
        p = run.profiles[i_band, i_dt, i_ps]
        p === nothing && continue
        n = length(p[:dz])
        valid[i_band, i_dt, i_ps] = n
        rtime[i_band, i_dt, i_ps] = t_end
        for v in PROFILE_VARIABLES
            data[v][1:n, i_band, i_dt, i_ps] = p[v]
        end
    end

    for v in PROFILE_VARIABLES
        g[string(v)][:, :, :, :] = data[v]
    end
    g["valid_layers"][:, :, :] = valid
    g["restart_time"][:, :, :] = rtime
    return nothing
end

# The run grid axes of a tile file, in the order `_assert_run_grid_matches` compares them.
const TILE_RUN_GRID_AXES = ("band_center", "delta_temperature", "precipitation_scaling")

_assert_tile_grid_matches(ds, run::GlacierTileRun) =
    _assert_run_grid_matches("the existing file",
                             Tuple(collect(Float64, ds[name][:]) for name in TILE_RUN_GRID_AXES),
                             ([b.center for b in run.bands], run.delta_temperatures,
                              run.precipitation_scalings);
                             axes = TILE_RUN_GRID_AXES, what = "tile")
