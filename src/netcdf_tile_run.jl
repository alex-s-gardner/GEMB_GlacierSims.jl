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

Write a [`GlacierTileRun`](@ref) to a CF-1.11 netCDF-4 file at `path`, overwriting any existing file.
The `time` dimension is unlimited so [`append_glacier_tile_netcdf`](@ref) can extend it.

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

    NCDatasets.NCDataset(path, "c") do ds
        NCDatasets.defDim(ds, "time", Inf)          # unlimited, for append
        NCDatasets.defDim(ds, "band", length(run.bands))
        NCDatasets.defDim(ds, "delta_temperature", length(run.delta_temperatures))
        NCDatasets.defDim(ds, "precipitation_scaling", length(run.precipitation_scalings))
        NCDatasets.defDim(ds, "layer", n_layer)

        _write_tile_run_globals!(ds, run; institution, references)
        _write_tile_run_coordinates!(ds, run, n_layer)
        _write_tile_band_series!(ds, run, precision, deflatelevel)
        _write_tile_totals!(ds, run, deflatelevel)
        _write_tile_restart_group!(ds, run, n_layer)
    end

    return path
end

"""
    append_glacier_tile_netcdf(path, run::GlacierTileRun) -> path

Append a continuation run to an existing tile file: extend the unlimited `time` dimension with the new
series and overwrite the restart group with the new final profiles.

The new record must start strictly after the file's last time, and the run grid (bands, temperature
deltas, precipitation scalings) must match the file's — a mismatch means the arrays would not line up,
so it is refused rather than reconciled.
"""
function append_glacier_tile_netcdf(path::AbstractString, run::GlacierTileRun)
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
            ds[string(v)][idx, :, :, :] = run.bands_series[v]
        end
        for v in keys(run.totals)
            ds[_tile_total_variable(v)][idx, :, :] = run.totals[v]
        end

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

"""
    read_glacier_tile_status(path) -> NamedTuple or nothing

What an existing tile file already covers, read from its coordinates and attributes only — **no series
and no firn state**. `nothing` when the file does not exist.

Returns `(; time, band_centers, delta_temperatures, precipitation_scalings, geotile_id, parameters)`.
`time` is the last output time in the file, i.e. the point a continuation must start after.

This exists to be cheap, for the same reason [`read_downscaling_tile_status`](@ref) does: a sweep over
819 tiles decides "is this tile already done" before touching forcing, and that decision is what makes
a re-run cost a file open per tile instead of a forcing pass.
"""
function read_glacier_tile_status(path::AbstractString)
    isfile(path) || return nothing
    return NCDatasets.NCDataset(path, "r") do ds
        n = length(ds["time"])
        (; time = n == 0 ? nothing : _nc_decode_time(ds["time"][n]),
         n_timesteps = n,
         band_centers = collect(Float64, ds["band_center"][:]),
         delta_temperatures = collect(Float64, ds["delta_temperature"][:]),
         precipitation_scalings = collect(Float64, ds["precipitation_scaling"][:]),
         geotile_id = get(ds.attrib, "geotile_id", ""),
         parameters = _read_run_parameters(ds))
    end
end

"""
    read_glacier_tile_restart(path) -> NamedTuple or nothing

Read the firn state and run grid back out of a tile file, in the form [`gemb_glacier_tile`](@ref) would
resume from. `nothing` when the file does not exist.

Returns `(; time, band_centers, delta_temperatures, precipitation_scalings, profiles, parameters,
provenance)`, where `profiles` maps `(i_band, i_dt, i_ps)` to a `DimStack` carrying
[`PROFILE_VARIABLES`](@ref). A run slot that produced no output has no entry.
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

        profiles = Dict{Tuple{Int,Int,Int},DimStack}()
        for i_ps in eachindex(scalings), i_dt in eachindex(deltas), i_band in eachindex(band_centers)
            n = valid[i_band, i_dt, i_ps]
            (ismissing(n) || n <= 0) && continue     # band that produced no output
            n = Int(n)
            zdim = Z(1:n; metadata = layer_metadata)
            cols = NamedTuple(
                v => DimArray(collect(Float64, @view columns[v][1:n, i_band, i_dt, i_ps]), (zdim,))
                for v in PROFILE_VARIABLES)
            profiles[(i_band, i_dt, i_ps)] =
                DimStack(cols; layermetadata = cf_layermetadata(cols; time_axis = false),
                         metadata = provenance)
        end

        (; time = n_time == 0 ? nothing : _nc_decode_time(ds["time"][n_time]),
         band_centers, delta_temperatures = deltas, precipitation_scalings = scalings,
         profiles, parameters = _read_run_parameters(ds), provenance)
    end
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
    t = NCDatasets.defVar(ds, "time", Float64, ("time",))
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
        v.attrib["standard_name"] = "height_above_reference_ellipsoid"
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
    # and `k` in particular is resolved at each band's own centre.
    keys_seen = Set{String}()
    for p in run.band_provenance, k in keys(p)
        push!(keys_seen, k)
    end
    for key in sort!(collect(keys_seen))
        values = [Float64(get(p, key, NaN)) for p in run.band_provenance]
        v = NCDatasets.defVar(ds, "band_" * key, Float64, ("band",); fillvalue = NC_FILL)
        v.attrib["units"] = occursin("lapse_rate", key) && !occursin("_n_", key) ? "K km-1" :
                            (startswith(key, "extrapolation") ? "m" : "1")
        v.attrib["long_name"] = replace(key, "_" => " ")
        v[:] = values
    end
    haskey(ds, "band_extrapolation_above_reanalysis") &&
        (ds["band_extrapolation_above_reanalysis"].attrib["comment"] =
            "Signed distance from this band's center to the highest reanalysis surface that fed " *
            "it. Positive means the band's forcing is a lapse extrapolation above every " *
            "contributing cell, which for glaciers is the norm rather than an error — they occupy " *
            "the high ground within a 0.1 degree cell — and is the dominant uncertainty in this " *
            "forcing, so a fit against altimetry should weight a band by it.")

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

function _write_tile_band_series!(ds, run::GlacierTileRun, precision::Type, deflatelevel::Int)
    dimnames = ("time", "band", "delta_temperature", "precipitation_scaling")
    deflate = deflatelevel > 0 ? deflatelevel : nothing

    for name in TILE_MASS_VARIABLES
        v = NCDatasets.defVar(ds, string(name), precision, dimnames;
                              deflatelevel = deflate, fillvalue = precision(NC_FILL))
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
         "at. Element 1 already carries the first output interval, so subtract it for an anomaly " *
         "referenced to the first output time."),
        (:dh_mass, "surface elevation change from surface mass balance",
         "Cumulative SMB divided by density_ice, referenced to the first output time."),
        (:dh_water, "surface elevation change from column liquid storage",
         "Minus the change in stored water divided by density_ice. Water in a pore adds mass " *
         "without adding thickness, so it belongs to the mass budget but not to the height."),
        (:dh_firn, "surface elevation change from firn compaction",
         "Change in firn air content, referenced to the first output time."),
        (:dh_residual, "closure residual of the height change decomposition",
         "dh - dh[1] minus the sum of the mass, water and firn terms. Should sit at the rounding " *
         "floor; a growing residual means the column's base is no longer at ice density and the " *
         "height series under-counts compaction."),
        (:firn_air_content, "firn air content", ""),
    )
        v = NCDatasets.defVar(ds, string(name), precision, dimnames;
                              deflatelevel = deflate, fillvalue = precision(NC_FILL))
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

    # Float64 whatever `precision` says: these are what a regional total is summed from, and a tile
    # sum is the point at which a truncation stops being local.
    for name in sort!(collect(keys(run.totals)))
        v = NCDatasets.defVar(ds, _tile_total_variable(name), Float64, dimnames;
                              deflatelevel = deflate, fillvalue = NC_FILL)
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
        "Every total_* variable except total_fac is CUMULATIVE from the first output time. The " *
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
        "trajectory diverges slightly at the seam. Separately, a height series resumed from here " *
        "restarts its datum: dh is measured against the column as it was at the seam, so a " *
        "continuation's dh must be offset by the previous record's last value before the two are " *
        "read as one series."

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
