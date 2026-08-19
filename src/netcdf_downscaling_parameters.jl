# CF-compliant netCDF output for one downscaling-parameter tile.
#
# netCDF-4 *is* HDF5 on disk, so these files read with `h5py`/`h5ls` as well as with any netCDF
# reader, while carrying the units, calendar and provenance a bare HDF5 dataset would not. The time
# encoding, the attribute encoder and the CF globals are shared with the per-cell writer in
# `netcdf_output.jl` rather than restated, so the two output families cannot drift apart.
#
# What a tile file has to be sufficient for is the thing that shapes the schema: a consumer must be
# able to re-evaluate `k` at an arbitrary elevation without re-deriving anything. `k` is not a single
# series — the fit carries a `glm × z` interaction, so it is a function of elevation — which is why
# the four fitted coefficients are stored (in Float64, see `_write_decoupling!`) alongside the series
# itself, and why the constants `decoupling_factor_at_elevation` closes over are recorded as global
# attributes. The lapse rate, by contrast, genuinely is one series per tile.
#
# Both fits are written **raw**: `NaN` wherever the tile's forcing supported no fit, nothing
# substituted and nothing clamped. That is upstream's contract and this file does not soften it. The
# one derived variable, `fit_usable`, records the joint screen the two series must be read through —
# see `_write_lapse_rate!` for why that is a variable rather than a comment.

# The settings a re-run has to match before an existing tile file may be reused. Recorded under this
# roster (the same convention `_write_run_parameters!` uses for a cell run) so the status reader
# names them from the file rather than recomputing a key set that a future version might extend.
const _TILE_PARAMETER_KEYS = ("climate_model", "tile_size", "tile_buffer", "min_cells",
                              "area_minimum", "elevation_interval_forcing")

"""
    write_downscaling_tile_netcdf(path, tile, p; climate_model, time_range, tile_size, buffer,
                                  elevation_intervals = nothing, min_cells, area_minimum = 0.0,
                                  precision = Float32, deflatelevel = 4,
                                  institution = nothing, references = nothing) -> path

Write one tile's downscaling parameters to a CF-compliant netCDF-4 (HDF5) file.

`tile` is an entry from [`downscaling_tiles`](@ref) and `p` the result of
[`derive_downscaling_parameters`](@ref) over `tile.buffered`.

The file holds, on a shared `time` axis at the forcing's **native** resolution:

- the decoupling fit — `decoupling_factor` (`k` at the tile's reference elevation), the four
  coefficients `coef_alpha`/`coef_beta`/`coef_gamma`/`coef_delta` that make `k(z)` re-evaluable at
  any elevation via [`decoupling_factor_at_elevation`](@ref), and the `decoupling_r2`,
  `ambient_excess` and `decoupling_n_cells` diagnostics;
- the lapse fit — `lapse_rate` (one series for the whole tile), `lapse_rate_n_cells`,
  `elevation_spread`, and the derived `fit_usable` screen;
- the tile's **core** cells (`cell_*`), which are the cells these parameters *apply to*, and the
  cells the regressions were actually fitted from (`fit_cell_*`), which is the buffered selection
  minus anything whose forcing was unusable.

Both fits are stored raw — `NaN` where the tile's forcing supported no fit, nothing clamped. Read
them through `fit_usable`; see [`derive_downscaling_parameters`](@ref) on why the two series must be
screened together rather than separately.

Pass `elevation_intervals` (a materialized vector from `p.elevation_interval_forcing`) to also store
the glacier-area weighted per-elevation-interval forcing; see
[`write_downscaling_tile_netcdf`](@ref)'s `elevation_interval` variables below. Omitted by default
because it is by far the bulk of the output.

# Keywords
- `precision = Float32`: element type for the fitted series and diagnostics. The four coefficients
  are always Float64 regardless — they are evaluated against elevations up to ~9 km and then divided
  by a thresholded quantity, so truncating them changes `k` far from the reference elevation.
- `deflatelevel = 4`: zlib level. The fit series are `NaN`-dominated and compress heavily.
- `min_cells`, `area_minimum`: recorded as run parameters, so a re-run with different ones re-derives
  rather than reading a file that means something else.

There is no append counterpart. Extending the time window means re-reading every cell's forcing and
re-running a pooled cross-cell regression, so a window change is a fresh derivation that rewrites
the file rather than a continuation of it.
"""
function write_downscaling_tile_netcdf(path::AbstractString, tile, p;
                                       climate_model,
                                       time_range,
                                       tile_size::Real,
                                       buffer::Real,
                                       elevation_intervals = nothing,
                                       min_cells::Int = _MIN_CELLS_DEFAULT,
                                       area_minimum::Real = 0.0,
                                       precision::Type = Float32,
                                       deflatelevel::Int = 4,
                                       institution = nothing,
                                       references = nothing)
    mkpath(dirname(path))
    n_time = length(p.time)

    NCDatasets.NCDataset(path, "c") do ds
        # Fixed-length, not unlimited: there is no append path (see the docstring), and a fixed
        # length is what lets every time variable be chunked and compressed properly.
        NCDatasets.defDim(ds, "time", n_time)
        NCDatasets.defDim(ds, "cell", nrow(tile.core))
        NCDatasets.defDim(ds, "fit_cell", nrow(p.grid_cells))

        _write_tile_time!(ds, p.time, deflatelevel)
        _write_decoupling!(ds, p, precision, deflatelevel)
        _write_lapse_rate!(ds, p, precision, deflatelevel)
        _write_tile_cells!(ds, tile, p)
        if elevation_intervals !== nothing
            _write_elevation_intervals!(ds, elevation_intervals, precision, deflatelevel)
        end
        _write_tile_globals!(ds, tile, p; climate_model, time_range, tile_size, buffer,
                             min_cells, area_minimum,
                             n_intervals = elevation_intervals === nothing ? nothing :
                                           length(elevation_intervals),
                             institution, references)
    end
    return path
end

"""
    write_sparse_downscaling_tile_netcdf(path, tile; climate_model, time_range, tile_size, buffer,
                                         min_cells, area_minimum = 0.0, reason,
                                         institution = nothing, references = nothing) -> path

Write a tile that could not be fitted: too few grid cells for the cross-cell regressions, or no cell
with usable forcing at all.

A file is written rather than skipped because a *missing* file cannot be told apart from a tile that
was never run, which is what a resumable sweep needs to know. The `cell` dimension is still
populated, so the point→tile mapping holds for every cell on Earth whether or not its tile could be
fitted.

The `time` dimension has length **zero**. A tile below `min_cells` never loads any forcing, so its
native timestep is genuinely unknown — there is nothing to write an axis from. The requested window
is recorded in `requested_time_coverage_start`/`_end` instead, and
[`read_downscaling_tile_status`](@ref) judges a sparse tile's coverage from those.
"""
function write_sparse_downscaling_tile_netcdf(path::AbstractString, tile;
                                             climate_model,
                                             time_range,
                                             tile_size::Real,
                                             buffer::Real,
                                             min_cells::Int = _MIN_CELLS_DEFAULT,
                                             area_minimum::Real = 0.0,
                                             reason::AbstractString,
                                             elevation_interval_forcing::Bool = false,
                                             institution = nothing,
                                             references = nothing)
    mkpath(dirname(path))
    NCDatasets.NCDataset(path, "c") do ds
        NCDatasets.defDim(ds, "time", 0)
        NCDatasets.defDim(ds, "cell", nrow(tile.core))
        NCDatasets.defDim(ds, "fit_cell", 0)

        _write_tile_time!(ds, DateTime[], 0)
        _write_tile_cells!(ds, tile, nothing)
        _write_tile_globals!(ds, tile, nothing; climate_model, time_range, tile_size, buffer,
                             min_cells, area_minimum, n_intervals = nothing,
                             sparse_reason = reason,
                             elevation_interval_forcing, institution, references)
    end
    return path
end

function _write_tile_time!(ds, time, deflatelevel)
    t = NCDatasets.defVar(ds, "time", Float64, ("time",);
                          deflatelevel = deflatelevel > 0 ? deflatelevel : nothing)
    t.attrib["units"] = NC_TIME_UNITS
    t.attrib["calendar"] = NC_CALENDAR
    t.attrib["standard_name"] = "time"
    t.attrib["long_name"] = "forcing timestep the fits were taken at"
    t.attrib["comment"] = "The forcing's own native resolution. Both fits are per-timestep " *
                          "cross-cell regressions, so grouping them (by month, by hour) is a " *
                          "downstream decision made on this axis."
    isempty(time) || (t[1:length(time)] = _nc_encode_time.(time))
    return nothing
end

# The decoupling fit. Eight variables, because that is what the fit reports and dropping any of them
# would make the file a summary rather than a record.
function _write_decoupling!(ds, p, precision, deflatelevel)
    d = p.decoupling
    # The four coefficients are Float64 no matter what `precision` says. `k(z)` is
    # ((α+γ) + (β+δ)z) / (α + βz) evaluated at elevations up to ~9000 m, with the denominator
    # thresholded near 0.5 K: β is a small number multiplied by a large one, so a Float32 β shifts
    # `k` at elevations far from where it was fitted — which is exactly where the glacier is.
    for (name, layer, units, long_name) in (
        ("coef_alpha", d.coef_alpha, "K", "ambient warm-excess regression intercept"),
        ("coef_beta", d.coef_beta, "K m-1", "ambient warm-excess elevation coefficient"),
        ("coef_gamma", d.coef_gamma, "K", "warm-excess glacier-fraction coefficient"),
        ("coef_delta", d.coef_delta, "K m-1",
         "warm-excess glacier-fraction x elevation interaction coefficient"))
        _defvar_time!(ds, name, Float64, layer, units, long_name, deflatelevel;
                      comment = "Coefficient of the per-timestep regression " *
                                "E = alpha + beta*z + gamma*glm + delta*glm*z of the warm excess " *
                                "E = max(T - 273.15, 0) across the tile's cells. Stored so k can " *
                                "be re-evaluated at any elevation; see the " *
                                "decoupling_factor_formula global attribute.")
    end

    _defvar_time!(ds, "decoupling_factor", precision, d.decoupling_factor, "1",
                  "on-glacier air temperature decoupling factor at the reference elevation",
                  deflatelevel;
                  comment = "Raw per-timestep fit, NaN where the tile's forcing supported none. " *
                            "Evaluated at reference_elevation; k varies with elevation, so use " *
                            "the coef_* variables to evaluate it at the elevation of interest " *
                            "rather than taking this value as the tile's k.")
    _defvar_time!(ds, "decoupling_r2", precision, d.r2, "1",
                  "coefficient of determination of the warm-excess regression", deflatelevel)
    _defvar_time!(ds, "ambient_excess", precision, d.ambient_excess, "K",
                  "fitted ambient warm excess at the reference elevation", deflatelevel;
                  comment = "alpha + beta*z at reference_elevation. A fit needs at least " *
                            "min_ambient_excess K here to be identifiable at all.")
    _defvar_time!(ds, "decoupling_n_cells", Int32, d.n_cells, "1",
                  "grid cells contributing to the decoupling fit at this timestep", deflatelevel)
    return nothing
end

function _write_lapse_rate!(ds, p, precision, deflatelevel)
    lr = p.lapse_rate
    _defvar_time!(ds, "lapse_rate", precision, lr.lapse_rate, "K km-1",
                  "on-glacier air temperature lapse rate", deflatelevel;
                  comment = "Positive means cooling with height, the sign convention " *
                            "climate_adjust_for_elevation expects. ONE series for the whole tile: " *
                            "a per-timestep pooled regression of on-glacier temperature on " *
                            "elevation across the tile's cells, so it does not vary with " *
                            "elevation. Raw — NaN where unfittable, never clamped.")
    _defvar_time!(ds, "lapse_rate_n_cells", Int32, lr.n_cells, "1",
                  "grid cells contributing to the lapse-rate fit at this timestep", deflatelevel)
    _defvar_time!(ds, "elevation_spread", precision, lr.elevation_spread, "m",
                  "standard deviation of contributing cell reanalysis elevations", deflatelevel;
                  comment = "How much elevation range the slope was identified from at this " *
                            "timestep. A small spread means a poorly constrained lapse rate even " *
                            "where the fit itself succeeded.")

    # The joint screen, as a variable rather than a comment.
    #
    # `derive_lapse_rate` consumes `k` to bring the cells to a common on-glacier state, so a `k`
    # outside (0, 1] corrupts that timestep's *slope* as well — and the corruption is a smooth false
    # gradient rather than scatter, so it is invisible in the slope series alone and no robust
    # estimator removes it. A reader who screens the two series separately therefore gets a
    # plausible, wrong answer. One byte per timestep makes the correct screen impossible to miss;
    # the alternative (storing a second, pre-screened copy of `lapse_rate`) would double the largest
    # variable in the file to say the same thing.
    k = collect(p.decoupling.decoupling_factor)
    usable = Int8[isfinite(x) && 0.0 < x <= 1.0 for x in k]
    v = NCDatasets.defVar(ds, "fit_usable", Int8, ("time",);
                          deflatelevel = deflatelevel > 0 ? deflatelevel : nothing)
    v.attrib["units"] = "1"
    v.attrib["long_name"] = "whether this timestep's fits are within the domain their consumers accept"
    v.attrib["flag_values"] = Int8[0, 1]
    v.attrib["flag_meanings"] = "unusable usable"
    v.attrib["comment"] =
        "1 where the fitted k is finite and in (0, 1], the domain climate_adjust_for_glacier " *
        "accepts. SCREEN THE TWO SERIES TOGETHER: derive_lapse_rate consumes k, so an " *
        "out-of-domain k corrupts that timestep's lapse_rate as a smooth false gradient that no " *
        "robust estimator can see. Timesteps where k is NaN are flagged 0 here but their " *
        "lapse_rate is the uncorrected ambient slope, which is still meaningful."
    isempty(usable) || (v[1:length(usable)] = usable)
    return nothing
end

# Per-cell variables. Two sets, because they answer two different questions: `cell_*` is what this
# tile's parameters apply to (its core, i.e. the cells it owns in the global partition), and
# `fit_cell_*` is what they were fitted from (the buffered selection, minus cells whose forcing was
# unusable). Conflating them would make a tile's area double-count against its neighbours'.
function _write_tile_cells!(ds, tile, p)
    core = tile.core
    _defvar_cells!(ds, "cell", "cell_latitude", Float64,
                   Float64.(core[!, :latitude]), "degrees_north", "core cell center latitude";
                   standard_name = "latitude")
    _defvar_cells!(ds, "cell", "cell_longitude", Float64,
                   [wrap_lon(Float64(x)) for x in core[!, :longitude]],
                   "degrees_east", "core cell center longitude"; standard_name = "longitude")
    _defvar_cells!(ds, "cell", "cell_glacier_area", Float64,
                   glacier_area_column(core), "km2", "total glacier area in the core cell")
    _defvar_cells!(ds, "cell", "cell_glm", Float64,
                   [_row_glm(r) for r in eachrow(core)], "1",
                   "ERA5-Land glacier mask fraction of the core cell")
    if hasproperty(core, :chunk_id)
        # `missing` as -1 rather than as a fill value: this is an identifier, not a measurement, and
        # -1 is not a valid chunk id.
        _defvar_cells!(ds, "cell", "cell_chunk_id", Int32,
                       [ismissing(v) ? Int32(-1) : Int32(v) for v in core[!, :chunk_id]], "1",
                       "ERA5-Land download chunk the core cell belongs to";
                       comment = "-1 where the table carried no chunk id.")
    end

    p === nothing && return nothing

    fit = p.grid_cells
    _defvar_cells!(ds, "fit_cell", "fit_cell_latitude", Float64,
                   Float64.(fit[!, :latitude]), "degrees_north", "fitted cell center latitude";
                   standard_name = "latitude")
    _defvar_cells!(ds, "fit_cell", "fit_cell_longitude", Float64,
                   [wrap_lon(Float64(x)) for x in fit[!, :longitude]],
                   "degrees_east", "fitted cell center longitude"; standard_name = "longitude")
    _defvar_cells!(ds, "fit_cell", "fit_cell_elevation", Float64,
                   collect(Float64, p.cell_elevations), "m",
                   "reanalysis surface elevation of the fitted cell";
                   standard_name = "surface_altitude",
                   comment = "The elevations the cross-cell regressions were taken over. Their " *
                             "spread is what identifies the lapse rate; their area-weighted mean " *
                             "is reference_elevation.")
    _defvar_cells!(ds, "fit_cell", "fit_cell_glacier_area", Float64,
                   collect(Float64, p.cell_areas), "km2", "total glacier area in the fitted cell")
    _defvar_cells!(ds, "fit_cell", "fit_cell_glm", Float64,
                   [_row_glm(r) for r in eachrow(fit)], "1",
                   "ERA5-Land glacier mask fraction of the fitted cell")
    return nothing
end

# The optional bulk: glacier-area weighted forcing per elevation interval.
#
# This is the one part of the output that *applies* the fits rather than reporting them, so it is
# also the only part affected by the fill/clamp settings — which is why the applied `k` is stored
# alongside the forcing instead of being left to be recomputed from the coefficients. It cannot be
# recomputed: it has been filled where the fit was absent and clamped into the domain its consumer
# accepts, and neither is recoverable from the raw fit.
function _write_elevation_intervals!(ds, intervals, precision, deflatelevel)
    NCDatasets.defDim(ds, "elevation_interval", length(intervals))
    n_time = ds.dim["time"]
    deflate = deflatelevel > 0 ? deflatelevel : nothing

    for (name, values, units, long_name, standard_name) in (
        ("interval_center", [Float64(iv.center) for iv in intervals], "m",
         "elevation interval center", "height_above_reference_ellipsoid"),
        ("interval_lower", [Float64(iv.lo) for iv in intervals], "m",
         "elevation interval lower edge", "height_above_reference_ellipsoid"),
        ("interval_upper", [Float64(iv.hi) for iv in intervals], "m",
         "elevation interval upper edge", "height_above_reference_ellipsoid"),
        ("interval_area", [Float64(iv.area) for iv in intervals], "km2",
         "glacier area in the elevation interval", ""))
        v = NCDatasets.defVar(ds, name, Float64, ("elevation_interval",))
        v.attrib["units"] = units
        v.attrib["long_name"] = long_name
        isempty(standard_name) || (v.attrib["standard_name"] = standard_name)
        v[:] = values
    end
    ds["interval_area"].attrib["comment"] =
        "Sums over the intervals to the total glacier area of the cells whose forcing was usable, " *
        "so no ice is dropped between the cells and the intervals."

    iv_n = NCDatasets.defVar(ds, "interval_n_cells", Int32, ("elevation_interval",))
    iv_n.attrib["units"] = "1"
    iv_n.attrib["long_name"] = "grid cells contributing ice to the elevation interval"
    iv_n[:] = Int32[iv.n_cells for iv in intervals]

    meta(iv, key, default = NaN) = get(DimensionalData.metadata(iv.forcing), key, default)

    ex = NCDatasets.defVar(ds, "interval_extrapolation_above_reanalysis", Float64,
                           ("elevation_interval",))
    ex.attrib["units"] = "m"
    ex.attrib["long_name"] = "interval center above the highest contributing reanalysis surface"
    ex.attrib["comment"] =
        "Positive means this interval's forcing is a lapse extrapolation above every reanalysis " *
        "cell that fed it. For glaciers that is the norm rather than an error — they occupy the " *
        "high ground within a 0.1 degree cell — and it is the dominant uncertainty in this forcing."
    ex[:] = [Float64(meta(iv, "extrapolation_above_reanalysis")) for iv in intervals]

    for (name, key, long_name) in (
        ("interval_decoupling_factor_n_held", "glacier_decoupling_factor_n_held",
         "timesteps whose k was evaluated at a held elevation"),
        ("interval_decoupling_factor_n_filled", "glacier_decoupling_factor_n_filled",
         "timesteps whose k was filled because the fit measured none"),
        ("interval_decoupling_factor_n_clamped", "glacier_decoupling_factor_n_clamped",
         "timesteps whose k was clamped into the accepted domain"))
        v = NCDatasets.defVar(ds, name, Int32, ("elevation_interval",))
        v.attrib["units"] = "1"
        v.attrib["long_name"] = long_name
        v[:] = Int32[Int(meta(iv, key, 0)) for iv in intervals]
    end
    ds["interval_decoupling_factor_n_filled"].attrib["comment"] =
        "How much of the applied k was substituted rather than measured. A mostly-filled interval " *
        "carries decoupling_factor_fill, not a fit."

    # The applied k, per interval per timestep.
    #
    # Float64 regardless of `precision`, and this one is not a precision nicety but a correctness
    # requirement. `_make_applicable` clamps a below-domain fit to `nextfloat(0.0)` — 5e-324, the
    # smallest Float64 above zero — precisely so the value still satisfies the half-open `(0, 1]`
    # domain `climate_adjust_for_glacier` validates. Float32's smallest subnormal is ~1e-45, so that
    # sentinel underflows to exactly 0.0 on the way to disk and the value read back is *outside* the
    # domain it was clamped into: the stored forcing would be rejected by the very check the clamp
    # exists to pass. Observed on real Wrangell forcing, where a clamped timestep is ordinary.
    ak = NCDatasets.defVar(ds, "interval_decoupling_factor", Float64,
                           ("time", "elevation_interval"); deflatelevel = deflate,
                           fillvalue = NC_FILL)
    ak.attrib["units"] = "1"
    ak.attrib["long_name"] = "decoupling factor applied to this interval's forcing"
    ak.attrib["comment"] =
        "k evaluated at this interval's center, then filled and clamped into (0, 1]. NOT " *
        "recoverable from the coef_* variables, which are the raw fit: this is what was actually " *
        "applied to the temperature below. Not yet weighted by each cell's glm, which the " *
        "aggregation applies per cell as 1 - (1 - k)*(1 - glm). Stored double-precision because a " *
        "clamped value can sit one float above zero, which single precision would flatten to zero " *
        "and so out of the domain its consumer accepts."
    for (j, iv) in enumerate(intervals)
        ak[:, j] = convert(Vector{Float64}, collect(iv.decoupling_factor))
    end

    # The forcing itself: same seven layers as `climate_forcing` output, so an interval's column
    # drops into a sweep unchanged.
    for var in _FORCING_VARIABLES
        v = NCDatasets.defVar(ds, string(var), precision, ("time", "elevation_interval");
                              deflatelevel = deflate, fillvalue = precision(NC_FILL))
        layer_meta = DimensionalData.metadata(first(intervals).forcing[var])
        for key in ("units", "long_name", "standard_name")
            haskey(layer_meta, key) && (v.attrib[key] = layer_meta[key])
        end
        for (j, iv) in enumerate(intervals)
            v[:, j] = convert(Vector{precision}, collect(iv.forcing[var]))
        end
    end
    ds["temperature_air"].attrib["comment"] =
        "Glacier-area weighted mean over the cells holding ice in this interval, each lapsed to " *
        "the interval center and then decoupled — in that order, which is the order " *
        "climate_adjust_for_glacier requires."
    return nothing
end

# A variable on the shared time axis. `NaN` is the fill for the floating-point ones, which is what
# the fits themselves use for "unmeasurable", so an absent fit reads back as an absent fit rather
# than as a number.
function _defvar_time!(ds, name, T, values, units, long_name, deflatelevel; comment = "")
    v = NCDatasets.defVar(ds, name, T, ("time",);
                          deflatelevel = deflatelevel > 0 ? deflatelevel : nothing,
                          fillvalue = T <: AbstractFloat ? T(NC_FILL) : nothing)
    v.attrib["units"] = units
    v.attrib["long_name"] = long_name
    isempty(comment) || (v.attrib["comment"] = comment)
    n = length(values)
    n == 0 || (v[1:n] = convert(Vector{T}, collect(values)))
    return v
end

function _defvar_cells!(ds, dim, name, T, values, units, long_name;
                        standard_name = "", comment = "")
    v = NCDatasets.defVar(ds, name, T, (dim,))
    v.attrib["units"] = units
    v.attrib["long_name"] = long_name
    isempty(standard_name) || (v.attrib["standard_name"] = standard_name)
    isempty(comment) || (v.attrib["comment"] = comment)
    n = length(values)
    n == 0 || (v[1:n] = convert(Vector{T}, collect(values)))
    return v
end

function _write_tile_globals!(ds, tile, p; climate_model, time_range, tile_size, buffer,
                              min_cells, area_minimum, n_intervals,
                              sparse_reason = "", elevation_interval_forcing = nothing,
                              institution = nothing, references = nothing)
    for (k, v) in GEMB_CF_GLOBAL_ATTRIBUTES
        ds.attrib[k] = v
    end
    ds.attrib["history"] = "$(Dates.format(now(), "yyyy-mm-ddTHH:MM:SS")): created by " *
                           "GEMB_GlacierSims.write_downscaling_tile_netcdf"
    ds.attrib["title"] = "Regional glacier downscaling parameters for one grid tile"
    ds.attrib["featureType"] = "timeSeries"
    institution === nothing || (ds.attrib["institution"] = institution)
    references === nothing || (ds.attrib["references"] = references)

    # --- tiling geometry ---
    ds.attrib["tile_index_lon"] = tile.index[1]
    ds.attrib["tile_index_lat"] = tile.index[2]
    ds.attrib["tile_lon_min"] = tile.bounds.lon_min
    ds.attrib["tile_lon_max"] = tile.bounds.lon_max
    ds.attrib["tile_lat_min"] = tile.bounds.lat_min
    ds.attrib["tile_lat_max"] = tile.bounds.lat_max
    ds.attrib["buffered_lon_min"] = tile.buffered_bounds.lon_min
    ds.attrib["buffered_lon_max"] = tile.buffered_bounds.lon_max
    ds.attrib["buffered_lat_min"] = tile.buffered_bounds.lat_min
    ds.attrib["buffered_lat_max"] = tile.buffered_bounds.lat_max
    ds.attrib["longitude_convention"] = "(-180, 180]"
    ds.attrib["tile_assignment"] =
        "A cell belongs to the tile (fld(lon, tile_size)*tile_size, fld(lat, tile_size)*tile_size) " *
        "with lon wrapped to (-180, 180] and an exact 180.0 folded to -180. Assignment is total " *
        "and non-overlapping, so every cell of the source table belongs to exactly one tile and " *
        "the cell_* variables of all tiles partition it."
    ds.attrib["tile_edges_comment"] =
        "tile_size divides 360, so tile edges fall exactly on +/-180 and no tile straddles the " *
        "antimeridian. A tile against the seam does have a buffer that crosses it: buffered_lon_min " *
        "may be < -180 or buffered_lon_max > 180, which is the honest span rather than a wrapped " *
        "one. Membership was tested by modular longitude distance, not against these bounds."
    ds.attrib["buffer_comment"] =
        "The parameters in this file apply to the tile's core cells (cell_*) but were fitted from " *
        "the buffered selection (fit_cell_*), because both fits are regressions ACROSS cells and a " *
        "bare tile often carries too little elevation and glacier-fraction range to constrain " *
        "them. Buffers of neighbouring tiles overlap; cores do not."

    # --- derivation settings (the roster a re-run must match) ---
    ds.attrib["climate_model"] = string(climate_model)
    ds.attrib["tile_size"] = Float64(tile_size)
    ds.attrib["tile_buffer"] = Float64(buffer)
    ds.attrib["min_cells"] = min_cells
    ds.attrib["area_minimum"] = Float64(area_minimum)
    has_intervals = elevation_interval_forcing === nothing ? n_intervals !== nothing :
                    elevation_interval_forcing
    ds.attrib["elevation_interval_forcing"] = _encode_attribute(has_intervals)
    ds.attrib["downscaling_parameters"] = join(_TILE_PARAMETER_KEYS, " ")
    ds.attrib["downscaling_parameters_comment"] =
        "The settings that define this derivation. A sweep re-derives the tile rather than reusing " *
        "this file when any of them differs, because the file name records only the tile corner " *
        "and cannot distinguish two griddings or two forcing windows."

    # Requested vs actual window. The requested one is what a sparse tile has instead of a time
    # axis, so it is written for every tile rather than only for those.
    ds.attrib["requested_time_coverage_start"] = _encode_attribute(first(time_range))
    ds.attrib["requested_time_coverage_end"] = _encode_attribute(last(time_range))

    # --- counts and the sparse flag ---
    ds.attrib["n_cells_core"] = nrow(tile.core)
    ds.attrib["n_cells_buffered"] = nrow(tile.buffered)
    is_sparse = p === nothing
    ds.attrib["sparse"] = _encode_attribute(is_sparse)
    ds.attrib["sparse_reason"] = sparse_reason
    if is_sparse
        ds.attrib["n_cells_used"] = 0
        ds.attrib["n_timesteps"] = 0
        ds.attrib["n_decoupling_factor_fitted"] = 0
        ds.attrib["n_lapse_rate_fitted"] = 0
        ds.attrib["sparse_comment"] =
            "No fit was attempted or none succeeded, so there is no time axis: a tile below " *
            "min_cells never loads forcing and its native timestep is therefore unknown. Read " *
            "requested_time_coverage_* for the window that was asked for. The cell_* variables " *
            "are still complete, so this tile's cells are still accounted for globally."
        return nothing
    end

    ds.attrib["n_cells_used"] = nrow(p.grid_cells)
    ds.attrib["n_timesteps"] = length(p.time)
    ds.attrib["time_coverage_start"] = _encode_attribute(first(p.time))
    ds.attrib["time_coverage_end"] = _encode_attribute(last(p.time))
    n_intervals === nothing || (ds.attrib["n_elevation_intervals"] = n_intervals)
    for key in ("n_decoupling_factor_fitted", "n_lapse_rate_fitted", "n_elevation_intervals",
                "mean_elevation", "adjustment_order")
        haskey(p.provenance, key) &&
            (ds.attrib[key] = _encode_attribute(p.provenance[key]))
    end
    ds.attrib["n_fit_usable"] =
        count(x -> isfinite(x) && 0.0 < x <= 1.0, p.decoupling.decoupling_factor)

    # --- what makes k(z) reproducible from this file alone ---
    dmeta = DimensionalData.metadata(p.decoupling)
    haskey(dmeta, "reference_elevation") &&
        (ds.attrib["reference_elevation"] = Float64(dmeta["reference_elevation"]))
    ds.attrib["decoupling_factor_formula"] =
        "k(z) = ((alpha + gamma) + (beta + delta)*z) / (alpha + beta*z)"
    ds.attrib["decoupling_factor_comment"] =
        "Evaluate with the coef_* variables at the elevation of interest, which is what " *
        "GEMB_ClimateForcing.decoupling_factor_at_elevation does. Where the ambient excess " *
        "alpha + beta*z falls below min_ambient_excess the evaluation elevation is held at the " *
        "nearest elevation where it does not, rather than reverting to 1."
    ds.attrib["min_ambient_excess"] = _MIN_AMBIENT_EXCESS
    ds.attrib["decoupling_reference_temperature"] = _DECOUPLING_REFERENCE_TEMPERATURE
    ds.attrib["decoupling_factor_limits"] = collect(Float64, _DECOUPLING_FACTOR_LIMITS)
    ds.attrib["lapse_rate_limits"] = collect(Float64, _LAPSE_RATE_LIMITS)
    ds.attrib["fits_are_raw"] =
        "Both fitted series are reported as measured: NaN where the tile's forcing supported no " *
        "fit, with nothing substituted and nothing clamped. A lapse rate outside lapse_rate_limits " *
        "or a k outside decoupling_factor_limits is evidence about that timestep, not an error to " *
        "be hidden. Screen with fit_usable; see its comment for why the two series must be " *
        "screened jointly."
    return nothing
end

"""
    read_downscaling_tile_status(path) -> NamedTuple or nothing

What an existing tile file covers, read from its attributes and `time` coordinate only — **no
forcing is touched**. `nothing` when the file does not exist.

Returns `(; sparse, sparse_reason, n_timesteps, time_first, time_last, n_cells_core,
n_cells_buffered, n_cells_used, n_decoupling_factor_fitted, n_lapse_rate_fitted, parameters)`.

This is what makes a re-run cheap, and the reason it reads nothing but metadata: deciding "this tile
is already done" has to happen before the forcing pass, since that pass is the entire cost of the
sweep.

For a **sparse** tile `time_first`/`time_last` come from the requested window rather than from the
(empty) time axis — see [`write_sparse_downscaling_tile_netcdf`](@ref).
"""
function read_downscaling_tile_status(path::AbstractString)
    isfile(path) || return nothing
    return NCDatasets.NCDataset(path, "r") do ds
        n = length(ds["time"])
        sparse = get(ds.attrib, "sparse", "false") == "true"
        # A sparse tile has no time axis at all, and a non-sparse one written with a zero-length
        # window would be indistinguishable from it here — both fall back to the requested window,
        # which is the only thing either can be judged on.
        if sparse || n == 0
            first_t = _tile_attr_time(ds, "requested_time_coverage_start")
            last_t = _tile_attr_time(ds, "requested_time_coverage_end")
        else
            first_t = _nc_decode_time(ds["time"][1])
            last_t = _nc_decode_time(ds["time"][n])
        end
        (; sparse,
         sparse_reason = get(ds.attrib, "sparse_reason", ""),
         n_timesteps = n,
         time_first = first_t,
         time_last = last_t,
         n_cells_core = Int(get(ds.attrib, "n_cells_core", 0)),
         n_cells_buffered = Int(get(ds.attrib, "n_cells_buffered", 0)),
         n_cells_used = Int(get(ds.attrib, "n_cells_used", 0)),
         n_decoupling_factor_fitted = Int(get(ds.attrib, "n_decoupling_factor_fitted", 0)),
         n_lapse_rate_fitted = Int(get(ds.attrib, "n_lapse_rate_fitted", 0)),
         parameters = _read_tile_parameters(ds))
    end
end

# The stored derivation settings, named by the file's own roster so a file written by an older or
# newer version is still read correctly. Values come back in their encoded form, which is what the
# sweep's comparison is written against.
function _read_tile_parameters(ds)
    haskey(ds.attrib, "downscaling_parameters") || return Dict{String,Any}()
    names = split(ds.attrib["downscaling_parameters"])
    return Dict{String,Any}(k => ds.attrib[k] for k in names if haskey(ds.attrib, k))
end

function _tile_attr_time(ds, key)
    haskey(ds.attrib, key) || return nothing
    v = ds.attrib[key]
    v isa DateTime && return v
    return DateTime(String(v))
end

"""
    read_downscaling_tile(path) -> NamedTuple

Read a tile file back into the shape [`derive_downscaling_parameters`](@ref) returns, so a stored
tile and a freshly derived one are consumed by the same code.

Returns `(; time, decoupling, lapse_rate, fit_usable, cells, fit_cells, intervals, attributes)`.
`decoupling` is a `DimStack` carrying the same layer names the fit does — including the four
coefficients — so it can be handed straight to [`decoupling_factor_at_elevation`](@ref) to
re-evaluate `k` at any elevation. `intervals` is `nothing` unless the file stores the optional
elevation-interval forcing.
"""
function read_downscaling_tile(path::AbstractString)
    return NCDatasets.NCDataset(path, "r") do ds
        time = _nc_decode_time.(ds["time"][:])
        ti = Ti(collect(DateTime, time))
        get1(name) = haskey(ds, name) ? collect(ds[name][:]) : nothing

        # `NaN` for the missing values rather than `missing`: that is what the fits use and what the
        # arithmetic downstream expects, and NCDatasets hands back `missing` for a fill value.
        # A sparse tile carries no fit variables at all, and its `time` axis is empty, so an absent
        # variable reads as the empty vector rather than as `nothing`. That keeps a sparse tile a
        # *readable* tile — same fields, same types, zero rows — so a consumer iterating a directory
        # does not need a separate branch for it. (An absent *cell* variable is a different matter;
        # see `fit_usable` below, which is genuinely `nothing` when the file has no fits.)
        f64(name) = (v = get1(name); v === nothing ? Float64[] :
                     Float64[x === missing ? NaN : Float64(x) for x in v])
        i32(name) = (v = get1(name); v === nothing ? Int[] :
                     Int[x === missing ? 0 : Int(x) for x in v])

        dmeta = Dict{String,Any}(k => ds.attrib[k] for k in keys(ds.attrib))
        decoupling = DimStack((decoupling_factor = DimArray(f64("decoupling_factor"), (ti,)),
                               n_cells = DimArray(i32("decoupling_n_cells"), (ti,)),
                               r2 = DimArray(f64("decoupling_r2"), (ti,)),
                               ambient_excess = DimArray(f64("ambient_excess"), (ti,)),
                               coef_alpha = DimArray(f64("coef_alpha"), (ti,)),
                               coef_beta = DimArray(f64("coef_beta"), (ti,)),
                               coef_gamma = DimArray(f64("coef_gamma"), (ti,)),
                               coef_delta = DimArray(f64("coef_delta"), (ti,)));
                              metadata = dmeta)
        lapse_rate = DimStack((lapse_rate = DimArray(f64("lapse_rate"), (ti,)),
                               n_cells = DimArray(i32("lapse_rate_n_cells"), (ti,)),
                               elevation_spread = DimArray(f64("elevation_spread"), (ti,)));
                              metadata = dmeta)

        cells = DataFrame(latitude = f64("cell_latitude"), longitude = f64("cell_longitude"),
                          glacier_area = f64("cell_glacier_area"), glm = f64("cell_glm"))
        haskey(ds, "cell_chunk_id") && (cells[!, :chunk_id] = i32("cell_chunk_id"))
        fit_cells = DataFrame(latitude = f64("fit_cell_latitude"),
                              longitude = f64("fit_cell_longitude"),
                              elevation = f64("fit_cell_elevation"),
                              glacier_area = f64("fit_cell_glacier_area"),
                              glm = f64("fit_cell_glm"))

        intervals = haskey(ds.dim, "elevation_interval") ? _read_tile_intervals(ds, ti) : nothing

        (; time, decoupling, lapse_rate,
         fit_usable = (v = get1("fit_usable"); v === nothing ? nothing : Bool[x == 1 for x in v]),
         cells, fit_cells, intervals, attributes = dmeta)
    end
end

# The stored interval forcing, rebuilt as one `DimStack` per interval so each element matches what
# `elevation_interval_forcing` yields and drops into a sweep the same way.
function _read_tile_intervals(ds, ti)
    n = ds.dim["elevation_interval"]
    col(name, j) = Float64[x === missing ? NaN : Float64(x) for x in ds[name][:, j]]
    return [begin
        layers = NamedTuple(v => DimArray(col(string(v), j), (ti,)) for v in _FORCING_VARIABLES)
        meta = Dict{String,Any}(
            "elevation" => Float64(ds["interval_center"][j]),
            "glacier_area" => Float64(ds["interval_area"][j]),
            "n_grid_cells" => Int(ds["interval_n_cells"][j]),
            "elevation_interval_lower" => Float64(ds["interval_lower"][j]),
            "elevation_interval_upper" => Float64(ds["interval_upper"][j]),
            "extrapolation_above_reanalysis" =>
                Float64(ds["interval_extrapolation_above_reanalysis"][j]))
        (; lo = Int(ds["interval_lower"][j]), hi = Int(ds["interval_upper"][j]),
         center = Float64(ds["interval_center"][j]),
         area = Float64(ds["interval_area"][j]),
         n_cells = Int(ds["interval_n_cells"][j]),
         decoupling_factor = col("interval_decoupling_factor", j),
         forcing = DimStack(layers; metadata = meta))
    end for j in 1:n]
end
