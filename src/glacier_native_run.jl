"""
GEMB at the native resolution of the forcing: every ERA5-Land cell of a tile on its own forcing and its
own hypsometry, under the tile's calibrated forcing correction.

A tile run ([`gemb_glacier_tile`](@ref)) forces each elevation band with the glacier-area weighted mean
of its cells' forcing, each lapsed to the band center and decoupled by the tile's
[`AppliedDownscaling`](@ref). A native run makes exactly the same per-cell adjustment and stops there:
each cell is a one-cell tile, so [`elevation_interval_forcing`](@ref) over that cell alone yields the
cell's own adjusted forcing per band, and [`gemb_glacier_tile`](@ref) runs it at the single
`(ΔT, pscale)` the calibration chose. The tile run's band forcing is the area-weighted mean of these
per-cell forcings, so the two runs share every adjustment and differ only in whether GEMB sees the cells
separately or averaged — which is the information a native run adds.

Each cell's output is a tile-format file ([`write_glacier_tile_netcdf`](@ref)) with one-element
`delta_temperature` and `precipitation_scaling` axes holding the correction, so the reader, the restart
and the append path are the tile run's own. A changed correction changes those axes, which makes an
existing file a different experiment rather than one to extend.
"""

"""
    native_cells(tile) -> Vector{NamedTuple}

Each core cell of a [`downscaling_tiles`](@ref) entry as a one-cell tile: `(; index, bounds, name, core,
buffered, latitude, longitude)`, where `core` is the cell's own one-row table and `index`, `bounds` and
`buffered` are the tile's. `name` is [`cell_output_name`](@ref) of the cell center.

Keeping the tile's bounds is what makes the cell's `geotile_id` the geotile it lies in, and keeping the
tile's buffered neighbourhood is what lets a masked cell borrow forcing exactly as it would in the tile
run.
"""
function native_cells(tile)
    core = tile.core
    return [(; tile.index, tile.bounds, tile.buffered,
             core = core[i:i, :],
             name = cell_output_name(core.latitude[i], core.longitude[i]),
             latitude = Float64(core.latitude[i]), longitude = Float64(core.longitude[i]))
            for i in 1:nrow(core)]
end

"""
    gemb_native_cell(cell, applied, mp; correction, climate_model, time_range, kwargs...) -> GlacierTileRun

Run GEMB for every populated elevation band of one cell from [`native_cells`](@ref), under the tile's
`applied` downscaling and the calibrated `correction` (`(; pscale, ΔT, group, donor, distance_km)`, as
[`forcing_correction`](@ref) returns).

The forcing is [`elevation_interval_forcing`](@ref) over the cell alone, with donors from the tile's
buffered neighbourhood, so a cell outside the reanalysis land mask borrows the forcing it would borrow
in the tile run. The run is [`gemb_glacier_tile`](@ref) at `delta_temperatures = [correction.ΔT]` and
`precipitation_scalings = [correction.pscale]`; the offset and the scaling are therefore applied after
the downscaling, as they are in the tile runs the correction was fitted against.

Keywords other than `correction` are those of [`elevation_interval_forcing`](@ref) (`climate_model`,
`time_range`, `token`, `cache_path`, `forcing_loader`, `donor_max_distance_km`) and
[`gemb_glacier_tile`](@ref) (`spinup_window`, `simulation_years_maximum`, `convergence_drift_fac`,
`drift_window`, `threaded`, `restart`, `force_restart`, `spinup_seed`). The cell's coordinates, its
`glm` and the correction are added to the run's provenance.

[`native_cell_spinup_seed`](@ref) builds a `spinup_seed` from this cell's own file at a different
correction, to shorten the spinup of a perturbation probe around an already-run point.

Throws `ForcingUnavailable` when the cell has ice in no band of `applied`, or when neither it nor any
donor within `donor_max_distance_km` has usable forcing.
"""
function gemb_native_cell(cell, applied::AppliedDownscaling, mp::ModelParameters;
                          correction,
                          climate_model::Symbol,
                          time_range,
                          token = nothing,
                          cache_path = nothing,
                          forcing_loader = climate_forcing,
                          donor_max_distance_km::Real = _DONOR_MAX_DISTANCE_KM_DEFAULT,
                          spinup_window = nothing,
                          simulation_years_maximum::Real = 1000,
                          convergence_drift_fac = SPINUP_DRIFT_FAC,
                          drift_window::Int = SPINUP_DRIFT_WINDOW,
                          threaded::Bool = true,
                          restart = nothing,
                          force_restart::Bool = false,
                          spinup_seed = nothing)
    nrow(cell.core) == 1 || throw(ArgumentError(
        "a native cell carries exactly one grid cell; got $(nrow(cell.core)) (use `native_cells`)"))
    any(>(0), _interval_areas(first(eachrow(cell.core)), applied.bands)) || throw(ForcingUnavailable(
        "cell $(cell.name) holds no ice in any elevation band of its tile's downscaling"))

    bands = try
        collect(elevation_interval_forcing(cell.core, applied;
                                           climate_model, time_range, token, cache_path,
                                           elevation_interval_batch = 0,
                                           forcing_loader,
                                           donor_cells = cell.buffered,
                                           donor_max_distance_km))
    catch e
        # A cell outside the reanalysis land mask with no usable donor within reach: the ice the tile
        # run counts as unrecovered. A property of the cell, so it is reported as unavailable forcing
        # rather than as a failure.
        e isa ArgumentError && occursin("yielded usable forcing", e.msg) || rethrow()
        throw(ForcingUnavailable("cell $(cell.name): " * e.msg))
    end

    run = gemb_glacier_tile(cell, applied, bands, mp;
                            delta_temperatures = [correction.ΔT],
                            precipitation_scalings = [correction.pscale],
                            spinup_window, simulation_years_maximum, convergence_drift_fac,
                            drift_window, threaded, restart, force_restart, spinup_seed,
                            donor_max_distance_km)

    # Where the cell is and which correction it ran under, beside the tile identifiers the run already
    # carries. The correction's values are the run's perturbation axes; what is recorded here is where
    # they came from, which a continuation does not have to match.
    row = first(eachrow(cell.core))
    run.provenance["native_resolution"] = true
    run.provenance["cell_latitude"] = cell.latitude
    run.provenance["cell_longitude"] = Float64(wrap_lon(cell.longitude))
    run.provenance["cell_glm"] = Float64(_row_glm(row))
    run.provenance["cell_glacier_area_km2"] = Float64(sum(glacier_area_column(cell.core)))
    run.provenance["forcing_correction_group"] = correction.group
    run.provenance["forcing_correction_donor"] = String(correction.donor)
    run.provenance["forcing_correction_distance_km"] = Float64(correction.distance_km)
    return run
end

"""
    native_cell_spinup_seed(path) -> Function or nothing

A `spinup_seed` for [`gemb_glacier_tile`](@ref)/[`gemb_native_cell`](@ref), built from a native cell
file already on disk at `path`: `i_band -> profile`, backed by that file's own saved columns. `nothing`
if the file does not exist or carries no restart group, so a caller can fall back to a cold spinup
unconditionally.

Meant to warm-start a run at a *different* `(ΔT, pscale)` from the one `path` was run at -- a native
cell file always has a single correction, so its restart has one profile per band. The column spun up
there is not the equilibrium of the new forcing, but `gemb_spinup` runs its own convergence loop on top
of it regardless, so a stale or mismatched seed only costs cycles, never correctness.
"""
function native_cell_spinup_seed(path::AbstractString)
    restart = read_glacier_tile_restart(path)
    restart === nothing && return nothing
    profiles = Dict(k[1] => v for (k, v) in restart.profiles if k[2] == 1 && k[3] == 1)
    isempty(profiles) && return nothing
    return i_band -> get(profiles, i_band, nothing)
end
