"""
    era5_land_invariant(; parameter=nothing, to=nothing)

Read ERA5-Land invariant field(s) and rewrap longitudes from the native
0–359.9°E grid to the `(-180, 180]` convention used by the Zarr climate grid
(`-179.9 … 180.0`).

- `parameter`: a single field (e.g. `:glm`) returns a `Raster`; an iterable of
  fields (e.g. `(:glm, :z)`) returns a `RasterStack`.
- `to`: an optional grid whose X/Y direction the result is reordered to match
  (e.g. `glaciers`). When `nothing`, only the longitude rewrap/sort is applied.
"""
function era5_land_invariant(; parameter=nothing, to=nothing)
    read1(p) = _rewrap_era5_lon(
        read(GEMB_ClimateForcing.climate_model_invariant(model=:era5_land, parameter=p)), to)
    if parameter isa Union{AbstractVector,Tuple,Base.Generator}
        return RasterStack((; (Symbol(p) => read1(p) for p in parameter)...))
    else
        return read1(parameter)
    end
end

"""
    wrap_lon(lon)

Wrap a native ERA5 longitude (0–359.9°E) to the `(-180, 180]` convention used by the
Zarr climate grid (`geo_chunk_map`/`glaciers`) and the Copernicus DEM.

NOTE: `mod(lon+180,360)-180` would send the 180.0° cell to -180.0 — one cell off from the
Zarr grid — so keep 180.0 as 180.0 and only wrap longitudes strictly greater than 180.
"""
wrap_lon(lon) = lon > 180 ? lon - 360 : lon

"""
    forcing_at_elevation(forcing_data, delta_elevation; lapse_rate=6.5, precip_scaling_method=nothing,
                         decoupling_factor=nothing)

Lapse-rate-adjust `forcing_data` by `delta_elevation` metres and convert it to a GEMB
`ClimateForcing`, ready to hand to `initialize_profile` / `gemb`.

This crosses the forcing→firn boundary once: [`climate_adjust_for_elevation`](@ref)
(GEMB_ClimateForcing) operates on the `forcing_data` `DimStack`, then `initialize_forcing`
(GEMB) converts the adjusted stack to a `ClimateForcing`. The adjustment always starts from
the passed `forcing_data`, so calling this repeatedly with different `delta_elevation` values
(e.g. once per glacier hypsometry bin) does not compound the deltas.

- `delta_elevation`: `z_target − z_reanalysis` in metres (positive to cool/raise to a higher target).
- `lapse_rate`: temperature lapse rate passed through to `climate_adjust_for_elevation` (K/km).
- `precip_scaling_method`: precipitation scaling passed through to `climate_adjust_for_elevation`
  (`nothing` leaves precipitation unchanged).
- `decoupling_factor`: on-glacier air temperature decoupling factor `k` in `(0, 1]`, applied by
  `climate_adjust_for_glacier` *after* the elevation adjustment — the order that function
  requires, since `k` scales an ambient temperature already at the glacier's elevation.
  `nothing` (the default) leaves the forcing ambient.
"""
function forcing_at_elevation(forcing_data, delta_elevation; lapse_rate=6.5,
                              precip_scaling_method=nothing, decoupling_factor=nothing)
    adjusted = climate_adjust_for_elevation(forcing_data, delta_elevation;
                                            lapse_rate, precip_scaling_method)
    decoupling_factor === nothing ||
        (adjusted = climate_adjust_for_glacier(adjusted, decoupling_factor))
    return initialize_forcing(adjusted)
end

# Rewrap glm (the ERA5-Land invariant grid) to the SAME (-180, 180] convention as the Zarr grid.
function _rewrap_era5_lon(ras, to)
    lon  = lookup(ras, X)
    lonw = @. round(wrap_lon(lon), digits=10)
    ras  = set(ras, X => lonw)          # relabel X (now non-monotonic)
    ras  = ras[X(sortperm(lonw))]       # physically sort X ascending
    return to === nothing ? ras : reorder(ras, to)   # match `to`'s X/Y direction
end
