# Default temperature lapse rate (K/km) for this package's elevation adjustments. Defined once so the
# per-bin adjustment, the sweep driver, and the interval aggregation's gap fill cannot drift apart.
const _DEFAULT_LAPSE_RATE = 6.5

# Is this exception a broken *caller* rather than a bad cell?
#
# A typo, an undefined name, a stale session whose loaded package predates a function being called, a
# signature that has moved: these fail identically for every cell of every region. Swallowed per cell
# they turn one bug into thousands of identical warnings that read like the *data* is bad — and worse,
# a sweep then records the region as unrunnable. So every per-cell `catch` in this package rethrows
# them, and the predicate lives here rather than being spelled out at each site: four copies of a type
# tuple is four places to forget one.
#
# `FieldError` (a mistyped or removed struct field) only exists from Julia 1.12; before that the same
# mistake surfaces as an `ErrorException`, which is too broad to rethrow on — this package supports
# 1.11, so the type is included only where it is defined rather than assumed. Without the guard the
# reference itself is an `UndefVarError` on 1.11, i.e. the handler becomes the bug it exists to catch.
const _CALLER_ERROR_TYPES = isdefined(Base, :FieldError) ?
    (UndefVarError, MethodError, getfield(Base, :FieldError)) : (UndefVarError, MethodError)

is_caller_error(e) = any(T -> e isa T, _CALLER_ERROR_TYPES)

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
    forcing_at_elevation(forcing_data, delta_elevation; lapse_rate=$(_DEFAULT_LAPSE_RATE),
                         precip_scaling_method=nothing, decoupling_factor=nothing)

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
function forcing_at_elevation(forcing_data, delta_elevation; lapse_rate=_DEFAULT_LAPSE_RATE,
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

# `glm` — the ERA5-Land glacier-mask fraction of a grid cell — as a finite fraction in [0, 1].
#
# The column is required: a table missing it cannot support the decoupling factor at all, whether
# that factor is looked up per cell or regressed across a region, so its absence is schema drift
# and throws. `missing`/`NaN` are the different case of a real data gap, and fall back to 0.0 —
# the uncorrected reanalysis assumption of a cell with no glacier.
function _row_glm(row)
    hasproperty(row, :glm) || throw(ArgumentError(
        "glacier elevation-class row has no `:glm` column, which the glacier decoupling factor " *
        "is weighted by (per cell) and regressed against (per region). A table built before the " *
        "current invariant column naming carries `:glm_frac`; run " *
        "`scripts/migrate_invariant_colnames.jl` to rename it in place."))
    v = row.glm
    v === missing && return 0.0
    g = Float64(v)
    return isfinite(g) ? clamp(g, 0.0, 1.0) : 0.0
end

# The share of a published decoupling factor `k` a cell still needs, given the ERA5-Land glacier-mask
# fraction `glm` its land-surface scheme has already accounted for:
#
#     k_eff = 1 - (1 - k) * (1 - glm)
#
# The full correction where the cell carries no glacier (`glm = 0`) and the identity where it is
# entirely glacier (`glm = 1`). Defined once because this is the package's single convention for
# that weighting, applied both per cell ([`cell_decoupling_factor`](@ref), one scalar `k` from the
# published table) and per elevation interval (`derive_decoupling_factor`, a fitted `k` per
# timestep) — and it is the same weighting `derive_lapse_rate` folds into its regression sums to
# undo. Two independent spellings would let those paths disagree about the same cell's forcing
# while both stayed finite and in-domain, so the drift would be invisible.
_effective_decoupling_factor(k, glm) = 1 - (1 - k) * (1 - glm)

# --- output file naming -------------------------------------------------------------------------
#
# Per-cell and per-tile output files are named by position, so a file is traceable to what it
# describes and a listing sorts geographically (by hemisphere, then by degree).
#
# The form follows ISO 6709's letter-prefix convention: hemisphere letter, then zero-padded
# magnitude — two integer digits of latitude, three of longitude, which `wrap_lon` bounds at 180.
# The letters carry the sign, so every name is the same length without spending a character on '+',
# and they make `lat`/`lon` prefixes redundant, since N/S can only be a latitude and E/W only a
# longitude. Keeping '-' out of filenames also avoids the corner where shell and `find` argument
# parsing treat a name as an option. The cost is that the degrees no longer `parse` straight out of
# the substring, so each name form is paired with its inverse below.
#
# The '_' between the two coordinates is redundant as a delimiter — the E/W letter already marks
# where latitude ends — but it makes the boundary visible without hunting for a letter among digits,
# and it means splitting a name into its two halves does not depend on the zero-padding being
# exactly 2-and-3 digits.

# ONE decimal for a cell, because that is all the forcing grid has: ERA5-Land is 0.1°, so every cell
# centre is exact at 1 dp and a second decimal is always '0' — a character of noise in every name.
# That makes the decimal point itself the only punctuation left, and it is worth keeping: dropping it
# would leave the scale implied ('N523' read as tenths), and the standard alternative for a
# point-free name is ISO 6709 degrees-minutes ('N5218'), which is less readable than decimal degrees
# for anyone working in this field.
#
# This ties the name to a 0.1° grid: a finer forcing grid (ERA5 at 0.25° is fine, but a 0.01° product
# would not be) needs another decimal here, or two of its cells would collide on one name.
const _CELL_NAME_DECIMALS = 1

function _degrees_tag(x, intdigits, positive, negative)
    # Round first, then split, so a cell just under a degree (52.97 at 1 dp) becomes 53.0 rather
    # than 52.10.
    scale = 10^_CELL_NAME_DECIMALS
    r = round(abs(x), digits = _CELL_NAME_DECIMALS)
    whole, frac = floor(Int, r), round(Int, scale * (r - floor(r)))
    # `x < 0` and not `signbit`: -0.0 is the equator/prime meridian, which is not a hemisphere.
    return (x < 0 ? negative : positive) * lpad(whole, intdigits, '0') * "." *
           lpad(frac, _CELL_NAME_DECIMALS, '0')
end

# A whole-degree tag, for a tile. No decimal point at all — which is also what keeps the tile and
# cell name forms unambiguous; see `tile_output_name`.
_degrees_tag_int(x::Integer, intdigits, positive, negative) =
    (x < 0 ? negative : positive) * lpad(abs(x), intdigits, '0')

"""
    cell_output_name(latitude, longitude; extension = ".nc") -> String

The file name for a glacier grid cell's output, named by cell centre: `"N52.3_W174.1.nc"`.

One decimal place, which is the ERA5-Land grid's own precision. [`parse_cell_lonlat`](@ref) is the
inverse. Longitude is wrapped with [`wrap_lon`](@ref), so a native 0–359.9°E value names the same
file as its `(-180, 180]` equivalent.
"""
cell_output_name(latitude, longitude; extension::AbstractString = ".nc") =
    _degrees_tag(Float64(latitude), 2, 'N', 'S') * "_" *
    _degrees_tag(wrap_lon(Float64(longitude)), 3, 'E', 'W') * extension

# Inverse of `cell_output_name`, and of `tile_output_name`. Anchored and fully specified so each
# rejects the other's names: a cell name always carries a decimal point and a tile name never does.
const _CELL_NAME_REGEX =
    Regex("^([NS])(\\d{2}\\.\\d{$_CELL_NAME_DECIMALS})_([EW])(\\d{3}\\.\\d{$_CELL_NAME_DECIMALS})\\.nc\$")
const _TILE_NAME_REGEX = r"^([NS])(\d{2})_([EW])(\d{3})\.nc$"

"""
    parse_cell_lonlat(path) -> (; latitude, longitude) or nothing

The cell centre a file written by [`cell_output_name`](@ref) is named for, or `nothing` if the name
is not one of ours (including a tile name, which carries no decimal point).

Rounded to the name's precision, so it identifies the cell but is not the exact centre — read the
file's attributes for that. Provided so downstream tooling can recover the cell from a filename
without re-implementing the format and drifting from it.
"""
function parse_cell_lonlat(path::AbstractString)
    m = match(_CELL_NAME_REGEX, basename(path))
    m === nothing && return nothing
    return (latitude = parse(Float64, m[2]) * (m[1] == "S" ? -1 : 1),
            longitude = parse(Float64, m[4]) * (m[3] == "W" ? -1 : 1))
end

"""
    tile_output_name(index; extension = ".nc") -> String

The file name for a downscaling-parameter tile, named by its southwest corner: `"N60_W142.nc"` for
`index = (-142, 60)`. [`parse_tile_index`](@ref) is the inverse.

Whole degrees, with no decimal point — which is what makes a tile name and a cell name
([`cell_output_name`](@ref)) impossible to confuse even in one directory, since a cell name always
carries its 0.1° decimal. It is also why `tile_size` must be a whole number of degrees: a fractional
grid would let two tiles collide on one name.

The name carries the corner but **not** the tile size, so one directory should hold one gridding.
The size and buffer are recorded in each file's attributes, and the tile writer's status check
compares them, so a directory reused across griddings re-derives rather than silently reading a
tile that means something else.
"""
tile_output_name(index; extension::AbstractString = ".nc") =
    _degrees_tag_int(Int(index[2]), 2, 'N', 'S') * "_" *
    _degrees_tag_int(Int(index[1]), 3, 'E', 'W') * extension

"""
    parse_tile_index(path) -> (Int, Int) or nothing

The tile index a file written by [`tile_output_name`](@ref) is named for, or `nothing` if the name
is not one of ours (including a per-cell name, which carries a decimal point).
"""
function parse_tile_index(path::AbstractString)
    m = match(_TILE_NAME_REGEX, basename(path))
    m === nothing && return nothing
    return (parse(Int, m[4]) * (m[3] == "W" ? -1 : 1),
            parse(Int, m[2]) * (m[1] == "S" ? -1 : 1))
end
