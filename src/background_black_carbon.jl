# Regional background light-absorbing carbon in snow, for GEMB's Gardner & Sharp (2010) albedo.
#
# GEMB takes one snow impurity loading per run, `ModelParameters.black_carbon_snow` [ppmw], and applies
# it to the surface snow cell at every step. This file supplies that value per RGI 7.0 first-order
# region from the vendored table `data/background_black_carbon.csv`: measured BC in fresh or dry,
# pre-melt snow, and for the dust-dominated regions a dust loading converted to a BC-equivalent. It is
# a background, constant in time; melt-season enrichment is not represented.
#
# A tile's region is the RGI 7.0 o1 region holding most of its glacier area, located against the
# vendored region outlines `data/rgi7_o1regions_simplified.geojson`.

"""
Path of the vendored background black-carbon table. One row per RGI 7.0 o1 region `"01"`–`"20"`,
plus reference rows for the Greenland (`"GrIS"`) and Antarctic (`"AIS"`) ice sheets that the lookup
does not return. Columns:

- `rgi_o1region`, `region_name`;
- `bc_ng_g`, `bc_low_ng_g`, `bc_high_ng_g`: BC in snow [ng g⁻¹], central value and range;
- `bc_method`, `bc_season`, `bc_source`: how, when and where the BC value was measured;
- `dust_ug_g`: dust in snow [µg g⁻¹] where a dust mass was measured, blank elsewhere;
- `nonbc_bc_equivalent_ng_g`: the BC-equivalent [ng g⁻¹] of non-BC absorption measured optically
  (ISSW `C_equiv − C_est`, or optical eBC minus SP2 rBC), for regions with no dust mass; blank
  elsewhere, and never set together with `dust_ug_g`;
- `dust_source`: the source of whichever of the two is set;
- `proxy_region`: the region whose measurement stands in for this one, blank when the region has its
  own;
- `notes`.

Fields containing commas are double-quoted.
"""
const BACKGROUND_BLACK_CARBON_TABLE =
    normpath(joinpath(@__DIR__, "..", "data", "background_black_carbon.csv"))

"""
Path of the vendored RGI 7.0 first-order region outlines (GTN-G 2023,
doi:10.5904/gtng-glacreg-2023-07), simplified by Douglas–Peucker at 0.01° with coordinates rounded to
1e-4°. Against the full outlines it assigns 7 of the 47,121 cells of the glacier elevation-class table
to a different region, each within 0.1° of a region border.
"""
const RGI7_O1_REGIONS_FILE =
    normpath(joinpath(@__DIR__, "..", "data", "rgi7_o1regions_simplified.geojson"))

"""
The two ways the table's values become `black_carbon_snow`:

- `:bc`: the measured BC alone;
- `:bc_dust`: BC plus the BC-equivalent of the region's dust: [`dust_bc_equivalent`](@ref) of
  `dust_ug_g`, or `nonbc_bc_equivalent_ng_g` as measured; in a region with neither, the same as
  `:bc`.
"""
const BACKGROUND_BLACK_CARBON_BASES = (:bc, :bc_dust)

# The RGI 7.0 o1 regions a lookup can return.
const _RGI7_O1_REGIONS = [lpad(i, 2, '0') for i in 1:20]

# The Gardner & Sharp (2010) fit spans 0–2 ppmw and GEMB asserts that range.
const _BLACK_CARBON_SNOW_MAXIMUM = 2.0

"""
    dust_bc_equivalent(dust_ug_g) -> Float64

The BC mass fraction [ng g⁻¹] with the same all-wave, clear-sky albedo reduction as `dust_ug_g`
[µg g⁻¹] of mineral dust in snow, from Dang, Brandt & Warren (2015, J. Geophys. Res. Atmos. 120,
doi:10.1002/2014JD022646), Eqs. 9–10 and Table 5:

    C_BC,eq = C_dust / f,   f = 159 + 23.94 x + 4.30 x²,   x = log10(C_dust / ppm)

`f` is fitted for 0.1 ≤ C_dust ≤ 100 ppm and is held at its value at the nearer end outside that range.
Dang's BC has the same 550 nm mass absorption cross-section (6.73 m² g⁻¹) as Gardner & Sharp's
light-absorbing carbon, so the result is in the units GEMB's `black_carbon_snow` expects (after
dividing by 1000).
"""
function dust_bc_equivalent(dust_ug_g::Real)
    dust_ug_g >= 0 || throw(DomainError(dust_ug_g, "dust concentration must be non-negative"))
    dust_ug_g == 0 && return 0.0
    x = log10(clamp(Float64(dust_ug_g), 0.1, 100.0))
    f = 159.0 + 23.94 * x + 4.30 * x^2
    # µg g⁻¹ → ng g⁻¹
    return 1000.0 * dust_ug_g / f
end

# BC-equivalent [ng g⁻¹] of a table row's non-BC absorber: from its dust mass, else as measured.
_dust_equivalent(row) = row.dust_ug_g isa Float64 ? dust_bc_equivalent(row.dust_ug_g) :
                        row.nonbc_bc_equivalent_ng_g isa Float64 ? row.nonbc_bc_equivalent_ng_g : 0.0

const _BACKGROUND_TABLE_CACHE = Ref{Any}(nothing)
const _RGI7_REGIONS_CACHE = Ref{Any}(nothing)

_parse_optional(s) = (s = strip(String(s)); isempty(s) ? missing : parse(Float64, s))

"""
    background_black_carbon_table(path = BACKGROUND_BLACK_CARBON_TABLE) -> DataFrame

The background black-carbon table with its numeric columns parsed (blank → `missing`) and checked:
every region `"01"`–`"20"` appears exactly once, every BC value is finite and non-negative with
`bc_low_ng_g ≤ bc_ng_g ≤ bc_high_ng_g`, every source is non-empty, and a `proxy_region` names a region
of the table that has its own measurement. The default path is read once per session.
"""
function background_black_carbon_table(path::AbstractString = BACKGROUND_BLACK_CARBON_TABLE)
    cached = path == BACKGROUND_BLACK_CARBON_TABLE ? _BACKGROUND_TABLE_CACHE[] : nothing
    cached === nothing || return cached
    isfile(path) || throw(ArgumentError("no background black-carbon table at $path"))
    raw = GeoDataFrames.read(path)
    required = ["rgi_o1region", "region_name", "bc_ng_g", "bc_low_ng_g", "bc_high_ng_g",
                "bc_method", "bc_season", "bc_source", "dust_ug_g", "nonbc_bc_equivalent_ng_g",
                "dust_source", "proxy_region", "notes"]
    missing_columns = setdiff(required, names(raw))
    isempty(missing_columns) || throw(ArgumentError(
        "background black-carbon table $path lacks column(s) $(join(missing_columns, ", "))"))
    table = DataFrame(rgi_o1region = String.(strip.(raw.rgi_o1region)),
                      region_name = String.(raw.region_name),
                      bc_ng_g = _parse_optional.(raw.bc_ng_g),
                      bc_low_ng_g = _parse_optional.(raw.bc_low_ng_g),
                      bc_high_ng_g = _parse_optional.(raw.bc_high_ng_g),
                      bc_method = String.(raw.bc_method),
                      bc_season = String.(raw.bc_season),
                      bc_source = String.(raw.bc_source),
                      dust_ug_g = _parse_optional.(raw.dust_ug_g),
                      nonbc_bc_equivalent_ng_g = _parse_optional.(raw.nonbc_bc_equivalent_ng_g),
                      dust_source = String.(raw.dust_source),
                      proxy_region = String.(strip.(raw.proxy_region)),
                      notes = String.(raw.notes))
    _validate_background_table(table, path)
    path == BACKGROUND_BLACK_CARBON_TABLE && (_BACKGROUND_TABLE_CACHE[] = table)
    return table
end

function _validate_background_table(table, path)
    for region in _RGI7_O1_REGIONS
        n = count(==(region), table.rgi_o1region)
        n == 1 || throw(ArgumentError(
            "background black-carbon table $path has $n rows for RGI region $region; it needs one"))
    end
    for row in eachrow(table)
        r = row.rgi_o1region
        bc, lo, hi = row.bc_ng_g, row.bc_low_ng_g, row.bc_high_ng_g
        (bc isa Float64 && lo isa Float64 && hi isa Float64) || throw(ArgumentError(
            "region $r of $path lacks a BC value or range"))
        all(isfinite, (bc, lo, hi)) && 0 <= lo <= bc <= hi || throw(ArgumentError(
            "region $r of $path needs 0 ≤ bc_low ≤ bc ≤ bc_high, got $lo, $bc, $hi"))
        isempty(strip(row.bc_source)) && throw(ArgumentError("region $r of $path has no BC source"))
        dust, nonbc = row.dust_ug_g, row.nonbc_bc_equivalent_ng_g
        dust isa Float64 && nonbc isa Float64 && throw(ArgumentError(
            "region $r of $path sets both dust_ug_g and nonbc_bc_equivalent_ng_g; it needs one"))
        for (name, x) in (("dust_ug_g", dust), ("nonbc_bc_equivalent_ng_g", nonbc))
            x isa Float64 || continue
            isfinite(x) && x >= 0 || throw(ArgumentError("region $r of $path has $name = $x"))
            isempty(strip(row.dust_source)) && throw(ArgumentError(
                "region $r of $path has $name but no dust_source"))
        end
        p = row.proxy_region
        if !isempty(p)
            i = findfirst(==(p), table.rgi_o1region)
            i === nothing && throw(ArgumentError(
                "region $r of $path is a proxy of $p, which the table does not have"))
            isempty(table.proxy_region[i]) || throw(ArgumentError(
                "region $r of $path is a proxy of $p, which is itself a proxy"))
        end
        value = (bc + _dust_equivalent(row)) / 1000
        value <= _BLACK_CARBON_SNOW_MAXIMUM || throw(ArgumentError(
            "region $r of $path gives $value ppmw, above the $(_BLACK_CARBON_SNOW_MAXIMUM) ppmw " *
            "the Gardner & Sharp (2010) fit spans"))
    end
    return nothing
end

"""
    background_black_carbon(region; basis = :bc) -> NamedTuple

The background snow impurity loading of RGI 7.0 o1 region `region` (`"01"`–`"20"`, or an integer),
as `(; rgi_o1region, region_name, basis, black_carbon_snow, bc_ng_g, dust_ug_g,
dust_bc_equivalent_ng_g, proxy_region, source)`.

`black_carbon_snow` is in ppmw, the unit of `GEMB.ModelParameters.black_carbon_snow`: `bc_ng_g / 1000`
under `basis = :bc`, plus `dust_bc_equivalent_ng_g / 1000` under `:bc_dust` where the region carries
a dust value. `proxy_region` is `"none"` for a region with its own measurement. `source` names the
measurement(s) the value rests on.
"""
function background_black_carbon(region::AbstractString; basis::Symbol = :bc)
    basis in BACKGROUND_BLACK_CARBON_BASES || throw(ArgumentError(
        "basis must be one of $(BACKGROUND_BLACK_CARBON_BASES), got :$basis"))
    region in _RGI7_O1_REGIONS || throw(ArgumentError(
        "region must be an RGI 7.0 o1 region \"01\"–\"20\", got \"$region\""))
    table = background_black_carbon_table()
    row = table[findfirst(==(region), table.rgi_o1region), :]
    dust = row.dust_ug_g isa Float64 ? row.dust_ug_g : 0.0
    dust_eq = basis === :bc_dust ? _dust_equivalent(row) : 0.0
    source = row.bc_source
    basis === :bc_dust && dust_eq > 0 && (source *= "; dust: " * row.dust_source)
    return (; rgi_o1region = row.rgi_o1region, region_name = row.region_name, basis,
            black_carbon_snow = (row.bc_ng_g + dust_eq) / 1000,
            bc_ng_g = row.bc_ng_g, dust_ug_g = basis === :bc_dust ? dust : 0.0,
            dust_bc_equivalent_ng_g = dust_eq,
            proxy_region = isempty(row.proxy_region) ? "none" : row.proxy_region,
            source)
end
background_black_carbon(region::Integer; kwargs...) =
    background_black_carbon(lpad(region, 2, '0'); kwargs...)

"""
    background_black_carbon(cells::AbstractDataFrame; basis = :bc) -> NamedTuple

[`background_black_carbon`](@ref) of the region [`tile_rgi_o1_region`](@ref) assigns `cells`.
"""
background_black_carbon(cells::AbstractDataFrame; basis::Symbol = :bc) =
    background_black_carbon(tile_rgi_o1_region(cells); basis)

function _rgi7_regions()
    cached = _RGI7_REGIONS_CACHE[]
    cached === nothing || return cached
    isfile(RGI7_O1_REGIONS_FILE) ||
        throw(ArgumentError("no RGI 7.0 region outlines at $RGI7_O1_REGIONS_FILE"))
    df = GeoDataFrames.read(RGI7_O1_REGIONS_FILE)
    regions = (; geometry = collect(df.geometry), code = String.(df.o1region),
               extent = [GeoInterface.extent(g) for g in df.geometry])
    _RGI7_REGIONS_CACHE[] = regions
    return regions
end

"""
    rgi_o1_region(lon, lat) -> String

The RGI 7.0 first-order region (`"01"`–`"20"`) containing the point (`lon`, `lat`) [degrees]; `lon`
may be in 0–360 or (−180, 180]. Throws when the point lies in no region.
"""
function rgi_o1_region(lon::Real, lat::Real)
    x = Float64(wrap_lon(lon))
    y = Float64(lat)
    regions = _rgi7_regions()
    point = GeoInterface.Point(x, y)
    for k in eachindex(regions.geometry)
        e = regions.extent[k]
        (e.X[1] <= x <= e.X[2] && e.Y[1] <= y <= e.Y[2]) || continue
        GO.intersects(regions.geometry[k], point) && return regions.code[k]
    end
    throw(ArgumentError("($lon, $lat) lies in no RGI 7.0 first-order region"))
end

"""
    tile_rgi_o1_region(cells::AbstractDataFrame) -> String

The RGI 7.0 o1 region holding the most glacier area ([`glacier_area_total`](@ref)) among `cells`,
rows of the glacier elevation-class table located by their Point `geometry`. A tie goes to the lower
region code. Throws when the cells hold no glacier area.
"""
function tile_rgi_o1_region(cells::AbstractDataFrame)
    nrow(cells) > 0 || throw(ArgumentError("no cells, so no region"))
    area = Dict{String,Float64}()
    for row in eachrow(cells)
        a = Float64(glacier_area_total(row))
        a > 0 || continue
        r = rgi_o1_region(GeoInterface.x(row.geometry), GeoInterface.y(row.geometry))
        area[r] = get(area, r, 0.0) + a
    end
    isempty(area) && throw(ArgumentError("the cells hold no glacier area, so no region"))
    best = maximum(values(area))
    return minimum(r for (r, a) in area if a == best)
end

# `mp` with `black_carbon_snow` replaced, everything else identical; see `_with_albedo_ice`.
_with_black_carbon_snow(mp::ModelParameters, value::Real) =
    ConstructionBase.setproperties(mp, (; black_carbon_snow = Float64(value)))

# The run settings a background loading adds to `tile_run_parameters`, beside the
# `model_black_carbon_snow` every run records.
function _record_background_black_carbon!(params, bbc)
    params["background_black_carbon_region"] = bbc.rgi_o1region
    params["background_black_carbon_basis"] = string(bbc.basis)
    params["background_black_carbon_proxy_region"] = bbc.proxy_region
    params["background_black_carbon_bc_ng_g"] = Float64(bbc.bc_ng_g)
    params["background_black_carbon_dust_bc_equivalent_ng_g"] = Float64(bbc.dust_bc_equivalent_ng_g)
    params["background_black_carbon_source"] = bbc.source
    return params
end

# A forcing stack whose impurity metadata equal `mp`'s, so `initialize_forcing` builds the impurity
# layers `gemb` checks against `ModelParameters` with the values the run uses. A stack that already
# carries a different value is refused rather than overwritten.
function _with_impurity_metadata(stack, mp::ModelParameters)
    meta = Dict{String,Any}(DimensionalData.metadata(stack))
    for (key, value) in (("black_carbon_snow", mp.black_carbon_snow),
                         ("black_carbon_ice", mp.black_carbon_ice))
        haskey(meta, key) && meta[key] != value && throw(ArgumentError(
            "the forcing carries $key = $(meta[key]) but the run's ModelParameters has $value"))
        meta[key] = value
    end
    return DimensionalData.rebuild(stack; metadata = meta)
end
