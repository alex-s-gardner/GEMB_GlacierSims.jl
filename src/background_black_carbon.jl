# Regional background light-absorbing particles in snow, for GEMB's Gardner & Sharp (2010) albedo.
#
# GEMB takes one snow impurity loading per run, `ModelParameters.black_carbon_snow` [ppmw], and applies
# it to the surface snow cell at every step. This file supplies that value per RGI 7.0 first-order
# region from the vendored table `data/background_black_carbon.csv`: measured BC in fresh, dry or
# pre-melt snow and firn, and dust (or optically measured non-BC absorption) where it was measured. It
# is a background, constant in time; melt-season enrichment is not represented.
#
# Gardner & Sharp's soot term darkens snow more per unit BC than radiative-transfer models do at
# background loadings, so the measured loading is converted to the Gardner & Sharp loading that gives
# the same broadband albedo reduction as a reference model: SNICAR-ADv3 (Flanner et al. 2021) through
# the vendored albedo table `data/snicar_adv3_albedo.csv`, or Dang et al. (2015) in closed form.
#
# A tile's loading is the glacier-area weighted mean of the loadings of the RGI 7.0 o1 regions its
# cells fall in, located against the vendored outlines `data/rgi7_o1regions_simplified.geojson`.

"""
Path of the vendored background black-carbon table. One row per RGI 7.0 o1 region `"01"`–`"20"`,
plus reference rows for the Greenland (`"GrIS"`) and Antarctic (`"AIS"`) ice sheets that the lookup
does not return. Columns:

- `rgi_o1region`, `region_name`;
- `bc_ng_g`, `bc_low_ng_g`, `bc_high_ng_g`: BC in snow or firn [ng g⁻¹], central value and range;
- `bc_method`, `bc_season`, `bc_source`: how, when and where the BC value was measured;
- `dust_ug_g`, `dust_low_ug_g`, `dust_high_ug_g`: dust [µg g⁻¹] where a dust mass was measured, and
  its range where one is given; blank elsewhere;
- `nonbc_bc_equivalent_ng_g`: the BC-equivalent [ng g⁻¹] of non-BC absorption measured optically
  (ISSW `C_equiv − C_est`, or optical eBC minus SP2 rBC), for regions with no dust mass; blank
  elsewhere, and never set together with `dust_ug_g`;
- `dust_source`: the source of whichever of the two is set;
- `proxy_region`: the region whose BC measurement stands in for this one, blank when the region has
  its own;
- `notes`.

Fields containing commas are double-quoted.
"""
const BACKGROUND_BLACK_CARBON_TABLE =
    normpath(joinpath(@__DIR__, "..", "data", "background_black_carbon.csv"))

"""
Path of the vendored SNICAR-ADv3 albedo table: broadband albedo of optically semi-infinite snow with
columns `illumination` (`direct`, `diffuse`), `radius_um`, `bc_ng_g`, `dust_ug_g`, `albedo`, over
radii 50–500 µm, BC 0–100 ng g⁻¹ and dust 0–50 µg g⁻¹. Written by `scripts/snicar_adv3_albedo_grid.m`
with SNICAR-ADv3 (Flanner et al. 2021, GMD 14, doi:10.5194/gmd-14-7673-2021; code
github.com/mflanner/SNICARv3 at a122b20) for spheres, uncoated BC, Saharan dust, the merged
Picard et al. (2016) ice refractive index, the sub-Arctic summer atmosphere, and a direct beam at a
49.5° solar zenith angle or diffuse illumination.
"""
const SNICAR_ADV3_ALBEDO_TABLE = normpath(joinpath(@__DIR__, "..", "data", "snicar_adv3_albedo.csv"))

"""
Path of the vendored RGI 7.0 first-order region outlines (GTN-G 2023,
doi:10.5904/gtng-glacreg-2023-07), simplified by Douglas–Peucker at 0.01° with coordinates rounded to
1e-4°. Against the full outlines it assigns 7 of the 47,121 cells of the glacier elevation-class table
to a different region, each within 0.1° of a region border.
"""
const RGI7_O1_REGIONS_FILE =
    normpath(joinpath(@__DIR__, "..", "data", "rgi7_o1regions_simplified.geojson"))

"""
The two ways the table's values become a loading:

- `:bc`: the measured BC alone;
- `:bc_dust`: BC plus the region's dust (`dust_ug_g`) or optically measured non-BC absorption
  (`nonbc_bc_equivalent_ng_g`); in a region with neither, the same as `:bc`.
"""
const BACKGROUND_BLACK_CARBON_BASES = (:bc, :bc_dust)

"""
The reference optics a measured loading is converted with:

- `:snicar_adv3`: the Gardner & Sharp (2010) loading with the same albedo reduction as SNICAR-ADv3
  ([`SNICAR_ADV3_ALBEDO_TABLE`](@ref)) gives the measured BC and dust;
- `:dang2015`: the same against Dang et al. (2015) ([`dang2015_albedo_reduction`](@ref));
- `:none`: no conversion; BC plus dust through Dang's mass equivalence
  ([`dust_bc_equivalent`](@ref)), as measured.
"""
const BACKGROUND_BLACK_CARBON_OPTICS = (:snicar_adv3, :dang2015, :none)

"""
Grain radius [µm] and geometry of the conversion to a Gardner & Sharp loading: semi-infinite snow
of 200 µm grains under a direct beam at a 49.5° solar zenith angle and clear sky (Dang et al. 2015's
clear-sky geometry). Against SNICAR-ADv3 the converted loading varies over 50–500 µm grains by at most
3 % for BC, 5 % for dust up to 20 µg g⁻¹ and 7.4 % at 50 µg g⁻¹, so one radius serves every grain size
in that range.
"""
const GS2010_EQUIVALENT_REFERENCE = (; radius_um = 200.0, solar_zenith_angle = 49.5,
                                     cloud_optical_thickness = 0.0, illumination = :direct)

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
"""
function dust_bc_equivalent(dust_ug_g::Real)
    dust_ug_g >= 0 || throw(DomainError(dust_ug_g, "dust concentration must be non-negative"))
    dust_ug_g == 0 && return 0.0
    x = log10(clamp(Float64(dust_ug_g), 0.1, 100.0))
    f = 159.0 + 23.94 * x + 4.30 * x^2
    # µg g⁻¹ → ng g⁻¹
    return 1000.0 * dust_ug_g / f
end

"""
    dang2015_albedo_reduction(bc_ng_g, dust_ug_g; radius_um, illumination = :direct) -> Float64

The all-wave broadband albedo reduction of semi-infinite snow of grain radius `radius_um` (50–2500 µm)
by `bc_ng_g` of BC and `dust_ug_g` of dust, from Dang, Brandt & Warren (2015): the dust is converted to
BC-equivalent with Eqs. 9–10 and Table 5, and the reduction is Eqs. 7–8 with Table 3,

    H = (C / 10⁻⁶)(r / 100 µm)^s,   log10 Δα = m₁ p² + m₂ p + m₃,   p = log10 H.

`illumination = :direct` uses the clear-sky coefficients (solar zenith angle 49.5°), `(m₁, m₂, m₃, s) =
(−0.042, 0.560, −1.110, 0.73)` and dust `(v₀, v₁, v₂) = (159, 23.94, 4.30)`; `:diffuse` the overcast
ones, `(−0.043, 0.547, −1.038, 0.73)` and `(155, 20.97, 2.41)`. Below the fitted range of `H`
(1.4 × 10⁻³ clear, 1.1 × 10⁻³ overcast) the reduction is taken proportional to `C`, as Dang's full
parameterization (Table 2) is linear in `C` at low loading.
"""
function dang2015_albedo_reduction(bc_ng_g::Real, dust_ug_g::Real; radius_um::Real,
                                   illumination::Symbol = :direct)
    illumination in (:direct, :diffuse) ||
        throw(ArgumentError("illumination must be :direct or :diffuse, got :$illumination"))
    50 <= radius_um <= 2500 ||
        throw(DomainError(radius_um, "Dang et al. (2015) is fitted for 50-2500 µm grains"))
    (bc_ng_g >= 0 && dust_ug_g >= 0) || throw(DomainError((bc_ng_g, dust_ug_g),
        "loadings must be non-negative"))
    clear = illumination === :direct
    m1, m2, m3, s, h_min = clear ? (-0.042, 0.560, -1.110, 0.73, 1.4e-3) :
                                   (-0.043, 0.547, -1.038, 0.73, 1.1e-3)
    v0, v1, v2 = clear ? (159.0, 23.94, 4.30) : (155.0, 20.97, 2.41)
    dust_eq = dust_ug_g > 0 ?
              (x = log10(clamp(Float64(dust_ug_g), 0.1, 100.0));
               1000.0 * dust_ug_g / (v0 + v1 * x + v2 * x^2)) : 0.0
    C = bc_ng_g + dust_eq                     # ng g⁻¹
    C > 0 || return 0.0
    # C / 10⁻⁶ with C as a mass fraction: ng g⁻¹ / 1000
    H = (C / 1000) * (radius_um / 100)^s
    reduction(h) = (p = log10(h); 10^(m1 * p^2 + m2 * p + m3))
    return H >= h_min ? reduction(H) : reduction(h_min) * H / h_min
end

const _BACKGROUND_TABLE_CACHE = Ref{Any}(nothing)
const _RGI7_REGIONS_CACHE = Ref{Any}(nothing)
const _SNICAR_TABLE_CACHE = Ref{Any}(nothing)

_parse_optional(s) = (s = strip(String(s)); isempty(s) ? missing : parse(Float64, s))

# Every field as a string, whatever else is loaded: the parsing below expects strings. With CSV.jl
# loaded, `GeoDataFrames.read(path)` (≥ 0.4.4) returns typed columns (`Float64`, `Int64`, `missing`);
# GDAL's CSV reader returns strings.
_read_csv_strings(path) = GeoDataFrames.read(GeoDataFrames.ArchGDALDriver(), path)

# The SNICAR-ADv3 albedo grid as `(; radii, bc, dust, albedo)`, with `albedo[illumination]` an
# array indexed (radius, bc, dust). Every node of the full outer product must be present.
function _snicar_table()
    cached = _SNICAR_TABLE_CACHE[]
    cached === nothing || return cached
    isfile(SNICAR_ADV3_ALBEDO_TABLE) ||
        throw(ArgumentError("no SNICAR-ADv3 albedo table at $SNICAR_ADV3_ALBEDO_TABLE"))
    raw = _read_csv_strings(SNICAR_ADV3_ALBEDO_TABLE)
    ill = Symbol.(strip.(raw.illumination))
    r = parse.(Float64, raw.radius_um)
    bc = parse.(Float64, raw.bc_ng_g)
    du = parse.(Float64, raw.dust_ug_g)
    a = parse.(Float64, raw.albedo)
    radii, bcs, dusts = sort(unique(r)), sort(unique(bc)), sort(unique(du))
    albedo = Dict(k => fill(NaN, length(radii), length(bcs), length(dusts)) for k in (:direct, :diffuse))
    for i in eachindex(ill, r, bc, du, a)
        haskey(albedo, ill[i]) ||
            throw(ArgumentError("unknown illumination $(ill[i]) in $SNICAR_ADV3_ALBEDO_TABLE"))
        albedo[ill[i]][searchsortedfirst(radii, r[i]), searchsortedfirst(bcs, bc[i]),
                       searchsortedfirst(dusts, du[i])] = a[i]
    end
    all(v -> all(isfinite, v), values(albedo)) || throw(ArgumentError(
        "$SNICAR_ADV3_ALBEDO_TABLE does not cover every radius, BC and dust node"))
    (first(bcs) == 0 && first(dusts) == 0) || throw(ArgumentError(
        "$SNICAR_ADV3_ALBEDO_TABLE must include the clean snow node (BC = 0, dust = 0)"))
    t = (; radii, bc = bcs, dust = dusts, albedo)
    _SNICAR_TABLE_CACHE[] = t
    return t
end

# Linear interpolation weights of `x` on the ascending `nodes`.
function _bracket(nodes, x, name)
    first(nodes) <= x <= last(nodes) || throw(DomainError(x,
        "$name outside the SNICAR-ADv3 table range $(first(nodes))-$(last(nodes))"))
    j = min(searchsortedlast(nodes, x), length(nodes) - 1)
    w = (x - nodes[j]) / (nodes[j + 1] - nodes[j])
    return j, w
end

"""
    snicar_adv3_albedo_reduction(bc_ng_g, dust_ug_g; radius_um, illumination = :direct) -> Float64

The broadband albedo reduction of semi-infinite snow by `bc_ng_g` of uncoated BC and `dust_ug_g` of
Saharan dust, relative to clean snow of the same grain radius, from the SNICAR-ADv3 table
[`SNICAR_ADV3_ALBEDO_TABLE`](@ref): bilinear in BC and dust between its nodes, at a tabulated radius.
Throws outside the table.
"""
function snicar_adv3_albedo_reduction(bc_ng_g::Real, dust_ug_g::Real; radius_um::Real,
                                      illumination::Symbol = :direct)
    t = _snicar_table()
    haskey(t.albedo, illumination) ||
        throw(ArgumentError("illumination must be :direct or :diffuse, got :$illumination"))
    k = findfirst(==(Float64(radius_um)), t.radii)
    k === nothing && throw(ArgumentError(
        "radius $radius_um µm is not tabulated; the SNICAR-ADv3 table has $(t.radii)"))
    A = t.albedo[illumination]
    i, wi = _bracket(t.bc, Float64(bc_ng_g), "BC $bc_ng_g ng/g")
    j, wj = _bracket(t.dust, Float64(dust_ug_g), "dust $dust_ug_g µg/g")
    a = (1 - wi) * (1 - wj) * A[k, i, j] + wi * (1 - wj) * A[k, i + 1, j] +
        (1 - wi) * wj * A[k, i, j + 1] + wi * wj * A[k, i + 1, j + 1]
    return A[k, 1, 1] - a
end

"""
    gs2010_equivalent(albedo_reduction; reference = GS2010_EQUIVALENT_REFERENCE) -> Float64

The loading `c` [ppmw] at which GEMB's Gardner & Sharp (2010) albedo of semi-infinite snow
(`GEMB._albedo_gardner`, the full albedo with its zenith and cloud terms) is lower than at `c = 0` by
`albedo_reduction`, for the grain radius and geometry of `reference`. Solved by bisection on 0–2 ppmw;
throws when the reduction is beyond what the fit gives at 2 ppmw.
"""
function gs2010_equivalent(albedo_reduction::Real; reference = GS2010_EQUIVALENT_REFERENCE)
    albedo_reduction >= 0 ||
        throw(DomainError(albedo_reduction, "an albedo reduction must be non-negative"))
    albedo_reduction == 0 && return 0.0
    r_mm = reference.radius_um / 1000
    albedo(c) = GEMB._albedo_gardner(r_mm, c, Float64(reference.solar_zenith_angle),
                                     Float64(reference.cloud_optical_thickness))
    clean = albedo(0.0)
    reduction(c) = clean - albedo(c)
    reduction(_BLACK_CARBON_SNOW_MAXIMUM) >= albedo_reduction || throw(DomainError(
        albedo_reduction, "beyond the $(reduction(_BLACK_CARBON_SNOW_MAXIMUM)) the Gardner & Sharp " *
                          "(2010) fit gives at its $(_BLACK_CARBON_SNOW_MAXIMUM) ppmw limit"))
    lo, hi = 0.0, _BLACK_CARBON_SNOW_MAXIMUM
    for _ in 1:100
        mid = (lo + hi) / 2
        reduction(mid) < albedo_reduction ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

"""
    background_black_carbon_table(path = BACKGROUND_BLACK_CARBON_TABLE) -> DataFrame

The background black-carbon table with its numeric columns parsed (blank → `missing`) and checked:
every region `"01"`–`"20"` appears exactly once, every BC value is finite and non-negative with
`bc_low_ng_g ≤ bc_ng_g ≤ bc_high_ng_g` (and likewise for a dust range), every source is non-empty,
and a `proxy_region` names a region of the table that has its own measurement. The default path is
read once per session.
"""
function background_black_carbon_table(path::AbstractString = BACKGROUND_BLACK_CARBON_TABLE)
    cached = path == BACKGROUND_BLACK_CARBON_TABLE ? _BACKGROUND_TABLE_CACHE[] : nothing
    cached === nothing || return cached
    isfile(path) || throw(ArgumentError("no background black-carbon table at $path"))
    raw = _read_csv_strings(path)
    required = ["rgi_o1region", "region_name", "bc_ng_g", "bc_low_ng_g", "bc_high_ng_g",
                "bc_method", "bc_season", "bc_source", "dust_ug_g", "dust_low_ug_g",
                "dust_high_ug_g", "nonbc_bc_equivalent_ng_g", "dust_source", "proxy_region",
                "notes"]
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
                      dust_low_ug_g = _parse_optional.(raw.dust_low_ug_g),
                      dust_high_ug_g = _parse_optional.(raw.dust_high_ug_g),
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
        dlo, dhi = row.dust_low_ug_g, row.dust_high_ug_g
        (dlo isa Float64) == (dhi isa Float64) || throw(ArgumentError(
            "region $r of $path sets one end of the dust range; it needs both or neither"))
        if dlo isa Float64
            dust isa Float64 && dlo <= dust <= dhi || throw(ArgumentError(
                "region $r of $path needs dust_low ≤ dust ≤ dust_high, got $dlo, $dust, $dhi"))
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

# BC-equivalent [ng g⁻¹] of a table row's non-BC absorber by Dang's mass equivalence: from its dust
# mass, else as measured optically.
_dust_equivalent(row) = row.dust_ug_g isa Float64 ? dust_bc_equivalent(row.dust_ug_g) :
                        row.nonbc_bc_equivalent_ng_g isa Float64 ? row.nonbc_bc_equivalent_ng_g : 0.0

"""
    background_black_carbon(region; basis = :bc, optics = :snicar_adv3) -> NamedTuple

The background snow impurity loading of RGI 7.0 o1 region `region` (`"01"`–`"20"`, or an integer),
as `(; rgi_o1region, region_name, basis, optics, black_carbon_snow, bc_ng_g, dust_ug_g,
nonbc_bc_equivalent_ng_g, dust_bc_equivalent_ng_g, reference_albedo_reduction, region_shares,
proxy_region, source)`.

`black_carbon_snow` [ppmw] is the value for `GEMB.ModelParameters.black_carbon_snow`:

- `optics = :snicar_adv3` or `:dang2015`: [`gs2010_equivalent`](@ref) of the albedo reduction the
  reference model gives the measured loading (`reference_albedo_reduction`), at
  [`GS2010_EQUIVALENT_REFERENCE`](@ref). Under `:bc_dust` the reference sees BC plus the optically
  measured non-BC BC-equivalent as BC, and the dust as dust.
- `optics = :none`: `(bc_ng_g + dust_bc_equivalent_ng_g) / 1000`, the measured loading with dust by
  Dang's mass equivalence ([`dust_bc_equivalent`](@ref)) and no conversion.

`bc_ng_g`, `dust_ug_g` and `nonbc_bc_equivalent_ng_g` are the measured values the loading rests on
(zero for dust and non-BC under `:bc`); `dust_bc_equivalent_ng_g` is their BC-equivalent by Dang's
mass equivalence. `region_shares` is `[region => 1.0]`. `proxy_region` is `"none"` for a region with
its own BC measurement.
"""
function background_black_carbon(region::AbstractString; basis::Symbol = :bc,
                                 optics::Symbol = :snicar_adv3)
    basis in BACKGROUND_BLACK_CARBON_BASES || throw(ArgumentError(
        "basis must be one of $(BACKGROUND_BLACK_CARBON_BASES), got :$basis"))
    optics in BACKGROUND_BLACK_CARBON_OPTICS || throw(ArgumentError(
        "optics must be one of $(BACKGROUND_BLACK_CARBON_OPTICS), got :$optics"))
    region in _RGI7_O1_REGIONS || throw(ArgumentError(
        "region must be an RGI 7.0 o1 region \"01\"–\"20\", got \"$region\""))
    table = background_black_carbon_table()
    row = table[findfirst(==(region), table.rgi_o1region), :]
    with_dust = basis === :bc_dust
    dust = with_dust && row.dust_ug_g isa Float64 ? row.dust_ug_g : 0.0
    nonbc = with_dust && row.nonbc_bc_equivalent_ng_g isa Float64 ? row.nonbc_bc_equivalent_ng_g : 0.0
    dust_eq = with_dust ? _dust_equivalent(row) : 0.0
    ref = GS2010_EQUIVALENT_REFERENCE
    reduction = optics === :snicar_adv3 ?
                snicar_adv3_albedo_reduction(row.bc_ng_g + nonbc, dust; radius_um = ref.radius_um,
                                             illumination = ref.illumination) :
                optics === :dang2015 ?
                dang2015_albedo_reduction(row.bc_ng_g + nonbc, dust; radius_um = ref.radius_um,
                                          illumination = ref.illumination) : NaN
    value = optics === :none ? (row.bc_ng_g + dust_eq) / 1000 : gs2010_equivalent(reduction)
    source = row.bc_source
    with_dust && dust_eq > 0 && (source *= "; dust: " * row.dust_source)
    return (; rgi_o1region = row.rgi_o1region, region_name = row.region_name, basis, optics,
            black_carbon_snow = value, bc_ng_g = row.bc_ng_g, dust_ug_g = dust,
            nonbc_bc_equivalent_ng_g = nonbc, dust_bc_equivalent_ng_g = dust_eq,
            reference_albedo_reduction = reduction, region_shares = [row.rgi_o1region => 1.0],
            proxy_region = isempty(row.proxy_region) ? "none" : row.proxy_region,
            source)
end
background_black_carbon(region::Integer; kwargs...) =
    background_black_carbon(lpad(region, 2, '0'); kwargs...)

"""
    background_black_carbon(cells::AbstractDataFrame; basis = :bc, optics = :snicar_adv3) -> NamedTuple

The loading of a tile whose cells are `cells`: the glacier-area weighted mean, over the RGI 7.0 o1
regions [`tile_rgi_o1_region_shares`](@ref) finds, of each region's [`background_black_carbon`](@ref).

`black_carbon_snow`, `bc_ng_g`, `dust_ug_g`, `nonbc_bc_equivalent_ng_g`, `dust_bc_equivalent_ng_g` and
`reference_albedo_reduction` are share-weighted means; `region_shares` lists every region and its share,
largest first; `rgi_o1region` and `region_name` are the largest share's; `proxy_region` lists the
proxies of the contributing regions (`"none"` if there are none); `source` names each region's
source with its share. A tile in one region gets exactly that region's loading.
"""
function background_black_carbon(cells::AbstractDataFrame; basis::Symbol = :bc,
                                 optics::Symbol = :snicar_adv3)
    shares = tile_rgi_o1_region_shares(cells)
    parts = [background_black_carbon(r; basis, optics) for (r, _) in shares]
    length(parts) == 1 && return only(parts)
    w = last.(shares)
    mean_of(f) = sum(w[i] * getproperty(parts[i], f) for i in eachindex(w, parts))
    proxies = unique(p.proxy_region for p in parts if p.proxy_region != "none")
    source = join(("region $(p.rgi_o1region) (share $(round(s; digits = 3))): $(p.source)"
                   for (p, s) in zip(parts, w)), " | ")
    return (; rgi_o1region = first(parts).rgi_o1region, region_name = first(parts).region_name,
            basis, optics, black_carbon_snow = mean_of(:black_carbon_snow),
            bc_ng_g = mean_of(:bc_ng_g), dust_ug_g = mean_of(:dust_ug_g),
            nonbc_bc_equivalent_ng_g = mean_of(:nonbc_bc_equivalent_ng_g),
            dust_bc_equivalent_ng_g = mean_of(:dust_bc_equivalent_ng_g),
            reference_albedo_reduction = mean_of(:reference_albedo_reduction),
            region_shares = shares,
            proxy_region = isempty(proxies) ? "none" : join(proxies, ","), source)
end

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
    tile_rgi_o1_region_shares(cells::AbstractDataFrame) -> Vector{Pair{String,Float64}}

The share of the glacier area ([`glacier_area_total`](@ref)) of `cells` in each RGI 7.0 o1 region,
as `region => share` pairs summing to 1, largest share first (ties by region code). `cells` are rows
of the glacier elevation-class table located by their Point `geometry`. Throws when the cells hold no
glacier area.
"""
function tile_rgi_o1_region_shares(cells::AbstractDataFrame)
    nrow(cells) > 0 || throw(ArgumentError("no cells, so no region"))
    area = Dict{String,Float64}()
    for row in eachrow(cells)
        a = Float64(glacier_area_total(row))
        a > 0 || continue
        r = rgi_o1_region(GeoInterface.x(row.geometry), GeoInterface.y(row.geometry))
        area[r] = get(area, r, 0.0) + a
    end
    isempty(area) && throw(ArgumentError("the cells hold no glacier area, so no region"))
    total = sum(values(area))
    shares = [r => a / total for (r, a) in area]
    return sort!(shares; by = p -> (-p.second, p.first))
end

"""
    tile_rgi_o1_region(cells::AbstractDataFrame) -> String

The RGI 7.0 o1 region holding the most glacier area among `cells` (the first of
[`tile_rgi_o1_region_shares`](@ref)).
"""
tile_rgi_o1_region(cells::AbstractDataFrame) = first(first(tile_rgi_o1_region_shares(cells)))

# `mp` with `black_carbon_snow` replaced, everything else identical; see `_with_albedo_ice`.
_with_black_carbon_snow(mp::ModelParameters, value::Real) =
    ConstructionBase.setproperties(mp, (; black_carbon_snow = Float64(value)))

# The run settings a background loading adds to `tile_run_parameters`, beside the
# `model_black_carbon_snow` every run records.
function _record_background_black_carbon!(params, bbc)
    params["background_black_carbon_region"] = bbc.rgi_o1region
    params["background_black_carbon_region_shares"] =
        join(("$r:$(round(s; digits = 4))" for (r, s) in bbc.region_shares), ",")
    params["background_black_carbon_basis"] = string(bbc.basis)
    params["background_black_carbon_optics"] = string(bbc.optics)
    params["background_black_carbon_proxy_region"] = bbc.proxy_region
    params["background_black_carbon_bc_ng_g"] = Float64(bbc.bc_ng_g)
    params["background_black_carbon_dust_ug_g"] = Float64(bbc.dust_ug_g)
    params["background_black_carbon_nonbc_bc_equivalent_ng_g"] = Float64(bbc.nonbc_bc_equivalent_ng_g)
    params["background_black_carbon_dust_bc_equivalent_ng_g"] = Float64(bbc.dust_bc_equivalent_ng_g)
    if bbc.optics !== :none
        ref = GS2010_EQUIVALENT_REFERENCE
        params["background_black_carbon_reference_albedo_reduction"] =
            Float64(bbc.reference_albedo_reduction)
        params["background_black_carbon_reference_radius_um"] = ref.radius_um
        params["background_black_carbon_reference_solar_zenith_angle"] = ref.solar_zenith_angle
    end
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
