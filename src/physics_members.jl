# A level with Flanner & Zender (2006) grain growth, the `B` (`:Arthern`) or `C` (`:Crocus` with
# destructive metamorphism) settling, and a constant fresh-snow density `ρ` [kg m-3].
function _constant_snow_level(settling::String, ρ::Float64)
    settle = settling == "B" ? (; densification_method = :Arthern, destructive_metamorphism = false) :
             settling == "C" ? (; densification_method = :Crocus, destructive_metamorphism = true) :
             throw(ArgumentError("settling must be \"B\" or \"C\", got \"$settling\""))
    words = settling == "B" ? "Arthern densification" :
            "Crocus densification with destructive metamorphism"
    return (; description = "Flanner & Zender (2006) grain growth; $(Int(ρ)) kg m-3 fresh snow; $words",
            model = (; grain_growth_method = :FlannerZender2006, new_snow_method = :Constant,
                     settle..., blowing_snow_method = :none, drift_fetch_length = 10.0,
                     new_snow_density = ρ, destructive_metamorphism_density = 175.0))
end

"""
Physics members: the unresolved physics choices a tile sweep can be run under, so that the calibrated
fluxes can be compared across them.

A member is one level of each of three factors, written `<grain>-<band decoupling>-<impurity>`, for
example `"C-kfit-table"`, optionally followed by a fourth, `-w<factor>`:

- grain growth, fresh snow and wind packing ([`PHYSICS_GRAIN_VARIANTS`](@ref)): `A`, `B`, `C`, `D1`,
  `S`, `B250`, `B300`, `C250`, `C300`, `C350`;
- the band decoupling ([`PHYSICS_BAND_DECOUPLING`](@ref)): `kfit`, `kshaw`;
- the fresh-snow impurity loading ([`PHYSICS_IMPURITY`](@ref)): `table`, `tbc`, `tdang`, `traw`,
  `bc0`, `cvis`;
- a wind correction: `w<factor>` multiplies every band's wind speed by `factor` (written in its
  shortest form, `w2`, `w1.5`) before GEMB, through `resolve_downscaling(...; wind_scaling)`; `ws`
  corrects it by the band's 2 km surface slope instead ([`slope_wind_correction`](@ref), through
  `resolve_downscaling(...; wind_correction = (; rule = :slope, band_slope))`). Without the token the
  wind is ERA5-Land's, a factor of 1.

Each level fixes every setting it is about, so a member names one model configuration whatever the
GEMB defaults are. [`tile_run_parameters`](@ref) records the member and its levels as `physics_*` run
parameters and refuses a member whose settings disagree with the model parameters, the downscaling
resolution or the impurity loading the run actually uses. The model settings
themselves stay in the `model_*`, `downscaling_*` and `*_black_carbon_*` parameters, which is what a
restart compares.
"""

"""
    PHYSICS_GRAIN_VARIANTS

Grain-growth, fresh-snow and wind-packing levels of a physics member, as the `ModelParameters` fields
each sets:

- `A`: `:hybrid` grain growth (Brun/Marbouty below 400 kg m⁻³), 350 kg m⁻³ fresh snow, `:Arthern`
  densification;
- `B`: Flanner & Zender (2006) grain growth (`:FlannerZender2006`), otherwise as `A`;
- `C`: Flanner & Zender (2006) grain growth with van Kampenhout et al. (2017) fresh-snow density,
  `:Crocus` densification and destructive metamorphism, the pairing GEMB documents for that density;
- `D1`: `C` with the Keenan et al. (2021) erosion–redeposition packing (`blowing_snow_method =
  :Keenan2021`) at a fetch length of 10 m (`drift_fetch_length`, their eq. 3);
- `S`: GEMB's default snow set, calibrated on SNOTEL seasonal snow: `C` with the fresh-snow
  temperature term refitted (`new_snow_method = :vanKampenhout2017SNOTEL`) and destructive
  metamorphism tapering above 100 kg m⁻³ (`destructive_metamorphism_density`);
- `B250`, `B300`: `B` with a constant fresh-snow density of 250 or 300 kg m⁻³
  (`new_snow_method = :Constant`, `new_snow_density`);
- `C250`, `C300`, `C350`: `C`'s settling (`:Crocus` densification with destructive metamorphism)
  under a constant fresh-snow density of 250, 300 or 350 kg m⁻³.

The letter of a level names its settling (`B`: `:Arthern` without destructive metamorphism; `C`:
`:Crocus` with it) and the number its constant deposition density. Levels without wind packing run
`blowing_snow_method = :none`, under which `drift_fetch_length` is not read and is held at GEMB's
default, and levels whose fresh snow is not `:Constant` hold `new_snow_density`, which only `:Constant`
reads, at GEMB's default; so every level fixes the same fields. Every level but `S` sets
`destructive_metamorphism_density` to 175 kg m⁻³, the value the A–D and constant-density levels were
run with, whether or not it reads it.
"""
const PHYSICS_GRAIN_VARIANTS = (
    A = (; description = "hybrid grain growth; 350 kg m-3 fresh snow; Arthern densification",
         model = (; grain_growth_method = :hybrid, new_snow_method = :Constant350,
                  densification_method = :Arthern, destructive_metamorphism = false,
                  blowing_snow_method = :none, drift_fetch_length = 10.0,
                  new_snow_density = 350.0, destructive_metamorphism_density = 175.0)),
    B = (; description = "Flanner & Zender (2006) grain growth; 350 kg m-3 fresh snow; " *
                         "Arthern densification",
         model = (; grain_growth_method = :FlannerZender2006, new_snow_method = :Constant350,
                  densification_method = :Arthern, destructive_metamorphism = false,
                  blowing_snow_method = :none, drift_fetch_length = 10.0,
                  new_snow_density = 350.0, destructive_metamorphism_density = 175.0)),
    C = (; description = "Flanner & Zender (2006) grain growth; van Kampenhout et al. (2017) " *
                         "fresh snow; Crocus densification with destructive metamorphism",
         model = (; grain_growth_method = :FlannerZender2006, new_snow_method = :vanKampenhout2017,
                  densification_method = :Crocus, destructive_metamorphism = true,
                  blowing_snow_method = :none, drift_fetch_length = 10.0,
                  new_snow_density = 350.0, destructive_metamorphism_density = 175.0)),
    D1 = (; description = "Flanner & Zender (2006) grain growth; van Kampenhout et al. (2017) " *
                          "fresh snow; Crocus densification with destructive metamorphism; " *
                          "Keenan et al. (2021) wind packing, 10 m fetch",
          model = (; grain_growth_method = :FlannerZender2006, new_snow_method = :vanKampenhout2017,
                   densification_method = :Crocus, destructive_metamorphism = true,
                   blowing_snow_method = :Keenan2021, drift_fetch_length = 10.0,
                   new_snow_density = 350.0, destructive_metamorphism_density = 175.0)),
    S = (; description = "Flanner & Zender (2006) grain growth; SNOTEL-calibrated van Kampenhout " *
                         "et al. (2017) fresh snow; Crocus densification with destructive " *
                         "metamorphism at 100 kg m-3",
         model = (; grain_growth_method = :FlannerZender2006, new_snow_method = :vanKampenhout2017SNOTEL,
                  densification_method = :Crocus, destructive_metamorphism = true,
                  blowing_snow_method = :none, drift_fetch_length = 10.0,
                  new_snow_density = 350.0, destructive_metamorphism_density = 100.0)),
    B250 = _constant_snow_level("B", 250.0),
    B300 = _constant_snow_level("B", 300.0),
    C250 = _constant_snow_level("C", 250.0),
    C300 = _constant_snow_level("C", 300.0),
    C350 = _constant_snow_level("C", 350.0),
)

"""
    PHYSICS_BAND_DECOUPLING

Band decoupling levels of a physics member, as the `band_decoupling` rule of
[`resolve_downscaling`](@ref):

- `kfit`: the band `k` fitted to ERA5-Land's glaciated cells, on the donor-converted ambient;
- `kshaw`: the framework of Shaw et al. (2025): the raw ERA5-Land ambient (no donor conversion),
  decoupled with each band's glacier-area weighted Shaw `k`, with
  [`SHAW_FILL_DECOUPLING_FACTOR`](@ref) for the ice Shaw does not cover
  ([`shaw_band_decoupling`](@ref)). The per-band `k` depends on the tile, so the rule here is the
  template the scripts fill in.
"""
const PHYSICS_BAND_DECOUPLING = (
    kfit = (; description = "fitted band k on the donor-converted ambient", rule = (; rule = :fitted)),
    kshaw = (; description = "Shaw et al. (2025) band k on the raw ERA5-Land ambient",
             rule = (; rule = :shaw)),
)

"""
    PHYSICS_IMPURITY

Fresh-snow impurity levels of a physics member:

- `table`: the regional background table, black carbon plus dust converted with SNICAR-ADv3 optics
  ([`background_black_carbon`](@ref), `basis = :bc_dust`, `optics = :snicar_adv3`), the production
  loading;
- `tbc`: the table's black carbon alone, SNICAR-ADv3 optics (`basis = :bc`);
- `tdang`, `traw`: black carbon plus dust converted with the Dang et al. (2015) optics, or
  unconverted (`optics = :dang2015`, `:none`);
- `bc0`: no loading (`black_carbon_snow = 0`);
- `cvis`: the effective loading fitted to MODIS shortwave albedo for the member's grain variant,
  capped at 0.045 ppmw, read from a regional table ([`effective_black_carbon`](@ref)). 0.045 ppmw is
  the largest absorber that MCD43C3 visible white-sky albedo allows on the Arctic ice caps once its
  deficit over nearly pure Antarctic snow is removed. Regions the table gives the background value
  keep it.
"""
const PHYSICS_IMPURITY = (
    table = (; description = "regional background table, BC + dust, SNICAR-ADv3 optics",
             basis = :bc_dust, optics = :snicar_adv3),
    tbc = (; description = "regional background table, BC only, SNICAR-ADv3 optics",
           basis = :bc, optics = :snicar_adv3),
    tdang = (; description = "regional background table, BC + dust, Dang et al. (2015) optics",
             basis = :bc_dust, optics = :dang2015),
    traw = (; description = "regional background table, BC + dust, unconverted",
            basis = :bc_dust, optics = :none),
    bc0 = (; description = "no snow impurity", basis = :off),
    cvis = (; description = "MODIS effective loading capped at 0.045 ppmw",
            basis = :effective, cap = 0.045),
)

# The `w` token of a wind factor, in its one spelling: `w2`, `w1.5`.
_wind_token(factor::Real) = "w" * (isinteger(factor) ? string(Int(factor)) : string(Float64(factor)))

"""
    physics_member(id::AbstractString) -> NamedTuple

The physics member `id` = `"<grain>-<band decoupling>-<impurity>"` or
`"<grain>-<band decoupling>-<impurity>-w<factor>"` (see the module notes above), as
`(; id, grain, band_decoupling, impurity, model, band_rule, impurity_level, wind_scaling,
wind_correction, description)`: the three level names as `Symbol`s, the `ModelParameters` fields the grain level sets,
the `band_decoupling` rule (`(; rule)`; a `:shaw` rule takes its per-band `k` from the tile), the
impurity level, the wind factor (1.0 without a `w<factor>` token) and the wind correction (`:slope`
for a `ws` token, `:scaling` otherwise). Throws for an unknown level or a malformed id.
"""
function physics_member(id::AbstractString)
    parts = split(id, '-')
    length(parts) in (3, 4) || throw(ArgumentError(
        "a physics member is \"<grain>-<band decoupling>-<impurity>\", e.g. \"C-kfit-table\", or " *
        "\"<grain>-<band decoupling>-<impurity>-w<factor>\" or \"...-ws\"; got \"$id\""))
    wind_scaling = 1.0
    wind_correction = :scaling
    if length(parts) == 4 && parts[4] == "ws"
        wind_correction = :slope
    elseif length(parts) == 4
        m = match(r"^w([0-9]+(\.[0-9]+)?)$", parts[4])
        m === nothing && throw(ArgumentError(
            "the fourth level of physics member \"$id\" must be a wind factor w<factor>, e.g. \"w2\", " *
            "or the slope correction \"ws\"; " *
            "got \"$(parts[4])\""))
        wind_scaling = parse(Float64, m[1])
        wind_scaling > 0 || throw(ArgumentError("the wind factor of physics member \"$id\" must be > 0"))
        parts[4] == _wind_token(wind_scaling) || throw(ArgumentError(
            "write the wind factor of physics member \"$id\" as \"$(_wind_token(wind_scaling))\""))
    end
    grain, k, impurity = Symbol.(parts[1:3])
    _check_level(grain, PHYSICS_GRAIN_VARIANTS, "grain variant", id)
    _check_level(k, PHYSICS_BAND_DECOUPLING, "band decoupling level", id)
    _check_level(impurity, PHYSICS_IMPURITY, "impurity level", id)
    g = PHYSICS_GRAIN_VARIANTS[grain]
    d = PHYSICS_BAND_DECOUPLING[k]
    i = PHYSICS_IMPURITY[impurity]
    wind = wind_correction === :slope ? "band wind × m(2 km slope)" :
           wind_scaling == 1 ? "ERA5-Land wind" : "band wind × $(_wind_token(wind_scaling)[2:end])"
    return (; id = String(id), grain, band_decoupling = k, impurity, model = g.model,
            band_rule = d.rule, impurity_level = i, wind_scaling, wind_correction,
            description = "$(g.description) | $(d.description) | $(i.description) | $wind")
end

_check_level(level, table, what, id) = haskey(table, level) || throw(ArgumentError(
    "unknown $what \"$level\" in physics member \"$id\"; one of $(join(keys(table), ", "))"))

"""
    physics_model_parameters(member; kwargs...) -> ModelParameters

`GEMB.initialize_parameters` under `member`'s grain level, with `kwargs` for every other field.
Throws when `kwargs` sets a field the member sets, so a run cannot be labelled with a member whose
grain settings it overrode.
"""
function physics_model_parameters(member; kwargs...)
    clash = intersect(keys(kwargs), keys(member.model))
    isempty(clash) || throw(ArgumentError(
        "physics member $(member.id) sets $(join(clash, ", ")); remove it from the keywords"))
    return GEMB.initialize_parameters(; kwargs..., member.model...)
end

"""
    physics_member_parameters(member) -> Dict{String,Any}

The run parameters that name `member`: `physics_member` (its id), `physics_grain`,
`physics_band_decoupling`, `physics_impurity` (its levels), `physics_wind_scaling` (its wind factor,
1.0 without a `w` level) and, for a `ws` member only, `physics_wind_correction = "slope"`.
"""
function physics_member_parameters(member)
    p = Dict{String,Any}(
        "physics_member" => member.id,
        "physics_grain" => string(member.grain),
        "physics_band_decoupling" => string(member.band_decoupling),
        "physics_impurity" => string(member.impurity),
        "physics_wind_scaling" => Float64(member.wind_scaling))
    _member_wind_correction(member) === :slope && (p["physics_wind_correction"] = "slope")
    return p
end

# A member's wind correction; a member tuple without the field corrects by `wind_scaling` alone.
_member_wind_correction(member) = hasproperty(member, :wind_correction) ? member.wind_correction : :scaling

# Throw unless a run under `mp`, `applied`, `background_black_carbon` and `effective_black_carbon` is
# the configuration `member` names.
function _assert_physics_member(member, mp::ModelParameters, applied, background_black_carbon,
                                effective_black_carbon)
    for (field, value) in pairs(member.model)
        getproperty(mp, field) == value || throw(ArgumentError(
            "physics member $(member.id) sets $field = $(repr(value)), but the run's " *
            "ModelParameters has $(repr(getproperty(mp, field)))"))
    end
    # An `AppliedDownscaling` built without a rule (by hand, or by a resolution predating the keyword)
    # applied none, which is the `:fitted` rule on the converted ambient.
    recorded = (; rule = get(applied.settings, "downscaling_band_decoupling_rule", "fitted"),
                donor_conversion = get(applied.settings, "downscaling_donor_conversion", "fitted"))
    expected = (; rule = string(member.band_rule.rule),
                donor_conversion = member.band_rule.rule === :shaw ? "none" : "fitted")
    recorded == expected || throw(ArgumentError(
        "physics member $(member.id) applies band decoupling $expected, but the downscaling was " *
        "resolved with $recorded"))
    wind = Float64(get(applied.settings, "downscaling_wind_scaling", 1.0))
    wind == member.wind_scaling || throw(ArgumentError(
        "physics member $(member.id) scales the band wind by $(member.wind_scaling), but the " *
        "downscaling was resolved with wind_scaling = $wind"))
    correction = get(applied.settings, "downscaling_wind_correction", "scaling")
    correction == string(_member_wind_correction(member)) || throw(ArgumentError(
        "physics member $(member.id) corrects the band wind by $(_member_wind_correction(member)), but the " *
        "downscaling was resolved with wind_correction = $correction"))
    level = member.impurity_level
    if level.basis === :off
        (background_black_carbon === nothing && effective_black_carbon === nothing) ||
            throw(ArgumentError("physics member $(member.id) runs without a snow impurity loading, " *
                                "but one was given"))
    elseif level.basis === :effective
        effective_black_carbon === nothing && throw(ArgumentError(
            "physics member $(member.id) needs an effective loading (`effective_black_carbon`)"))
        effective_black_carbon.cap == level.cap || throw(ArgumentError(
            "physics member $(member.id) caps the effective loading at $(level.cap) ppmw, but the " *
            "loading given is capped at $(effective_black_carbon.cap)"))
        background_black_carbon === nothing || throw(ArgumentError(
            "physics member $(member.id) takes the effective loading, not the background table"))
        startswith(effective_black_carbon.variant, string(member.grain)) || throw(ArgumentError(
            "physics member $(member.id) has grain variant $(member.grain), but the effective " *
            "loading was fitted for variant $(effective_black_carbon.variant)"))
    else
        background_black_carbon === nothing && throw(ArgumentError(
            "physics member $(member.id) needs the background loading (`background_black_carbon`)"))
        effective_black_carbon === nothing || throw(ArgumentError(
            "physics member $(member.id) takes the background table, not an effective loading"))
        (background_black_carbon.basis === level.basis &&
         background_black_carbon.optics === level.optics) || throw(ArgumentError(
            "physics member $(member.id) takes the background table with basis :$(level.basis) " *
            "and optics :$(level.optics), but the loading given has basis " *
            ":$(background_black_carbon.basis) and optics :$(background_black_carbon.optics)"))
    end
    return nothing
end

"""
    effective_black_carbon(cells, path; cap = Inf) -> NamedTuple

The effective fresh-snow impurity loading for the tile whose core is `cells`, from the regional table
at `path`: the glacier-area weighted mean, over the RGI 7.0 o1 regions of `cells`
([`tile_rgi_o1_region_shares`](@ref)), of each region's `black_carbon_snow` [ppmw, Gardner & Sharp
(2010) equivalent] capped at `cap`.

The table is a CSV with at least the columns `rgi_o1region`, `black_carbon_snow` and `variant` (the
grain variant the loading was fitted for, e.g. `C0`), one row per region, and one variant throughout.
Throws when a column is missing, a region of the tile has no row or more than one, a loading lies
outside GEMB's `[0, 2]` ppmw, or the table mixes variants.

Returns `(; black_carbon_snow, variant, cap, region_shares, region_values, table_file,
table_sha256)`, with `region_values` the capped values.
"""
function effective_black_carbon(cells::AbstractDataFrame, path::AbstractString; cap::Real = Inf)
    cap > 0 || throw(ArgumentError("the cap on an effective loading must be positive, got $cap"))
    isfile(path) || throw(ArgumentError("no effective-loading table at $path"))
    raw = _read_csv_strings(path)
    absent = setdiff(["rgi_o1region", "black_carbon_snow", "variant"], names(raw))
    isempty(absent) || throw(ArgumentError(
        "effective-loading table $path lacks column(s) $(join(absent, ", "))"))
    variants = unique(String.(strip.(raw.variant)))
    length(variants) == 1 || throw(ArgumentError(
        "effective-loading table $path holds variants $(join(variants, ", ")); it must hold one"))
    regions = String.(strip.(raw.rgi_o1region))
    shares = tile_rgi_o1_region_shares(cells)
    values = map(shares) do (region, _)
        rows = findall(==(region), regions)
        length(rows) == 1 || throw(ArgumentError(
            "effective-loading table $path has $(length(rows)) rows for RGI region $region; it " *
            "needs one"))
        v = tryparse(Float64, strip(String(raw.black_carbon_snow[only(rows)])))
        (v !== nothing && 0 <= v <= _BLACK_CARBON_SNOW_MAXIMUM) || throw(ArgumentError(
            "effective-loading table $path gives region $region a loading of " *
            "$(raw.black_carbon_snow[only(rows)]); it must be in [0, $(_BLACK_CARBON_SNOW_MAXIMUM)] " *
            "ppmw"))
        region => min(v, Float64(cap))
    end
    value = sum(last(s) * last(v) for (s, v) in zip(shares, values))
    return (; black_carbon_snow = value, variant = only(variants), cap = Float64(cap),
            region_shares = shares,
            region_values = values, table_file = basename(path),
            table_sha256 = bytes2hex(open(SHA.sha256, path)))
end

# The run settings an effective loading adds to `tile_run_parameters`, beside the
# `model_black_carbon_snow` every run records.
function _record_effective_black_carbon!(params, ebc)
    params["effective_black_carbon_variant"] = ebc.variant
    params["effective_black_carbon_cap"] = isfinite(ebc.cap) ? ebc.cap : "none"
    params["effective_black_carbon_region_shares"] =
        join(("$r:$(round(s; digits = 4))" for (r, s) in ebc.region_shares), ",")
    params["effective_black_carbon_region_values"] =
        join(("$r:$v" for (r, v) in ebc.region_values), ",")
    params["effective_black_carbon_table_file"] = ebc.table_file
    params["effective_black_carbon_table_sha256"] = ebc.table_sha256
    return params
end
