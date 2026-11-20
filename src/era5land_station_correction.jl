"""
ERA5-Land forcing corrected toward on-ice weather stations: a monthly additive offset of downward longwave
radiation and a monthly factor on the wind speed, by station group, applied to ERA5-Land forcing only.

Downward longwave is ERA5-Land's largest error over ice, and the one that sets the annual mean
surface temperature and so the firn temperature below the depth of the seasonal wave. The values
(`data/era5land_station_corrections.csv`) are medians over the stations of each group of the monthly
mean of station minus ERA5-Land downward longwave. The groups:

- `:antarctic_ice_sheet`: the Baseline Surface Radiation Network stations South Pole and Concordia
  (Driemel et al., 2018), on the plateau where the firn cores are;
- `:greenland_ice_sheet`: the PROMICE and GC-Net dry-snow and percolation stations (Fausto et al.,
  2021); it applies to every Greenland ice-sheet site;
- `:alpine`: the alpine stations and the Icelandic glacier stations; it applies to every glacier
  outside the Arctic.

[`era5land_correction_group`](@ref) gives `nothing` for the ice caps of Arctic Canada, the Greenland
periphery, Svalbard and the Russian Arctic, which are not corrected: the Arctic ice-cap stations lie
low on their ice caps, and their offset warms the firn at the summit of the Devon Ice Cap well above
its measured temperature.

The wind factors (`U` rows) are medians over the stations of each group of the monthly mean station 10 m wind over
ERA5-Land's. Their groups are those of the longwave and two more that take no longwave offset:
`:arctic_ice_cap` (the Arctic ice-cap stations) and `:greenland_ablation` (the Greenland ablation-zone stations);
the Antarctic Ice Sheet takes three wind groups by elevation, from the AntAWS stations (Wang et al., 2023), because
their wind over ERA5-Land's falls with elevation: `:antarctic_low` below 1500 m (the coast and ice shelves),
`:antarctic_slope` from 1500 to 2500 m and `:antarctic_plateau` above 2500 m. Over glaciers and ice caps
ERA5-Land's wind is a half to a third of the stations'; over the Greenland accumulation zone it agrees with them.

The correction describes ERA5-Land's errors, so [`era5land_station_correction`](@ref) refuses any
other forcing.

# References
- Driemel, A., et al. (2018). Baseline Surface Radiation Network (BSRN): structure and data
  description (1992-2017). *Earth Syst. Sci. Data* 10, 1491-1501.
- Fausto, R. S., et al. (2021). Programme for Monitoring of the Greenland Ice Sheet (PROMICE)
  automatic weather station data. *Earth Syst. Sci. Data* 13, 3819-3845.
- Wang, Y., et al. (2023). The AntAWS dataset: a compilation of Antarctic automatic weather station
  observations. *Earth Syst. Sci. Data* 15, 411-429.
"""

const ERA5LAND_STATION_CORRECTIONS_TABLE =
    normpath(joinpath(@__DIR__, "..", "data", "era5land_station_corrections.csv"))

# Station groups of the longwave correction.
const ERA5LAND_CORRECTION_GROUPS = (:antarctic_ice_sheet, :greenland_ice_sheet, :alpine)
# Station groups of the wind correction.
const ERA5LAND_WIND_CORRECTION_GROUPS = (:antarctic_low, :antarctic_slope, :antarctic_plateau, :greenland_ice_sheet,
                                         :greenland_ablation, :arctic_ice_cap, :alpine)
# Upper elevation [m] of each Antarctic wind group, lowest first.
const _ANTARCTIC_WIND_BANDS = ((1500.0, :antarctic_low), (2500.0, :antarctic_slope), (Inf, :antarctic_plateau))

# RGI 7.0 first-order regions of the Arctic ice caps, which are not corrected.
const _UNCORRECTED_RGI_REGIONS = ("03", "04", "05", "07", "09")

const _ERA5LAND_CORRECTION_CACHE = Ref{Union{Nothing,Dict{String,Dict{Symbol,Vector{Float64}}}}}(nothing)

# `var => group => 12 monthly values`: `"LW"` offsets [W m-2], `"U"` factors [-].
function _era5land_correction_tables()
    cached = _ERA5LAND_CORRECTION_CACHE[]
    cached === nothing || return cached
    isfile(ERA5LAND_STATION_CORRECTIONS_TABLE) || throw(ArgumentError(
        "no ERA5-Land station-correction table at $ERA5LAND_STATION_CORRECTIONS_TABLE"))
    raw = _read_csv_strings(ERA5LAND_STATION_CORRECTIONS_TABLE)
    tables = Dict("LW" => Dict{Symbol,Vector{Float64}}(), "U" => Dict{Symbol,Vector{Float64}}())
    for row in eachrow(raw)
        var = strip(row.var)
        haskey(tables, var) || throw(ArgumentError(
            "$ERA5LAND_STATION_CORRECTIONS_TABLE holds a $(var) row; the correction is LW and U only"))
        v = get!(() -> fill(NaN, 12), tables[var], Symbol(strip(row.group)))
        v[parse(Int, row.month)] = parse(Float64, row.median)
    end
    for (var, groups) in (("LW", ERA5LAND_CORRECTION_GROUPS), ("U", ERA5LAND_WIND_CORRECTION_GROUPS)), g in groups
        v = get(tables[var], g, nothing)
        (v !== nothing && all(isfinite, v)) || throw(ArgumentError(
            "$ERA5LAND_STATION_CORRECTIONS_TABLE lacks monthly $var values for group $g"))
    end
    all(v -> all(>(0), v), values(tables["U"])) ||
        throw(ArgumentError("$ERA5LAND_STATION_CORRECTIONS_TABLE holds a non-positive wind factor"))
    _ERA5LAND_CORRECTION_CACHE[] = tables
    return tables
end

# `group => 12 monthly downward longwave offsets` [W m-2].
_era5land_correction_table() = _era5land_correction_tables()["LW"]
# `group => 12 monthly wind factors` [-].
_era5land_wind_correction_table() = _era5land_correction_tables()["U"]

"""
    era5land_correction_group(lon, lat; ice_sheet::Bool) -> Union{Symbol,Nothing}

The station group of [`era5land_station_correction`](@ref) for a site at (`lon`, `lat`) [degrees], or
`nothing` for a site that is not corrected. An ice-sheet site (`ice_sheet = true`) is
`:antarctic_ice_sheet` south of the equator and `:greenland_ice_sheet` north of it. A glacier or ice
cap is `nothing` in RGI 7.0 first-order regions 03, 04, 05, 07 and 09 (Arctic Canada, the Greenland
periphery, Svalbard and Jan Mayen, the Russian Arctic; [`rgi_o1_region`](@ref)) and `:alpine`
everywhere else, Iceland included.
"""
function era5land_correction_group(lon::Real, lat::Real; ice_sheet::Bool)
    ice_sheet && return lat < 0 ? :antarctic_ice_sheet : :greenland_ice_sheet
    return rgi_o1_region(lon, lat) in _UNCORRECTED_RGI_REGIONS ? nothing : :alpine
end

"""
    era5land_wind_correction_group(lon, lat; ice_sheet::Bool, ablation::Bool = false, elevation = NaN) -> Symbol

The station group of the wind factor of [`era5land_station_correction`](@ref) for a site at (`lon`, `lat`)
[degrees]: for an Antarctic ice-sheet site, by its `elevation` [m], `:antarctic_low` below 1500 m,
`:antarctic_slope` from 1500 to 2500 m and `:antarctic_plateau` above; for a Greenland ice-sheet site
`:greenland_ablation` with `ablation = true` and `:greenland_ice_sheet` otherwise; for a glacier or ice cap
`:arctic_ice_cap` in RGI 7.0 regions 03, 04, 05, 07 and 09 and `:alpine` elsewhere, Iceland included.

Throws for an Antarctic ice-sheet site without a finite `elevation`.
"""
function era5land_wind_correction_group(lon::Real, lat::Real; ice_sheet::Bool, ablation::Bool = false,
                                        elevation::Real = NaN)
    if ice_sheet && lat < 0
        isfinite(elevation) || throw(ArgumentError(
            "an Antarctic ice-sheet site needs its elevation to choose an Antarctic wind group"))
        return last(first(b for b in _ANTARCTIC_WIND_BANDS if elevation < first(b)))
    end
    ice_sheet && return ablation ? :greenland_ablation : :greenland_ice_sheet
    return rgi_o1_region(lon, lat) in _UNCORRECTED_RGI_REGIONS ? :arctic_ice_cap : :alpine
end

"""
    era5land_station_correction(forcing::DimStack, group; wind_group = nothing) -> DimStack

`forcing` with its downward longwave radiation shifted by the calendar-month offset of station `group` (one of
`$(ERA5LAND_CORRECTION_GROUPS)`, or `nothing` for no longwave offset; see [`era5land_correction_group`](@ref)) and
held non-negative, and, with `wind_group` (one of `$(ERA5LAND_WIND_CORRECTION_GROUPS)`; see
[`era5land_wind_correction_group`](@ref)), its wind speed multiplied by that group's calendar-month factor, with the
metadata `"wind_speed_mean"` scaled by the factor's mean over the record's steps. Every other layer is left as it is.
The stack's metadata records the groups under `"era5land_station_correction"` and
`"era5land_wind_correction"`.

Throws unless `forcing` is ERA5-Land (`metadata["dataset"] == "ERA5-Land"`), if it has already been corrected, and
if both groups are `nothing`.
"""
function era5land_station_correction(forcing::DimStack, group::Union{Symbol,Nothing};
                                     wind_group::Union{Symbol,Nothing} = nothing)
    group === nothing || group in ERA5LAND_CORRECTION_GROUPS || throw(ArgumentError(
        "unknown correction group :$group; expected one of $(ERA5LAND_CORRECTION_GROUPS)"))
    wind_group === nothing || wind_group in ERA5LAND_WIND_CORRECTION_GROUPS || throw(ArgumentError(
        "unknown wind correction group :$wind_group; expected one of $(ERA5LAND_WIND_CORRECTION_GROUPS)"))
    group === nothing && wind_group === nothing &&
        throw(ArgumentError("neither a longwave nor a wind correction group was given"))
    meta = Dict{String,Any}(DimensionalData.metadata(forcing))
    dataset = get(meta, "dataset", nothing)
    dataset == "ERA5-Land" || throw(ArgumentError(
        "the station correction describes ERA5-Land's errors and applies to ERA5-Land forcing " *
        "only; this forcing's metadata gives dataset = $(repr(dataset))"))
    (haskey(meta, "era5land_station_correction") || haskey(meta, "era5land_wind_correction")) &&
        throw(ArgumentError("this forcing already carries the station correction for group " *
                            "$(get(meta, "era5land_station_correction", get(meta, "era5land_wind_correction", "")))"))
    m = month.(lookup(forcing, Ti))
    changed = Dict{Symbol,Any}()
    if group !== nothing
        dLW = _era5land_correction_table()[group]
        changed[:longwave_downward] = max.(parent(forcing[:longwave_downward]) .+ dLW[m], 0.0)
        meta["era5land_station_correction"] = String(group)
    end
    if wind_group !== nothing
        fU = _era5land_wind_correction_table()[wind_group][m]
        changed[:wind_speed] = parent(forcing[:wind_speed]) .* fU
        haskey(meta, "wind_speed_mean") && (meta["wind_speed_mean"] = meta["wind_speed_mean"] * mean(fU))
        meta["era5land_wind_correction"] = String(wind_group)
    end
    return GEMB_ClimateForcing._rebuild_forcing(forcing, meta; changed...)
end
