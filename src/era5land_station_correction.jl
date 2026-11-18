"""
ERA5-Land downward longwave radiation corrected toward on-ice weather stations: a monthly offset by
station group, applied to ERA5-Land forcing only.

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

The correction describes ERA5-Land's errors, so [`era5land_station_correction`](@ref) refuses any
other forcing.

# References
- Driemel, A., et al. (2018). Baseline Surface Radiation Network (BSRN): structure and data
  description (1992-2017). *Earth Syst. Sci. Data* 10, 1491-1501.
- Fausto, R. S., et al. (2021). Programme for Monitoring of the Greenland Ice Sheet (PROMICE)
  automatic weather station data. *Earth Syst. Sci. Data* 13, 3819-3845.
"""

const ERA5LAND_STATION_CORRECTIONS_TABLE =
    normpath(joinpath(@__DIR__, "..", "data", "era5land_station_corrections.csv"))

# Station groups of the correction table.
const ERA5LAND_CORRECTION_GROUPS = (:antarctic_ice_sheet, :greenland_ice_sheet, :alpine)

# RGI 7.0 first-order regions of the Arctic ice caps, which are not corrected.
const _UNCORRECTED_RGI_REGIONS = ("03", "04", "05", "07", "09")

const _ERA5LAND_CORRECTION_CACHE = Ref{Union{Nothing,Dict{Symbol,Vector{Float64}}}}(nothing)

# `group => 12 monthly downward longwave offsets` [W m-2].
function _era5land_correction_table()
    cached = _ERA5LAND_CORRECTION_CACHE[]
    cached === nothing || return cached
    isfile(ERA5LAND_STATION_CORRECTIONS_TABLE) || throw(ArgumentError(
        "no ERA5-Land station-correction table at $ERA5LAND_STATION_CORRECTIONS_TABLE"))
    raw = _read_csv_strings(ERA5LAND_STATION_CORRECTIONS_TABLE)
    table = Dict{Symbol,Vector{Float64}}()
    for row in eachrow(raw)
        strip(row.var) == "LW" || throw(ArgumentError(
            "$ERA5LAND_STATION_CORRECTIONS_TABLE holds a $(row.var) row; the correction is longwave only"))
        v = get!(() -> fill(NaN, 12), table, Symbol(strip(row.group)))
        v[parse(Int, row.month)] = parse(Float64, row.median)
    end
    for g in ERA5LAND_CORRECTION_GROUPS
        v = get(table, g, nothing)
        (v !== nothing && all(isfinite, v)) || throw(ArgumentError(
            "$ERA5LAND_STATION_CORRECTIONS_TABLE lacks monthly values for group $g"))
    end
    _ERA5LAND_CORRECTION_CACHE[] = table
    return table
end

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
    era5land_station_correction(forcing::DimStack, group::Symbol) -> DimStack

`forcing` with its downward longwave radiation shifted by the calendar-month offset of station
`group` (one of `$(ERA5LAND_CORRECTION_GROUPS)`; see [`era5land_correction_group`](@ref)) and held
non-negative. Every other layer is left as it is. The stack's metadata records the group under
`"era5land_station_correction"`.

Throws unless `forcing` is ERA5-Land (`metadata["dataset"] == "ERA5-Land"`), and if it has already
been corrected.
"""
function era5land_station_correction(forcing::DimStack, group::Symbol)
    group in ERA5LAND_CORRECTION_GROUPS || throw(ArgumentError(
        "unknown correction group :$group; expected one of $(ERA5LAND_CORRECTION_GROUPS)"))
    meta = Dict{String,Any}(DimensionalData.metadata(forcing))
    dataset = get(meta, "dataset", nothing)
    dataset == "ERA5-Land" || throw(ArgumentError(
        "the station correction describes ERA5-Land's errors and applies to ERA5-Land forcing " *
        "only; this forcing's metadata gives dataset = $(repr(dataset))"))
    haskey(meta, "era5land_station_correction") && throw(ArgumentError(
        "this forcing already carries the station correction for group " *
        "$(meta["era5land_station_correction"])"))
    dLW = _era5land_correction_table()[group]
    m = month.(lookup(forcing, Ti))
    meta["era5land_station_correction"] = String(group)
    return GEMB_ClimateForcing._rebuild_forcing(forcing, meta;
        longwave_downward = max.(parent(forcing[:longwave_downward]) .+ dLW[m], 0.0))
end
