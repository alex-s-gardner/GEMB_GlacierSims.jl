"""
    SHAW_GLACIER_BANDS_TABLE

The vendored table of the glaciers in the Shaw et al. (2025) decoupling table with their RGI 6.0 area
in each 100 m elevation band, written by `scripts/make_shaw_glacier_bands.jl`.
"""
const SHAW_GLACIER_BANDS_TABLE =
    normpath(joinpath(@__DIR__, "..", "data", "shaw2025_glacier_bands.csv.gz"))

"""
    SHAW_FILL_DECOUPLING_FACTOR

The `k` [`shaw_band_decoupling`](@ref) gives ice the Shaw et al. (2025) table does not cover (ice caps,
tidewater glaciers, RGI regions 05 and 19): 0.77, the glacier-area weighted mean of the table's
glacier-mean `k` with RGI 6.0 areas.
"""
const SHAW_FILL_DECOUPLING_FACTOR = 0.77

const _SHAW_BANDS_CACHE = Ref{Any}(nothing)

"""
    shaw_glacier_bands(path = SHAW_GLACIER_BANDS_TABLE) -> NamedTuple

The Shaw et al. (2025) glaciers with their area per 100 m band, as
`(; rgi_id, latitude, longitude, k, area_km2, band_lo, band_area_km2, sha256)`: `band_lo[i]` and
`band_area_km2[i]` list glacier `i`'s bands `[lo, lo + 100)` and its area in each. Validated on read:
every `k` in `(0, 1]`, every area non-negative, band edges on whole hundreds of metres, and each
glacier's band areas summing to its area to within 1e-4 relative. The default table is read once per
session.
"""
function shaw_glacier_bands(path::AbstractString = SHAW_GLACIER_BANDS_TABLE)
    cached = path == SHAW_GLACIER_BANDS_TABLE ? _SHAW_BANDS_CACHE[] : nothing
    cached === nothing || return cached
    isfile(path) || throw(ArgumentError("no Shaw glacier band table at $path; write it with " *
                                        "scripts/make_shaw_glacier_bands.jl"))
    rgi_id = String[]; latitude = Float64[]; longitude = Float64[]; k = Float64[]; area = Float64[]
    band_lo = Vector{Int}[]; band_area = Vector{Float64}[]
    header_seen = false
    open(CodecZlib.GzipDecompressorStream, path) do io
        for line in eachline(io)
            (isempty(line) || startswith(line, '#')) && continue
            if !header_seen
                line == "rgi_id,lat,lon,k,area_km2,band_lo,band_area_km2" || throw(ArgumentError(
                    "unexpected header in $path: $line"))
                header_seen = true
                continue
            end
            f = split(line, ',')
            length(f) == 7 || throw(ArgumentError("malformed row in $path: $line"))
            lo = parse.(Int, split(f[6], ';'))
            a = parse.(Float64, split(f[7], ';'))
            kv, av = parse(Float64, f[4]), parse(Float64, f[5])
            (0 < kv <= 1 && av >= 0 && length(lo) == length(a) && all(>=(0), a) &&
             all(x -> mod(x, 100) == 0, lo)) || throw(ArgumentError("invalid row in $path: $line"))
            isapprox(sum(a), av; rtol = 1e-4) || throw(ArgumentError(
                "band areas of $(f[1]) in $path sum to $(sum(a)), not its area $av"))
            push!(rgi_id, f[1]); push!(latitude, parse(Float64, f[2]))
            push!(longitude, parse(Float64, f[3])); push!(k, kv); push!(area, av)
            push!(band_lo, lo); push!(band_area, a)
        end
    end
    header_seen || throw(ArgumentError("no header in $path"))
    table = (; rgi_id, latitude, longitude, k, area_km2 = area, band_lo, band_area_km2 = band_area,
             sha256 = bytes2hex(open(SHA.sha256, path)))
    path == SHAW_GLACIER_BANDS_TABLE && (_SHAW_BANDS_CACHE[] = table)
    return table
end

# Whether a glacier centroid lies in tile `bounds`, by the tiling's own convention: longitude in
# (-180, 180] with 180 folded to -180, half-open intervals so adjacent tiles share no glacier.
function _in_tile_bounds(lat, lon, bounds)
    x = wrap_lon(lon)
    x == 180 && (x = -180.0)
    return bounds.lat_min <= lat < bounds.lat_max && bounds.lon_min <= x < bounds.lon_max
end

_tile_glaciers(bounds, table) =
    [i for i in eachindex(table.rgi_id) if _in_tile_bounds(table.latitude[i], table.longitude[i], bounds)]

"""
    shaw_decoupling_prior(bounds; table = shaw_glacier_bands()) -> Float64 or nothing

A tile's prior decoupling factor: the glacier-area weighted mean of the Shaw et al. (2025)
glacier-mean `k` over the glaciers whose centroid lies in the tile `bounds`
(`(; lon_min, lon_max, lat_min, lat_max)`), or `nothing` when the tile holds none (RGI regions 05 and
19, which the table does not cover, and tiles of ice caps alone). It is the `decoupling_factor_prior`
of [`resolve_downscaling`](@ref), and the whole band `k` of a tile whose cells carry no glacier fraction.
"""
function shaw_decoupling_prior(bounds; table = shaw_glacier_bands())
    g = _tile_glaciers(bounds, table)
    a = sum(table.area_km2[g]; init = 0.0)
    a > 0 || return nothing
    return sum(table.area_km2[i] * table.k[i] for i in g) / a
end

"""
    tile_interval_areas(cells, intervals) -> Vector{Float64}

The glacier area [km²] of `cells` (rows of the elevation-class table) in each of `intervals`
(from [`hypsometry_intervals`](@ref)).
"""
function tile_interval_areas(cells, intervals)
    areas = zeros(length(intervals))
    position = Dict((iv.lo, iv.hi) => i for (i, iv) in enumerate(intervals))
    for row in eachrow(cells), b in glacier_hypsometry(row; area_minimum = 0)
        i = get(position, (b.lo, b.hi), 0)
        i == 0 || (areas[i] += b.area)
    end
    return areas
end

"""
    shaw_band_decoupling(bounds, intervals, interval_area_km2; fill_k = SHAW_FILL_DECOUPLING_FACTOR,
                         table = shaw_glacier_bands()) -> NamedTuple

Per-band decoupling factors in the framework of Shaw et al. (2025): each band's `k` is the glacier-area
weighted mean of the glacier-mean `k` of the Shaw glaciers with their centroid in the tile `bounds`,
weighted by each glacier's area in the band, with the band's remaining ice
(`interval_area_km2[i]` minus the Shaw area, when positive) at `fill_k`. A band whose Shaw area
exceeds `interval_area_km2[i]` (RGI 6.0 and the tile's RGI 7.0 hypsometry differ) is all Shaw.

`intervals` must be the tile's 100 m bands. Returns `(; band_k, covered_fraction, shaw_area_km2)`, one
value per interval; `band_k` is what `(; rule = :shaw, band_k)` of [`resolve_downscaling`](@ref)
takes.
"""
function shaw_band_decoupling(bounds, intervals, interval_area_km2;
                              fill_k::Real = SHAW_FILL_DECOUPLING_FACTOR, table = shaw_glacier_bands())
    length(interval_area_km2) == length(intervals) || throw(DimensionMismatch(
        "$(length(interval_area_km2)) interval areas for $(length(intervals)) intervals"))
    all(iv -> iv.hi - iv.lo == 100 && mod(iv.lo, 100) == 0, intervals) || throw(ArgumentError(
        "shaw_band_decoupling needs 100 m bands on whole hundreds of metres"))
    0 < fill_k <= 1 || throw(ArgumentError("fill_k must be in (0, 1], got $fill_k"))
    position = Dict(iv.lo => i for (i, iv) in enumerate(intervals))
    shaw_area = zeros(length(intervals))
    shaw_k_area = zeros(length(intervals))
    for g in _tile_glaciers(bounds, table)
        for (lo, a) in zip(table.band_lo[g], table.band_area_km2[g])
            i = get(position, lo, 0)
            i == 0 && continue
            shaw_area[i] += a
            shaw_k_area[i] += a * table.k[g]
        end
    end
    band_k = similar(shaw_area); covered = similar(shaw_area)
    for i in eachindex(intervals, interval_area_km2)
        A = Float64(interval_area_km2[i])
        if A <= 0 && shaw_area[i] == 0
            band_k[i] = fill_k
            covered[i] = 0.0
        elseif shaw_area[i] >= A
            band_k[i] = shaw_k_area[i] / shaw_area[i]
            covered[i] = 1.0
        else
            band_k[i] = (shaw_k_area[i] + (A - shaw_area[i]) * fill_k) / A
            covered[i] = shaw_area[i] / A
        end
    end
    return (; band_k, covered_fraction = covered, shaw_area_km2 = shaw_area)
end
