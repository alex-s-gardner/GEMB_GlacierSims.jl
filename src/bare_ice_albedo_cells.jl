# Bare-ice albedo as a function of elevation, for ERA5-Land grid cells and the tiles over them.
#
# `bare_ice_albedo` (GEMB_ClimateForcing) returns MODIS MCD43A3 500 m cells for a geometry and
# `bare_ice_albedo_hyps` reduces those to albedo binned by elevation. What this file adds is the
# ERA5-Land-specific half: the box a grid cell occupies, which cell a MODIS cell falls in, the
# pooled per-tile derivation the downscaling sweep stores, and the resolution of that profile onto
# one `albedo_ice` per elevation class.

"""
Where a resolved `albedo_ice` came from, in the order [`resolve_albedo_ice`](@ref) tries them:

- `:observed` — the elevation bin's own MODIS cells.
- `:hold` — the nearest observed bin, because the class sits outside the sampled elevation range.
- `:fit` — the profile's linear albedo-elevation fit.
- `:clamped` — one of the above, moved to the nearest edge of [`GEMB_ALBEDO_ICE_RANGE`](@ref).
- `:default` — no observation resolved for this class, so the caller's own `albedo_ice` stands.

Deliberately **not** [`DOWNSCALING_SOURCES`](@ref): that vocabulary describes how a *time series* was
substituted where a fit was unusable, and has no notion of a value being clamped.
"""
const BARE_ICE_ALBEDO_SOURCES = (:observed, :hold, :fit, :clamped, :default)

"""
The range GEMB accepts for `albedo_ice`, `(0.2, 0.6)`.

`GEMB.validate_parameters` asserts it, so a value outside this is not a parameter GEMB will run —
it throws before the first timestep. [`resolve_albedo_ice`](@ref) therefore clamps into it and says
so, rather than handing over a number that cannot be used.

Observed bare-ice albedo genuinely exceeds `0.6` above the equilibrium line, where the
darkest-percentile statistic is measuring snow because there is no bare ice to see. Clamping loses
little: GEMB does not expose ice at those elevations either, so `albedo_ice` barely enters the
energy balance there.
"""
const GEMB_ALBEDO_ICE_RANGE = (0.2, 0.6)

"""
    era5_land_cell_polygon(lon, lat; cell_size = 0.1) -> GeoInterface.Polygon

The box an ERA5-Land grid cell occupies, centred on `(lon, lat)`, as a closed ring in
`(-180, 180]` longitude ([`wrap_lon`](@ref)) — the convention `bare_ice_albedo` expects.

`cell_size` is the grid spacing in degrees. The 0.1° default is ERA5-Land's, the same spacing
[`cell_output_name`](@ref) spells with one decimal.
"""
function era5_land_cell_polygon(lon, lat; cell_size::Real = 0.1)
    cell_size > 0 || throw(ArgumentError("cell_size must be positive, got $cell_size"))
    h = Float64(cell_size) / 2
    x = float(wrap_lon(Float64(lon)))
    y = Float64(lat)
    ring = GeoInterface.LinearRing([(x - h, y - h), (x + h, y - h), (x + h, y + h),
                                    (x - h, y + h), (x - h, y - h)])
    return GeoInterface.Polygon([ring])
end

"""
    era5_land_cell_key(lon, lat; cell_size = 0.1) -> Tuple{Int,Int}

Which ERA5-Land cell the point `(lon, lat)` falls in, as integer multiples of `cell_size`.

ERA5-Land cell centres sit exactly on multiples of 0.1°, so rounding a point's coordinates to the
nearest multiple names the cell whose box contains it. Longitude is [`wrap_lon`](@ref)ed first, so
a native 0–359.9°E centre and its `(-180, 180]` equivalent give the same key.

This is what lets one `bare_ice_albedo` call cover a whole tile: the MODIS cells it returns are
grouped by this key instead of the polygon of each grid cell being burned separately.

!!! note "Agreement with a per-cell burn is close, not exact"
    A per-cell burn tests each MODIS centre against the cell polygon *projected into the MODIS
    sinusoidal grid*, where a meridian edge becomes a curve that the projection chords
    vertex-to-vertex. Over 0.1° of latitude at 60°N that chord departs from the true meridian by
    about 3 m — 0.6 % of a 463 m cell — so the two agree except for a centre lying within a few
    metres of a cell edge, which goes to one of the two neighbouring cells either way.
"""
function era5_land_cell_key(lon, lat; cell_size::Real = 0.1)
    cell_size > 0 || throw(ArgumentError("cell_size must be positive, got $cell_size"))
    s = Float64(cell_size)
    return (round(Int, float(wrap_lon(Float64(lon))) / s), round(Int, Float64(lat) / s))
end

"""
    _bia_densified_box(xmin, xmax, ymin, ymax, step) -> GeoInterface.Polygon

A lon/lat box as a ring whose **meridian edges carry a vertex every `step` degrees** of latitude.

`bare_ice_albedo` reprojects a geometry into the MODIS sinusoidal grid and chords each edge
vertex-to-vertex. There `x = R·λ·cos(φ)`, so a parallel stays straight but a meridian becomes a
curve, and the chord across it falls inside the true edge by about `R·|λ|·cos(φ)·Δφ² / 8`. Over a
2.1° tile edge at 60°N and 141°W that is ~1.3 km — nearly three 463 m cells — which silently drops
MODIS cells along the tile's east and west margins. A vertex every 0.1° cuts it to a few metres,
since the error falls with the square of the segment.

The north and south edges are left as single segments because they need no densifying: latitude is
constant along them, so their projection is exactly straight.
"""
function _bia_densified_box(xmin::Real, xmax::Real, ymin::Real, ymax::Real, step::Real)
    step > 0 || throw(ArgumentError("step must be positive, got $step"))
    n = max(1, ceil(Int, (ymax - ymin) / step))
    ys = collect(range(Float64(ymin), Float64(ymax); length = n + 1))
    x0, x1 = Float64(xmin), Float64(xmax)
    # Counterclockwise from the southwest corner: south edge, east edge, north edge, west edge. The
    # west edge's last vertex is the southwest corner again, which closes the ring.
    pts = Tuple{Float64,Float64}[(x0, first(ys)), (x1, first(ys))]
    for y in ys[2:end]
        push!(pts, (x1, y))
    end
    push!(pts, (x0, last(ys)))
    for y in reverse(ys[1:end - 1])
        push!(pts, (x0, y))
    end
    return GeoInterface.Polygon([GeoInterface.LinearRing(pts)])
end

"""
    bare_ice_albedo_tile(tile, dem; kwargs...) -> (; per_cell, per_bin, hyps, census)

Bare-ice albedo binned by elevation for every cell a downscaling `tile` owns.

`tile` is one entry from [`downscaling_tiles`](@ref); only its `core` cells are reported, since a
cell's albedo profile is a property of that cell and not fitted from a neighbourhood. `dem` is a
lazy Copernicus DEM mosaic, e.g. `climate_model_invariant(model = :copernicus_dem_30m)`.

# How it is put together
One `bare_ice_albedo` call covers the whole tile rather than one per grid cell: each call
decompresses a hemisphere-wide table, so per-cell costs three orders of magnitude more for the
same cells. The returned MODIS centres are assigned to grid cells by
[`era5_land_cell_key`](@ref).

Elevations come from `Rasters.extract` at those centres, out of the DEM read one 1° window at a
time. Extracting straight from the lazy global mosaic instead costs one HTTP read per point —
tens of minutes for a single grid cell — whereas a window read is seconds and serves every point
in it.

# Keywords
- `bins = 0:100:10000`: elevation bin edges (m). Pass [`hypsometry_bin_edges`](@ref) of the
  elevation-class table to line the albedo bins up with the glacier-area bins.
- `sky = :bsa`, `min_cells = 3`, `extrapolate = 0`: passed to `bare_ice_albedo_hyps`.
- `boundary = :center`: how the tile polygon selects MODIS cells.
- `cell_size = 0.1`: ERA5-Land grid spacing, in degrees.
- `dem_fetch_concurrency = 8`: concurrent DEM window reads. GDAL's `/vsicurl` dataset open is not
  thread-safe under unbounded concurrency.
- `verbose = true`: progress bar and per-stage logging.

# Returns
- `per_cell`: one row per core cell — its coordinates, MODIS cell counts, median albedo, the fit,
  how many bins came from observation versus the fit, and the cell's glacier area.
- `per_bin`: long format, one row per (cell, bin) for every bin that holds either MODIS cells,
  glacier area, or a resolved albedo. Carries that bin's glacier area beside its albedo.
- `hyps`: the `BareIceAlbedoHyps` objects, aligned to `per_cell`'s rows.
- `census`: tile-level counts, including every reason a cell was dropped.

Longitudes in both tables are in the elevation-class table's native 0–359.9°E convention, as that
is what the forcing loader is keyed on.
"""
function bare_ice_albedo_tile(tile, dem;
                              bins = 0:100:10000,
                              sky::Symbol = :bsa,
                              min_cells::Integer = 3,
                              extrapolate::Union{Real,Symbol} = 0,
                              boundary::Symbol = :center,
                              cell_size::Real = 0.1,
                              dem_fetch_concurrency::Integer = 8,
                              verbose::Bool = true)
    core = tile.core
    nrow(core) > 0 || throw(ArgumentError(
        "tile $(tile.name) owns no cells, so there is nothing to derive"))

    s = _bia_sample(core, dem; label = tile.name, cell_size, boundary,
                    dem_fetch_concurrency, verbose)

    # One profile per core cell. The retrieval counts are handed over explicitly because the sliced
    # albedo is a plain vector and carries none — without them a bin resting on three retrievals is
    # indistinguishable from one resting on three thousand.
    edges = collect(Float64, bins)
    hyps = [bare_ice_albedo_hyps(s.albedo[idx], s.elevation[idx], edges;
                                 sky, min_cells, extrapolate, n_valid = s.n_valid[idx])
            for idx in s.points_of_cell]
    n_valid_of_row = [sum(Int, s.n_valid[idx]; init = 0) for idx in s.points_of_cell]

    per_cell = _bia_per_cell_table(core, hyps, s.points_of_cell, n_valid_of_row)
    per_bin = _bia_per_bin_table(core, hyps)

    census = (; s.census...,
              cells_with_albedo = count(h -> h.n_used > 0, hyps),
              cells_with_fit = count(h -> h.fit.n > 0, hyps),
              cells_with_observed_bin = count(h -> any(==(:observed), h.sources), hyps),
              bins_observed = sum(h -> count(==(:observed), h.sources), hyps),
              bins_fit = sum(h -> count(==(:fit), h.sources), hyps),
              bins_fit_beyond_qc = sum(_bia_fit_beyond_qc, hyps))
    if verbose
        @info "Tile $(tile.name) bare-ice albedo census" census...
        resolved = filter(!isnan, reduce(vcat, (h.albedo for h in hyps); init = Float64[]))
        isempty(resolved) ? @warn("no bin resolved anywhere in the tile") :
            @info "Resolved bins" n = length(resolved) albedo_min = round(minimum(resolved); digits = 3) albedo_median =
                round(median(resolved); digits = 3) albedo_max = round(maximum(resolved); digits = 3)
    end
    return (; per_cell, per_bin, hyps, census)
end

"""
    derive_bare_ice_albedo(cells; kwargs...) -> BareIceAlbedoHyps

**One** bare-ice albedo profile pooled over every glacierized MCD43A3 cell inside `cells`.

This is the tile-level downscaling parameter, the counterpart to [`derive_lapse_rate`](@ref) and
[`derive_decoupling_factor`](@ref): pass a tile's `buffered` cell selection
([`downscaling_tiles`](@ref)) and every elevation class of that tile reads its albedo from the one
profile returned here. At the 2°/1° default the buffered windows of neighbouring tiles overlap by
half their width, so the albedo varies smoothly across a tile seam rather than stepping at it.

Pooling is what makes the profile usable at every elevation. A single 0.1° cell spans a few hundred
metres and resolves a handful of the 100 m classes; a buffered tile spans the whole hypsometry, so
with `extrapolate = :hold` (the default here) every class gets a value drawn from real observations.

# Keywords
- `dem = nothing`: a lazy Copernicus DEM raster. Left `nothing` this builds one itself over the
  cells' own extent with `cache_tiles = true`, which is what a sweep wants — see below.
- `bins = 0:100:10000`: elevation bin edges (m). Pass [`hypsometry_bin_edges`](@ref) of the
  elevation-class table so albedo bins and glacier-area bins are the same bins.
- `sky = :bsa`, `min_cells = 3`, `extrapolate = :hold`: passed to `bare_ice_albedo_hyps`.
  `:hold` carries the end bins outward, so the answer outside the sampled range is still a value
  some bin measured.
- `boundary = :center`, `cell_size = 0.1`, `dem_fetch_concurrency = 8`, `verbose = true`: as for
  [`bare_ice_albedo_tile`](@ref).
- `dem_cache_path = nothing`, `dem_cache_tiles = true`: only read when `dem === nothing`.

# The DEM, and why the extent matters
With `dem === nothing` this calls

    climate_model_invariant(model = :copernicus_dem_30m, extent = <the cells' box>,
                            cache_path = dem_cache_path, cache_tiles = dem_cache_tiles)

`cache_tiles` writes whole 1° tiles to `cache_path/tiles/`, so a second pass over the same region —
a neighbouring tile, or a re-run — touches no network. It is bounded **only** because an `extent` is
given: on the global mosaic it would try to fetch every published tile. A 4°×4° buffered window is
~16–25 tiles at 19–40 MB each, and adjacent tiles reuse most of them.

`cache_tiles = true` requires a durable `cache_path` and throws otherwise, since tiles written under
`tempdir()` are reaped. Pass one, or set `ENV["GEMB_CACHE_PATH"]`.
"""
function derive_bare_ice_albedo(cells;
                                dem = nothing,
                                bins = 0:100:10000,
                                sky::Symbol = :bsa,
                                min_cells::Integer = 3,
                                extrapolate::Union{Real,Symbol} = :hold,
                                boundary::Symbol = :center,
                                cell_size::Real = 0.1,
                                dem_fetch_concurrency::Integer = 8,
                                dem_cache_path::Union{Nothing,AbstractString} = nothing,
                                dem_cache_tiles::Bool = true,
                                label::AbstractString = "region",
                                verbose::Bool = true)
    nrow(cells) > 0 || throw(ArgumentError(
        "no cells supplied, so there is no region to pool albedo over"))

    if isnothing(dem)
        box = _bia_cells_box(cells, Float64(cell_size))
        dem = climate_model_invariant(; model = :copernicus_dem_30m,
                                      extent = Extents.Extent(X = (box.xmin, box.xmax),
                                                              Y = (box.ymin, box.ymax)),
                                      cache_path = dem_cache_path,
                                      cache_tiles = dem_cache_tiles, verbose)
    end

    s = _bia_sample(cells, dem; label, cell_size, boundary, dem_fetch_concurrency, verbose)
    f = bare_ice_albedo_hyps(s.albedo, s.elevation, collect(Float64, bins);
                             sky, min_cells, extrapolate, n_valid = s.n_valid)
    if verbose
        @info "Pooled bare-ice albedo for $(label)" cells = nrow(cells) s.census...
        show(stderr, MIME"text/plain"(), f)
        println(stderr)
    end
    return f
end

# --------------------------------------------------------- resolving onto an elevation class

"""
    resolve_albedo_ice(profile, z; default, range = GEMB_ALBEDO_ICE_RANGE)
        -> (albedo::Float64, source::Symbol)

The `albedo_ice` to run one elevation class at, and where it came from.

`profile` is a `BareIceAlbedoHyps` from [`derive_bare_ice_albedo`](@ref), or `nothing`; `z` is the
class's centre elevation (m); `default` is the albedo to fall back on — pass `mp.albedo_ice`, so a
caller who tuned it gets their own value rather than a constant hidden here.

Resolution, in order:

1. The profile's value for the bin holding `z`, tagged with that bin's own
   [`BARE_ICE_ALBEDO_SOURCES`](@ref) provenance — `:observed`, `:hold` or `:fit`.
2. Clamped into `range`, and if the clamp moved it the source becomes `:clamped`. This is what keeps
   every class runnable: observed bare-ice albedo above the equilibrium line exceeds `0.6`, which
   `GEMB.validate_parameters` refuses.
3. `default` / `:default` when the profile is absent, has nothing for that bin, or `z` falls outside
   its binning altogether.

`default` is itself clamped, and reported as `:clamped` if that moved it, so this can never return a
value GEMB will reject.

# Example
```julia
f = derive_bare_ice_albedo(tile.buffered)
mp = initialize_parameters()
for band in hypsometry_intervals(tile.core)
    albedo, source = resolve_albedo_ice(f, band.center; default = mp.albedo_ice)
    @info "band" band.center albedo source
end
```
"""
function resolve_albedo_ice(profile, z::Real; default::Real,
                            range::Tuple{Real,Real} = GEMB_ALBEDO_ICE_RANGE)
    lo, hi = Float64(range[1]), Float64(range[2])
    lo < hi || throw(ArgumentError(
        "range must be (low, high) with low < high, got $(range)"))
    isfinite(default) || throw(ArgumentError(
        "default albedo_ice must be finite, got $(default); pass `mp.albedo_ice`"))

    value, source = NaN, :default
    if !isnothing(profile)
        b = _bia_hyps_bin(profile, z)
        if b != 0 && isfinite(profile.albedo[b])
            value, source = profile.albedo[b], profile.sources[b]
        end
    end
    isnan(value) && (value = Float64(default))

    clamped = clamp(value, lo, hi)
    # `:clamped` replaces the provenance rather than qualifying it: what GEMB ran on is the edge of
    # the range, not the bin's albedo, and a census that still called it `:observed` would overstate
    # how much of the tile the observation actually set.
    clamped == value || (source = :clamped)
    return (clamped, source)
end

"""
    _with_albedo_ice(mp, albedo) -> ModelParameters

`mp` with `albedo_ice` replaced, everything else identical.

`ConstructionBase.setproperties` rather than a rebuild through `initialize_parameters`:
`GEMB.ModelParameters` is an immutable `@kwdef` struct carrying a *derived* field (`dt_divisors`),
which a field-by-field reconstruction would either drop or recompute wrongly.

The result is deliberately **not** re-validated. `albedo` has already been through
[`resolve_albedo_ice`](@ref), which clamps into the range `GEMB.validate_parameters` asserts, and
`mp` was validated when the caller built it — so a `validate_parameters` call here would be a full
53-field pass per band per perturbation for a field that cannot be out of range.
"""
_with_albedo_ice(mp, albedo::Real) =
    ConstructionBase.setproperties(mp, (; albedo_ice = Float64(albedo)))

"""
    _record_bare_ice_albedo!(params, resolved)

Record the resolved bare-ice albedo of every band or bin in a run's stored parameters.

`resolved` is a vector of `(albedo, source)` pairs, as [`resolve_albedo_ice`](@ref) returns — the one
shape both the tile and the cell path produce, so the three attribute names and the code encoding are
written in exactly one place.

The values go in because `model_albedo_ice` cannot express them: that field is only the fallback for a
band the observations did not resolve. Re-deriving the albedo therefore invalidates a restart, as it
must — the same forcing under a different `albedo_ice` per band is a different experiment. The sources
go in so a finished file answers "which classes were measured and which fell back" on its own, without
the downscaling tile it came from.

Sources are stored as [`BARE_ICE_ALBEDO_SOURCES`](@ref) codes with the vocabulary beside them, since a
NetCDF attribute holds no symbols. An empty `resolved` writes nothing at all: a run with no albedo
product must still compare as current against the file it wrote before these keys existed.
"""
function _record_bare_ice_albedo!(params::AbstractDict, resolved)
    isempty(resolved) && return params
    params["applied_bare_ice_albedo"] = [Float64(first(r)) for r in resolved]
    params["applied_bare_ice_albedo_source"] =
        [bare_ice_albedo_source_code(last(r)) for r in resolved]
    params["applied_bare_ice_albedo_source_meanings"] = join(BARE_ICE_ALBEDO_SOURCES, " ")
    params["applied_bare_ice_albedo_comment"] =
        "One entry per band the downscaling resolved, in ascending elevation. This is the whole " *
        "resolution and not the set of bands a run produced, so it is what a continuation is " *
        "compared against; it is NOT indexed by the band dimension, which omits any band the " *
        "forcing could not drive. Read band_bare_ice_albedo for the per-band values of this file."
    return params
end

"""
    bare_ice_albedo_source_code(source) -> Int

`source`'s index in [`BARE_ICE_ALBEDO_SOURCES`](@ref), or `0` for one that is not in it.

Integer codes rather than symbols because these are stored as a NetCDF attribute on a run's output
and compared on restart. `bare_ice_albedo_source_name` is the inverse, so a stored vector can be read
back without the vocabulary being spelled a second time.
"""
bare_ice_albedo_source_code(source::Symbol) =
    something(findfirst(==(source), BARE_ICE_ALBEDO_SOURCES), 0)

"""
    bare_ice_albedo_source_name(code) -> Symbol

The [`BARE_ICE_ALBEDO_SOURCES`](@ref) entry a stored code names, or `:unknown` for one written by a
version with a different vocabulary — which must not be silently read as a source that *is* in this
one.

Takes a `Real`, not just an `Integer`: a code vector stored as a NetCDF attribute comes back as
`Float64`, because `_encode_attribute` normalizes every numeric attribute so that a cycle given as
integers and the same cycle given as floats compare equal on restart.
"""
function bare_ice_albedo_source_name(code::Real)
    isfinite(code) && isinteger(code) || return :unknown
    c = Int(code)
    return 1 <= c <= length(BARE_ICE_ALBEDO_SOURCES) ? BARE_ICE_ALBEDO_SOURCES[c] : :unknown
end


# ------------------------------------------------------------------------- shared sampling

# The lon/lat box a cell selection occupies, padded by half a grid cell so every cell's own box is
# inside it. Longitudes are wrapped to (-180, 180], which is what both `bare_ice_albedo` and the
# Copernicus DEM use.
function _bia_cells_box(cells, cell_size::Float64)
    h = cell_size / 2
    lons = [float(wrap_lon(Float64(x))) for x in cells.longitude]
    lats = [Float64(y) for y in cells.latitude]
    xmin, xmax = minimum(lons) - h, maximum(lons) + h
    ymin, ymax = max(-90.0, minimum(lats) - h), min(90.0, maximum(lats) + h)
    # The sinusoidal projection cannot hold an antimeridian-crossing ring as one shape, and a
    # silently wrong answer there would select cells across every intervening MODIS tile.
    (xmin < -180.0 || xmax > 180.0) && throw(ArgumentError(
        "this cell selection reaches past the antimeridian ($(round(xmin; digits=2))° … " *
        "$(round(xmax; digits=2))°), which the MODIS sinusoidal projection cannot represent as " *
        "one polygon. Split it at ±180° and derive each part."))
    return (; xmin, xmax, ymin, ymax)
end

# Every glacierized MCD43A3 cell inside `cells`, with an elevation each.
#
# Returns compacted vectors — only the cells that survived — plus `points_of_cell[r]`, the indices
# into them belonging to row `r` of `cells`. The pooled derivation uses the vectors whole and the
# per-cell one slices them, so the burn, the grouping and the DEM reads happen once either way.
#
# One `bare_ice_albedo` call covers the whole selection rather than one per grid cell: each call
# decompresses a hemisphere-wide table, so per-cell costs three orders of magnitude more for the
# same cells. Which cell a MODIS centre belongs to is then decided by `era5_land_cell_key`.
function _bia_sample(cells, dem; label::AbstractString, cell_size::Real,
                     boundary::Symbol, dem_fetch_concurrency::Integer, verbose::Bool)
    box = _bia_cells_box(cells, Float64(cell_size))
    poly = _bia_densified_box(box.xmin, box.xmax, box.ymin, box.ymax, Float64(cell_size))
    verbose && @info "Querying bare-ice albedo for $(label)" cells = nrow(cells) bounds =
        (round(box.xmin; digits = 2), round(box.xmax; digits = 2),
         round(box.ymin; digits = 2), round(box.ymax; digits = 2))

    # A selection with no glacierized MODIS cell at all is a real outcome, not an error: ERA5-Land's
    # glacier mask and the RGI 7.0 outlines do not agree everywhere.
    a = try
        bare_ice_albedo(poly; boundary)
    catch err
        err isa ArgumentError && occursin("selects no MCD43A3 cell", err.msg) || rethrow()
        @warn "$(label) selects no MCD43A3 cell; every elevation bin is unresolved"
        nothing
    end

    n_cells = nrow(cells)
    row_of_key = Dict(era5_land_cell_key(cells.longitude[i], cells.latitude[i]; cell_size) => i
                      for i in 1:n_cells)
    points_of_cell = [Int[] for _ in 1:n_cells]
    keep = Int[]
    n_modis_total = 0
    n_off_product = 0
    n_outside_cells = 0
    if !isnothing(a)
        lon = parent(a[:longitude])
        lat = parent(a[:latitude])
        # Screened on black-sky albedo whatever `sky` the caller reduces: the two share every
        # retrieval, so a cell unresolved in one is unresolved in the other.
        alb = parent(a[:albedo_bsa])
        in_product = parent(a[:in_product])
        n_modis_total = length(lon)
        for i in 1:n_modis_total
            if !in_product[i] || isnan(alb[i])
                n_off_product += 1
                continue
            end
            r = get(row_of_key, era5_land_cell_key(lon[i], lat[i]; cell_size), 0)
            if r == 0
                n_outside_cells += 1
            else
                push!(keep, i)
                push!(points_of_cell[r], length(keep))
            end
        end
    end

    albedo_bsa = isnothing(a) ? Float64[] : Float64.(parent(a[:albedo_bsa])[keep])
    albedo_wsa = isnothing(a) ? Float64[] : Float64.(parent(a[:albedo_wsa])[keep])
    n_valid = isnothing(a) ? Int[] : Int.(parent(a[:n_valid_bsa])[keep])
    elevation = fill(NaN, length(keep))
    n_no_dem = 0

    if !isempty(keep)
        lon = parent(a[:longitude])[keep]
        lat = parent(a[:latitude])[keep]
        windows = Dict{Tuple{Int,Int},Vector{Int}}()
        for j in eachindex(lon)
            push!(get!(() -> Int[], windows, _tile_key(lon[j], lat[j])), j)
        end
        window_list = collect(windows)
        verbose && @info "Reading Copernicus DEM" points = length(keep) windows =
            length(window_list)

        pool = Base.Semaphore(dem_fetch_concurrency)
        prog = Progress(length(window_list); desc = "DEM windows: ", enabled = verbose)
        Threads.@threads for w in eachindex(window_list)
            key, idx = window_list[w]
            # Pad by a whole DEM pixel so a point on the window's own edge is still covered.
            _bia_fill_elevation!(elevation, dem, idx, lon, lat, key, 1 / 3600, pool)
            next!(prog)
        end
        finish!(prog)
        n_no_dem = count(isnan, elevation)
        n_no_dem > 0 && verbose &&
            @warn "Copernicus DEM has no value at some MODIS cells" points = n_no_dem of =
                length(keep)
    end

    census = (; region = String(label), cells = n_cells,
              modis_cells = n_modis_total,
              modis_off_product = n_off_product,
              modis_outside_cells = n_outside_cells,
              modis_used = length(keep),
              modis_no_dem = n_no_dem)
    return (; albedo = albedo_bsa, albedo_wsa, n_valid, elevation, points_of_cell, census)
end

# Read one 1° DEM window and write the elevations of the points in `idx` into `elevation`.
#
# Threads call this with disjoint `idx`, so the shared `elevation` writes never overlap. A tile the
# Copernicus DEM does not publish (it omits ocean) leaves its points `NaN` rather than taking the
# whole sweep down, which is the same concession `tile_hypsometry!` makes.
function _bia_fill_elevation!(elevation, dem, idx, lon, lat, key, pad, pool)
    xlo, xhi = key[1] - pad, key[1] + 1 + pad
    ylo, yhi = key[2] - pad, key[2] + 1 + pad
    Base.acquire(pool)
    window = try
        read(view(dem, X = (xlo .. xhi), Y = (ylo .. yhi)))
    catch err
        (err isa ArgumentError && occursin("No Copernicus DEM tiles", err.msg)) || rethrow()
        return nothing
    finally
        Base.release(pool)
    end
    pts = [GeoInterface.Point(float(wrap_lon(lon[i])), Float64(lat[i])) for i in idx]
    sampled = extract(window, pts; geometry = false, progress = false)
    for (j, i) in enumerate(idx)
        v = first(sampled[j])
        elevation[i] = (ismissing(v) || !isfinite(v)) ? NaN : Float64(v)
    end
    return nothing
end

# One row per core cell: coverage, the median albedo actually resolved, and the fit.
function _bia_per_cell_table(core, hyps, cells_of_row, n_valid_of_row)
    n = nrow(core)
    df = DataFrame()
    df[!, :longitude] = collect(core.longitude)
    df[!, :latitude] = collect(core.latitude)
    df[!, :glacier_area_km2] = glacier_area_column(core)
    df[!, :n_modis_cells] = [length(cells_of_row[r]) for r in 1:n]
    df[!, :n_used] = [hyps[r].n_used for r in 1:n]
    df[!, :n_no_elevation] = [hyps[r].n_no_elevation for r in 1:n]
    df[!, :n_valid] = n_valid_of_row
    df[!, :n_bins_observed] = [count(==(:observed), hyps[r].sources) for r in 1:n]
    df[!, :n_bins_fit] = [count(==(:fit), hyps[r].sources) for r in 1:n]
    df[!, :albedo_median] = [_bia_median(hyps[r].albedo) for r in 1:n]
    df[!, :albedo_min] = [_bia_reduce(minimum, hyps[r].albedo) for r in 1:n]
    df[!, :albedo_max] = [_bia_reduce(maximum, hyps[r].albedo) for r in 1:n]
    df[!, :slope_per_km] = [hyps[r].fit.slope_per_km for r in 1:n]
    df[!, :intercept] = [hyps[r].fit.intercept for r in 1:n]
    df[!, :slope_stderr] = [hyps[r].fit.slope_stderr for r in 1:n]
    df[!, :r2] = [hyps[r].fit.r2 for r in 1:n]
    df[!, :fit_n] = [hyps[r].fit.n for r in 1:n]
    df[!, :elevation_min] = [hyps[r].fit.elevation_range[1] for r in 1:n]
    df[!, :elevation_max] = [hyps[r].fit.elevation_range[2] for r in 1:n]
    return df
end

_bia_reduce(f, v) = (w = filter(!isnan, v); isempty(w) ? NaN : f(w))
_bia_median(v) = _bia_reduce(median, v)

# Fit-filled bins whose albedo lies outside the range any single retrieval was allowed to take
# (`POOLED_ICE_ALBEDO_RANGE`, currently (0.25, 1.0)).
#
# A linear fit can leave that envelope even inside the elevation span that informed it — a steep
# slope over a kilometre of relief is enough. Such a value is not a measurement of anything: no
# observation behind it was that dark or that bright. Counted rather than clamped, because whether
# to clamp, refit, or drop the cell depends on what the profile is being used for.
function _bia_fit_beyond_qc(f::BareIceAlbedoHyps)
    lo, hi = POOLED_ICE_ALBEDO_RANGE
    return count(b -> f.sources[b] === :fit && !isnan(f.albedo[b]) &&
                      (f.albedo[b] < lo || f.albedo[b] > hi), eachindex(f.albedo))
end

# Long format, one row per (cell, bin). Only bins that carry something are emitted: a full outer
# product of cells and a 0-10000 m binning is mostly empty, and an empty row says nothing that the
# absence of the row does not.
function _bia_per_bin_table(core, hyps)
    lon = Float64[]; lat = Float64[]
    lo = Int[]; hi = Int[]; center = Float64[]
    albedo = Float64[]; observed = Float64[]
    n_cells = Int[]; n_valid = Int[]; source = String[]
    area = Float64[]
    for r in 1:nrow(core)
        f = hyps[r]
        row = core[r, :]
        # Every bin of the row's hypsometry, zero-area ones included, so a bin with albedo but no
        # glacier area is visible rather than dropped.
        area_of = Dict((h.lo, h.hi) => h.area for h in glacier_hypsometry(row; area_minimum = -1))
        t = bare_ice_albedo_table(f)
        for b in eachindex(t.lo)
            key = (Int(t.lo[b]), Int(t.hi[b]))
            bin_area = get(area_of, key, 0.0)
            (t.n_cells[b] > 0 || bin_area > 0 || !isnan(t.albedo[b])) || continue
            push!(lon, Float64(row.longitude)); push!(lat, Float64(row.latitude))
            push!(lo, key[1]); push!(hi, key[2]); push!(center, t.center[b])
            push!(albedo, t.albedo[b]); push!(observed, t.observed[b])
            push!(n_cells, t.n_cells[b]); push!(n_valid, t.n_valid[b])
            push!(source, String(t.source[b])); push!(area, bin_area)
        end
    end
    return DataFrame(; longitude = lon, latitude = lat, lo, hi, center,
                     albedo, observed, n_cells, n_valid, source, glacier_area_km2 = area)
end
