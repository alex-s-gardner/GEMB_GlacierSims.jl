# Tiling a global glacier elevation-class table into the regions the downscaling-parameter fits are
# derived over, and sweeping those regions.
#
# `derive_downscaling_parameters` fits one region at a time. This file is what turns "every glacier
# grid cell on Earth" into a list of regions: a `tile_size`° grid whose cells partition the table,
# each fitted from a `buffer`°-widened selection. The buffer is not decoration — both fits are
# regressions *across* cells, so a bare 2° tile often carries too little elevation and glacier-mask
# range to constrain a slope at all, while its 1°-widened neighbourhood does.
#
# Terminology, kept distinct from `downscaling_parameters.jl`'s because a third object appears here:
#   grid cell — one row of the table, i.e. one 0.1° ERA5-Land cell.
#   tile      — one `tile_size`° box. Its *core* cells are those it owns; its *buffered* cells are
#               those within `buffer` of it, which is what the fits actually see.
#
# **Why the tiling is arithmetic rather than a polygon per tile.** `grid_cells_in_region` is the
# right tool for a caller's arbitrary geometry and the wrong one for a partition, for two reasons
# that are both properties of this data rather than of that function:
#
#   1. `GO.contains` is strictly interior (pinned in the test suite). Tile edges are whole degrees
#      and ERA5-Land cell centers are 0.1° multiples, so *every* cell sitting exactly on a tile edge
#      would fall into no tile at all. For an arbitrary ROI that is the correct conservative answer;
#      for "derive parameters for all cells" it is silent data loss.
#   2. A tile against the antimeridian has a buffer that crosses it, and a region spanning the seam
#      is precisely what `grid_cells_in_region` refuses (a >180° extent is either an antimeridian
#      crossing or a convention mistake, and it cannot tell which).
#
# `fld(wrap_lon(lon), tile_size) * tile_size` has neither problem: it is total (every cell lands in
# exactly one tile), it is exact at the edges (no floating-point containment test), and taking
# longitude *separations* modulo 360 makes the seam ordinary arithmetic rather than a special case.

# Signed shortest separation between two longitudes, in degrees, in `[-180, 180)`.
#
# This is the whole of the antimeridian handling. A cell at 179.9°E and a tile centred on 179.0°W
# are 1.1° apart, not 358.9°, and only the modular form says so — which is why buffer membership is
# expressed as a distance from the tile centre rather than as a pair of longitude bounds a cell must
# lie between. Bounds would need the seam split into two intervals; a distance does not.
_lon_delta(a, b) = mod(a - b + 180.0, 360.0) - 180.0

"""
    tile_index(lon, lat; tile_size = 2) -> (Int, Int)

The tile a cell centre belongs to, as the integer degrees of the tile's southwest corner.

`lon` may be in either the table's native 0–359.9°E convention or the `(-180, 180]` one; it is
[`wrap_lon`](@ref)ed first, so the returned index is always on `(-180, 180]`. With
`tile_size = 2` the longitude index runs -180, -178, … 178 and the latitude index -90, -88, … 88.

Assignment is `fld`, so it is **total and non-overlapping**: every cell centre maps to exactly one
tile, including one sitting exactly on a tile edge (which belongs to the tile it is the *lower*
edge of). That totality is the point — see this file's header for why the tiling is not done with
polygons and [`grid_cells_in_region`](@ref).
"""
function tile_index(lon, lat; tile_size::Real = 2)
    _check_tile_size(tile_size)
    ix = Int(fld(wrap_lon(Float64(lon)), tile_size) * tile_size)
    # `wrap_lon` deliberately keeps 180.0 as 180.0 rather than folding it to -180 (it is a real grid
    # column of the Zarr forcing, one cell away from -180.0), so a cell exactly on the antimeridian
    # lands on index 180 — a tile that does not otherwise exist, since the indices run to
    # `180 - tile_size`. Folded here: 180 ≡ -180, and the tile starting at -180 is the one whose
    # lower edge that cell sits on. Without this the tile column at +180 would be a duplicate of the
    # one at -180 and the partition would gain a spurious, nearly-empty tile.
    ix == 180 && (ix = -180)
    return (ix, Int(fld(Float64(lat), tile_size) * tile_size))
end

"""
    tile_bounds(index; tile_size = 2, buffer = 0) -> (; lon_min, lon_max, lat_min, lat_max)

The bounds of the tile at `index`, optionally widened by `buffer` degrees.

Longitude bounds are **not** wrapped back onto `(-180, 180]` when a buffer takes them past the
seam: the tile at longitude index -180 buffered by 1° reports `lon_min = -181.0`, which is the
honest description of the window and reads correctly as a span. Do not test cell membership against
these bounds — use the distance form the tiler uses (`_lon_delta`), which handles the seam. They
are here to be *recorded*, so a written tile file says what geometry produced it.

Latitude bounds are clamped to `[-90, 90]`, since there is no cell beyond a pole to include.
"""
function tile_bounds(index; tile_size::Real = 2, buffer::Real = 0)
    _check_tile_size(tile_size)
    buffer >= 0 || throw(ArgumentError("buffer must be >= 0, got $buffer"))
    b = Float64(buffer)
    return (lon_min = index[1] - b, lon_max = index[1] + tile_size + b,
            lat_min = max(-90.0, index[2] - b),
            lat_max = min(90.0, index[2] + tile_size + b))
end

# A tile size has to be a whole number of degrees that divides 360.
#
# Integer, because `tile_output_name` spells a tile with integer degrees and a fractional size would
# let two tiles collide on one filename. Divides 360, because otherwise the column of tiles against
# +180 is narrower than every other and its fit covers a different-sized region while claiming the
# same `tile_size` — one anomalous tile in a global sweep is worse than a refusal at the call that
# would have produced it.
function _check_tile_size(tile_size::Real)
    # An integer-valued Float64 (`2.0`) is accepted — a caller writing `tile_size = 2.0` means the
    # same grid — but a fractional one is not, for the naming reason below.
    isinteger(tile_size) || throw(ArgumentError(
        "tile_size must be a whole number of degrees, got $tile_size: tile file names " *
        "(`tile_output_name`) spell whole degrees, so a fractional size would let two tiles " *
        "collide on one name"))
    tile_size > 0 || throw(ArgumentError("tile_size must be positive, got $tile_size"))
    360 % tile_size == 0 || throw(ArgumentError(
        "tile_size must divide 360 evenly, got $tile_size: otherwise the tiles against the " *
        "antimeridian are narrower than the rest while reporting the same tile_size. Use one of " *
        "$(join([d for d in 1:180 if 360 % d == 0], ", "))."))
    return nothing
end

"""
    downscaling_tiles(glacier_elevation_classes; tile_size = 2, buffer = 1,
                      area_minimum = 0.0, order = :chunk)

Tile a glacier elevation-class table into the regions [`derive_downscaling_parameters`](@ref) is
derived over, one entry per tile holding at least one qualifying grid cell.

Each entry is

    (; index, bounds, buffered_bounds, core, buffered, core_rows, buffered_rows, name)

- `index`: the tile's southwest corner in whole degrees, as [`tile_index`](@ref) returns.
- `bounds`, `buffered_bounds`: the tile's own and its widened extent ([`tile_bounds`](@ref)).
- `core`: the cells this tile **owns**, as a `SubDataFrame` view. Across the returned tiles the
  cores partition the (area-screened) table: every qualifying row appears in exactly one.
- `buffered`: the cells within `buffer`° of the tile — what the fits are derived from. A superset
  of `core`. Buffered sets of neighbouring tiles overlap; that is the point of them.
- `core_rows`, `buffered_rows`: the row indices those two views select, **into the frame passed to
  this function**. Provided because `parentindices` of a view resolves all the way to the ultimate
  parent, which is a different frame when the caller passed a view of a larger table.
- `name`: the tile's file name stem ([`tile_output_name`](@ref)).

`core` is what a tile's parameters *apply to* and `buffered` is what they are *fitted from*. Keeping
them separate is what lets the fits see a wide neighbourhood while each cell's glacier area is still
attributed to exactly one tile.

# Keywords
- `tile_size = 2`: grid spacing in whole degrees; must divide 360 (see [`tile_index`](@ref)).
- `buffer = 1`: how far past the tile to reach for cells to fit from. `0` is legal and makes
  `buffered == core`.
- `area_minimum = 0.0`: skip cells holding less than this total glacier area (km²), via
  [`glacier_area_column`](@ref). Applied before tiling, so a screened-out cell is in no tile's core
  *or* buffer and the partition is over the qualifying rows.
- `order = :chunk`: tile order. `:chunk` groups by the tile's dominant `chunk_id` so a sweep reads
  each ERA5-Land download chunk while it is still warm — neighbouring tiles share buffered cells,
  so the locality is real. `:lonlat` sorts geographically, `:none` leaves discovery order.

Both selections are views over the passed table with ascending row indices, so a `chunk_id`-sorted
table stays sorted within every tile and a streaming pass over one keeps its Zarr cache locality.

The table must carry `:latitude`/`:longitude` columns, since that is what the forcing loader is
keyed on; see the error message for the two lines that add them.
"""
function downscaling_tiles(glacier_elevation_classes; tile_size::Real = 2, buffer::Real = 1,
                           area_minimum::Real = 0.0, order::Symbol = :chunk)
    _check_tile_size(tile_size)
    buffer >= 0 || throw(ArgumentError("buffer must be >= 0, got $buffer"))
    order in (:chunk, :lonlat, :none) ||
        throw(ArgumentError("order must be :chunk, :lonlat or :none, got :$order"))
    # Checked up front rather than at the first forcing load: without these columns the failure is a
    # `FieldError` from inside the loader closure, tens of thousands of cell loads into a sweep.
    for col in (:latitude, :longitude)
        hasproperty(glacier_elevation_classes, col) || throw(ArgumentError(
            "glacier elevation-class table has no `:$col` column, which the forcing loader is " *
            "keyed on. The cached table carries only the Point geometry; add both with\n" *
            "    table[!, :longitude] = GeoInterface.x.(table.geometry)\n" *
            "    table[!, :latitude]  = GeoInterface.y.(table.geometry)\n" *
            "leaving longitudes in the table's native 0-359.9°E convention."))
    end

    n = nrow(glacier_elevation_classes)
    lat = convert(Vector{Float64}, glacier_elevation_classes[!, :latitude])
    # Wrapped once here, into a local vector: the table keeps its native 0-359.9°E longitudes
    # (that is what `climate_forcing` is called with), and every geometric decision below is on
    # `(-180, 180]`. Doing it per comparison instead would wrap the same value ~800 times.
    lonw = [wrap_lon(Float64(x)) for x in glacier_elevation_classes[!, :longitude]]

    # As a column, not row by row — on the global table this is ~40 ms against ~1.8 s, and it runs
    # before any tiling so a screened-out cell is in no tile at all.
    qualifying = area_minimum > 0 ? glacier_area_column(glacier_elevation_classes) .>= area_minimum :
                 trues(n)

    # Core assignment: one bucket per tile, filled in row order so every bucket stays ascending.
    cores = Dict{Tuple{Int,Int},Vector{Int}}()
    for i in 1:n
        qualifying[i] || continue
        push!(get!(() -> Int[], cores, tile_index(lonw[i], lat[i]; tile_size)), i)
    end
    isempty(cores) && return NamedTuple[]

    # Buffered selection, bucketed by tile so the scan is over a neighbourhood rather than the whole
    # table per tile. A cell can only be within `buffer` of a tile whose index is within
    # `ceil(buffer / tile_size)` tiles of its own, so only those candidate buckets are examined —
    # which keeps this O(n * (2r+1)^2) rather than O(n * n_tiles), the difference between
    # milliseconds and minutes on a global table with a fine `tile_size`.
    reach = ceil(Int, Float64(buffer) / tile_size)
    lon_span = tile_size / 2 + Float64(buffer)
    n_lon = 360 ÷ tile_size

    tiles = NamedTuple[]
    for (index, core_rows) in cores
        lon_center = index[1] + tile_size / 2
        bounds = tile_bounds(index; tile_size)
        buffered_bounds = tile_bounds(index; tile_size, buffer)

        buffered_rows = Int[]
        if buffer == 0
            # `buffered == core` exactly, and cheaply: no scan, and the same view object semantics
            # as every other tile so a caller cannot tell the degenerate case apart structurally.
            buffered_rows = core_rows
        else
            # Candidate tiles within reach, deduplicated: at a coarse `tile_size` the longitude
            # neighbourhood can wrap onto itself, and a bucket visited twice would list its rows
            # twice in `buffered`.
            seen = Set{Tuple{Int,Int}}()
            for dlat in -reach:reach, dlon in -reach:reach
                # Longitude index wraps around the globe; latitude does not (there is nothing past
                # a pole), so an out-of-range latitude neighbour is simply absent.
                jx = index[1] + dlon * tile_size
                jx = Int(mod(jx + 180, 360) - 180)
                key = (jx, index[2] + dlat * tile_size)
                key in seen && continue
                push!(seen, key)
                rows = get(cores, key, nothing)
                rows === nothing && continue
                for i in rows
                    abs(_lon_delta(lonw[i], lon_center)) <= lon_span &&
                        bounds.lat_min - buffer <= lat[i] <= bounds.lat_max + buffer &&
                        push!(buffered_rows, i)
                end
            end
            # Row order, not bucket order: a `chunk_id`-sorted table must stay sorted inside the
            # view, which is what keeps a streaming pass over it cache-local.
            sort!(buffered_rows)
        end

        push!(tiles, (; index, bounds, buffered_bounds,
                      core = view(glacier_elevation_classes, core_rows, :),
                      buffered = view(glacier_elevation_classes, buffered_rows, :),
                      # Row indices into the frame that was passed in. Carried explicitly because
                      # `parentindices` of the views above resolves to the ultimate parent, which is a
                      # different frame whenever the caller passed a view of a larger table.
                      core_rows = core_rows, buffered_rows = buffered_rows,
                      name = tile_output_name(index)))
    end

    return _order_tiles!(tiles, order)
end

# Tile sweep order. `:chunk` is the default because it is the one that costs nothing and saves
# real time: neighbouring tiles share buffered cells, so visiting a download chunk's tiles together
# reads each chunk's forcing while it is still in the local cache.
function _order_tiles!(tiles, order::Symbol)
    order === :none && return tiles
    if order === :lonlat
        sort!(tiles; by = t -> (t.index[2], t.index[1]))
    else
        sort!(tiles; by = t -> (_dominant_chunk_id(t.buffered), t.index[2], t.index[1]))
    end
    return tiles
end

# The chunk most of a tile's buffered cells come from, as a sort key. Mode rather than minimum: a
# tile straddling two chunks should sort with the one it mostly reads. A table without the column
# (or with `missing`) sorts as 0, which just degrades this to a geographic sort.
function _dominant_chunk_id(cells)
    hasproperty(cells, :chunk_id) || return 0
    counts = Dict{Int,Int}()
    for v in cells[!, :chunk_id]
        ismissing(v) && continue
        c = Int(v)
        counts[c] = get(counts, c, 0) + 1
    end
    isempty(counts) && return 0
    # `argmax` over pairs would pick by count then by an arbitrary key; break ties on the smaller
    # chunk id so the order is deterministic across runs and platforms.
    best, best_n = 0, -1
    for (c, k) in counts
        (k > best_n || (k == best_n && c < best)) && ((best, best_n) = (c, k))
    end
    return best
end

# ------------------------------------------------------------------------------------------------
# the sweep
# ------------------------------------------------------------------------------------------------

"""
    derive_downscaling_parameter_tiles(climate_model, time_range, glacier_elevation_classes,
                                       output_dir; token, cache_path, kwargs...) -> summary

Derive downscaling parameters for **every** grid cell of a glacier elevation-class table, tile by
tile, writing one netCDF-4 (HDF5) file per tile into `output_dir`.

Every cell of the table belongs to exactly one tile's core ([`downscaling_tiles`](@ref)), so the
tile files together cover the whole table; `output_dir/tiles_index.parquet` records that mapping as
one row per cell, and the sweep asserts its row count against the table's before writing it.

Returns a `DataFrame` summary, one row per tile: `name`, `index_lon`, `index_lat`, `n_cells_core`,
`n_cells_buffered`, `n_cells_used`, `n_timesteps`, `n_decoupling_factor_fitted`,
`n_lapse_rate_fitted`, `n_elevation_intervals`, `status`, `sparse_reason`, `seconds`, `error`. It is
also written to `output_dir/tiles_summary.parquet`. `status` is one of:

- `:written` — derived and written.
- `:skipped` — an existing file already covers this request (see below).
- `:sparse` — too few cells to fit, or no cell with usable forcing. A file is still written, with
  the cells listed and no time axis; see [`write_sparse_downscaling_tile_netcdf`](@ref).
- `:failed` — something else went wrong; the message is in `error` and the sweep continued.

**Resumability.** A tile is skipped when its file's stored settings match this request and its window
covers the requested one, decided from metadata alone with no forcing read — which is what makes a
re-run cheap, since the forcing pass is essentially the whole cost. A changed `tile_size`, `buffer`,
`min_cells`, `area_minimum`, `climate_model` or `retain_elevation_interval_forcing`, or a wider
window, re-derives and overwrites; there is no append (see the writer). `force = true` re-derives
unconditionally.

# Keywords
- `tile_size = 2`, `buffer = 1`: the grid, in whole degrees, and how far past each tile to reach for
  cells to fit from. See [`downscaling_tiles`](@ref).
- `area_minimum = 0.0`, `min_cells = $(_MIN_CELLS_DEFAULT)`: the per-cell area screen and the fewest
  cells a tile needs before either fit is attempted.
- `retain_elevation_interval_forcing = false`: also store each tile's glacier-area weighted
  per-elevation-interval forcing. **This is the expensive half** — it is most of the output volume,
  and it costs a second pass over the tile's cells (or more, see `elevation_interval_batch`) on top
  of the fit pass. Off by default; the fits alone are enough to regenerate it later.
- `elevation_interval_batch = 0`: how many intervals to accumulate per pass when the above is on.
  `0` means one pass holding every interval at once, which is the fewest forcing reads and the most
  memory. Ignored when interval forcing is off.
- `decoupling_factor_fill`, `lapse_rate_fill`, `clamp_to_valid_domain`: how an unmeasurable timestep
  is handled where the fits are *applied*. These affect **only** the interval forcing, since that is
  the only thing here that applies a fit; the stored fits themselves are always raw.
- `precision = Float32`, `deflatelevel = 4`: storage for the written series. The `k` coefficients are
  always Float64 — see [`write_downscaling_tile_netcdf`](@ref).
- `order = :chunk`: tile visit order; `:chunk` keeps the forcing cache warm across neighbouring
  tiles.
- `tile_limit = Inf`: stop after this many tiles. Useful for a first pass over a global table.
- `forcing_loader = climate_forcing`: injectable loader, so a sweep can be exercised with no CDS
  token.

The sweep is serial: it is I/O bound on the forcing store, and running tiles concurrently would work
against the cache locality `order = :chunk` is there to exploit.
"""
function derive_downscaling_parameter_tiles(climate_model::Symbol, time_range,
                                            glacier_elevation_classes, output_dir::AbstractString;
                                            token,
                                            cache_path,
                                            tile_size::Real = 2,
                                            buffer::Real = 1,
                                            area_minimum::Real = 0.0,
                                            min_cells::Int = _MIN_CELLS_DEFAULT,
                                            retain_elevation_interval_forcing::Bool = false,
                                            elevation_interval_batch::Int = 0,
                                            decoupling_factor_fill::Real = 1.0,
                                            lapse_rate_fill::Real = _DEFAULT_LAPSE_RATE,
                                            clamp_to_valid_domain::Bool = true,
                                            precision::Type = Float32,
                                            deflatelevel::Int = 4,
                                            order::Symbol = :chunk,
                                            force::Bool = false,
                                            tile_limit = Inf,
                                            institution = nothing,
                                            references = nothing,
                                            forcing_loader = climate_forcing)
    tiles = downscaling_tiles(glacier_elevation_classes; tile_size, buffer, area_minimum, order)
    isempty(tiles) && throw(ArgumentError(
        "no grid cells with at least $area_minimum km² of glacier area, so there are no tiles"))

    mkpath(output_dir)
    # Written before the sweep, not after: it is derived from the tiling alone, and a sweep that is
    # interrupted (or that fails on some tiles) should still leave behind the complete point→tile
    # mapping for the tiles it did write.
    _write_tiles_index(tiles, output_dir, glacier_elevation_classes, area_minimum)

    selected = tiles
    if !(tile_limit === nothing || isinf(tile_limit))
        selected = first(tiles, Int(tile_limit))
    end

    @info "Deriving downscaling parameters by tile" tiles=length(selected) of=length(tiles) tile_size buffer time_range interval_forcing=retain_elevation_interval_forcing

    rows = NamedTuple[]
    for (i, tile) in enumerate(selected)
        path = joinpath(output_dir, tile.name)
        t0 = time()
        try
            # --- pre-flight, before any forcing I/O ------------------------------------------
            # The whole point of reading the status first: for a re-run most tiles are already done,
            # and the forcing pass is the entire cost. Deciding from metadata makes that free.
            status = read_downscaling_tile_status(path)
            if !force && status !== nothing &&
               _tile_is_current(status, time_range; climate_model, tile_size, buffer, min_cells,
                                area_minimum, retain_elevation_interval_forcing)
                @info "Tile already covers the request; skipping" tile=i path last_time=status.time_last
                push!(rows, _tile_summary_row(tile, status, :skipped, time() - t0))
                continue
            end

            # --- too few cells to fit at all ------------------------------------------------
            # Pre-flighted rather than left to the derivation, so a known-sparse tile costs one file
            # write instead of a forcing pass it cannot use.
            if nrow(tile.buffered) < min_cells
                write_sparse_downscaling_tile_netcdf(path, tile; climate_model, time_range,
                                                     tile_size, buffer, min_cells, area_minimum,
                                                     reason = "below_min_cells",
                                                     elevation_interval_forcing =
                                                         retain_elevation_interval_forcing,
                                                     institution, references)
                @info "Tile has too few cells to fit; wrote sparse tile" tile=i path cells=nrow(tile.buffered) min_cells
                push!(rows, _tile_summary_row(tile, nothing, :sparse, time() - t0;
                                              sparse_reason = "below_min_cells"))
                continue
            end

            p = derive_downscaling_parameters(climate_model, time_range, tile.buffered;
                                             token, cache_path,
                                             region_extent = tile.buffered_bounds,
                                             area_minimum, min_cells,
                                             elevation_interval_batch,
                                             decoupling_factor_fill, lapse_rate_fill,
                                             clamp_to_valid_domain, forcing_loader)

            # `elevation_interval_forcing` is a lazy iterator: leaving it alone reads no forcing at
            # all, which is why the fits-only sweep costs one pass over the tile's cells. Collecting
            # it is what makes this the expensive mode, so it happens only when asked for.
            intervals = retain_elevation_interval_forcing ?
                        collect(p.elevation_interval_forcing) : nothing

            write_downscaling_tile_netcdf(path, tile, p; climate_model, time_range,
                                          tile_size, buffer, min_cells, area_minimum,
                                          elevation_intervals = intervals,
                                          precision, deflatelevel, institution, references)

            @info "Wrote tile" tile=i path cells_used=nrow(p.grid_cells) steps=length(p.time) k_fitted=p.provenance["n_decoupling_factor_fitted"]
            push!(rows, _tile_summary_row(tile, nothing, :written, time() - t0; p, intervals))
        catch e
            e isa InterruptException && rethrow()
            # A tile with no usable forcing anywhere is unrunnable, not broken: ERA5-Land is
            # land-only, so a tile of small island glaciers can have every cell centre on water.
            # Recorded the same way a below-min_cells tile is, so the mapping stays complete.
            if e isa RegionForcingUnavailable
                write_sparse_downscaling_tile_netcdf(path, tile; climate_model, time_range,
                                                     tile_size, buffer, min_cells, area_minimum,
                                                     reason = "no_usable_forcing",
                                                     elevation_interval_forcing =
                                                         retain_elevation_interval_forcing,
                                                     institution, references)
                @info "Tile has no usable forcing; wrote sparse tile" tile=i path cells=nrow(tile.buffered)
                push!(rows, _tile_summary_row(tile, nothing, :sparse, time() - t0;
                                              sparse_reason = "no_usable_forcing"))
                continue
            end
            # A broken driver — a typo, a stale session, a signature that moved — fails identically
            # for every one of the hundreds of tiles. Swallowed per tile it becomes hundreds of
            # identical warnings that read like bad *data*, so abort on the first one and let the
            # real error be what gets seen.
            is_caller_error(e) && rethrow()
            @warn "Tile failed; continuing" tile=i path exception=(e, catch_backtrace())
            push!(rows, _tile_summary_row(tile, nothing, :failed, time() - t0;
                                          error = sprint(showerror, e)))
        end
    end

    summary = DataFrame(rows)
    _log_tile_census(summary)
    Parquet2.writefile(joinpath(output_dir, "tiles_summary.parquet"), summary)
    return summary
end

# Does an existing tile file already answer this request? Settings first, then coverage.
#
# Compared against the file's own stored roster, in the encoded form the attributes hold, so the
# comparison does not depend on a key list this version happens to know about.
function _tile_is_current(status, time_range; climate_model, tile_size, buffer, min_cells,
                          area_minimum, retain_elevation_interval_forcing)
    stored = status.parameters
    isempty(stored) && return false
    requested = Dict{String,Any}(
        "climate_model" => string(climate_model),
        "tile_size" => Float64(tile_size),
        "tile_buffer" => Float64(buffer),
        "min_cells" => min_cells,
        "area_minimum" => Float64(area_minimum),
        "elevation_interval_forcing" => _encode_attribute(retain_elevation_interval_forcing))
    for (k, want) in requested
        haskey(stored, k) || return false
        got = stored[k]
        # Numeric attributes come back as numbers and string ones as strings; `==` across those is
        # `false` rather than an error, so normalize the numeric case before comparing.
        if want isa Real && got isa Real
            Float64(got) == Float64(want) || return false
        else
            string(got) == string(want) || return false
        end
    end
    # Coverage, not equality: a file spanning more than was asked for still answers the request.
    status.time_first === nothing && return false
    status.time_last === nothing && return false
    return status.time_first <= first(time_range) && status.time_last >= last(time_range)
end

function _tile_summary_row(tile, status, status_symbol, seconds;
                           p = nothing, intervals = nothing, sparse_reason = "", error = "")
    return (; name = tile.name,
            index_lon = tile.index[1], index_lat = tile.index[2],
            n_cells_core = nrow(tile.core),
            n_cells_buffered = nrow(tile.buffered),
            n_cells_used = p === nothing ? (status === nothing ? 0 : status.n_cells_used) :
                           nrow(p.grid_cells),
            n_timesteps = p === nothing ? (status === nothing ? 0 : status.n_timesteps) :
                          length(p.time),
            n_decoupling_factor_fitted =
                p === nothing ? (status === nothing ? 0 : status.n_decoupling_factor_fitted) :
                Int(p.provenance["n_decoupling_factor_fitted"]),
            n_lapse_rate_fitted =
                p === nothing ? (status === nothing ? 0 : status.n_lapse_rate_fitted) :
                Int(p.provenance["n_lapse_rate_fitted"]),
            n_elevation_intervals = intervals === nothing ? 0 : length(intervals),
            status = String(status_symbol),
            sparse_reason = status === nothing ? sparse_reason : status.sparse_reason,
            seconds = round(seconds, digits = 2),
            error)
end

# One line at the end, because a per-tile log over hundreds of tiles is unreadable and the counts are
# what a reader actually checks. Sparse is broken out by reason: "too few cells" is a property of the
# glacier distribution and expected, while "no usable forcing" says cells landed on water.
function _log_tile_census(summary)
    isempty(summary) && return nothing
    count_of(s) = count(==(s), summary.status)
    sparse_rows = summary[summary.status .== "sparse", :]
    @info "Tile sweep complete" tiles=nrow(summary) written=count_of("written") skipped=count_of("skipped") sparse=count_of("sparse") failed=count_of("failed") sparse_below_min_cells=count(==("below_min_cells"), sparse_rows.sparse_reason) sparse_no_forcing=count(==("no_usable_forcing"), sparse_rows.sparse_reason)
    failed = summary[summary.status .== "failed", :]
    isempty(failed) || @warn "Some tiles failed" n=nrow(failed) names=collect(failed.name)
    return nothing
end

# The point→tile mapping, as one row per grid cell.
#
# This is what makes "parameters for every cell" checkable with a join rather than by opening every
# tile file. The row-count assertion is the tiling's totality property tested against the real data
# on every run: if it ever fails, cells have been lost or duplicated between the table and the tiles.
function _write_tiles_index(tiles, output_dir, glacier_elevation_classes, area_minimum)
    rows = Int[]
    names = String[]
    for tile in tiles
        # `tile.core_rows`, not `parentindices(tile.core)`: the core is a view over whatever frame
        # was handed to `downscaling_tiles`, so when *that* is itself a `SubDataFrame` (a sweep over
        # a region of the global table) `parentindices` resolves a level too far and returns indices
        # into the ultimate parent. Those then index the passed frame out of bounds. The tiler
        # records the indices it selected against its own argument, which is the frame indexed below.
        append!(rows, tile.core_rows)
        append!(names, fill(tile.name, length(tile.core_rows)))
    end
    order = sortperm(rows)
    rows, names = rows[order], names[order]

    expected = area_minimum > 0 ?
               count(>=(area_minimum), glacier_area_column(glacier_elevation_classes)) :
               nrow(glacier_elevation_classes)
    length(rows) == expected || error(
        "tiling lost or duplicated cells: $(length(rows)) tiled rows against $expected " *
        "qualifying rows in the table. The tile assignment is supposed to be a partition, so this " *
        "is a bug in `downscaling_tiles` rather than a data problem.")
    allunique(rows) || error("tiling assigned a cell to more than one tile core")

    index = DataFrame(row = rows, tile_name = names)
    src = view(glacier_elevation_classes, rows, :)
    index[!, :latitude] = Float64.(src[!, :latitude])
    index[!, :longitude] = Float64.(src[!, :longitude])
    hasproperty(src, :chunk_id) && (index[!, :chunk_id] = src[!, :chunk_id])
    Parquet2.writefile(joinpath(output_dir, "tiles_index.parquet"), index)
    return index
end
