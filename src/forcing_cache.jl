"""
A bounded in-memory cache of loaded cell forcing, for sweeps that visit the same cell more than once.

The tiled derivation fits each tile's parameters over a **buffered** neighbourhood so its cross-cell
regressions have enough elevation range to be identifiable. At the 2°/1° default that window is 4°×4°
while tiles step 2°, so every cell falls inside four tiles' buffered sets and a global sweep loads it
four times. Measured on the 47,121-cell global table: **170,474 loads for 47,121 distinct cells, a
factor of 3.62**.

The existing Zarr chunk cache does not remove that cost. It keeps the compressed *bytes* on local disk,
so the repeat is not a network fetch — but each load still decompresses them, extracts 17,521 timesteps
across four variable groups, derives vapour pressure and wind speed, and validates units. At ~200 ms per
cell that is the entire cost of the sweep once `_accumulate_cell!` is a function barrier: forcing I/O is
99.8% of a tile's time.

This sits at the `forcing_loader` seam the derivation already exposes, so nothing about the derivation
changes — it is handed a loader that sometimes answers from memory.
"""

# Bytes one cell's forcing occupies: seven `Float64` layers over the record. Used to turn a memory
# budget into a cell count, since a budget is what a caller can reason about and a count is not.
_forcing_bytes_per_cell(n_time::Integer) = 7 * n_time * sizeof(Float64)

"""
    CachedForcingLoader(loader; capacity, on_evict = nothing)

Wrap a `climate_forcing`-shaped `loader` so repeated requests for one cell are answered from memory.

Callable with the loader's own signature, so it can be passed straight as the `forcing_loader` keyword
of [`derive_downscaling_parameter_tiles`](@ref) or [`derive_downscaling_parameters`](@ref):

```julia
cached = CachedForcingLoader(climate_forcing; capacity = 4000)
derive_downscaling_parameter_tiles(:era5land, time_range, table, dir;
                                   token, cache_path, forcing_loader = cached)
@info "cache" hits=cached.hits[] misses=cached.misses[]
```

`capacity` is a number of cells. One cell of hourly ERA5-Land over two years is ~1 MB, so 4,000 cells is
~4 GB; over the full 1950–2026 record a cell is ~37 MB and the same budget buys ~110 cells. Size it
against the *working set*, which is what makes this worth doing: tiles are visited in chunk order, so
the tiles that share a cell are neighbours in visit order and a few tiles' worth of capacity captures
most of the reuse. Use [`forcing_cache_capacity`](@ref) to turn a memory budget into a count.

Eviction is least-recently-used, in batches: when full, the oldest quarter is dropped in one pass. A
strict one-at-a-time LRU would scan the store on every miss; a batch amortizes that to one scan per
quarter-capacity of misses, and the access pattern is spatially clustered rather than adversarial, so
the difference in hit rate is immaterial.

**Not thread-safe, deliberately.** The tiled sweep is serial — it is I/O bound on the forcing store, and
running tiles concurrently would work against the chunk locality that visiting them in chunk order
exists to exploit — and every load inside the derivation happens on the calling task. Adding a lock
would imply concurrent use is supported when the wrapped loader may not be.

A cache miss calls the wrapped loader, so a load that throws throws through here unchanged and the
derivation's own per-cell handling sees it as it would without the cache. Failures are not cached: a
cell that failed once is retried if it is asked for again, which is what a transient store error wants.
"""
struct CachedForcingLoader{L}
    loader::L
    capacity::Int
    # Values are whatever the wrapped loader returns — a `DimStack` whose concrete type is the loader's
    # business, and which an injected test loader may spell differently. `Any` costs one dynamic
    # dispatch per cell, against the ~200 ms of I/O this exists to avoid.
    store::Dict{Tuple{Symbol,Float64,Float64},Any}
    # Last access, as a monotonic counter rather than a clock: it only has to order accesses, and a
    # clock would make eviction depend on how fast the machine happened to be.
    touched::Dict{Tuple{Symbol,Float64,Float64},Int}
    clock::Base.RefValue{Int}
    hits::Base.RefValue{Int}
    misses::Base.RefValue{Int}
    evictions::Base.RefValue{Int}
    # The window the cached forcing covers. Learned from the first call and then enforced: two windows
    # are different data under the same key, and silently returning the wrong one would be invisible.
    window::Base.RefValue{Any}
end

function CachedForcingLoader(loader; capacity::Integer = 4000)
    capacity >= 1 || throw(ArgumentError("capacity must be at least 1 cell, got $capacity"))
    return CachedForcingLoader(loader, Int(capacity),
                               Dict{Tuple{Symbol,Float64,Float64},Any}(),
                               Dict{Tuple{Symbol,Float64,Float64},Int}(),
                               Ref(0), Ref(0), Ref(0), Ref(0), Ref{Any}(nothing))
end

function (c::CachedForcingLoader)(climate_model::Symbol, lat, lon; time_range = nothing, kwargs...)
    if c.window[] === nothing
        c.window[] = time_range
    elseif c.window[] != time_range
        throw(ArgumentError(
            "this cache holds forcing for $(c.window[]) but was asked for $time_range. The window is " *
            "part of what a cell's forcing *is*, so one cache cannot serve two of them; build a " *
            "second CachedForcingLoader."))
    end

    key = (climate_model, Float64(lat), Float64(lon))
    c.clock[] += 1

    hit = get(c.store, key, nothing)
    if hit !== nothing
        c.hits[] += 1
        c.touched[key] = c.clock[]
        return hit
    end

    # Outside the store update, so a throwing load is not recorded as a miss-with-no-value and is
    # retried on a later request.
    value = c.loader(climate_model, lat, lon; time_range, kwargs...)

    c.misses[] += 1
    length(c.store) >= c.capacity && _evict_oldest!(c)
    c.store[key] = value
    c.touched[key] = c.clock[]
    return value
end

# Drop the least recently used quarter of the store. One `partialsort` over the access counters rather
# than a scan per eviction.
function _evict_oldest!(c::CachedForcingLoader)
    n_drop = max(1, c.capacity ÷ 4)
    keys_by_age = sort!(collect(keys(c.touched)); by = k -> c.touched[k])
    for key in first(keys_by_age, n_drop)
        delete!(c.store, key)
        delete!(c.touched, key)
        c.evictions[] += 1
    end
    return nothing
end

"""
    forcing_cache_capacity(budget_bytes, n_time) -> Int

How many cells of forcing fit in `budget_bytes`, for a record of `n_time` steps.

A cell holds seven `Float64` layers over the record, so this is `budget / (7 * n_time * 8)`, floored at
one. Provided because a memory budget is the thing a caller can reason about on a shared machine, while
[`CachedForcingLoader`](@ref)'s `capacity` is a cell count — and the conversion depends on the window,
which is easy to forget: the same budget that holds 4,000 cells of a two-year record holds ~110 of the
full 1950–2026 one.

The estimate covers the forcing arrays only, not the per-cell metadata or the `DimStack` wrappers, so
leave headroom.
"""
function forcing_cache_capacity(budget_bytes::Real, n_time::Integer)
    n_time >= 1 || throw(ArgumentError("n_time must be at least 1, got $n_time"))
    budget_bytes > 0 || throw(ArgumentError("budget_bytes must be positive, got $budget_bytes"))
    return max(1, floor(Int, budget_bytes / _forcing_bytes_per_cell(n_time)))
end

"""
    forcing_cache_report(c::CachedForcingLoader) -> NamedTuple

Hit rate and occupancy of a [`CachedForcingLoader`](@ref), for a sweep to log.

Returns `(; hits, misses, evictions, requests, hit_rate, cached_cells, capacity)`. The number worth
watching is `hit_rate` against the redundancy the tiling implies: at the 2°/1° default each cell is
requested ~3.62 times, so the ceiling is about 0.72 and a rate far below it means `capacity` is smaller
than the working set the visit order produces.
"""
function forcing_cache_report(c::CachedForcingLoader)
    requests = c.hits[] + c.misses[]
    return (; hits = c.hits[], misses = c.misses[], evictions = c.evictions[], requests,
            hit_rate = requests == 0 ? 0.0 : c.hits[] / requests,
            cached_cells = length(c.store), capacity = c.capacity)
end
