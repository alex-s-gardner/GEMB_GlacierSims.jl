# The bare-ice albedo profiles of a directory of downscaling tile files, re-derived and filled in place.
#
# A profile is a function of MODIS and the DEM alone, never of the forcing, so it is rewritten without
# touching a tile's fits. The fill needs every tile's observations at once, which is why it is a pass
# over the whole directory rather than part of the per-tile derivation.

"""
    bare_ice_albedo_is_current(path; sky = :wsa, min_glacier_fraction = BARE_ICE_ALBEDO_MIN_GLACIER_FRACTION)
        -> Bool

Whether the tile file at `path` holds a bare-ice albedo profile derived with these settings.

A file with no profile, or one whose `bare_ice_albedo_sky` or `bare_ice_albedo_min_glacier_fraction`
differs or is absent, is not current. A file written before those attributes existed is therefore
re-derived rather than assumed to match.
"""
function bare_ice_albedo_is_current(path::AbstractString; sky::Symbol = :wsa,
                                    min_glacier_fraction::Real = BARE_ICE_ALBEDO_MIN_GLACIER_FRACTION)
    return NCDatasets.NCDataset(path, "r") do ds
        haskey(ds.dim, "bare_ice_albedo_bin") || return false
        get(ds.attrib, "bare_ice_albedo_sky", "") == String(sky) || return false
        stored = get(ds.attrib, "bare_ice_albedo_min_glacier_fraction", NaN)
        return Float64(stored) == Float64(min_glacier_fraction)
    end
end

"""
    rederive_bare_ice_albedo_tiles(parameter_dir, glacier_elevation_classes; kwargs...) -> DataFrame

Re-derive the bare-ice albedo profile of every non-sparse tile file in `parameter_dir`, then fill each
profile's unresolved bins from the nearest tile with an observed bin
([`fill_bare_ice_albedo_neighbors`](@ref)).

Each tile's profile is derived with [`derive_bare_ice_albedo`](@ref) over the same buffered cells
[`derive_downscaling_parameter_tiles`](@ref) used, on the binning of the tile's own hypsometry, and
written into the existing file. The forcing fits are left as they are, so no CDS token is needed —
only the Copernicus DEM.

A tile whose profile is already current ([`bare_ice_albedo_is_current`](@ref)) is not re-derived
unless `force = true`. The neighbor fill always runs over the whole directory, because adding or
re-deriving one tile can change which observation is nearest to another tile's empty bin.

# Keywords
- `tile_size = 2`, `buffer = 1`: the grid the files were derived on, as for
  [`downscaling_tiles`](@ref).
- `sky = :wsa`, `min_glacier_fraction = BARE_ICE_ALBEDO_MIN_GLACIER_FRACTION`: passed to
  [`derive_bare_ice_albedo`](@ref) and recorded in each file.
- `dem_archive = nothing`, `dem_cache_path = nothing`: where the Copernicus DEM is read, as for
  [`derive_bare_ice_albedo`](@ref).
- `force = false`: re-derive every profile, current or not.
- `tile_gate = nothing`: called as `tile_gate(tile, path)` before a tile is re-derived; `false` skips
  it. This is how the pass is split across processes.
- `fill = tile_gate === nothing`: run the neighbor fill after re-deriving. Off by default when a gate
  is given, since the fill rewrites every tile file and must run in one process, after every worker
  has finished; passing both is an error.

Returns one row per tile file: `name`, `status` (`"rederived"`, `"current"`, `"sparse"`, `"missing"`
or `"failed"`), `error`, and `n_bins_observed`.
"""
function rederive_bare_ice_albedo_tiles(parameter_dir::AbstractString, glacier_elevation_classes;
                                        tile_size::Real = 2, buffer::Real = 1,
                                        sky::Symbol = :wsa,
                                        min_glacier_fraction::Real =
                                            BARE_ICE_ALBEDO_MIN_GLACIER_FRACTION,
                                        dem_archive::Union{Nothing,AbstractString} = nothing,
                                        dem_cache_path::Union{Nothing,AbstractString} = nothing,
                                        force::Bool = false, tile_gate = nothing,
                                        fill::Bool = tile_gate === nothing)
    fill && tile_gate !== nothing && throw(ArgumentError(
        "fill = true with a tile_gate would run the neighbor fill in every worker at once; run the " *
        "workers with fill = false, then fill_bare_ice_albedo_neighbors once"))
    tiles = downscaling_tiles(glacier_elevation_classes; tile_size, buffer, order = :lonlat)
    rows = NamedTuple{(:name, :status, :error, :n_bins_observed),Tuple{String,String,String,Int}}[]
    for tile in tiles
        path = joinpath(parameter_dir, tile.name)
        if !isfile(path)
            push!(rows, (tile.name, "missing", "", 0))
            continue
        end
        status = read_downscaling_tile_status(path)
        if status.sparse
            push!(rows, (tile.name, "sparse", "", 0))
            continue
        end
        if !force && bare_ice_albedo_is_current(path; sky, min_glacier_fraction)
            push!(rows, (tile.name, "current", "", 0))
            continue
        end
        tile_gate === nothing || tile_gate(tile, path) || continue
        try
            f = derive_bare_ice_albedo(tile.buffered; bins = hypsometry_bin_edges(tile.buffered),
                                       sky, min_glacier_fraction, dem_archive, dem_cache_path,
                                       label = tile.name, verbose = false)
            NCDatasets.NCDataset(path, "a") do ds
                _write_bare_ice_albedo!(ds, f, 4; min_glacier_fraction)
            end
            push!(rows, (tile.name, "rederived", "", count(==(:observed), f.sources)))
        catch err
            err isa InterruptException && rethrow()
            is_caller_error(err) && rethrow()
            @warn "Could not re-derive this tile's bare-ice albedo" tile = tile.name exception = err
            push!(rows, (tile.name, "failed", sprint(showerror, err), 0))
        end
    end
    summary = DataFrame(rows)
    @info "Bare-ice albedo re-derivation" tiles = nrow(summary) (Symbol(s) => count(==(s), summary.status)
                                                              for s in unique(summary.status))...
    fill && fill_bare_ice_albedo_neighbors(parameter_dir)
    return summary
end

"""
    fill_bare_ice_albedo_neighbors(parameter_dir) -> DataFrame

Fill every bin a tile's own observations leave unresolved (`:none`, or a previous `:neighbor` fill)
with the value of the nearest tile, by great-circle distance between tile centres, whose profile
holds an `:observed` value for that same bin. The filled bins are marked `:neighbor` and record the
donor tile in `bare_ice_albedo_donor`. A non-sparse tile with no profile at all (its derivation
failed) is given one with every bin filled this way, under the donors' sky and screen.

Only `:observed` bins are ever donated, and earlier fills are recomputed rather than kept, so running
this again — after a tile is added or re-derived — gives the same result as running it once. A
`:hold` or `:fit` bin is the tile's own estimate and is not replaced.

Every profile must share one binning; a file on another binning is an error, since a bin index would
then mean a different elevation in the donor.

Returns one row per filled tile: `name`, `n_bins_filled`, and `max_donor_distance_km`.
"""
function fill_bare_ice_albedo_neighbors(parameter_dir::AbstractString)
    tiles = _bare_ice_albedo_profiles(parameter_dir)
    pool = filter(t -> t.profile !== nothing && any(==(:observed), t.profile.sources), tiles)
    isempty(pool) && error("no tile file in $(parameter_dir) holds an observed bare-ice albedo bin")
    edges = first(pool).profile.edges
    sky = first(pool).profile.sky
    screen = first(pool).min_glacier_fraction
    for t in tiles
        t.profile === nothing && continue
        (t.profile.sky === sky && t.min_glacier_fraction == screen) || error(
            "$(t.name) was derived under sky = :$(t.profile.sky), min_glacier_fraction = " *
            "$(t.min_glacier_fraction) and $(first(pool).name) under :$(sky), $(screen); re-derive " *
            "every tile with rederive_bare_ice_albedo_tiles before filling")
        t.profile.edges == edges || error(
            "$(t.name) is binned on different elevation edges from $(first(pool).name); the neighbor " *
            "fill needs every profile on one binning")
    end
    n_bins = length(edges) - 1
    # For each bin, the tiles holding an observation there — the only candidate donors.
    donors_of_bin = [[k for (k, p) in pairs(pool) if p.profile.sources[b] === :observed]
                     for b in 1:n_bins]

    rows = NamedTuple{(:name, :n_bins_filled, :max_donor_distance_km),Tuple{String,Int,Float64}}[]
    for p in tiles
        f = something(p.profile, bare_ice_albedo_hyps(Float64[], Float64[], edges; sky))
        albedo = copy(f.albedo)
        sources = copy(f.sources)
        donors = fill("", n_bins)
        far = 0.0
        for b in 1:n_bins
            sources[b] in (:none, :neighbor) || continue
            sources[b], albedo[b] = :none, NaN
            best, best_d = 0, Inf
            for k in donors_of_bin[b]
                pool[k].name == p.name && continue
                d = _haversine_km(p.latitude, p.longitude, pool[k].latitude, pool[k].longitude)
                d < best_d && ((best, best_d) = (k, d))
            end
            best == 0 && continue
            albedo[b] = pool[best].profile.albedo[b]
            sources[b] = :neighbor
            donors[b] = pool[best].name
            far = max(far, best_d)
        end
        filled = BareIceAlbedoHyps(f.edges, albedo, f.observed, f.n_cells, f.n_valid, sources,
                                   f.fit, f.sky, f.n_input, f.n_used, f.n_no_albedo,
                                   f.n_no_elevation)
        NCDatasets.NCDataset(p.path, "a") do ds
            _write_bare_ice_albedo!(ds, filled, 4; donors, min_glacier_fraction = screen)
        end
        push!(rows, (p.name, count(==(:neighbor), sources), far))
    end
    summary = DataFrame(rows)
    @info "Bare-ice albedo neighbor fill" tiles = nrow(summary) bins_filled = sum(summary.n_bins_filled; init = 0) max_donor_distance_km =
        round(maximum(summary.max_donor_distance_km; init = 0.0); digits = 0)
    return summary
end

# Every non-sparse tile file in `parameter_dir`, with its centre in degrees, its profile (`nothing`
# when the file holds none) and that profile's glacier-fraction screen.
function _bare_ice_albedo_profiles(parameter_dir::AbstractString)
    entry_type = NamedTuple{(:name, :path, :longitude, :latitude, :profile, :min_glacier_fraction),
                            Tuple{String,String,Float64,Float64,Union{Nothing,BareIceAlbedoHyps},
                                  Float64}}
    all = entry_type[]
    for name in sort!(filter(endswith(".nc"), readdir(parameter_dir)))
        index = parse_tile_index(name)
        index === nothing && continue
        path = joinpath(parameter_dir, name)
        entry = NCDatasets.NCDataset(path, "r") do ds
            get(ds.attrib, "sparse", "false") == "true" && return nothing
            b = tile_bounds(index; tile_size = ds.attrib["tile_size"])
            longitude, latitude = (b.lon_min + b.lon_max) / 2, (b.lat_min + b.lat_max) / 2
            haskey(ds.dim, "bare_ice_albedo_bin") ||
                return (; name, path, longitude, latitude, profile = nothing,
                        min_glacier_fraction = NaN)
            haskey(ds.attrib, "bare_ice_albedo_min_glacier_fraction") || error(
                "$(path) holds a bare-ice albedo profile with no recorded glacier-fraction screen; " *
                "re-derive it with rederive_bare_ice_albedo_tiles before filling")
            (; name, path, longitude, latitude, profile = _read_bare_ice_albedo(ds),
             min_glacier_fraction = Float64(ds.attrib["bare_ice_albedo_min_glacier_fraction"]))
        end
        entry === nothing || push!(all, entry)
    end
    return all
end
