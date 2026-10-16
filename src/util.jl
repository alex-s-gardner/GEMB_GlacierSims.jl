# Default temperature lapse rate (K/km) for this package's elevation adjustments. Defined once so the
# per-bin adjustment, the sweep driver, and the interval aggregation's gap fill cannot drift apart.
const _DEFAULT_LAPSE_RATE = 6.5

# Fallback bare-ice albedo for an elevation class the observations cannot resolve, matching GEMB's
# own `ModelParameters` default. Restated here rather than read from GEMB because `resolve_downscaling`
# resolves a tile without ever being handed a `ModelParameters`; a caller who tuned `albedo_ice` should
# pass `mp.albedo_ice` as `albedo_ice_default` so their value is what the gaps fall back to.
const _DEFAULT_ALBEDO_ICE = 0.48

# Is this exception a broken *caller* rather than a bad cell?
#
# A typo, an undefined name, a stale session whose loaded package predates a function being called, a
# signature that has moved: these fail identically for every cell of every region. Swallowed per cell
# they turn one bug into thousands of identical warnings that read like the *data* is bad — and worse,
# a sweep then records the region as unrunnable. So every per-cell `catch` in this package rethrows
# them, and the predicate lives here rather than being spelled out at each site: four copies of a type
# tuple is four places to forget one.
#
# `FieldError` (a mistyped or removed struct field) only exists from Julia 1.12; before that the same
# mistake surfaces as an `ErrorException`, which is too broad to rethrow on — this package supports
# 1.11, so the type is included only where it is defined rather than assumed. Without the guard the
# reference itself is an `UndefVarError` on 1.11, i.e. the handler becomes the bug it exists to catch.
const _CALLER_ERROR_TYPES = isdefined(Base, :FieldError) ?
    (UndefVarError, MethodError, getfield(Base, :FieldError)) : (UndefVarError, MethodError)

is_caller_error(e) = any(T -> e isa T, _CALLER_ERROR_TYPES)

"""
    era5_land_invariant(; parameter=nothing, to=nothing)

Read ERA5-Land invariant field(s) and rewrap longitudes from the native
0–359.9°E grid to the `(-180, 180]` convention used by the Zarr climate grid
(`-179.9 … 180.0`).

- `parameter`: a single field (e.g. `:glm`) returns a `Raster`; an iterable of
  fields (e.g. `(:glm, :z)`) returns a `RasterStack`.
- `to`: an optional grid whose X/Y direction the result is reordered to match
  (e.g. `glaciers`). When `nothing`, only the longitude rewrap/sort is applied.
"""
function era5_land_invariant(; parameter=nothing, to=nothing)
    read1(p) = _rewrap_era5_lon(
        read(GEMB_ClimateForcing.climate_model_invariant(model=:era5_land, parameter=p)), to)
    if parameter isa Union{AbstractVector,Tuple,Base.Generator}
        return RasterStack((; (Symbol(p) => read1(p) for p in parameter)...))
    else
        return read1(parameter)
    end
end

"""
    wrap_lon(lon)

Wrap a native ERA5 longitude (0–359.9°E) to the `(-180, 180]` convention used by the
Zarr climate grid (`geo_chunk_map`/`glaciers`) and the Copernicus DEM.

NOTE: `mod(lon+180,360)-180` would send the 180.0° cell to -180.0 — one cell off from the
Zarr grid — so keep 180.0 as 180.0 and only wrap longitudes strictly greater than 180.
"""
wrap_lon(lon) = lon > 180 ? lon - 360 : lon

"""
    forcing_at_elevation(forcing_data, delta_elevation; lapse_rate=$(_DEFAULT_LAPSE_RATE),
                         precip_scaling_method=nothing, decoupling_factor=nothing,
                         glacier_fraction=0.0)

Lapse-rate-adjust `forcing_data` by `delta_elevation` metres and convert it to a GEMB
`ClimateForcing`, ready to hand to `initialize_profile` / `gemb`.

This crosses the forcing→firn boundary once: [`climate_adjust_for_elevation`](@ref)
(GEMB_ClimateForcing) operates on the `forcing_data` `DimStack`, then `initialize_forcing`
(GEMB) converts the adjusted stack to a `ClimateForcing`. The adjustment always starts from
the passed `forcing_data`, so calling this repeatedly with different `delta_elevation` values
(e.g. once per glacier hypsometry bin) does not compound the deltas.

- `delta_elevation`: `z_target − z_reanalysis` in metres (positive to cool/raise to a higher target).
- `lapse_rate`: temperature lapse rate passed through to `climate_adjust_for_elevation` (K/km).
- `precip_scaling_method`: precipitation scaling passed through to `climate_adjust_for_elevation`
  (`nothing` leaves precipitation unchanged).
- `decoupling_factor`: on-glacier air temperature decoupling factor `k` in `(0, 1]`. The forcing is
  first brought to the ice-free state with `k` and `glacier_fraction`, then lapsed, then decoupled
  with `k` by `climate_adjust_for_glacier` — the order the decoupling requires, since `k` scales an
  ambient temperature already at the glacier's elevation. `nothing` (the default) leaves the
  forcing ambient.
- `glacier_fraction`: the fraction `glm` of the reanalysis cell its land-surface scheme already runs
  as ice, whose warm excess is therefore `[1 - glm·(1 - k)]·E_ambient`. Only read with a
  `decoupling_factor`; `0` treats the forcing as ambient already.

With one scalar `k` and no regional fit, the ambient excess is read off that mixing model directly,
`E_ambient = E_obs / (1 - glm·(1 - k))` (`_ambient_forcing`, with `k` at every timestep). The
division multiplies the observed excess by at most `1/k` (at `glm = 1`), so a cell with
`glacier_fraction > 0` needs `k` inside `$(_DONOR_DECOUPLING_FACTOR_LIMITS)`, the range that bounds
that factor, and a smaller `k` throws. With `glacier_fraction = 0` there is nothing to undo, and
any `k` in `(0, 1]` is accepted.
"""
function forcing_at_elevation(forcing_data, delta_elevation; lapse_rate=_DEFAULT_LAPSE_RATE,
                              precip_scaling_method=nothing, decoupling_factor=nothing,
                              glacier_fraction=0.0)
    decoupling_factor === nothing &&
        return initialize_forcing(climate_adjust_for_elevation(forcing_data, delta_elevation;
                                                               lapse_rate, precip_scaling_method))
    k = Float64(decoupling_factor)
    glm = Float64(glacier_fraction)
    _DECOUPLING_FACTOR_LIMITS[1] < k <= _DECOUPLING_FACTOR_LIMITS[2] || throw(ArgumentError(
        "decoupling_factor must be in (0, 1] (got $(decoupling_factor))"))
    0 <= glm <= 1 || throw(ArgumentError("glacier_fraction must be in [0, 1] (got $(glacier_fraction))"))
    ambient = glm == 0 ? forcing_data :
              _ambient_forcing(forcing_data, glm, fill(k, length(forcing_data[:temperature_air])))
    lapsed = climate_adjust_for_elevation(ambient, delta_elevation; lapse_rate, precip_scaling_method)
    return initialize_forcing(climate_adjust_for_glacier(lapsed, decoupling_factor))
end

# Rewrap glm (the ERA5-Land invariant grid) to the SAME (-180, 180] convention as the Zarr grid.
function _rewrap_era5_lon(ras, to)
    lon  = lookup(ras, X)
    lonw = @. round(wrap_lon(lon), digits=10)
    ras  = set(ras, X => lonw)          # relabel X (now non-monotonic)
    ras  = ras[X(sortperm(lonw))]       # physically sort X ascending
    return to === nothing ? ras : reorder(ras, to)   # match `to`'s X/Y direction
end

# `glm` — the ERA5-Land glacier-mask fraction of a grid cell — as a finite fraction in [0, 1].
#
# The column is required: a table missing it cannot support the decoupling factor at all, whether
# that factor is looked up per cell or regressed across a region, so its absence is schema drift
# and throws. `missing`/`NaN` are the different case of a real data gap, and fall back to 0.0 —
# the uncorrected reanalysis assumption of a cell with no glacier.
function _row_glm(row)
    hasproperty(row, :glm) || throw(ArgumentError(
        "glacier elevation-class row has no `:glm` column, which the glacier decoupling factor " *
        "is weighted by (per cell) and regressed against (per region). A table built before the " *
        "current invariant column naming carries `:glm_frac`; run " *
        "`scripts/migrate_invariant_colnames.jl` to rename it in place."))
    v = row.glm
    v === missing && return 0.0
    g = Float64(v)
    return isfinite(g) ? clamp(g, 0.0, 1.0) : 0.0
end

# How far a cell that carries ice but no forcing of its own may borrow a neighbour's, by default:
# about four to five ERA5-Land cells (the grid is 0.1°, ~11 km). Loose enough to reach the nearest
# land cell on an Antarctic or Arctic-archipelago coast, where a masked cell's neighbours are often
# 20–40 km away, and tight enough that a donor still shares the synoptic regime. Every substitution
# records its actual distance, so a stricter screen can be applied afterwards without re-running.
# Consumed by `elevation_interval_forcing`.
const _DONOR_MAX_DISTANCE_KM_DEFAULT = 50.0

# `_haversine_km` is borrowed from GEMB_ClimateForcing (see the import in GEMB_GlacierSims.jl) rather
# than reimplemented: it is one formula on one Earth radius, and two copies would be free to drift.
#
# Longitudes need no seam handling and no `wrap_lon` before calling it: the formula uses
# `sin(Δλ/2)^2`, which is unchanged by adding 360° to Δλ, so the table's native 0–359.9°E values give
# the same distance as the (-180, 180] convention would. The tiler next door does need explicit seam
# logic (`_lon_delta`) because membership in a longitude interval is not periodic in the same way.

# --- output file naming -------------------------------------------------------------------------
#
# Per-cell and per-tile output files are named by position, so a file is traceable to what it
# describes and a listing sorts geographically (by hemisphere, then by degree).
#
# The form follows ISO 6709's letter-prefix convention: hemisphere letter, then zero-padded
# magnitude — two integer digits of latitude, three of longitude, which `wrap_lon` bounds at 180.
# The letters carry the sign, so every name is the same length without spending a character on '+',
# and they make `lat`/`lon` prefixes redundant, since N/S can only be a latitude and E/W only a
# longitude. Keeping '-' out of filenames also avoids the corner where shell and `find` argument
# parsing treat a name as an option. The cost is that the degrees no longer `parse` straight out of
# the substring, so each name form is paired with its inverse below.
#
# The '_' between the two coordinates is redundant as a delimiter — the E/W letter already marks
# where latitude ends — but it makes the boundary visible without hunting for a letter among digits,
# and it means splitting a name into its two halves does not depend on the zero-padding being
# exactly 2-and-3 digits.

# ONE decimal for a cell, because that is all the forcing grid has: ERA5-Land is 0.1°, so every cell
# centre is exact at 1 dp and a second decimal is always '0' — a character of noise in every name.
# That makes the decimal point itself the only punctuation left, and it is worth keeping: dropping it
# would leave the scale implied ('N523' read as tenths), and the standard alternative for a
# point-free name is ISO 6709 degrees-minutes ('N5218'), which is less readable than decimal degrees
# for anyone working in this field.
#
# This ties the name to a 0.1° grid: a finer forcing grid (ERA5 at 0.25° is fine, but a 0.01° product
# would not be) needs another decimal here, or two of its cells would collide on one name.
const _CELL_NAME_DECIMALS = 1

function _degrees_tag(x, intdigits, positive, negative)
    # Round first, then split, so a cell just under a degree (52.97 at 1 dp) becomes 53.0 rather
    # than 52.10.
    scale = 10^_CELL_NAME_DECIMALS
    r = round(abs(x), digits = _CELL_NAME_DECIMALS)
    whole, frac = floor(Int, r), round(Int, scale * (r - floor(r)))
    # `x < 0` and not `signbit`: -0.0 is the equator/prime meridian, which is not a hemisphere.
    return (x < 0 ? negative : positive) * lpad(whole, intdigits, '0') * "." *
           lpad(frac, _CELL_NAME_DECIMALS, '0')
end

# A whole-degree tag, for a tile. No decimal point at all — which is also what keeps the tile and
# cell name forms unambiguous; see `tile_output_name`.
_degrees_tag_int(x::Integer, intdigits, positive, negative) =
    (x < 0 ? negative : positive) * lpad(abs(x), intdigits, '0')

"""
    cell_output_name(latitude, longitude; extension = ".nc") -> String

The file name for a glacier grid cell's output, named by cell centre: `"N52.3_W174.1.nc"`.

One decimal place, which is the ERA5-Land grid's own precision. [`parse_cell_lonlat`](@ref) is the
inverse. Longitude is wrapped with [`wrap_lon`](@ref), so a native 0–359.9°E value names the same
file as its `(-180, 180]` equivalent.
"""
cell_output_name(latitude, longitude; extension::AbstractString = ".nc") =
    _degrees_tag(Float64(latitude), 2, 'N', 'S') * "_" *
    _degrees_tag(wrap_lon(Float64(longitude)), 3, 'E', 'W') * extension

# Inverse of `cell_output_name`, and of `tile_output_name`. Anchored and fully specified so each
# rejects the other's names: a cell name always carries a decimal point and a tile name never does.
const _CELL_NAME_REGEX =
    Regex("^([NS])(\\d{2}\\.\\d{$_CELL_NAME_DECIMALS})_([EW])(\\d{3}\\.\\d{$_CELL_NAME_DECIMALS})\\.nc\$")
const _TILE_NAME_REGEX = r"^([NS])(\d{2})_([EW])(\d{3})\.nc$"

"""
    parse_cell_lonlat(path) -> (; latitude, longitude) or nothing

The cell centre a file written by [`cell_output_name`](@ref) is named for, or `nothing` if the name
is not one of ours (including a tile name, which carries no decimal point).

Rounded to the name's precision, so it identifies the cell but is not the exact centre — read the
file's attributes for that. Provided so downstream tooling can recover the cell from a filename
without re-implementing the format and drifting from it.
"""
function parse_cell_lonlat(path::AbstractString)
    m = match(_CELL_NAME_REGEX, basename(path))
    m === nothing && return nothing
    return (latitude = parse(Float64, m[2]) * (m[1] == "S" ? -1 : 1),
            longitude = parse(Float64, m[4]) * (m[3] == "W" ? -1 : 1))
end

"""
    tile_output_name(index; extension = ".nc") -> String

The file name for a downscaling-parameter tile, named by its southwest corner: `"N60_W142.nc"` for
`index = (-142, 60)`. [`parse_tile_index`](@ref) is the inverse.

Whole degrees, with no decimal point — which is what makes a tile name and a cell name
([`cell_output_name`](@ref)) impossible to confuse even in one directory, since a cell name always
carries its 0.1° decimal. It is also why `tile_size` must be a whole number of degrees: a fractional
grid would let two tiles collide on one name.

The name carries the corner but **not** the tile size, so one directory should hold one gridding.
The size and buffer are recorded in each file's attributes, and the tile writer's status check
compares them, so a directory reused across griddings re-derives rather than silently reading a
tile that means something else.
"""
tile_output_name(index; extension::AbstractString = ".nc") =
    _degrees_tag_int(Int(index[2]), 2, 'N', 'S') * "_" *
    _degrees_tag_int(Int(index[1]), 3, 'E', 'W') * extension

"""
    parse_tile_index(path) -> (Int, Int) or nothing

The tile index a file written by [`tile_output_name`](@ref) is named for, or `nothing` if the name
is not one of ours (including a per-cell name, which carries a decimal point).
"""
function parse_tile_index(path::AbstractString)
    m = match(_TILE_NAME_REGEX, basename(path))
    m === nothing && return nothing
    return (parse(Int, m[4]) * (m[3] == "W" ? -1 : 1),
            parse(Int, m[2]) * (m[1] == "S" ? -1 : 1))
end

"""
    claim_tile!(claim_dir, tile_name, output_path; stale_after = Hour(24)) -> Bool

Take ownership of a tile for this process, or report that another already has it.

`mkdir` is atomic on POSIX — exactly one caller can create a given name — so a directory per tile is a
whole mutual exclusion mechanism with no lock server, no shared file to corrupt, and nothing to clean up
beyond the directory. That is what lets a sweep run as N independent processes that steal work from each
other rather than being handed a fixed partition up front, which matters when per-tile cost is not
predictable in advance.

A claim left behind by a process that died would drop its tile from the sweep silently, so a claim whose
heartbeat is older than `stale_after` and whose tile still has no output is reclaimed: removed and
re-created, the re-creation deciding the winner if two callers race. The heartbeat is the modification
time of the claim's `heartbeat` file, written when the claim is taken and refreshed for as long as the
owner runs [`start_claim_heartbeat`](@ref); a claim without the file is judged by the directory's own
modification time. A running tile is therefore never reclaimed however long it takes, provided
`stale_after` is several heartbeat intervals, and a crashed owner's claim becomes reclaimable
`stale_after` after its last heartbeat. `stale_after <= 0` disables reclaiming.
"""
function claim_tile!(claim_dir::AbstractString, tile_name::AbstractString,
                     output_path::AbstractString; stale_after::Period = Hour(24))
    path = joinpath(claim_dir, tile_name)
    try
        mkdir(path)
        touch(joinpath(path, CLAIM_HEARTBEAT_FILE))
        return true
    catch err
        err isa Base.IOError || rethrow()
    end

    stale_after <= Hour(0) && return false
    isfile(output_path) && return false            # finished, not abandoned
    age = try
        Millisecond(round(Int, (time() - _claim_heartbeat_time(path)) * 1000))
    catch
        return false                               # vanished under us: another caller owns it
    end
    age > stale_after || return false

    @warn "Reclaiming a stale tile claim" tile=tile_name age=canonicalize(age)
    try
        rm(path; force = true, recursive = true)
        mkdir(path)                                # atomic: only one reclaimer wins
        touch(joinpath(path, CLAIM_HEARTBEAT_FILE))
        return true
    catch err
        err isa Base.IOError || rethrow()
        return false
    end
end

# The file inside a claim whose modification time is the owner's heartbeat.
const CLAIM_HEARTBEAT_FILE = "heartbeat"

function _claim_heartbeat_time(path)
    beat = joinpath(path, CLAIM_HEARTBEAT_FILE)
    return isfile(beat) ? mtime(beat) : (isdir(path) ? mtime(path) : error("no claim at $path"))
end

"""
    ClaimHeartbeat

A running heartbeat for one claim, from [`start_claim_heartbeat`](@ref). Stop it with
[`stop_claim_heartbeat!`](@ref).
"""
struct ClaimHeartbeat
    path::String
    process::Base.Process
end

"""
    start_claim_heartbeat(claim_dir, tile_name; interval = Minute(5), owner = getpid())
        -> ClaimHeartbeat

Refresh the heartbeat of the claim [`claim_tile!`](@ref) took on `tile_name` every `interval` until
[`stop_claim_heartbeat!`](@ref). The claim is then never stale while its owner runs, however long the
tile takes, and a claim whose owner died stops beating.

The refresh runs in a separate `sh` process that touches the claim's heartbeat file and exits within
one `interval` of the process `owner` (by default this one) ending, however it ends. It is a process and not a task because a tile
run computes for hours without yielding, and a Julia task waiting on a timer would not run until it did.
"""
function start_claim_heartbeat(claim_dir::AbstractString, tile_name::AbstractString;
                               interval::Period = Minute(5), owner::Integer = getpid())
    seconds = Dates.value(convert(Millisecond, interval)) / 1000
    seconds > 0 || throw(ArgumentError("interval must be positive, got $interval"))
    path = joinpath(claim_dir, tile_name, CLAIM_HEARTBEAT_FILE)
    isdir(dirname(path)) || throw(ArgumentError("no claim for $tile_name under $claim_dir"))
    # `$1` is the heartbeat path and `$2` the owner's process id.
    loop = """
        while kill -0 "\$2" 2>/dev/null; do
            touch "\$1" || exit 1
            sleep $(seconds)
        done
        """
    process = run(pipeline(`sh -c $loop claim-heartbeat $path $owner`;
                           stdout = devnull, stderr = stderr); wait = false)
    return ClaimHeartbeat(path, process)
end

"""
    stop_claim_heartbeat!(heartbeat::ClaimHeartbeat)

Stop refreshing the claim's heartbeat. Throws if the heartbeat had already stopped on its own, since
the claim may then have gone stale while its owner ran.
"""
function stop_claim_heartbeat!(heartbeat::ClaimHeartbeat)
    p = heartbeat.process
    running = process_running(p)
    running && kill(p)
    wait(p)
    running || error("the heartbeat of the claim at $(dirname(heartbeat.path)) stopped while its " *
                     "owner ran (exit code $(p.exitcode)), so the claim may have been reclaimed")
    return nothing
end

"""
    with_claim_heartbeat(f, claim_dir, tile_name; interval = Minute(5))

Run `f()` with the claim's heartbeat refreshed throughout, and stop it however `f` exits.
"""
function with_claim_heartbeat(f, claim_dir::AbstractString, tile_name::AbstractString;
                              interval::Period = Minute(5))
    heartbeat = start_claim_heartbeat(claim_dir, tile_name; interval)
    try
        return f()
    finally
        stop_claim_heartbeat!(heartbeat)
    end
end

"""
    weight_balanced_blocks(w, k) -> Vector{UnitRange{Int}}

Split `1:length(w)` into `k` contiguous ranges of as near equal total weight as the item granularity
allows, by closing each range as the running total crosses its share.

**Always exactly `k` ranges**, so `blocks[i]` is in bounds for every `i in 1:k`. Ranges may be empty
when `k` exceeds the number of items the crossing rule can separate — two equal weights split into
four blocks yields two singletons and two empties — so callers must tolerate an empty block, but never
a missing one.

Contiguous rather than scattered because the tile lists these partition are ordered by forcing chunk:
neighbours share cached cells, and a modulo split would make each process miss what another already
holds.
"""
function weight_balanced_blocks(w, k::Int)
    k >= 1 || throw(ArgumentError("k must be at least 1, got $k"))
    total = sum(w)
    blocks = UnitRange{Int}[]
    lo = 1
    acc = zero(eltype(w))
    for i in eachindex(w)
        acc += w[i]
        if length(blocks) < k - 1 && acc >= total * (length(blocks) + 1) / k
            push!(blocks, lo:i)
            lo = i + 1
        end
    end
    push!(blocks, lo:length(w))
    # The crossing rule closes a range only when the running total reaches the next share, so it can
    # run out of items before it has opened `k - 1` of them: two equal weights over four blocks closes
    # twice and leaves two short. Padding past the end keeps the count exactly `k`, which is what lets
    # `claim_order` index by block number without a bounds check of its own.
    n = length(w)
    while length(blocks) < k
        push!(blocks, n + 1:n)
    end
    return blocks
end

"""
    claim_order(weights, block, n_blocks) -> Vector{Int}

The order a worker should try to claim tiles in: its own contiguous weight-balanced block first, then
everything else heaviest-first.

Affinity first keeps a worker's run spatially coherent, which is what the chunk ordering buys. Stealing
afterwards removes the tail a static partition cannot: a worker that finishes early takes the heaviest
tile still unclaimed instead of idling, and heaviest-first (longest-processing-time-first) is what keeps
the last tile started small.
"""
function claim_order(weights, block::Int, n_blocks::Int)
    n_blocks >= 1 || throw(ArgumentError("n_blocks must be at least 1, got $n_blocks"))
    1 <= block <= n_blocks ||
        throw(ArgumentError("block must be in 1..$n_blocks, got $block"))
    n_blocks == 1 && return collect(eachindex(weights))
    mine = weight_balanced_blocks(weights, n_blocks)[block]
    rest = setdiff(eachindex(weights), mine)
    sort!(rest; by = i -> weights[i], rev = true)
    return vcat(collect(mine), rest)
end

"""
    software_provenance() -> Dict{String,Any}

The versions and source trees of GEMB, GEMB_ClimateForcing and GEMB_GlacierSims this process runs,
as `software_<package>_version` and `software_<package>_tree`. Both [`run_parameters`](@ref) and
[`tile_run_parameters`](@ref) include them, so a run under different code compares as a different
experiment.

The tree is the git tree hash of the package's `src` directory at `HEAD` when the package is a git
checkout — with `+dirty:` and a CRC-32C of the uncommitted changes to `src` and `Project.toml`
(including untracked files under `src`) when there are any — otherwise the `git-tree-sha1` the
active manifest records for it, and failing both `crc32c:` and a CRC-32C of every file under `src`
and of `Project.toml`. Computed once per process.
"""
function software_provenance()
    p = lock(_SOFTWARE_PROVENANCE_LOCK) do
        cached = _SOFTWARE_PROVENANCE[]
        cached === nothing || return cached
        p = Dict{String,Any}()
        for (label, mod) in (("gemb", GEMB), ("gemb_climateforcing", GEMB_ClimateForcing),
                             ("gemb_glaciersims", @__MODULE__))
            p["software_$(label)_version"] = string(pkgversion(mod))
            p["software_$(label)_tree"] = _package_tree(mod)
        end
        _SOFTWARE_PROVENANCE[] = p
    end
    return copy(p)
end

# Filled by the first call; the lock keeps concurrent first calls from each running the git queries.
const _SOFTWARE_PROVENANCE = Ref{Union{Nothing,Dict{String,Any}}}(nothing)
const _SOFTWARE_PROVENANCE_LOCK = ReentrantLock()

function _package_tree(mod::Module)
    dir = pkgdir(mod)
    dir === nothing && throw(ArgumentError("$(mod) has no package directory to identify"))
    git(args...) = Cmd(`git -C $dir $args`; ignorestatus = true)
    if isdir(joinpath(dir, ".git")) || isfile(joinpath(dir, ".git"))
        tree = readchomp(Cmd(`git -C $dir rev-parse HEAD:./src`))
        dirty = readchomp(git("status", "--porcelain", "--untracked-files=all", "--", "src",
                              "Project.toml"))
        isempty(dirty) && return tree
        untracked = filter(!isempty, split(readchomp(git("ls-files", "--others", "--exclude-standard",
                                                         "--", "src")), '\n'))
        crc = _crc_files(dir, untracked, Base._crc32c(read(git("diff", "HEAD", "--", "src", "Project.toml"))))
        return tree * "+dirty:" * string(crc; base = 16, pad = 8)
    end
    uuid = string(Base.PkgId(mod).uuid)
    for env in Base.load_path()
        manifest = Base.project_file_manifest_path(env)
        (manifest === nothing || !isfile(manifest)) && continue
        deps = get(Base.parsed_toml(manifest), "deps", Dict{String,Any}())
        for entry in get(deps, string(nameof(mod)), [])
            get(entry, "uuid", "") == uuid || continue
            haskey(entry, "git-tree-sha1") && return entry["git-tree-sha1"]
        end
    end
    # A plain directory, such as a `dev`ed copy outside git: identify it by its contents.
    files = [relpath(joinpath(root, f), dir) for (root, _, fs) in walkdir(joinpath(dir, "src")) for f in fs]
    crc = _crc_files(dir, sort!(push!(files, "Project.toml")), UInt32(0))
    return "crc32c:" * string(crc; base = 16, pad = 8)
end

# CRC-32C over each file's path (relative to `dir`) and contents, continuing from `crc`.
function _crc_files(dir, files, crc::UInt32)
    for f in files
        crc = Base._crc32c(f, crc)
        crc = Base._crc32c(read(joinpath(dir, f)), crc)
    end
    return crc
end

"""
    era5_land_record_span(cache_path) -> (; start, stop)

The first and last hourly stamps of the ERA5-Land record held in the disk cache at `cache_path`, the
`cache_path` passed to `climate_forcing`, read as `climate_forcing` reads it: the time axis of the
cached `sfc-2m-temperature` group, the only group whose time axis it reads, checked against the
record length of every (lon, lat, time) variable of every group in the cached consolidated metadata.

The cache is only read: nothing is fetched and nothing is written. `climate_forcing` serves a cached
record as it was when the cache was first filled and refuses any request past its end, so this is the
latest end a run against the cache can ask for. Throws when a variable group has no cached
consolidated metadata, when the time axis cannot be read in full, when it is not strictly increasing
in whole hours, or when a variable's record length differs from it: each of those is a cache
`climate_forcing` would also refuse.
"""
function era5_land_record_span(cache_path::AbstractString)
    isdir(cache_path) || throw(ArgumentError("no ERA5-Land disk cache at $cache_path"))
    groups = Dict(group => _open_cached_group(joinpath(cache_path, group))
                  for group in keys(GEMB_ClimateForcing.ERA5_LAND_STORES))
    dir = joinpath(cache_path, "sfc-2m-temperature")
    temperature = groups["sfc-2m-temperature"]
    haskey(temperature.arrays, "time") || throw(ArgumentError("the cached group $dir has no time axis"))
    t = temperature["time"]
    units = get(t.attrs, "units", "")
    m = match(r"hours since (\d{4}-\d{2}-\d{2})", units)
    m === nothing && throw(ArgumentError(
        "the time axis of $dir is in \"$units\", not hours since a date"))
    epoch = DateTime(m.captures[1])
    hours = try
        collect(Float64, t[:])
    catch e
        e isa InterruptException && rethrow()
        throw(ArgumentError("the time axis of $dir cannot be read in full from the cache " *
                            "($(sprint(showerror, e)))"))
    end
    isempty(hours) && throw(ArgumentError("the time axis of $dir is empty"))
    all(isfinite, hours) && all(isinteger, hours) && all(>(0), diff(hours)) || throw(ArgumentError(
        "the time axis of $dir is not strictly increasing in whole hours"))
    start = epoch + Hour(Int(first(hours)))
    stop = epoch + Hour(Int(last(hours)))
    variables = [a for g in values(groups) for a in values(g.arrays) if ndims(a) == 3]
    GEMB_ClimateForcing._check_cached_record(cache_path, hours, epoch, (start, stop), variables)
    return (; start, stop)
end

# One variable group of the ERA5-Land disk cache, read-only. `climate_forcing` reaches the cache
# through `Zarr.CachingStore`, which stores only the keys it read: the group's `.zmetadata` and the
# chunks read, with no `.zgroup` or `.zarray` for Zarr's format detection to find. ClimateForcing
# declares its own stores Zarr v2 instead of detecting the format, and this store does the same (Zarr
# detects the format of a consolidated group from the store under it, whatever `zopen` is told). It
# serves only what is on disk and refuses writes, so reading the record fetches and writes nothing.
struct _CachedGroupStore <: Zarr.AbstractStore
    dir::Zarr.DirectoryStore
end
Base.getindex(s::_CachedGroupStore, k::String) = s.dir[k]
Base.setindex!(::_CachedGroupStore, v, k::String) =
    throw(ArgumentError("the ERA5-Land disk cache is read here, not written; refusing to write $k"))
Zarr.subdirs(s::_CachedGroupStore, p) = Zarr.subdirs(s.dir, p)
Zarr.subkeys(s::_CachedGroupStore, p) = Zarr.subkeys(s.dir, p)
Zarr.storagesize(s::_CachedGroupStore, p) = Zarr.storagesize(s.dir, p)
Zarr.ZarrFormat(::_CachedGroupStore, path) = Zarr.ZarrFormat(2)

function _open_cached_group(dir)
    isfile(joinpath(dir, ".zmetadata")) || throw(ArgumentError(
        "the ERA5-Land cache has no consolidated metadata at $dir; load forcing through " *
        "`climate_forcing` with this cache_path once to fill it"))
    return Zarr.zopen(_CachedGroupStore(Zarr.DirectoryStore(dir)); consolidated = true,
                      fill_as_missing = false)
end

"""
    run_time_range(args, record; start_default = DateTime(1950, 1, 1)) -> (; time_range, source)

The `(start, stop)` a run script covers, from its command-line `args` and the forcing `record`
(`(; start, stop)`, as [`era5_land_record_span`](@ref) returns). `args[1]` is the start and `args[2]`
the end, each a year (that year's 1 January, 00:00) or an ISO date-time such as
`2026-08-14T23:00:00`. Without `args[2]` the run ends at the end of the record. Throws when the range
is empty or reaches outside the record, naming the record, so a run refuses at start-up rather than
in every tile. `source` says where each end came from, for the settings table.
"""
function run_time_range(args, record; start_default::DateTime = DateTime(1950, 1, 1))
    start = length(args) >= 1 ? _parse_run_time(args[1]) : start_default
    stop = length(args) >= 2 ? _parse_run_time(args[2]) : record.stop
    start < stop || throw(ArgumentError("the run starts at $start, not before its end $stop"))
    (record.start <= start && stop <= record.stop) || throw(ArgumentError(
        "the run $start to $stop reaches outside the cached ERA5-Land record, which runs from " *
        "$(record.start) to $(record.stop); end the run at or before $(record.stop), or refresh " *
        "the cache"))
    source = (length(args) >= 1 ? "start from ARGS" : "start default") * ", " *
             (length(args) >= 2 ? "end from ARGS" : "end of the cached record")
    return (; time_range = (start, stop), source)
end

function _parse_run_time(s::AbstractString)
    occursin(r"^\d{4}$", s) && return DateTime(parse(Int, s), 1, 1)
    t = tryparse(DateTime, s)
    t === nothing && throw(ArgumentError(
        "\"$s\" is neither a year nor an ISO date-time such as 2026-08-14T23:00:00"))
    return t
end
