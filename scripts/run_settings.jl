# The settings table `derive_downscaling_parameter_tiles.jl` and `gemb_tile_sweep.jl` print before
# either does any work.
#
# Nearly every value those two run under is overridable from the environment, so the value in force is
# not always the value in the file. Each row therefore states where its value came from, which is what
# makes a stale `export` in the launching shell visible instead of silently shaping a run measured in
# hours. `SHOW_SETTINGS=1` prints the table and stops, which is how the launcher shows it once on the
# terminal before starting the workers.
#
# Included rather than added to the package: this is how a script reports itself, not part of the
# library's interface.

const SHOW_SETTINGS = get(ENV, "SHOW_SETTINGS", "0") == "1"

# The directory under `CLIMATE_CACHE` holding one directory of tile runs per physics member
# (`tile_run_dir`), which the tile sweep writes by default and the native run takes its reference tile
# runs from, so the two agree without either being told. `run_tile_sweep.sh` carries the same name. A
# new name is a new set of tile runs: GlobalGlacierAnalysis.jl reads every file in the directory its
# GEMB run id and physics member name.
const TILE_RUN_DIR = "tile_runs_v2"

# The last calendar year of the default spinup climatology window, shared by the tile sweep and the
# native run so that their windows, which the native run's reference check compares, agree. See
# `SPINUP_CLIMATOLOGY_WINDOW` in `gemb_tile_sweep.jl`.
const SPINUP_CLIMATOLOGY_LAST_YEAR = 1980

# The physics member the tile sweep and the native run are run as (`GEMB_GlacierSims.physics_member`):
# its grain-growth level sets the GEMB snow settings, its band decoupling level the `band_decoupling`
# rule of `resolve_downscaling`, and its impurity level the snow impurity loading. Shared so that a
# native cell runs as its reference tile's member. `PHYSICS_MEMBER` overrides the default, which is
# Flanner & Zender grain growth with 350 kg m-3 fresh snow, the fitted band k, and the production
# background impurity table; `run_tile_sweep.sh` and `run_native_sweep.sh` carry the same default.
# Each member's tiles go to a directory of their own (`tile_run_dir`), so no two members' tiles share
# a directory.
import GEMB_GlacierSims
const DEFAULT_PHYSICS_MEMBER = "B-kfit-table"
const PHYSICS_MEMBER = GEMB_GlacierSims.physics_member(get(ENV, "PHYSICS_MEMBER", DEFAULT_PHYSICS_MEMBER))

"""
    tile_run_dir(climate_cache) -> String

The directory of tile runs for `PHYSICS_MEMBER` under `climate_cache`:
`<climate_cache>/<TILE_RUN_DIR>/<member>`. The tile sweep writes it by default and the native run
takes its reference tiles from it.
"""
tile_run_dir(climate_cache) = joinpath(climate_cache, TILE_RUN_DIR, PHYSICS_MEMBER.id)

"""
    physics_model_parameters(; kwargs...) -> ModelParameters

`GEMB.initialize_parameters` under `PHYSICS_MEMBER`'s grain level, with `kwargs` for every other
field.
"""
physics_model_parameters(; kwargs...) =
    GEMB_GlacierSims.physics_model_parameters(PHYSICS_MEMBER; kwargs...)

# The regional background snow impurity loading (`background_black_carbon`), from the member's impurity
# level: the production table (`bc_dust`, `snicar_adv3`) for `table`, the table's sensitivity forms for
# `tbc` (`bc`), `tdang` (`dang2015` optics) and `traw` (`none`), and `off` for `bc0` and `cvis`.
# `BACKGROUND_BLACK_CARBON` and `BACKGROUND_BLACK_CARBON_OPTICS`, when set, must name the same loading:
# a stale `export` that disagrees with the member stops the script here instead of running a tile under
# a member it is not.
const BACKGROUND_BLACK_CARBON, BACKGROUND_BLACK_CARBON_OPTICS = let level = PHYSICS_MEMBER.impurity_level
    basis = level.basis === :effective ? :off : level.basis
    optics = basis === :off ? :snicar_adv3 : level.optics
    for (var, value) in (("BACKGROUND_BLACK_CARBON", basis), ("BACKGROUND_BLACK_CARBON_OPTICS", optics))
        haskey(ENV, var) || continue
        basis === :off && var == "BACKGROUND_BLACK_CARBON_OPTICS" && continue
        ENV[var] == string(value) || error(
            "$var=$(ENV[var]) disagrees with physics member $(PHYSICS_MEMBER.id), whose impurity " *
            "level $(PHYSICS_MEMBER.impurity) sets $(value); choose the loading with PHYSICS_MEMBER " *
            "and unset $var")
    end
    basis, optics
end

# The effective loading table a `cvis` member reads (`EFFECTIVE_LOADING_TABLE`), and `nothing` for any
# other member. Required for a `cvis` member and refused for the others, so a member is always run at
# the loading its name says.
const EFFECTIVE_LOADING_TABLE = let level = PHYSICS_MEMBER.impurity_level
    path = get(ENV, "EFFECTIVE_LOADING_TABLE", nothing)
    if level.basis === :effective
        path === nothing && error("physics member $(PHYSICS_MEMBER.id) takes an effective loading; " *
                                  "set EFFECTIVE_LOADING_TABLE to its regional table")
        isfile(path) || error("EFFECTIVE_LOADING_TABLE=$path does not exist")
    else
        path === nothing || error("EFFECTIVE_LOADING_TABLE is set, but physics member " *
                                  "$(PHYSICS_MEMBER.id) takes no effective loading")
    end
    path
end

"""
    tile_background_black_carbon(cells) -> NamedTuple or nothing

The `background_black_carbon` keyword for a tile whose core is `cells`, under
`BACKGROUND_BLACK_CARBON` and `BACKGROUND_BLACK_CARBON_OPTICS`: `nothing` when the former is `off`.
"""
tile_background_black_carbon(cells) = BACKGROUND_BLACK_CARBON === :off ? nothing :
    GEMB_GlacierSims.background_black_carbon(cells; basis = BACKGROUND_BLACK_CARBON,
                                             optics = BACKGROUND_BLACK_CARBON_OPTICS)

"""
    tile_effective_black_carbon(cells) -> NamedTuple or nothing

The `effective_black_carbon` keyword for a tile whose core is `cells`: the loading in
`EFFECTIVE_LOADING_TABLE`, or `nothing` when the member takes none.
"""
tile_effective_black_carbon(cells) = EFFECTIVE_LOADING_TABLE === nothing ? nothing :
    GEMB_GlacierSims.effective_black_carbon(cells, EFFECTIVE_LOADING_TABLE;
                                            cap = PHYSICS_MEMBER.impurity_level.cap)

"""
    tile_band_decoupling(tile, intervals) -> NamedTuple

The `band_decoupling` rule of `resolve_downscaling` for `tile` under `PHYSICS_MEMBER`: the member's rule,
with the tile's per-band Shaw `k` (`shaw_band_decoupling` over its core hypsometry) for a `:shaw`
member.
"""
function tile_band_decoupling(tile, intervals)
    PHYSICS_MEMBER.band_rule.rule === :shaw || return PHYSICS_MEMBER.band_rule
    areas = GEMB_GlacierSims.tile_interval_areas(tile.core, intervals)
    return (; rule = :shaw,
            band_k = GEMB_GlacierSims.shaw_band_decoupling(tile.bounds, intervals, areas).band_k)
end

"""
    tile_decoupling_factor_prior(tile) -> Float64 or nothing

The `decoupling_factor_prior` of `resolve_downscaling` for `tile`: the glacier-area weighted Shaw et al.
(2025) `k` of the glaciers in it (`shaw_decoupling_prior`). It is the whole band `k` of a tile whose
cells carry no glacier fraction, and the fallback for timesteps no fit or climatology resolves.
"""
tile_decoupling_factor_prior(tile) = GEMB_GlacierSims.shaw_decoupling_prior(tile.bounds)

# Column profiles for the WGMS density test (`ColumnProfileRecorder`), taken only for the bands that
# hold a WGMS density record and only when `COLUMN_PROFILES` names the table of those records: a CSV with
# columns `tile` (e.g. N46_E010), `date` (yyyy-mm-dd) and `z` (m, the record's elevation). Each tile
# with records gets `column_profiles/<tile>.nc` under the sweep's output directory; tile files
# themselves are unchanged. Off (`nothing`) by default.
const COLUMN_PROFILES = let path = get(ENV, "COLUMN_PROFILES", nothing)
    path === nothing || isfile(path) || error("COLUMN_PROFILES=$path does not exist")
    path === nothing ? nothing : begin
        raw = GEMB_GlacierSims._read_csv_strings(path)
        absent = setdiff(["tile", "date", "z"], names(raw))
        isempty(absent) || error("COLUMN_PROFILES table $path lacks column(s) $(join(absent, ", "))")
        [(; tile = String(strip(r.tile)), date = Dates.DateTime(Dates.Date(strip(r.date))),
            z = parse(Float64, strip(r.z))) for r in eachrow(raw)]
    end
end

"""
    tile_column_profiles(tile_name, intervals) -> ColumnProfileRecorder or nothing

The recorder for the tile named `tile_name` (`N46_E010`): the dates of its `COLUMN_PROFILES` records
and the bands among `intervals` holding their elevations, or `nothing` when profiles are off or the
tile has no record inside its bands.
"""
function tile_column_profiles(tile_name, intervals)
    COLUMN_PROFILES === nothing && return nothing
    records = [r for r in COLUMN_PROFILES if r.tile == tile_name]
    bands = Float64[iv.center for iv in intervals if any(r -> iv.lo <= r.z < iv.hi, records)]
    isempty(bands) && return nothing
    dates = [r.date for r in records if any(iv -> iv.lo <= r.z < iv.hi, intervals)]
    return GEMB_GlacierSims.ColumnProfileRecorder(bands, dates)
end

column_profiles_setting() =
    env_setting("column profiles (WGMS density bands)",
                COLUMN_PROFILES === nothing ? "off" :
                "$(length(COLUMN_PROFILES)) records in $(length(unique(r.tile for r in COLUMN_PROFILES))) tiles",
                "COLUMN_PROFILES")

physics_member_setting() =
    env_setting("physics member", "$(PHYSICS_MEMBER.id) ($(PHYSICS_MEMBER.description))",
                "PHYSICS_MEMBER")

background_black_carbon_setting() =
    setting("background snow black carbon",
            BACKGROUND_BLACK_CARBON === :off ? "off" :
            "$(BACKGROUND_BLACK_CARBON), $(BACKGROUND_BLACK_CARBON_OPTICS) optics", "physics member")

effective_black_carbon_setting() =
    env_setting("effective snow impurity table",
                EFFECTIVE_LOADING_TABLE === nothing ? "none" : EFFECTIVE_LOADING_TABLE,
                "EFFECTIVE_LOADING_TABLE")

"""
    setting(label, value, source = "default") -> NamedTuple

One row of a settings table: what the option is, the value in force, and where that value came from.
"""
setting(label, value, source::AbstractString = "default") =
    (; label = String(label), value = _setting_value(value), source = String(source))

"""
    env_setting(label, value, var) -> NamedTuple

A [`setting`](@ref) whose source is `ENV[var]` when that variable is set, and the script's own default
otherwise. `var` is named here as well as at the constant it overrides, so a row cannot claim an
override the script does not honour.
"""
env_setting(label, value, var::AbstractString) =
    setting(label, value, haskey(ENV, var) ? "ENV[$var]" : "default")

# `string` alone reads badly for the types these tables carry: a DateTime keeps a `T00:00:00` no
# setting here resolves to, a vector keeps its brackets, and `true`/`false` states a switch less
# plainly than yes/no.
_setting_value(x) = string(x)
_setting_value(x::Bool) = x ? "yes" : "no"
_setting_value(x::Dates.DateTime) =
    Dates.format(x, x == Dates.DateTime(Dates.Date(x)) ? "yyyy-mm-dd" : "yyyy-mm-ddTHH:MM")
_setting_value(x::AbstractVector) = join(map(_setting_value, x), ", ")
_setting_value(x::Tuple{Dates.DateTime,Dates.DateTime}) =
    "$(_setting_value(x[1])) .. $(_setting_value(x[2]))"

"""
    print_settings([io = stdout], title, rows)

Print `rows` under `title` as an aligned table of option, value and source.

Goes to `stdout` rather than through `@info`, so it stays one block of text in a log a launcher
redirects, instead of being interleaved with the log lines of a run already under way.
"""
function print_settings(io::IO, title::AbstractString, rows)
    isempty(rows) && throw(ArgumentError("a settings table needs at least one row"))
    w_label = maximum(length(r.label) for r in rows)
    w_value = maximum(length(r.value) for r in rows)
    width = max(length(title), 2 + w_label + 2 + w_value + 2 + maximum(length(r.source) for r in rows))
    println(io)
    println(io, title)
    println(io, '─'^width)
    for r in rows
        println(io, "  ", rpad(r.label, w_label), "  ", rpad(r.value, w_value), "  ", r.source)
    end
    println(io, '─'^width)
    flush(io)
    return nothing
end

print_settings(title::AbstractString, rows) = print_settings(stdout, title, rows)

"""
    exit_on_failures(summary, what)

End the process with exit status 1 when any row of `summary` (a table with a `status` column, or
`nothing`) has status `"failed"`, after printing how many to `stderr`. A worker that exits 0 has run
every unit it claimed without a failure, which is what a launcher's own exit status is built from.
"""
function exit_on_failures(summary, what::AbstractString)
    summary === nothing && return nothing
    n = isempty(summary) ? 0 : count(==("failed"), summary.status)
    n == 0 && return nothing
    println(stderr, "$n $what failed; see the log and the summary for each error")
    flush(stderr)
    exit(1)
end
