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
_setting_value(x::Dates.DateTime) = Dates.format(x, "yyyy-mm-dd")
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
