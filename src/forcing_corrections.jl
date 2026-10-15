"""
Calibrated forcing corrections: the precipitation scaling and temperature offset fitted per 2° geotile
by GlobalGlacierAnalysis.jl's `gemb_calibration`, looked up for the cells a native-resolution run
covers.

The fit is made against tile runs ([`gemb_glacier_tile`](@ref)) over a (ΔT × pscale) grid, so a
correction is only meaningful for forcing built the same way: lapsed and decoupled by the tile's
[`AppliedDownscaling`](@ref), with the offset and the scaling applied afterwards. That is what
[`gemb_native_cell`](@ref) does.

Corrections are constant within a geotile, because that is the resolution the altimetry constrains
them at. A geotile the calibration did not fit — one with no altimetry, or whose ice the binning
dropped — borrows the nearest fitted geotile's correction, and the loan is recorded with its distance.
"""

"""
    read_forcing_corrections(path) -> Dict{String,@NamedTuple{pscale::Float64, ΔT::Float64, group::Int}}

The per-geotile corrections in a `gemb_calibration` output (`*_gembfit.arrow`), keyed by geotile id
(`"lat[+60+62]lon[-142-140]"`, the form [`geotile_id`](@ref) produces).

Throws if a required column is missing, if any `pscale` is not positive and finite or any `ΔT` is not
finite, or if one id appears twice with different values: each of those is a calibration that cannot
be applied, and applying it anyway would fail far from the cause.
"""
function read_forcing_corrections(path::AbstractString)
    isfile(path) || throw(ArgumentError("no forcing-correction file at $path"))
    df = GeoDataFrames.read(path)
    required = ("id", "pscale", "ΔT", "group")
    absent = [c for c in required if !(c in names(df))]
    isempty(absent) || throw(ArgumentError(
        "forcing-correction file $path lacks column(s) $(join(absent, ", ")); expected the " *
        "`gemb_calibration` output with columns $(join(required, ", "))"))
    isempty(df.id) && throw(ArgumentError("forcing-correction file $path holds no geotiles"))

    corrections = Dict{String,@NamedTuple{pscale::Float64, ΔT::Float64, group::Int}}()
    for row in eachrow(df)
        id = String(row.id)
        pscale = Float64(row.pscale)
        ΔT = Float64(row.ΔT)
        (isfinite(pscale) && pscale > 0) || throw(ArgumentError(
            "geotile $id in $path has precipitation scaling $pscale; a scaling must be positive " *
            "and finite"))
        isfinite(ΔT) || throw(ArgumentError(
            "geotile $id in $path has temperature offset $ΔT; an offset must be finite"))
        value = (; pscale, ΔT, group = Int(row.group))
        if haskey(corrections, id)
            corrections[id] == value || throw(ArgumentError(
                "geotile $id appears twice in $path with different corrections " *
                "($(corrections[id]) and $value)"))
            continue
        end
        corrections[id] = value
    end
    return corrections
end

"""
    forcing_correction(geotile_id, corrections; max_distance_km = 1000)
        -> (; pscale, ΔT, group, donor, distance_km)

The correction a cell in `geotile_id` runs under. A geotile present in `corrections` is its own donor at
zero distance. One that is not borrows the fitted geotile whose center is nearest by great-circle
distance, which is how the 2° GEMB ensemble treats a geotile with no tile of its own.

`max_distance_km` is the backstop on that loan: wide enough for an isolated island beside a fitted
neighbour, and tight enough that a correction file missing a whole region throws rather than spreading
one distant fit across it.
"""
function forcing_correction(geotile_id::AbstractString, corrections; max_distance_km::Real = 1000)
    isempty(corrections) && throw(ArgumentError("no forcing corrections to draw from"))
    if haskey(corrections, geotile_id)
        c = corrections[geotile_id]
        return (; c.pscale, c.ΔT, c.group, donor = String(geotile_id), distance_km = 0.0)
    end

    lat, lon = _geotile_center(geotile_id)
    best_id, best_d = "", Inf
    # Sorted so a tie between two equidistant donors resolves the same way on every call.
    for id in sort!(collect(keys(corrections)))
        d = _haversine_km(lat, lon, _geotile_center(id)...)
        d < best_d && ((best_id, best_d) = (id, d))
    end
    best_d <= max_distance_km || throw(ArgumentError(
        "geotile $geotile_id has no forcing correction, and the nearest fitted geotile " *
        "($best_id) is $(round(Int, best_d)) km away, beyond the $(max_distance_km) km cap"))
    c = corrections[best_id]
    return (; c.pscale, c.ΔT, c.group, donor = best_id, distance_km = best_d)
end

# Geotile center as (latitude, longitude), the argument order `_haversine_km` takes, parsed from the id
# `geotile_id` writes. Parsed by pattern rather than by character offset so a malformed id throws here.
const _GEOTILE_ID_REGEX = r"^lat\[([+-]\d{2})([+-]\d{2})\]lon\[([+-]\d{3})([+-]\d{3})\]$"

function _geotile_center(id::AbstractString)
    m = match(_GEOTILE_ID_REGEX, id)
    m === nothing && throw(ArgumentError("not a geotile id: \"$id\" (expected e.g. lat[+60+62]lon[-142-140])"))
    lat_min, lat_max, lon_min, lon_max = parse.(Int, m.captures)
    return ((lat_min + lat_max) / 2, (lon_min + lon_max) / 2)
end

"""
    reference_forcing_corrections(reference_ensemble_file) -> String

The `gemb_calibration` output GlobalGlacierAnalysis.jl writes for the ensemble member whose aligned
binned file is `reference_ensemble_file` (`…_aligned.jld2`): `…_synthesized_gembfit.arrow` beside it.
GlobalGlacierAnalysis.jl names the synthesized file by replacing `aligned.jld2` with `synthesized.jld2`
and its fit by replacing `.jld2` with `_gembfit.arrow`. Throws when the name does not end in
`_aligned.jld2` or the file does not exist.
"""
function reference_forcing_corrections(reference_ensemble_file::AbstractString)
    endswith(reference_ensemble_file, "_aligned.jld2") || throw(ArgumentError(
        "the reference ensemble member $reference_ensemble_file is not an `…_aligned.jld2` file, " *
        "so its `gemb_calibration` output cannot be named"))
    path = replace(reference_ensemble_file, r"aligned\.jld2$" => "synthesized_gembfit.arrow")
    isfile(path) || throw(ArgumentError(
        "no `gemb_calibration` output for the reference ensemble member at $path; run the " *
        "calibration for that member, or name the correction file with FORCING_CORRECTIONS"))
    return path
end

"""
    gga_reference_ensemble_file(gga_dir) -> String

The reference ensemble member that the GlobalGlacierAnalysis.jl checkout at `gga_dir` declares, read
from the `const reference_ensemble_file = "…"` in its `src/utilities_project.jl` without loading that
package. Throws unless the file declares it exactly once, as a string literal.
"""
function gga_reference_ensemble_file(gga_dir::AbstractString)
    path = joinpath(gga_dir, "src", "utilities_project.jl")
    isfile(path) || throw(ArgumentError(
        "no GlobalGlacierAnalysis.jl source at $path, so the reference ensemble member is unknown; " *
        "set GGA_PROJECT to its checkout, or REFERENCE_ENSEMBLE_FILE or FORCING_CORRECTIONS"))
    found = Any[]
    _find_const_assignments!(found, Meta.parseall(read(path, String); filename = path),
                             :reference_ensemble_file)
    length(found) == 1 || throw(ArgumentError(
        "$path declares `reference_ensemble_file` $(length(found)) times; expected once"))
    only(found) isa String || throw(ArgumentError(
        "`reference_ensemble_file` in $path is not a string literal, so it cannot be read without " *
        "loading GlobalGlacierAnalysis.jl; set REFERENCE_ENSEMBLE_FILE"))
    return only(found)
end

# The right-hand sides of top-level `const name = …`, including those a trailing `;` wraps in a
# `:toplevel` expression.
function _find_const_assignments!(found, ex, name::Symbol)
    ex isa Expr || return found
    if ex.head === :toplevel
        foreach(a -> _find_const_assignments!(found, a, name), ex.args)
    elseif ex.head === :const
        a = only(ex.args)
        a isa Expr && a.head === :(=) && a.args[1] === name && push!(found, a.args[2])
    end
    return found
end

"""
    forcing_corrections_provenance_path(corrections) -> String

Where the tile-run provenance of a `gemb_calibration` output is recorded: beside it, with `.arrow`
replaced by `_provenance.toml`.

The file records the tile sweep the corrections were fitted on:

```toml
tile_dir = "/mnt/bylot-r3/data/era5land/tile_runs_v2"   # the directory of tile runs the fit read

[tile_run_parameters]          # run parameters every tile file the fit read records, with one value
software_gemb_tree = "…"
software_gemb_climateforcing_tree = "…"
software_gemb_glaciersims_tree = "…"
# any further run parameter, e.g. software_*_version or model_*
```
"""
forcing_corrections_provenance_path(corrections::AbstractString) =
    replace(corrections, r"\.arrow$" => "_provenance.toml")

# The run parameters a corrections provenance file must record: the source trees of the code that
# produced the tile runs, which is what distinguishes one sweep's physics from another's.
const _CORRECTIONS_REQUIRED_PARAMETERS =
    Tuple("software_$(p)_tree" for p in ("gemb", "gemb_climateforcing", "gemb_glaciersims"))

"""
    assert_corrections_fitted_on(corrections, reference_dir, reference_parameters = nothing)

Throw unless the forcing corrections at `corrections` were fitted on the tile sweep in `reference_dir`,
from the provenance file [`forcing_corrections_provenance_path`](@ref) names: its `tile_dir` must be
`reference_dir` (after resolving links), and every run parameter it records, which must include the
three `software_*_tree` keys, must equal the same key of `reference_parameters` (the `parameters` of a
reference tile file's [`read_glacier_tile_status`](@ref)). With `reference_parameters = nothing` only the
directory is compared.

A native-resolution run applies the corrections to its own forcing; corrections fitted on a sweep run
by other code, or on another directory of tile runs, would be applied to a model they were not fitted
for, and nothing in the corrections themselves shows it.
"""
function assert_corrections_fitted_on(corrections::AbstractString, reference_dir::AbstractString,
                                      reference_parameters = nothing)
    path = forcing_corrections_provenance_path(corrections)
    isfile(path) || throw(ArgumentError(
        "the forcing corrections $corrections carry no tile-run provenance ($path is absent), so " *
        "nothing shows which tile sweep they were fitted on; `gemb_calibration` must record it"))
    prov = TOML.parsefile(path)
    tile_dir = get(prov, "tile_dir", nothing)
    tile_dir isa AbstractString || throw(ArgumentError("$path records no `tile_dir`"))
    isdir(tile_dir) || throw(ArgumentError(
        "the forcing corrections were fitted on $tile_dir ($path), which does not exist here"))
    realpath(tile_dir) == realpath(reference_dir) || throw(ArgumentError(
        "the forcing corrections were fitted on the tile runs in $tile_dir ($path), not on the " *
        "reference sweep $reference_dir"))
    recorded = get(prov, "tile_run_parameters", nothing)
    recorded isa AbstractDict || throw(ArgumentError("$path records no [tile_run_parameters]"))
    absent = [k for k in _CORRECTIONS_REQUIRED_PARAMETERS if !haskey(recorded, k)]
    isempty(absent) || throw(ArgumentError(
        "$path does not record $(join(absent, ", ")), so the code the fitted tile runs came from " *
        "is unknown"))
    reference_parameters === nothing && return nothing
    differences = String[]
    for key in sort!(collect(keys(recorded)))
        stored = get(reference_parameters, key, nothing)
        stored !== nothing && _attribute_value(stored) == _attribute_value(recorded[key]) && continue
        push!(differences, "$key: fitted on $(repr(recorded[key])), reference $(repr(stored))")
    end
    isempty(differences) || throw(ArgumentError(
        "the forcing corrections were fitted on tile runs that differ from the reference sweep " *
        "$reference_dir ($path): $(join(differences, "; "))"))
    return nothing
end
