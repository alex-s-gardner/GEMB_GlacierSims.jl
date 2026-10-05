"""
    merge_tile_perturbations(out_path, paths; force = false) -> out_path

Join tile files that ran **different temperature offsets of the same experiment** into one file whose
`delta_temperature` axis is the union of theirs, ascending.

This is how the perturbation grid is widened without re-running the points that already exist. Extending
`delta_temperature` cannot be done in place — unlike `time` it is a fixed dimension, and it indexes every
series, every per-run provenance variable and the restart group — so a new file is written and the inputs
are left alone.

# What must agree, and why it is checked rather than assumed

A merged file claims its `dv` came from one experiment sampled at more offsets. That is only true if the
passes differ in nothing but the offsets, so this refuses unless

- the run parameters match, by the same [`run_parameter_differences`](@ref) the restart path uses — a
  changed spinup window, model parameter or downscaling policy makes them different experiments, and
  splicing those together would produce a response surface with a discontinuity that looks physical;
- the time axis, band grid and precipitation scalings are identical, since they are shared axes.
  `layer` is exempt: it is the padded width of the restart profiles, so it is set by the deepest
  column a pass happened to produce rather than by any setting. The merged width is the largest of
  the sources and shorter profiles keep their fill padding, which is the convention the restart
  group already documents and reads back through `valid_layers`;
- the offsets are disjoint, so no value has two sources. `force` downgrades this to a warning and takes
  the first file that carries the value.

# Provenance

Every global attribute is carried over from the first file, and `history` gains a line naming each source
and the offsets it contributed, so the file records which points were run when. Per-run spinup provenance
(`spinup_cycles`, `spinup_converged`, … — one value per band × offset × scaling) travels with its own
offsets rather than being reduced or dropped, which is what keeps a merged file as answerable about how a
column was settled as an unmerged one.

Merging is driven by each variable's declared dimensions, not by a list of names: a variable is merged
along `delta_temperature` if it has that dimension and copied otherwise. A series added to the writer is
therefore merged without touching this function.
"""
function merge_tile_perturbations(out_path::AbstractString,
                                  paths::AbstractVector{<:AbstractString}; force::Bool = false)
    length(paths) >= 2 || throw(ArgumentError("merging needs at least two files, got $(length(paths))"))
    for p in paths
        isfile(p) || throw(ArgumentError("no such tile file: $p"))
    end

    datasets = [NCDatasets.NCDataset(p, "r") for p in paths]
    try
        _assert_mergeable(datasets, paths)

        # Where each source's offsets land in the merged axis. Built before anything is written so a
        # duplicate or a disjointness failure is reported before a partial file exists.
        offsets = [collect(Float64, ds["delta_temperature"][:]) for ds in datasets]
        merged = sort!(unique(reduce(vcat, offsets)))
        _assert_offsets_disjoint(offsets, paths, force)
        # First source wins a duplicate, which only arises under `force`.
        target = [[findfirst(≈(x), merged)::Int for x in o] for o in offsets]

        # Widest restart profile across the sources; shorter ones keep their fill padding.
        n_layers = maximum(ds.dim["layer"] for ds in datasets)

        mkpath(dirname(abspath(out_path)))
        NCDatasets.NCDataset(out_path, "c") do out
            _copy_tile_dims!(out, first(datasets), length(merged), n_layers)
            _copy_tile_globals!(out, first(datasets), paths, offsets, merged)
            _merge_group!(out, datasets, target, merged)

            # The restart group is state rather than series, but it is indexed by offset like everything
            # else, so a merged file that omitted it would be unrestartable at exactly the new points.
            if haskey(first(datasets).group, "restart")
                gout = NCDatasets.defGroup(out, "restart")
                gsrc = [ds.group["restart"] for ds in datasets]
                for (k, v) in first(gsrc).attrib
                    gout.attrib[k] = v
                end
                _merge_group!(gout, gsrc, target, merged)
            end
        end
    finally
        close.(datasets)
    end

    return out_path
end

# Every check that must pass before a merge is a merge and not a splice.
function _assert_mergeable(datasets, paths)
    ref, refp = first(datasets), first(paths)
    for (ds, p) in zip(datasets[2:end], paths[2:end])
        get(ds.attrib, "geotile_id", "") == get(ref.attrib, "geotile_id", "") ||
            throw(ArgumentError("$p is a different geotile ($(get(ds.attrib, "geotile_id", "?"))) " *
                                "than $refp ($(get(ref.attrib, "geotile_id", "?")))"))
        # Raw stored values, not decoded times: a decode difference is not a disagreement.
        for v in ("time", "band_center", "band_lower", "band_upper", "precipitation_scaling")
            haskey(ref, v) || continue
            ds[v][:] == ref[v][:] ||
                throw(ArgumentError("$v differs between $refp and $p; these are shared axes and a " *
                                    "merge cannot reconcile them"))
        end
        diffs = run_parameter_differences(_read_run_parameters(ref), _read_run_parameters(ds))
        isempty(diffs) || throw(ArgumentError(
            "$p was run under different settings than $refp, so they are different experiments and " *
            "must not share a response surface: " * string(diffs)))
    end
    return nothing
end

function _assert_offsets_disjoint(offsets, paths, force)
    seen = Dict{Float64,String}()
    for (o, p) in zip(offsets, paths), x in o
        if haskey(seen, x)
            msg = "delta_temperature $x appears in both $(seen[x]) and $p"
            force || throw(ArgumentError(msg * "; pass force = true to take the first"))
            @warn msg * "; taking the value from $(seen[x])"
        else
            seen[x] = p
        end
    end
    return nothing
end

# Dimensions of the output: the reference file's, with `delta_temperature` and `layer` resized.
# `time` stays unlimited so a merged file can still be extended by `append_glacier_tile_netcdf`.
function _copy_tile_dims!(out, ref, n_offsets::Int, n_layers::Int)
    for (name, len) in ref.dim
        if name == "delta_temperature"
            NCDatasets.defDim(out, name, n_offsets)
        elseif name == "layer"
            NCDatasets.defDim(out, name, n_layers)
        elseif name == "time"
            NCDatasets.defDim(out, name, Inf)
        else
            NCDatasets.defDim(out, name, len)
        end
    end
    return nothing
end

function _copy_tile_globals!(out, ref, paths, offsets, merged)
    for (k, v) in ref.attrib
        k == "history" && continue
        out.attrib[k] = v
    end
    contributed = join(("$(basename(p)) [" * join(string.(o), ", ") * "]"
                        for (p, o) in zip(paths, offsets)), "; ")
    out.attrib["history"] = get(ref.attrib, "history", "") *
        "\n$(Dates.format(now(), "yyyy-mm-ddTHH:MM:SS")): delta_temperature axis merged to [" *
        join(string.(merged), ", ") * "] from " * contributed *
        " by GEMB_GlacierSims.merge_tile_perturbations"
    return nothing
end

# Copy every variable of `srcs[1]`, merging along `delta_temperature` where a variable has it.
function _merge_group!(out, srcs, target, merged)
    ref = first(srcs)
    for name in keys(ref)
        vref = ref[name]
        dims = NCDatasets.dimnames(vref)
        axis = findfirst(==("delta_temperature"), dims)

        vout = _def_like(out, name, vref, dims)

        # A scalar carries no axis to merge along and cannot be sliced.
        if isempty(dims)
            vout[] = vref[]
            continue
        end
        if name == "delta_temperature"
            vout[1:length(merged)] = merged
            continue
        end
        if axis === nothing
            # Not offset-indexed, so it is a shared axis and `_assert_mergeable` has already
            # established the sources agree on it.
            data = vref[:]
            vout[_ranges(data)...] = data
            continue
        end

        # Explicit ranges rather than colons throughout, for two reasons: `time` is an unlimited
        # dimension and is length zero in a freshly created file, so a colon there selects nothing
        # and the write is silently dropped; and a source's `layer` extent may be narrower than the
        # merged one, where writing only the source's own width is what leaves the tail at the fill
        # value that `valid_layers` implies.
        for (src, idx) in zip(srcs, target)
            vsrc = src[name]
            for (i_src, i_out) in enumerate(idx)
                sel_src = Any[Colon() for _ in dims]
                sel_src[axis] = i_src
                slice = vsrc[sel_src...]
                sel_out = Any[1:n for n in size(slice)]
                insert!(sel_out, axis, i_out)
                vout[sel_out...] = slice
            end
        end
    end
    return nothing
end

# A variable in `out` matching `vref`'s type, dimensions, attributes, fill value and compression, so a
# merged file is structurally the same artifact as an unmerged one.
function _def_like(out, name, vref, dims)
    attrib = Dict{String,Any}()
    fill = nothing
    for (k, v) in vref.attrib
        k == "_FillValue" ? (fill = v) : (attrib[k] = v)
    end
    T = eltype(vref.var)
    kw = Dict{Symbol,Any}(:attrib => attrib)
    fill === nothing || (kw[:fillvalue] = fill)
    # `deflate` reports `(shuffle, is_deflated, level)`, not a level — reading it as one silently
    # passes a tuple to `defVar` and fails inside the C call.
    shuffle, deflated, level = NCDatasets.deflate(vref.var)
    shuffle && (kw[:shuffle] = true)
    deflated && (kw[:deflatelevel] = Int(level))
    return NCDatasets.defVar(out, name, T, dims; kw...)
end

# `1:n` per dimension of `a`. Writing an unlimited dimension needs the extent stated, so colons are
# avoided everywhere a variable is written for the first time.
_ranges(a) = ntuple(i -> 1:size(a, i), ndims(a))
