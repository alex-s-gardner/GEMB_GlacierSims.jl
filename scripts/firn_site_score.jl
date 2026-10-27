# Grade firn-site sweeps against the cores, by the registered benchmark.
#
#     julia --project=. scripts/firn_site_score.jl <library> <runs_dir> <reference> <out_dir>
#
# `<runs_dir>` holds one `<config>.jls` per configuration, as `firn_site_sweep.jl` writes them, and
# `<reference>` names the configuration the others are compared with. Metrics, errors, site weights and
# the decision rule are those of `gemb_review_2026-10-14/reports/firn-benchmark-registration.md`:
#
# - per core: `bias_0_1`, `bias_1_5`, `bias_5_10`, `bias_10_20` (mean model - core over the core rows
#   whose midpoint falls in the bin, given at least 1 m of sample; model density is the mean over each
#   row's own interval), `fac10`, `fac20` (model - core firn air content over the core's span, when its
#   rows cover 90 % of it), `z550`, `z830` (model - core depth of first reaching the density, when both
#   do: the shallowest depth below 3 m from which the median of all deeper samples is past the
#   density, so that an ice layer is not taken for the transition), `rmse` (reported only);
# - each residual divided by sqrt(σ_obs² + σ_rep²). σ_obs is the row error of the density method, plus
#   the digitizing error, over √n rows, floored at 10 kg m-3; for the FAC and horizon metrics, where the
#   registration states no σ_obs, σ_rep alone, which for a horizon is 0.35 of the core's horizon depth,
#   at least 0.5 m. Rows flagged `suspect` and records whose depth reference
#   is `other` are excluded;
# - χ² per core is the mean squared normalized residual; cores sharing an ERA5-Land cell and within
#   100 m of elevation form one site, whose χ² is their mean; sites are weighted by the inverse of a
#   Gaussian kernel density in (log accumulation, mean temperature, melt index), bandwidth one standard
#   deviation per coordinate, capped at 3× the median weight; the accumulation is the reference run's
#   mean snowfall, which is the core's measured accumulation when the runs match it;
# - the objective is the weighted mean site χ². Against the reference, each configuration reports the
#   bootstrap over sites (1,000 resamples) of the objective difference and, per RGI 7 o1 region (a
#   record with `position.ice_sheet`, or an ice-sheet site outside every region, is in its ice sheet's
#   region), the change in the region's mean χ², and
#   whether every region stays within 1 + 2·s/√n of it (n the region's sites, s the robust spread,
#   1.4826·MAD, of the configuration's site χ² changes).
#
# The runs are all of Test 1 (snowfall matched to the core) or all of Test 2 (end to end, the
# forcing's own snowfall; `firn_site_sweep.jl` with `MATCH_ACCUMULATION=false`); a mix is refused.
#
# Writes `<out_dir>/cores.tsv` (every metric per core and configuration), `<out_dir>/sites.tsv` (each
# site's cores, region, weight and χ² per configuration, the input of the Stage B emulator) and
# `<out_dir>/summary.md`.

using GEMB_GlacierSims, Serialization, DelimitedFiles, TOML, Statistics, Printf, Random

length(ARGS) == 4 || error("usage: firn_site_score.jl <library> <runs_dir> <reference> <out_dir>")
const EX = joinpath(ARGS[1], "extracted")
const RUNS = ARGS[2]
const REFERENCE = ARGS[3]
const OUT = ARGS[4]
const ρI = 917.0
const BINS = ((0, 1), (1, 5), (5, 10), (10, 20))
const SIGMA_REP = (; bias = (52.0, 28.0, 47.0, 43.0), fac10 = 0.36, fac20 = 0.52,
                    horizon_relative = 0.35, horizon_minimum = 0.5)
const SIGMA_METHOD = Dict("gravimetric_core" => 20.0, "ct" => 10.0, "gamma" => 10.0,
                          "pit_cutter" => 15.0)
const SIGMA_UNSTATED = 30.0
const SIGMA_BIN_FLOOR = 10.0
const N_BOOT = 1000

function core_rows(id)
    t, h = readdlm(joinpath(EX, id, "profile.csv"), ',', Any; header = true)
    c(n) = findfirst(==(n), vec(h))
    num(x) = x isa Real ? Float64(x) : (v = tryparse(Float64, string(x)); v === nothing ? NaN : v)
    top = num.(t[:, c("top_m")]); bot = num.(t[:, c("bottom_m")]); ρ = num.(t[:, c("density_kg_m3")])
    rs = num.(t[:, c("reading_sigma_kg_m3")]); ds = num.(t[:, c("density_sigma_kg_m3")])
    keep = string.(t[:, c("flag")]) .!= "suspect"
    # A point sample (top = bottom) stands for 0.1 m around it.
    pt = bot .<= top .+ 1e-9
    top[pt] .-= 0.05; bot[pt] .+= 0.05
    top .= max.(top, 0.0)
    return (; top = top[keep], bot = bot[keep], ρ = ρ[keep], reading = rs[keep], stated = ds[keep])
end

# Mean modeled density over [top, bottom]; NaN where the column does not reach.
function model_mean(p, top, bot)
    zb = cumsum(p.dz); zt = zb .- p.dz
    m = h = 0.0
    for k in eachindex(p.dz)
        o = min(bot, zb[k]) - max(top, zt[k])
        o > 0 || continue
        m += o * p.density[k]; h += o
    end
    return h >= 0.999 * (bot - top) ? m / h : NaN
end

model_density_at(p, z) = (k = searchsortedfirst(cumsum(p.dz), z); k > length(p.dz) ? NaN : p.density[k])

# Core density at depth `z` by linear interpolation between row midpoints, held flat beyond them.
function core_density_at(o, z)
    mid = (o.top .+ o.bot) ./ 2
    p = sortperm(mid); mid, ρ = mid[p], o.ρ[p]
    z <= first(mid) && return first(ρ)
    z >= last(mid) && return last(ρ)
    k = searchsortedlast(mid, z)
    return ρ[k] + (ρ[k + 1] - ρ[k]) * (z - mid[k]) / (mid[k + 1] - mid[k])
end

function fac_pair(o, p, D)
    cover = sum(max.(0, min.(o.bot, D) .- max.(o.top, 0)))
    (cover >= 0.9D && minimum((o.top .+ o.bot) ./ 2) <= 1.0 && sum(p.dz) >= D) || return NaN
    z = 0.05:0.1:D
    return sum(0.1 * (1 - model_density_at(p, x) / ρI) for x in z) -
           sum(0.1 * (1 - core_density_at(o, x) / ρI) for x in z)
end

# Depth [m] of the transition through `ρc`: the shallowest sample below `HORIZON_MINIMUM_DEPTH` from
# which the median density of every deeper sample is at least `ρc`, interpolated against the sample
# above it. The deeper median, rather than the first sample to reach `ρc`, so that an ice layer in
# lighter firn is not taken for the transition; `NaN` when the profile never makes it.
const HORIZON_MINIMUM_DEPTH = 3.0

function first_reach(z, ρ, ρc)
    deeper = [median(@view ρ[i:end]) for i in eachindex(ρ)]
    k = findfirst(i -> z[i] >= HORIZON_MINIMUM_DEPTH && ρ[i] >= ρc && deeper[i] >= ρc, eachindex(z))
    k === nothing && return NaN
    (k == 1 || z[k - 1] < HORIZON_MINIMUM_DEPTH || ρ[k - 1] >= ρc) && return z[k]
    return z[k - 1] + (ρc - ρ[k - 1]) * (z[k] - z[k - 1]) / (ρ[k] - ρ[k - 1])
end

# Model - core horizon depth [m], and the core's horizon depth, which scales its error.
function horizon_pair(o, p, ρc)
    mid = (o.top .+ o.bot) ./ 2; s = sortperm(mid)
    zc = first_reach(mid[s], o.ρ[s], ρc)
    zm = first_reach(cumsum(p.dz) .- p.dz ./ 2, p.density, ρc)
    return zm - zc, zc
end

function row_sigma(o, method)
    base = get(SIGMA_METHOD, method, SIGMA_UNSTATED)
    return [isfinite(o.stated[i]) ? o.stated[i] :
            sqrt(base^2 + (isfinite(o.reading[i]) ? o.reading[i]^2 : 0.0)) for i in eachindex(o.ρ)]
end

# Every metric and its normalized residual for one core under one run.
function core_metrics(id, r, meta)
    o = core_rows(id); p = r.profile
    σrow = row_sigma(o, get(get(meta, "density", Dict()), "method", ""))
    mod = [model_mean(p, o.top[i], o.bot[i]) for i in eachindex(o.ρ)]
    mid = (o.top .+ o.bot) ./ 2; w = o.bot .- o.top
    m = Dict{Symbol,Float64}(); z = Dict{Symbol,Float64}()
    for (j, (lo, hi)) in enumerate(BINS)
        s = findall(i -> lo <= mid[i] < hi && isfinite(mod[i]), eachindex(mid))
        key = Symbol("bias_$(lo)_$(hi)")
        if sum(w[s]; init = 0.0) >= 1.0
            m[key] = sum(w[s] .* (mod[s] .- o.ρ[s])) / sum(w[s])
            σobs = max(sqrt(mean(σrow[s] .^ 2)) / sqrt(length(s)), SIGMA_BIN_FLOOR)
            z[key] = m[key] / sqrt(σobs^2 + SIGMA_REP.bias[j]^2)
        else
            m[key] = NaN
        end
    end
    for (key, D, σ) in ((:fac10, 10.0, SIGMA_REP.fac10), (:fac20, 20.0, SIGMA_REP.fac20))
        m[key] = fac_pair(o, p, D)
        isfinite(m[key]) && (z[key] = m[key] / σ)
    end
    for (key, ρc) in ((:z550, 550.0), (:z830, 830.0))
        m[key], zc = horizon_pair(o, p, ρc)
        σ = max(SIGMA_REP.horizon_relative * zc, SIGMA_REP.horizon_minimum)
        isfinite(m[key]) && (z[key] = m[key] / σ)
    end
    ok = isfinite.(mod)
    m[:rmse] = any(ok) ? sqrt(mean((mod[ok] .- o.ρ[ok]) .^ 2)) : NaN
    chi2 = isempty(z) ? NaN : mean(abs2, values(z))
    return m, chi2
end

function climate_table()
    t, h = readdlm(joinpath(EX, "CLIMATE.tsv"), '\t', Any; header = true)
    c(n) = findfirst(==(n), vec(h))
    return Dict(string(t[i, 1]) => (; cell = (t[i, c("grid_lat")], t[i, c("grid_lon")]),
                                    T = Float64(t[i, c("t_mean_c")]),
                                    melt = Float64(t[i, c("melt_accumulation_ratio")]))
                for i in axes(t, 1))
end

function region_of(site, meta)
    sheet = get(meta["position"], "ice_sheet", "")
    isempty(sheet) || return sheet == "greenland" ? "Greenland ice sheet" : "Antarctic ice sheet"
    try
        return rgi_o1_region(site.longitude, site.latitude)
    catch e
        e isa ArgumentError || rethrow()
        return site.latitude < 0 ? "Antarctic ice sheet" : "Greenland ice sheet"
    end
end

function site_weights(sites)
    x = [[log(s.acc), s.T, s.melt] for s in sites]
    h = [std(getindex.(x, k)) for k in 1:3]
    dens = [sum(exp(-0.5 * sum(((a .- b) ./ h) .^ 2)) for b in x) for a in x]
    w = 1 ./ dens
    return min.(w, 3 * median(w))
end

function main()
    mkpath(OUT)
    configs = sort([replace(f, ".jls" => "") for f in readdir(RUNS) if endswith(f, ".jls")])
    REFERENCE in configs || error("no reference run $REFERENCE.jls in $RUNS")
    clim = climate_table()
    runs = Dict(c => deserialize(joinpath(RUNS, c * ".jls")) for c in configs)
    ids = sort(collect(keys(runs[REFERENCE])))
    tests = unique(get(r, :match_accumulation, true) for c in configs for r in values(runs[c])
                   if !haskey(r, :error))
    length(tests) == 1 || error("the runs mix accumulation-matched and unmatched results")
    test = only(tests) ? "Test 1 (snowfall matched to each core)" :
           "Test 2 (end to end, the forcing's own snowfall)"
    metas = Dict(id => TOML.parsefile(joinpath(EX, id, "meta.toml")) for id in ids)
    excluded = [id for id in ids if get(metas[id]["timing"], "depth_reference", "") == "other"]
    ids = setdiff(ids, excluded)

    # Sites: one ERA5-Land cell, elevation within 100 m.
    groups = Vector{Vector{String}}()
    for id in ids
        s = runs[REFERENCE][id].site
        g = findfirst(gr -> (r0 = runs[REFERENCE][gr[1]].site;
                             clim[gr[1]].cell == clim[id].cell && abs(r0.elevation - s.elevation) <= 100),
                      groups)
        g === nothing ? push!(groups, [id]) : push!(groups[g], id)
    end
    sites = [(; ids = g, acc = mean(runs[REFERENCE][i].snowfall for i in g),
              T = clim[g[1]].T, melt = clim[g[1]].melt, region = region_of(runs[REFERENCE][g[1]].site, metas[g[1]]))
             for g in groups]
    w = site_weights(sites)

    lines = String[]
    metric_names = (:bias_0_1, :bias_1_5, :bias_5_10, :bias_10_20, :fac10, :fac20, :z550, :z830, :rmse)
    push!(lines, join(["config", "record_id", "chi2", string.(metric_names)...], '\t'))
    site_chi2 = Dict{String,Vector{Float64}}()
    for c in configs
        core_chi2 = Dict{String,Float64}()
        for id in ids
            r = get(runs[c], id, nothing)
            (r === nothing || haskey(r, :error)) && error("configuration $c has no result for $id")
            m, x = core_metrics(id, r, metas[id])
            core_chi2[id] = x
            push!(lines, join([c, id, @sprintf("%.3f", x),
                               [@sprintf("%.3f", m[k]) for k in metric_names]...], '\t'))
        end
        site_chi2[c] = [mean(filter(isfinite, [core_chi2[i] for i in s.ids])) for s in sites]
    end
    write(joinpath(OUT, "cores.tsv"), join(lines, '\n') * '\n')
    open(joinpath(OUT, "sites.tsv"), "w") do io
        println(io, join(["site", "region", "weight", "record_ids", configs...], '\t'))
        for (k, s) in enumerate(sites)
            println(io, join([k, s.region, @sprintf("%.6g", w[k]), join(s.ids, ","),
                              [@sprintf("%.6g", site_chi2[c][k]) for c in configs]...], '\t'))
        end
    end

    objective(v, idx) = (k = filter(i -> isfinite(v[i]), idx); sum(w[k] .* v[k]) / sum(w[k]))
    all_idx = collect(eachindex(sites))
    rng = Xoshiro(20261026)
    boots = [rand(rng, all_idx, length(all_idx)) for _ in 1:N_BOOT]
    regions = sort(unique(s.region for s in sites))
    md = String["# Firn benchmark: $test", "",
                "Reference `$REFERENCE`. $(length(ids)) cores at $(length(sites)) sites" *
                (isempty(excluded) ? "" : "; excluded for depth reference `other`: $(join(excluded, ", "))") * ".",
                "", "| configuration | objective | Δ vs reference | Δ 95 % interval | share of resamples improving | worst region Δχ² (region) | regions past 1 + 2·s/√n |",
                "|---|---|---|---|---|---|---|"]
    ref = site_chi2[REFERENCE]
    for c in sort(configs; by = c -> objective(site_chi2[c], all_idx))
        v = site_chi2[c]
        Δ = [objective(v, b) - objective(ref, b) for b in boots]
        d = [v[i] - ref[i] for i in all_idx]
        dd = filter(isfinite, d)
        s = 1.4826 * median(abs.(dd .- median(dd)))
        reg = map(regions) do rg
            x = filter(isfinite, [d[i] for i in all_idx if sites[i].region == rg])
            (rg, mean(x), 1 + 2 * s / sqrt(length(x)))
        end
        worst = reg[argmax(getindex.(reg, 2))]
        past = [@sprintf("%s (%+.2f > %.2f)", r...) for r in reg if r[2] > r[3]]
        push!(md, @sprintf("| `%s` | %.3f | %+.3f | %+.3f … %+.3f | %.3f | %+.2f (%s) | %s |", c,
                           objective(v, all_idx), objective(v, all_idx) - objective(ref, all_idx),
                           quantile(Δ, 0.025), quantile(Δ, 0.975), mean(Δ .< 0), worst[2], worst[1],
                           c == REFERENCE ? "" : isempty(past) ? "none" : join(past, ", ")))
    end
    push!(md, "", "Per-core metrics: `cores.tsv`. The seasonal-snow regression guard (plan §6.5) is run " *
              "separately on the finalists.")
    write(joinpath(OUT, "summary.md"), join(md, '\n') * '\n')
    println(join(md, '\n'))
end

main()
