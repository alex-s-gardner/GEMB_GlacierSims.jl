# Before/after comparison of one tile's GEMB run, for the configurations of the 2026-10 pipeline
# review: A production (GEMB 0cda823, GEMB_ClimateForcing 352d7db, GlacierSims
# feat/native-resolution-run), B GEMB fixes only, C ClimateForcing and GlacierSims fixes only,
# D all fixes. One invocation runs one configuration on one tile, at ΔT = 0, pscale = 1 and at the
# tile's calibrated correction, with the band-based tile run (`gemb_glacier_tile`), and writes a
# summary for `summarize`.
#
# It uses only API present in every configuration, so the same file runs in each environment;
# the environment decides which code is compared.
#
# Run (one environment per configuration, each `Pkg.develop`ing the three packages at its commits):
#   julia -t 2 --project=<env> scripts/audit/before_after_tiles.jl run <config> <tile> <pscale> <ΔT> <ceiling_years> [out_dir]
#   julia --project=<env> scripts/audit/before_after_tiles.jl summarize <out_dir> <markdown_path>
# e.g. `run D N60_W140 1.9 3.5 500`. `ceiling_years` is the spinup ceiling the configuration's sweep
# uses: 100 on feat/native-resolution-run, 500 on fix/pipeline-audit. `out_dir` defaults to
# $BEFORE_AFTER_OUT, else ./before_after_tiles.
#
# Reads (never writes) the ERA5-Land Zarr cache, the downscaling parameter tiles and, for context,
# existing native-resolution runs, all under CLIMATE_CACHE (default /mnt/bylot-r3/data/era5land).
# A forcing chunk missing from the cache would be fetched from CDS into it.
using GEMB_GlacierSims, GEMB_ClimateForcing, GEMB, DataFrames, Dates, DimensionalData, Statistics,
      Serialization
import GeoDataFrames
const GS = GEMB_GlacierSims
const NCDatasets = GS.NCDatasets

const CLIMATE_CACHE = get(ENV, "CLIMATE_CACHE", "/mnt/bylot-r3/data/era5land")
# The cached ERA5-Land record: its first step and the end of the cache.
# `BEFORE_AFTER_END_YEAR` shortens both, for a quick check of the script itself.
const END_YEAR = parse(Int, get(ENV, "BEFORE_AFTER_END_YEAR", "0"))
const TIME_RANGE = (DateTime(1950, 1, 2), END_YEAR == 0 ? DateTime(2026, 8, 14, 23) : DateTime(END_YEAR, 1, 1))
# The production sweep's spinup settings, apart from the ceiling.
const SPINUP_WINDOW = (DateTime(1950, 1, 1), DateTime(1980, 12, 31))
const SPINUP_DRIFT_FAC = 1e-2
const DONOR_MAX_DISTANCE_KM = 50.0
const OUTPUT_FREQUENCY = :weekly
# The window rates are reported over.
const WINDOW = END_YEAR == 0 ? (DateTime(2000, 1, 1), DateTime(2024, 12, 31, 23)) :
               (DateTime(END_YEAR - 3, 1, 1), DateTime(END_YEAR - 1, 12, 31, 23))
# Ablation season by hemisphere: June-August north, December-February south.
ablation_months(lat) = lat >= 0 ? (6, 7, 8) : (12, 1, 2)

function load_tile(name)
    table = GeoDataFrames.read(joinpath(pkgdir(GS), "data", "era5land_glacier_elevation_classes.parquet"))
    table[!, :longitude] = GeoDataFrames.GeoInterface.x.(table.geometry)
    table[!, :latitude] = GeoDataFrames.GeoInterface.y.(table.geometry)
    return only(filter(t -> t.name == name * ".nc", downscaling_tiles(table; tile_size = 2, buffer = 1)))
end

# Mean of `x` over each output interval: output `k` closes the forcing steps after output `k - 1`.
function interval_means(x, forcing_time, out_time)
    stops = [searchsortedlast(forcing_time, t) for t in out_time]
    starts = [1; stops[1:end-1] .+ 1]
    return [mean(view(x, a:b)) for (a, b) in zip(starts, stops)]
end

# Per-column diagnostics an `on_output` hook can take from the full output stack before it is
# discarded: SW-weighted albedo over bare-ice intervals and ablation-season turbulent fluxes, both
# over WINDOW, as sums so bands can be combined afterwards.
function column_diagnostics(output, band, lat)
    out_time = collect(lookup(output, Ti))
    sw = interval_means(collect(band.forcing[:shortwave_downward]), collect(lookup(band.forcing, Ti)), out_time)
    rho_ice = Float64(DimensionalData.metadata(output)["density_ice"])
    surface_density = parent(output[:density])[1, :]
    albedo = collect(output[:albedo_broadband])
    shf, lhf = collect(output[:heat_flux_sensible]), collect(output[:heat_flux_latent])
    in_window = [WINDOW[1] <= t <= WINDOW[2] for t in out_time]
    bare = in_window .& (surface_density .>= rho_ice - 1)
    ablation = in_window .& [month(t) in ablation_months(lat) for t in out_time]
    return (; albedo_sw = sum(albedo[bare] .* sw[bare]), sw = sum(sw[bare]), n_bare = count(bare),
            n_window = count(in_window), shf = sum(shf[ablation]), lhf = sum(lhf[ablation]),
            n_ablation = count(ablation))
end

# The output intervals inside WINDOW are those after `i0` up to `i1`, spanning `window_years` years.
window_bounds(time) = (findlast(<(WINDOW[1]), time), findlast(<=(WINDOW[2]), time))
function window_years(time)
    i0, i1 = window_bounds(time)
    return Dates.value(Millisecond(time[i1] - time[i0])) / (365.25 * 86_400_000)
end

# Rate over WINDOW of a cumulative total, per year.
function window_rate(cumulative, time)
    i0, i1 = window_bounds(time)
    return (cumulative[i1] - cumulative[i0]) / window_years(time)
end

function run_config(config, tile_name, pscale, ΔT, ceiling, out_dir)
    t_start = time()
    tile = load_tile(tile_name)
    lat = mean(tile.core.latitude)
    token = GEMB_ClimateForcing.get_cds_api_key()
    cache = joinpath(CLIMATE_CACHE, "cache")
    fit = read_downscaling_tile(joinpath(CLIMATE_CACHE, "downscaling_parameters", tile.name))
    intervals = hypsometry_intervals(tile.core)
    row = first(tile.core)
    probe_time = collect(dims(climate_forcing(:era5land, row.latitude, row.longitude;
                                              time_range = TIME_RANGE, token, cache_path = cache), Ti))
    applied = resolve_downscaling(fit, intervals, probe_time; basis = :climatology,
                                  decoupling_factor_prior = decoupling_factor_prior(tile.core))
    bands = collect(elevation_interval_forcing(tile.core, applied; climate_model = :era5land,
                                               time_range = TIME_RANGE, token, cache_path = cache,
                                               elevation_interval_batch = 0, donor_cells = tile.buffered,
                                               donor_max_distance_km = DONOR_MAX_DISTANCE_KM))
    t_forcing = time() - t_start
    mp = GEMB.initialize_parameters(output_frequency = OUTPUT_FREQUENCY)

    results = Dict{String,Any}("config" => config, "tile" => tile_name, "forcing_seconds" => t_forcing,
                               "time_start" => first(probe_time), "time_end" => last(probe_time),
                               "ceiling_years" => ceiling, "software" => string.([pkgversion(GEMB), pkgdir(GEMB),
                                                                                  pkgdir(GEMB_ClimateForcing), pkgdir(GS)]))
    for (label, dT, ps) in (("baseline", 0.0, 1.0), ("calibrated", ΔT, pscale))
        t0 = time()
        diagnostics = Dict{Float64,Any}()
        lk = ReentrantLock()
        hook = (output; band, delta, pscale) -> begin
            d = column_diagnostics(output, band, lat)
            lock(() -> (diagnostics[band.center] = d), lk)
        end
        run = gemb_glacier_tile(tile, applied, bands, mp; delta_temperatures = [dT],
                                precipitation_scalings = [ps], spinup_window = SPINUP_WINDOW,
                                simulation_years_maximum = ceiling, convergence_drift_fac = SPINUP_DRIFT_FAC,
                                donor_max_distance_km = DONOR_MAX_DISTANCE_KM, on_output = hook)
        area = [b.area for b in run.bands]
        A = sum(area)
        rate(name) = window_rate(run.totals[name][:, 1, 1], run.time)
        gt = Dict(string(v) => rate(v) for v in (:dm, :melt, :runoff, :refreeze, :rain, :precipitation,
                                                 :evaporation_condensation, :blowing_snow))
        # Band mass balance over WINDOW, m w.e. per year: SMB fluxes summed over the window's intervals.
        in_window = [WINDOW[1] <= t <= WINDOW[2] for t in run.time]
        years = window_years(run.time)
        s = run.bands_series
        smb = [sum(s[:precipitation][in_window, b, 1, 1] .+ s[:evaporation_condensation][in_window, b, 1, 1] .+
                   s[:blowing_snow][in_window, b, 1, 1] .- s[:runoff][in_window, b, 1, 1]) / years / 1000
               for b in eachindex(area)]
        d = [diagnostics[b.center] for b in run.bands]
        albedo_sw = sum(x.albedo_sw * a for (x, a) in zip(d, area)) / sum(x.sw * a for (x, a) in zip(d, area))
        bare_fraction = sum(x.n_bare / max(x.n_window, 1) * a for (x, a) in zip(d, area)) / A
        shf = sum(x.shf / max(x.n_ablation, 1) * a for (x, a) in zip(d, area)) / A
        lhf = sum(x.lhf / max(x.n_ablation, 1) * a for (x, a) in zip(d, area)) / A
        years_sim = run.run_provenance["spinup_years_simulated"][:, 1, 1]
        converged = run.run_provenance["spinup_converged"][:, 1, 1]
        results[label] = Dict{String,Any}(
            "delta_temperature" => dT, "precipitation_scaling" => ps, "area_km2" => A,
            "n_bands" => length(run.bands), "n_bands_dropped" => run.provenance["n_bands_dropped"],
            "rates_gt" => gt, "dm_mwe" => gt["dm"] * 1e3 / A,
            "albedo_bare_sw" => albedo_sw, "bare_interval_fraction" => bare_fraction,
            "ablation_shf" => shf, "ablation_lhf" => lhf,
            "band_center" => [b.center for b in run.bands], "band_area" => area, "band_smb_mwe" => smb,
            "spinup_years_max" => maximum(years_sim), "spinup_years_mean" => sum(years_sim .* area) / A,
            "spinup_n_unconverged" => count(==(0.0), converged),
            "spinup_unconverged_area" => sum(area[converged .== 0.0]; init = 0.0),
            "seconds" => time() - t0)
        @info "finished" config tile_name label seconds = round(time() - t0) dm_gt = gt["dm"]
    end
    mkpath(out_dir)
    serialize(joinpath(out_dir, "$(config)_$(tile_name).jls"), results)
    return results
end

# Mass and energy closure over one ablation season, with GEMB's per-step budget checks on
# (`verbose = true`, which throws on a violation), for the band nearest the tile's median area.
function closure_check(tile_name, pscale, ΔT, out_dir)
    tile = load_tile(tile_name)
    lat = mean(tile.core.latitude)
    token = GEMB_ClimateForcing.get_cds_api_key()
    cache = joinpath(CLIMATE_CACHE, "cache")
    fit = read_downscaling_tile(joinpath(CLIMATE_CACHE, "downscaling_parameters", tile.name))
    intervals = hypsometry_intervals(tile.core)
    row = first(tile.core)
    probe_time = collect(dims(climate_forcing(:era5land, row.latitude, row.longitude;
                                              time_range = TIME_RANGE, token, cache_path = cache), Ti))
    applied = resolve_downscaling(fit, intervals, probe_time; basis = :climatology,
                                  decoupling_factor_prior = decoupling_factor_prior(tile.core))
    bands = collect(elevation_interval_forcing(tile.core, applied; climate_model = :era5land,
                                               time_range = TIME_RANGE, token, cache_path = cache,
                                               elevation_interval_batch = 0, donor_cells = tile.buffered,
                                               donor_max_distance_km = DONOR_MAX_DISTANCE_KM))
    band = bands[argmax([b.area for b in bands])]
    season = lat >= 0 ? (DateTime(2020, 5, 1), DateTime(2020, 9, 30, 23)) :
                        (DateTime(2019, 11, 1), DateTime(2020, 3, 31, 23))
    mp = GS._with_albedo_ice(GEMB.initialize_parameters(output_frequency = :daily), band.albedo_ice)
    full = initialize_forcing(precipitation_adjust(temperature_adjust(band.forcing, ΔT), pscale))
    cf_spinup = GS._spinup_climatology(full, SPINUP_WINDOW, mp)
    initial = initialize_profile(mp, cf_spinup)
    profile = gemb_spinup(initial, cf_spinup, mp; simulation_years_maximum = 30,
                          convergence_delta_density = nothing)
    seasonal = band.forcing[Ti(season[1] .. season[2])]
    cf = initialize_forcing(precipitation_adjust(temperature_adjust(seasonal, ΔT), pscale))
    out = gemb(profile, cf, mp; verbose = true)
    c = height_change_components(out; initial_profile = profile)
    result = Dict{String,Any}("tile" => tile_name, "band_center" => band.center, "season" => season,
                              "steps" => length(lookup(seasonal, Ti)), "verbose_checks_passed" => true,
                              "max_abs_dh_residual_m" => maximum(abs, c.residual),
                              "mass_tolerance_kg_m2" => GEMB.M_TOLERANCE,
                              "energy_tolerance_floor_J_m2" => GEMB.E_TOLERANCE)
    serialize(joinpath(out_dir, "closure_$(tile_name).jls"), result)
    return result
end

# Window rates of the tile totals in a directory of native-resolution cell files (summed over cells).
function native_rates(dir)
    files = filter(endswith(".nc"), readdir(dir))
    totals = Dict{String,Float64}()
    area = 0.0
    ps = dT = NaN
    for f in files
        NCDatasets.NCDataset(joinpath(dir, f)) do ds
            time = collect(ds["time"][:])
            area += ds["glacier_area_total"][]
            ps, dT = ds["precipitation_scaling"][1], ds["delta_temperature"][1]
            for v in ("dm", "melt", "runoff", "refreeze", "rain", "precipitation",
                      "evaporation_condensation", "blowing_snow")
                totals[v] = get(totals, v, 0.0) + window_rate(collect(Float64, ds["total_" * v][:, 1, 1]), time)
            end
        end
    end
    return (; totals, area, pscale = ps, ΔT = dT, n_cells = length(files))
end

function summarize(out_dir, markdown)
    runs = Dict((r["config"], r["tile"]) => r for r in
                (deserialize(joinpath(out_dir, f)) for f in readdir(out_dir)
                 if endswith(f, ".jls") && !startswith(f, "closure_")))
    configs = sort!(unique(first.(collect(keys(runs)))))
    tiles = sort!(unique(last.(collect(keys(runs)))))
    io = IOBuffer()
    fmt(x; d = 3) = x isa Real && isfinite(x) ? string(round(x; digits = d)) : "n/a"
    println(io, "# Before/after tile comparison\n")
    println(io, "Rates over $(Date(WINDOW[1])) to $(Date(WINDOW[2])), tile totals of the band-based tile ",
            "run. Gt yr⁻¹ unless noted; accumulation is precipitation minus rain; fluxes are W m⁻², ",
            "area-weighted means over ablation-season (", "JJA north, DJF south) weekly intervals.\n")
    for tile in tiles, label in ("baseline", "calibrated")
        cols = [c for c in configs if haskey(runs, (c, tile))]
        r1 = runs[(first(cols), tile)][label]
        native = label == "calibrated" && tile == "S50_E068" ?
                 [("LW-only native", joinpath(CLIMATE_CACHE, "native_runs_lwfix", tile)),
                  ("production native", joinpath(CLIMATE_CACHE, "native_runs", tile))] : []
        nat = [(n, native_rates(p)) for (n, p) in native if isdir(p)]
        println(io, "## $tile, $label (ΔT = $(r1["delta_temperature"]) K, pscale = $(r1["precipitation_scaling"]))\n")
        println(io, "| quantity | ", join(cols, " | "), (isempty(nat) ? "" : " | " * join(first.(nat), " | ")), " |")
        println(io, "|---|", repeat("---|", length(cols) + length(nat)))
        row(name, f; natf = nothing) = println(io, "| $name | ",
            join([f(runs[(c, tile)][label]) for c in cols], " | "),
            isempty(nat) ? "" : " | " * join([natf === nothing ? "n/a" : natf(n) for (_, n) in nat], " | "), " |")
        row("mass change (Gt/yr)", r -> fmt(r["rates_gt"]["dm"]); natf = n -> fmt(n.totals["dm"]))
        row("mass change (m w.e./yr)", r -> fmt(r["dm_mwe"]); natf = n -> fmt(n.totals["dm"] * 1e3 / n.area))
        for (n, k) in (("melt", "melt"), ("runoff", "runoff"), ("refreeze", "refreeze"), ("rain", "rain"),
                       ("evaporation/condensation", "evaporation_condensation"))
            row(n, r -> fmt(r["rates_gt"][k]); natf = x -> fmt(x.totals[k]))
        end
        row("accumulation", r -> fmt(r["rates_gt"]["precipitation"] - r["rates_gt"]["rain"]);
            natf = x -> fmt(x.totals["precipitation"] - x.totals["rain"]))
        row("SW-weighted bare-ice albedo", r -> fmt(r["albedo_bare_sw"]))
        row("bare-ice share of intervals", r -> fmt(r["bare_interval_fraction"]))
        row("ablation-season sensible flux", r -> fmt(r["ablation_shf"]; d = 1))
        row("ablation-season latent flux", r -> fmt(r["ablation_lhf"]; d = 1))
        row("glacier area (km²)", r -> fmt(r["area_km2"]; d = 1); natf = n -> fmt(n.area; d = 1))
        row("bands run / dropped", r -> "$(r["n_bands"]) / $(r["n_bands_dropped"])")
        row("spinup years, max / area mean", r -> "$(round(Int, r["spinup_years_max"])) / $(round(Int, r["spinup_years_mean"]))")
        row("spinup unconverged columns (area km²)", r -> "$(r["spinup_n_unconverged"]) ($(fmt(r["spinup_unconverged_area"]; d = 1)))")
        row("run time (min)", r -> fmt(r["seconds"] / 60; d = 1))
        println(io)
        if all(c -> haskey(runs, (c, tile)), ("A", "D"))
            a, d = runs[("A", tile)][label], runs[("D", tile)][label]
            println(io, "Band mass balance (m w.e. yr⁻¹), A against D:\n")
            println(io, "| band (m) | area (km²) | A | D |\n|---|---|---|---|")
            for z in sort!(union(a["band_center"], d["band_center"]))
                ia, id = findfirst(==(z), a["band_center"]), findfirst(==(z), d["band_center"])
                area = something(ia === nothing ? nothing : a["band_area"][ia], d["band_area"][id])
                println(io, "| $(round(Int, z)) | $(fmt(area; d = 1)) | ",
                        ia === nothing ? "n/a" : fmt(a["band_smb_mwe"][ia]), " | ",
                        id === nothing ? "n/a" : fmt(d["band_smb_mwe"][id]), " |")
            end
            println(io)
        end
    end
    for f in filter(f -> startswith(f, "closure_"), readdir(out_dir))
        c = deserialize(joinpath(out_dir, f))
        println(io, "Closure, configuration D, $(c["tile"]) band $(c["band_center"]) m, $(Date(c["season"][1])) ",
                "to $(Date(c["season"][2])) ($(c["steps"]) steps): GEMB's per-step verbose mass and energy ",
                "checks passed (mass tolerance $(c["mass_tolerance_kg_m2"]) kg m⁻², energy floor ",
                "$(c["energy_tolerance_floor_J_m2"]) J m⁻²); max |height residual| $(c["max_abs_dh_residual_m"]) m.\n")
    end
    mkpath(dirname(markdown))
    write(markdown, take!(io))
    return markdown
end

if abspath(PROGRAM_FILE) == @__FILE__
    mode = ARGS[1]
    out_default = get(ENV, "BEFORE_AFTER_OUT", "before_after_tiles")
    if mode == "run"
        run_config(ARGS[2], ARGS[3], parse(Float64, ARGS[4]), parse(Float64, ARGS[5]), parse(Float64, ARGS[6]),
                   length(ARGS) >= 7 ? ARGS[7] : out_default)
    elseif mode == "closure"
        closure_check(ARGS[2], parse(Float64, ARGS[3]), parse(Float64, ARGS[4]),
                      length(ARGS) >= 5 ? ARGS[5] : out_default)
    elseif mode == "summarize"
        summarize(ARGS[2], ARGS[3])
    else
        error("mode must be run, closure or summarize; got $mode")
    end
end
