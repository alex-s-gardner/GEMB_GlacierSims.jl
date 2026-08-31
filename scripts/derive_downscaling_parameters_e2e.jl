# End-to-end check of `derive_downscaling_parameters` against real ERA5-Land forcing.
#
# The offline suite in `test/runtests.jl` fits `:synthetic` forcing through an injected
# `forcing_loader`, so it proves the estimator inverts its own forward model but never exercises the
# real loader. This script does, and reports the two things only real forcing can show: how much a
# region's per-timestep fits vary once resolved by season and by hour of day, which is what says
# whether their spread is physical structure or fit error; and whether the elevation-interval forcing
# they produce drops into `gemb_glacier_cell` unchanged.
#
# Wrangell-St Elias has the elevation range to make both fits identifiable across its cells, so it is
# the region checked here.
#
# Needs CDS credentials (`~/.cdsapirc` or `ENV["CDS_API_KEY"]`) and network. Downloads are cached as
# Zarr chunks under `CLIMATE_CACHE`, shared across cells, so a re-run is nearly free.
#
# Run:  julia --project=. scripts/derive_downscaling_parameters_e2e.jl [years]

using GEMB_GlacierSims
using GEMB_ClimateForcing
using DataFrames
using Dates
using DimensionalData
using Statistics
import GeoDataFrames
const GI = GeoDataFrames.GeoInterface

const CLIMATE_MODEL = :era5land
const PARQUET = joinpath(@__DIR__, "..", "data", "era5land_glacier_elevation_classes.parquet")

# Shared persistent forcing cache: the ERA5-Land chunks are tens of GB and expensive to re-fetch, so
# they live off `tempdir()` and survive a reboot. Override with `ENV["CLIMATE_CACHE"]` elsewhere.
const CLIMATE_CACHE = get(ENV, "CLIMATE_CACHE",
                          joinpath("/mnt/bylot-r3/data", string(CLIMATE_MODEL)))

# Wrangell-St Elias, Alaska (RGI region 01). A ~1° box, 64 grid cells, ~1400 km² of ice.
const REGION_NAME = "Wrangell-St Elias"
const REGION_BOX = (-142.5, 61.0, -141.5, 61.6)

# Two full years by default: enough for a seasonal cycle with a second year to show it repeats,
# while keeping the download to a few hundred MB of Zarr chunks.
const N_YEARS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2
const TIME_RANGE = (DateTime(2018, 1, 1), DateTime(2018 + N_YEARS, 1, 1))

box_polygon((x0, y0, x1, y1)) =
    GI.Polygon([GI.LinearRing([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)])])

# Median over a group's finite values; `NaN` when a group measured nothing, which is the report's
# own convention rather than a substituted number.
function med(v)
    f = filter(isfinite, v)
    return isempty(f) ? NaN : Statistics.median(f)
end

fmt(x; digits = 2) = isfinite(x) ? string(round(x; digits)) : "  --"

function main()
    token = GEMB_ClimateForcing.get_cds_api_key()
    token === nothing && error("no CDS API key; set ENV[\"CDS_API_KEY\"] or write ~/.cdsapirc")
    cache = joinpath(CLIMATE_CACHE, "cache")

    table = GeoDataFrames.read(PARQUET)
    # `derive_downscaling_parameters` loads forcing by `row.latitude`/`row.longitude`; the cached table
    # carries only the Point geometry, so derive them the way `era5_example.jl` does.
    table[!, :longitude] = GI.x.(table.geometry)
    table[!, :latitude] = GI.y.(table.geometry)

    region = box_polygon(REGION_BOX)
    @info "Region" REGION_NAME box=REGION_BOX time_range=TIME_RANGE

    t0 = time()
    p = derive_downscaling_parameters(CLIMATE_MODEL, TIME_RANGE, table, region;
                                 token, cache_path = cache)
    @info "Derived" seconds=round(time() - t0; digits = 1)

    # Each report flushed as it lands: the interval pass touches upstream validation that can throw,
    # and a buffered stdout would lose the fit diagnostics that explain why.
    for r in (report_fits, report_seasonal, report_diurnal, report_intervals, report_gemb)
        try
            r(p)
        catch e
            e isa InterruptException && rethrow()
            println("\n!! $(nameof(r)) FAILED: ", sprint(showerror, e))
        end
        flush(stdout)
    end
    return p
end

function report_fits(p)
    println("\n", "="^78, "\n RAW FITS\n", "="^78)
    k = p.decoupling.decoupling_factor
    lr = p.lapse_rate.lapse_rate
    n = length(p.time)

    println("timesteps:            $n  ($(first(p.time)) .. $(last(p.time)))")
    println("grid cells used:      $(p.provenance["n_grid_cells_used"]) " *
            "of $(p.provenance["n_grid_cells_in_region"]) in region")
    println("elevation intervals:  $(p.provenance["n_elevation_intervals"])")
    println("mean cell elevation:  $(fmt(p.provenance["mean_elevation"]; digits = 1)) m")

    kf, lf = filter(isfinite, k), filter(isfinite, lr)
    println("\nk fitted:             $(length(kf))/$n " *
            "($(round(100 * length(kf) / n; digits = 1))%)")
    println("lapse rate fitted:    $(length(lf))/$n " *
            "($(round(100 * length(lf) / n; digits = 1))%)")

    # Spread over the whole series. Both parameters vary with season and time of day — the seasonal
    # and diurnal tables below are this same spread resolved along those axes — so a wide range here
    # is structure, and reading it as fit error would be a mistake. The per-timestep diagnostics
    # printed next are what says whether an individual fit is trustworthy.
    for (name, v, unit) in (("k", kf, ""), ("lapse rate", lf, " K/km"))
        isempty(v) && continue
        q25, q75 = Statistics.quantile(v, 0.25), Statistics.quantile(v, 0.75)
        println("\n$name:")
        println("  median  $(fmt(Statistics.median(v)))$unit   " *
                "IQR $(fmt(q75 - q25))$unit   range $(fmt(minimum(v)))..$(fmt(maximum(v)))$unit")
    end

    # Per-timestep fit quality, summarized: enough to judge a fit on its own terms.
    r2 = filter(isfinite, p.decoupling.r2)
    ae = filter(isfinite, p.decoupling.ambient_excess)
    isempty(r2) || println("\nfit diagnostics:")
    isempty(r2) || println("  r2              median $(fmt(Statistics.median(r2)))  " *
                           "q10 $(fmt(Statistics.quantile(r2, 0.10)))")
    isempty(ae) || println("  ambient_excess  median $(fmt(Statistics.median(ae))) K  " *
                           "min $(fmt(minimum(ae))) K   (fits need >= 0.5 K)")
    sp = filter(isfinite, p.lapse_rate.elevation_spread)
    isempty(sp) || println("  elev. spread    median $(fmt(Statistics.median(sp); digits = 1)) m")

    # Fits outside the domain their consumer accepts are reported raw here; clamping happens only
    # where the parameters are applied. Their count measures how much of the series is hard to fit.
    n_high = count(x -> isfinite(x) && x > 1.0, k)
    n_low = count(x -> isfinite(x) && x <= 0.0, k)
    n_lr_out = count(x -> isfinite(x) && (x < -30 || x > 25), lr)
    println("\noutside the domain their consumer accepts (reported raw, clamped only on apply):")
    println("  k > 1:        $n_high      k <= 0:  $n_low")
    println("  lapse rate outside [-30, 25] K/km:  $n_lr_out")
end

# The seasonal cycle. `k` is only identifiable when there is a warm ambient excess to damp, so it is
# fitted in the melt season and absent outside it; the lapse rate is fitted year-round, and its
# month-to-month range is what one constant per region would have to stand in for.
function report_seasonal(p)
    println("\n", "="^78, "\n SEASONAL CYCLE   (groupby(p.lapse_rate, Ti => month))\n", "="^78)
    lr_by = groupby(p.lapse_rate, Ti => month)
    k_by = groupby(p.decoupling, Ti => month)
    println("month   lapse K/km    n_fit      k        n_fit    ambient_excess K")
    lr_med = Float64[]
    for m in 1:12
        lrg = lr_by[At(m)]
        kg = k_by[At(m)]
        l = med(lrg.lapse_rate)
        push!(lr_med, l)
        println(rpad(monthabbr(m), 6),
                lpad(fmt(l), 9), lpad(count(isfinite, lrg.lapse_rate), 10),
                lpad(fmt(med(kg.decoupling_factor)), 10),
                lpad(count(isfinite, kg.decoupling_factor), 11),
                lpad(fmt(med(kg.ambient_excess)), 14))
    end
    finite = findall(isfinite, lr_med)
    if !isempty(finite)
        lo = finite[argmin(lr_med[finite])]
        hi = finite[argmax(lr_med[finite])]
        println("\nshallowest: $(monthabbr(lo)) ($(fmt(lr_med[lo])) K/km)   " *
                "steepest: $(monthabbr(hi)) ($(fmt(lr_med[hi])) K/km)   " *
                "seasonal range $(fmt(lr_med[hi] - lr_med[lo])) K/km")
        summer = [lr_med[m] for m in (6, 7, 8) if isfinite(lr_med[m])]
        winter = [lr_med[m] for m in (12, 1, 2) if isfinite(lr_med[m])]
        if !isempty(summer) && !isempty(winter)
            println("summer (JJA) $(fmt(mean(summer)))  vs  winter (DJF) $(fmt(mean(winter))) K/km")
        end
    end
end

# The diurnal cycle, the other axis the fits resolve. Where nighttime inversions form they flatten or
# reverse the lapse rate; the hourly medians are what shows whether this region has them.
function report_diurnal(p)
    println("\n", "="^78, "\n DIURNAL CYCLE   (groupby(p.lapse_rate, Ti => hour))\n", "="^78)
    lr_by = groupby(p.lapse_rate, Ti => hour)
    k_by = groupby(p.decoupling, Ti => hour)
    println("hour UTC  lapse K/km    n_fit       k       n_fit")
    lr_med = Float64[]
    hours = Int[]
    for h in 0:23
        g = try
            lr_by[At(h)]
        catch
            continue
        end
        l = med(g.lapse_rate)
        push!(lr_med, l); push!(hours, h)
        kg = k_by[At(h)]
        println(lpad(h, 5), lpad(fmt(l), 13), lpad(count(isfinite, g.lapse_rate), 10),
                lpad(fmt(med(kg.decoupling_factor)), 10),
                lpad(count(isfinite, kg.decoupling_factor), 11))
    end
    finite = findall(isfinite, lr_med)
    if !isempty(finite)
        lo = finite[argmin(lr_med[finite])]
        hi = finite[argmax(lr_med[finite])]
        println("\nflattest: $(hours[lo])h ($(fmt(lr_med[lo])) K/km)   " *
                "steepest: $(hours[hi])h ($(fmt(lr_med[hi])) K/km)   " *
                "diurnal range $(fmt(lr_med[hi] - lr_med[lo])) K/km")
        n_inv = count(x -> isfinite(x) && x < 0, lr_med)
        n_inv > 0 && println("$n_inv hour(s) of day show an inversion (median lapse rate < 0)")
    end
end

# What the sweep actually consumes. The invariants asserted offline, re-checked on real forcing, plus
# the fill/clamp counts that say how much of each applied series was measured rather than substituted.
function report_intervals(p)
    println("\n", "="^78, "\n ELEVATION INTERVAL FORCING\n", "="^78)
    ivs = collect(p.elevation_interval_forcing)
    total_area = sum(glacier_area_total(r) for r in eachrow(p.grid_cells))
    println("intervals: $(length(ivs))   area $(fmt(sum(iv.area for iv in ivs); digits = 3)) km² " *
            "vs $(fmt(total_area; digits = 3)) km² in the cells " *
            "(conserved: $(isapprox(sum(iv.area for iv in ivs), total_area; atol = 1e-9)))")
    # `measured/warm` is the number that matters. `k` scales `max(T - 273.15, 0)`, so below freezing
    # every value gives identical forcing and a substitution there costs nothing; restricting the
    # count to the above-freezing timesteps is what says how much of the *melt-relevant* forcing
    # rested on a measurement. `fit_ok` and `held` describe the fits behind it: `held` approaching
    # `fit_ok` means this interval's `k` is the ceiling value the tile's ambient excess still
    # supports rather than a fit at this elevation.
    println("\n  interval m    area km²  cells   mean T K   k mean  fit_ok  held  " *
            "measured/warm  extrap m")
    for iv in ivs
        m = DimensionalData.metadata(iv.forcing)
        warm = m["n_timesteps_above_freezing"]
        measured = m["glacier_decoupling_factor_n_fitted_above_freezing"] +
                   m["glacier_decoupling_factor_n_held_above_freezing"]
        println("  ", lpad("$(iv.lo)-$(iv.hi)", 11),
                lpad(fmt(iv.area; digits = 2), 11), lpad(iv.n_cells, 7),
                lpad(fmt(m["temperature_air_mean"]), 11),
                lpad(fmt(m["glacier_decoupling_factor_mean"]; digits = 3), 9),
                lpad(m["glacier_decoupling_factor_n_fit_in_domain"], 8),
                lpad(m["glacier_decoupling_factor_n_fit_held"], 6),
                lpad("$measured/$warm", 15),
                lpad(fmt(m["extrapolation_above_reanalysis"]; digits = 0), 10))
    end

    ok = all(forcing_is_complete(iv.forcing) for iv in ivs)
    at_center = all(DimensionalData.metadata(iv.forcing)["elevation"] == iv.center for iv in ivs)
    in_domain = all(all(x -> 0 < x <= 1, iv.decoupling_factor) for iv in ivs)
    cools = issorted([mean(iv.forcing[:temperature_air]) for iv in ivs], rev = true)
    println("\nforcing_is_complete on every interval:      $ok")
    println("metadata[\"elevation\"] == interval center:   $at_center")
    println("applied k within (0, 1] on every interval:  $in_domain")
    println("temperature decreases with elevation:       $cools")
    m1 = DimensionalData.metadata(first(ivs).forcing)
    lapse_by_source = join(["$source $(m1["temperature_lapse_rate_n_$source"])"
                            for source in DOWNSCALING_SOURCES
                            if m1["temperature_lapse_rate_n_$source"] > 0], ", ")
    println("lapse rate applied: $(fmt(m1["temperature_lapse_rate"])) K/km mean  " *
            "($lapse_by_source of $(length(p.time)))")
end

# The last link: an interval's forcing has to drop into `gemb_glacier_cell` unchanged. Run the
# highest-area interval's cell through a short simulation rather than the whole region.
function report_gemb(p)
    println("\n", "="^78, "\n gemb_glacier_cell ACCEPTS AN INTERVAL'S FORCING\n", "="^78)
    ivs = collect(p.elevation_interval_forcing)
    iv = ivs[argmax([x.area for x in ivs])]
    # The row whose hypsometry the run is weighted by; any cell of the region carries the same
    # invariant schema, so take the first.
    row = first(eachrow(p.grid_cells))
    println("interval $(iv.lo)-$(iv.hi) m, $(fmt(iv.area; digits = 2)) km², " *
            "$(length(p.time)) timesteps")
    try
        mp = GEMB.initialize_parameters()
        run = gemb_glacier_cell(row, iv.forcing, mp;
                                delta_temperatures = [0.0], precipitation_scalings = [1.0],
                                coverage = 0.5,
                                # The region's own fitted lapse rate, as a sweep would use it. The
                                # interval forcing is already at the interval and already decoupled,
                                # so `glacier_decoupling = false`: applying the table's `k` on top
                                # would damp the warm excess a second time.
                                lapse_rate = DimensionalData.metadata(
                                    iv.forcing)["temperature_lapse_rate"],
                                glacier_decoupling = false,
                                threaded = false)
        println("accepted: run returned $(typeof(run))")
        println("  bins $(length(run.bins))  timesteps $(length(run.time))")
    catch e
        e isa InterruptException && rethrow()
        println("REJECTED: ", sprint(showerror, e))
    end
end

import GEMB
main()
