# End-to-end check of the tile GEMB driver against real ERA5-Land forcing.
#
# Runs one 2° tile's elevation bands through `gemb_glacier_tile` over a small perturbation grid, writes
# the result, reads it back, and reports the three things that say whether the chain is sound:
#
#   1. how much of each band's applied downscaling parameters was measured rather than assumed, over the
#      timesteps where the parameter changed the forcing at all;
#   2. whether the height-change decomposition closes — the numerical check on every dh in the file;
#   3. volume change per band and for the tile, which is the sanity check a glaciologist can read: the
#      ablation zone must lose mass, the accumulation zone must gain it, and the tile total must sit in a
#      credible range for its area.
#
# Needs CDS credentials (`~/.cdsapirc` or `ENV["CDS_API_KEY"]`) and the cached downscaling-parameter
# tiles. Forcing is read from the shared Zarr cache under `CLIMATE_CACHE`, so a tile whose parameters
# have already been derived costs no network.
#
# Run:  julia --project=. -t 100 scripts/gemb_tile_e2e.jl [tile_name] [start_year] [end_year]
#
# Figures land in `figures/<tile>/` as PNGs; the netCDF lands in `$CLIMATE_CACHE/tile_runs/`.

using GEMB_GlacierSims
using GEMB_ClimateForcing
using DataFrames
using Dates
using DimensionalData
using Statistics
using CairoMakie
import GEMB
import GeoDataFrames
const GI = GeoDataFrames.GeoInterface

const CLIMATE_MODEL = :era5land
const PARQUET = joinpath(@__DIR__, "..", "data", "$(CLIMATE_MODEL)_glacier_elevation_classes.parquet")
const CLIMATE_CACHE = get(ENV, "CLIMATE_CACHE", joinpath("/mnt/bylot-r3/data", string(CLIMATE_MODEL)))
const PARAMETER_DIR = joinpath(CLIMATE_CACHE, "downscaling_parameters")
const OUTPUT_DIR = joinpath(CLIMATE_CACHE, "tile_runs")

const TILE_SIZE = 2
const BUFFER = 1

# N60_W142 is St Elias, Alaska: the most glacierized tile on Earth at this gridding (335 core cells,
# ~13,000 km² of ice over 60 bands) and one with a strong seasonal melt cycle, so every part of the
# chain is exercised at once.
const TILE_NAME = length(ARGS) >= 1 ? ARGS[1] : "N60_W142"
const START_YEAR = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2018
const END_YEAR = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 2020
const TIME_RANGE = (DateTime(START_YEAR, 1, 1), DateTime(END_YEAR, 1, 1))

# Small enough to run in minutes, wide enough to show the sign of the response in both directions. The
# production sweep uses the eight precipitation scalings the earlier MATLAB GEMB runs used.
const DELTA_TEMPERATURES = [-1.0, 0.0, 1.0]
const PRECIPITATION_SCALINGS = [1.0, 1.5, 2.0]

# Enough cycles for a temperate column to settle. `convergence_delta_density` stops earlier where it
# can; the report says how many bands actually converged, which is the thing to watch — a cold-started
# column drifts several metres over a run and that drift is not a response to the forcing.
const SPINUP_MAX_ITERATIONS = 400
const SPINUP_CONVERGENCE = 0.01

fmt(x; digits = 2) = isfinite(x) ? string(round(x; digits)) : "  --"

function main()
    token = GEMB_ClimateForcing.get_cds_api_key()
    token === nothing && error("no CDS API key; set ENV[\"CDS_API_KEY\"] or write ~/.cdsapirc")
    cache = joinpath(CLIMATE_CACHE, "cache")

    isfile(PARQUET) || error("no glacier elevation-class table at $PARQUET")
    parameter_path = joinpath(PARAMETER_DIR, TILE_NAME * ".nc")
    isfile(parameter_path) || error("no downscaling parameters for $TILE_NAME at $parameter_path; " *
                                    "run scripts/derive_downscaling_parameter_tiles.jl first")

    table = GeoDataFrames.read(PARQUET)
    # `climate_forcing` is keyed on these, and the cached table carries only the Point geometry. Left in
    # the native 0-359.9°E convention, which is what the loader expects.
    table[!, :longitude] = GI.x.(table.geometry)
    table[!, :latitude] = GI.y.(table.geometry)

    tiles = downscaling_tiles(table; tile_size = TILE_SIZE, buffer = BUFFER)
    matching = filter(t -> t.name == TILE_NAME * ".nc", tiles)
    isempty(matching) && error("no tile named $TILE_NAME in the tiling of this table")
    tile = only(matching)

    fit = read_downscaling_tile(parameter_path)
    intervals = hypsometry_intervals(tile.core)

    @info "Tile" name=TILE_NAME geotile=geotile_id(tile.bounds) core_cells=nrow(tile.core) buffered_cells=nrow(tile.buffered) bands=length(intervals) area_km2=round(sum(glacier_area_column(tile.core)); digits = 1) fit_steps=length(fit.time) time_range=TIME_RANGE

    if isempty(intervals)
        @warn "This tile has no populated elevation bands, so there is nothing to run" TILE_NAME
        return nothing
    end

    # The authoritative run time axis, from the forcing itself rather than assumed to be hourly. One
    # cell load off the warm cache, which also fails early and clearly if the cache or the token is
    # wrong — before the tile-wide pass.
    probe = climate_forcing(CLIMATE_MODEL, first(tile.core).latitude, first(tile.core).longitude;
                            time_range = TIME_RANGE, token, cache_path = cache)
    run_time = collect(dims(probe, Ti))

    # `:fitted` prefers each timestep's own fit and needs the fit and the run on one axis; `:climatology`
    # reduces the fits to a month-by-hour median field and so applies to any window. Chosen by
    # comparing the axes rather than assumed, and reported, because the two are different experiments.
    basis = length(fit.time) == length(run_time) && collect(fit.time) == run_time ?
            :fitted : :climatology
    @info "Downscaling basis" basis reason=(basis === :fitted ?
        "the fit covers the run window exactly" :
        "the fit covers $(length(fit.time)) steps against the run's $(length(run_time))")

    prior = decoupling_factor_prior(tile.core)
    @info "Prior decoupling factor" k=(prior === nothing ? "none (RGI 05/19 are uncovered)" :
                                       round(prior; digits = 4))

    applied = resolve_downscaling(fit, intervals, run_time; basis, decoupling_factor_prior = prior)

    t0 = time()
    # `elevation_interval_batch = 0` is one pass over the tile's cells holding every band at once, which
    # is the fewest forcing reads. At this window that is a few hundred MB; a full-record sweep must
    # batch instead.
    #
    # `tile.core`, not `tile.buffered`: the parameters are fitted over the buffered neighbourhood so the
    # cross-cell regressions are identifiable, but the *area* must be the core only, or the ice in the
    # buffer is counted again by the neighbouring tile.
    bands = collect(elevation_interval_forcing(tile.core, applied;
                                               climate_model = CLIMATE_MODEL,
                                               time_range = TIME_RANGE, token, cache_path = cache,
                                               elevation_interval_batch = 0))
    @info "Band forcing built" seconds=round(time() - t0; digits = 1) bands=length(bands) area_km2=round(sum(b.area for b in bands); digits = 1)

    report_parameters(bands, applied)

    mp = GEMB.initialize_parameters(output_frequency = :monthly)
    t0 = time()
    run = gemb_glacier_tile(tile, applied, bands, mp;
                            delta_temperatures = DELTA_TEMPERATURES,
                            precipitation_scalings = PRECIPITATION_SCALINGS,
                            max_iterations = SPINUP_MAX_ITERATIONS,
                            convergence_delta_density = SPINUP_CONVERGENCE)
    simulations = length(run.bands) * length(DELTA_TEMPERATURES) * length(PRECIPITATION_SCALINGS)
    @info "GEMB finished" minutes=round((time() - t0) / 60; digits = 2) simulations seconds_per_simulation=round((time() - t0) / simulations; digits = 2)

    report_closure(run)
    report_volume(run)

    path = write_glacier_tile_netcdf(joinpath(OUTPUT_DIR, TILE_NAME * ".nc"), run;
                                    institution = "NASA Jet Propulsion Laboratory")
    report_roundtrip(path, run)
    write_figures(run)

    return run
end

# How much of each band's applied parameters was measured. The `above freezing` restriction is the
# point: `k` scales `max(T - 273.15, 0)`, so below freezing every value gives bit-identical forcing and
# counting those as substitutions describes the fits rather than the run.
function report_parameters(bands, applied)
    println("\n", "="^100, "\n APPLIED DOWNSCALING PARAMETERS\n", "="^100)
    lapse = downscaling_source_counts(applied.lapse_rate_source)
    println("lapse rate: median $(fmt(median(applied.lapse_rate))) K/km, " *
            "range $(fmt(minimum(applied.lapse_rate)))..$(fmt(maximum(applied.lapse_rate)))")
    println("  sources: ", join(["$k $(lapse[k])" for k in DOWNSCALING_SOURCES if lapse[k] > 0], ", "))

    println("\n  band m       area km²  cells   mean T K   k mean   fit_ok   held   " *
            "measured/warm   extrap m")
    for band in bands
        m = DimensionalData.metadata(band.forcing)
        warm = m["n_timesteps_above_freezing"]
        measured = m["glacier_decoupling_factor_n_fitted_above_freezing"] +
                   m["glacier_decoupling_factor_n_held_above_freezing"]
        println("  ", lpad("$(band.lo)-$(band.hi)", 11),
                lpad(fmt(band.area), 11), lpad(band.n_cells, 7),
                lpad(fmt(m["temperature_air_mean"]), 11),
                lpad(fmt(m["glacier_decoupling_factor_mean"]; digits = 3), 9),
                lpad(m["glacier_decoupling_factor_n_fit_in_domain"], 9),
                lpad(m["glacier_decoupling_factor_n_fit_held"], 7),
                lpad(warm == 0 ? "n/a (never warm)" : "$measured/$warm", 16),
                lpad(fmt(m["extrapolation_above_reanalysis"]; digits = 0), 11))
    end

    # A band whose `k` is entirely held carries the ceiling value the tile's ambient excess still
    # supports rather than a fit at its own elevation. Worth naming, but only where it mattered.
    held_and_warm = [(band.lo, band.hi) for band in bands
                     if (m = DimensionalData.metadata(band.forcing);
                         m["n_timesteps_above_freezing"] > 0 &&
                         m["glacier_decoupling_factor_n_fit_held"] > 0.9 *
                             max(m["glacier_decoupling_factor_n_fit_in_domain"], 1))]
    isempty(held_and_warm) ||
        println("\nbands whose k is essentially all held AND that see melt: ", held_and_warm)
end

# The numerical check on every height series in the file. `dh` is `-cumsum(ice_flux)`, and the identity
# says it equals the mass, water and firn terms summed; a residual above the rounding floor means the
# column's base is no longer at ice density and the series under-counts compaction.
function report_closure(run)
    println("\n", "="^100, "\n HEIGHT CHANGE CLOSURE\n", "="^100)
    res = run.bands_series[:dh_residual]
    println("max |dh - (mass + water + firn)| over every band and perturbation: ",
            maximum(abs, res), " m")
    dh_scale = maximum(abs, run.bands_series[:dh])
    println("largest |dh| in the run: ", fmt(dh_scale; digits = 3), " m")
    println("relative closure: ", dh_scale > 0 ?
            fmt(maximum(abs, res) / dh_scale; digits = 12) : "n/a")
    println(maximum(abs, res) < 1e-6 ? "CLOSES at the rounding floor." :
            "DOES NOT CLOSE — check column_reaches_ice_density and column_depth_max.")
end

function report_volume(run)
    println("\n", "="^100, "\n VOLUME CHANGE\n", "="^100)
    i_dt = findfirst(==(0.0), run.delta_temperatures)
    i_ps = findfirst(==(1.0), run.precipitation_scalings)
    years = (last(run.time) - first(run.time)).value / (365.25 * 86_400_000)

    if i_dt !== nothing && i_ps !== nothing
        println("baseline (dT = 0, pscale = 1) over $(fmt(years)) yr, cumulative over the record:")
        for (name, unit) in ((:dv, "km3 i.e."), (:dm, "Gt"), (:precipitation, "Gt"),
                             (:melt, "Gt"), (:runoff, "Gt"), (:refreeze, "Gt"),
                             (:evaporation_condensation, "Gt"))
            total = run.totals[name][end, i_dt, i_ps]
            println("  ", rpad(name, 26), lpad(fmt(total; digits = 4), 12), " ", unit,
                    "   (", fmt(total / years; digits = 4), " $unit/yr)")
        end
        # A state, not a flux: the column's absolute firn air volume. Reported without a rate, because
        # dividing a state by the record length means nothing.
        println("  ", rpad("fac (state, not a flux)", 26),
                lpad(fmt(run.totals[:fac][end, i_dt, i_ps]; digits = 4), 12), " km3",
                "   (", fmt(run.totals[:fac][end, i_dt, i_ps] /
                            sum(b.area for b in run.bands) * 1000; digits = 3),
                " m of air per column)")
        println("\nper-band surface height change (m over the record), baseline:")
        println("  band m        area km²      dh      mass     water      firn")
        for (i, b) in enumerate(run.bands)
            println("  ", lpad("$(b.lo)-$(b.hi)", 11), lpad(fmt(b.area), 13),
                    lpad(fmt(run.bands_series[:dh][end, i, i_dt, i_ps]; digits = 3), 9),
                    lpad(fmt(run.bands_series[:dh_mass][end, i, i_dt, i_ps]; digits = 3), 10),
                    lpad(fmt(run.bands_series[:dh_water][end, i, i_dt, i_ps]; digits = 3), 10),
                    lpad(fmt(run.bands_series[:dh_firn][end, i, i_dt, i_ps]; digits = 3), 10))
        end
    end

    # The shape of the eventual fit against altimetry: which (pscale, dT) pairs give which volume
    # trend. A monotone response in both directions is what makes the inverse problem well posed.
    println("\ntile volume change rate (km3 i.e. / yr) across the perturbation grid:")
    print("  pscale \\ dT ")
    for dt in run.delta_temperatures
        print(lpad(fmt(dt; digits = 1) * " K", 11))
    end
    println()
    for (j, ps) in enumerate(run.precipitation_scalings)
        print("  ", lpad(fmt(ps; digits = 2), 11))
        for i in eachindex(run.delta_temperatures)
            print(lpad(fmt(run.totals[:dv][end, i, j] / years; digits = 4), 11))
        end
        println()
    end
end

function report_roundtrip(path, run)
    println("\n", "="^100, "\n NETCDF ROUND TRIP\n", "="^100)
    println("wrote ", path, "  (", round(filesize(path) / 1e6; digits = 1), " MB)")
    status = read_glacier_tile_status(path)
    println("status: geotile ", status.geotile_id, "  last time ", status.time,
            "  bands ", length(status.band_centers),
            "  grid ", length(status.delta_temperatures), "x",
            length(status.precipitation_scalings))
    println("band centers match:  ", status.band_centers == [b.center for b in run.bands])
    println("delta grid matches:  ", status.delta_temperatures == run.delta_temperatures)
    println("pscale grid matches: ", status.precipitation_scalings == run.precipitation_scalings)
    restart = read_glacier_tile_restart(path)
    expected = count(!isnothing, run.profiles)
    println("restart profiles:    ", length(restart.profiles), " of ", expected, " expected")
    # The restart columns must round-trip exactly, or `gemb` cannot resume: it pins the column depth
    # and cell count from the restored `dz`.
    exact = all(k -> collect(restart.profiles[k][:dz]) == collect(run.profiles[k...][:dz]),
                keys(restart.profiles))
    println("dz bit-exact:        ", exact)
end

function write_figures(run)
    dir = joinpath(@__DIR__, "..", "figures", TILE_NAME)
    mkpath(dir)
    i_dt = something(findfirst(==(0.0), run.delta_temperatures), 1)
    i_ps = something(findfirst(==(1.0), run.precipitation_scalings), 1)
    t = run.time
    centers = [b.center for b in run.bands]
    years = (last(t) - first(t)).value / (365.25 * 86_400_000)

    # 1. Per-band height change, coloured by elevation. The ablation zone should fall steeply and the
    # accumulation zone rise, with the crossover somewhere near the equilibrium line.
    fig = Figure(size = (900, 560))
    ax = Axis(fig[1, 1], xlabel = "time", ylabel = "surface height change (m)",
              title = "$(TILE_NAME): per-band dh, dT = $(run.delta_temperatures[i_dt]) K, " *
                      "pscale = $(run.precipitation_scalings[i_ps])")
    cmap = cgrad(:viridis)
    lo, hi = extrema(centers)
    for (i, b) in enumerate(run.bands)
        frac = hi > lo ? (b.center - lo) / (hi - lo) : 0.5
        lines!(ax, t, run.bands_series[:dh][:, i, i_dt, i_ps]; color = cmap[frac], linewidth = 1.5)
    end
    Colorbar(fig[1, 2], colormap = cmap, limits = (lo, hi), label = "band center (m)")
    save(joinpath(dir, "dh_by_band.png"), fig)

    # 2. Tile volume change for the baseline, in the unit the altimetry products use.
    fig = Figure(size = (900, 460))
    ax = Axis(fig[1, 1], xlabel = "time", ylabel = "volume change (km³ i.e.)",
              title = "$(TILE_NAME): tile volume change, baseline " *
                      "($(fmt(sum(b.area for b in run.bands); digits = 0)) km² of ice)")
    lines!(ax, t, run.totals[:dv][:, i_dt, i_ps]; color = :black, linewidth = 2, label = "total")
    lines!(ax, t, run.totals[:dv_mass][:, i_dt, i_ps]; linewidth = 1.5, label = "mass term")
    lines!(ax, t, run.totals[:dv_firn][:, i_dt, i_ps]; linewidth = 1.5, label = "firn term")
    axislegend(ax; position = :lb)
    save(joinpath(dir, "volume_change.png"), fig)

    # 3. The decomposition for the highest-area band, which carries the most weight in the total. The
    # terms must sum to dh, and the residual is the check that they do.
    i_band = argmax([b.area for b in run.bands])
    b = run.bands[i_band]
    fig = Figure(size = (900, 460))
    ax = Axis(fig[1, 1], xlabel = "time", ylabel = "height change (m)",
              title = "$(TILE_NAME): dh decomposition, modal band $(b.lo)-$(b.hi) m " *
                      "($(fmt(b.area; digits = 0)) km²)")
    for (name, label) in ((:dh, "dh (total)"), (:dh_mass, "mass"), (:dh_water, "water storage"),
                          (:dh_firn, "firn compaction"))
        lines!(ax, t, run.bands_series[name][:, i_band, i_dt, i_ps];
               linewidth = name === :dh ? 2.5 : 1.5, label = label,
               color = name === :dh ? :black : Makie.wong_colors()[findfirst(
                   ==(name), [:dh, :dh_mass, :dh_water, :dh_firn])])
    end
    axislegend(ax; position = :lb)
    save(joinpath(dir, "dh_decomposition.png"), fig)

    # 4. The perturbation grid, which is the shape the fit against altimetry searches.
    trend = [run.totals[:dv][end, i, j] / years
             for i in eachindex(run.delta_temperatures), j in eachindex(run.precipitation_scalings)]
    fig = Figure(size = (700, 520))
    ax = Axis(fig[1, 1], xlabel = "temperature offset (K)", ylabel = "precipitation scaling",
              title = "$(TILE_NAME): volume change rate (km³ i.e. / yr)")
    hm = heatmap!(ax, run.delta_temperatures, run.precipitation_scalings, trend;
                  colormap = :RdBu, colorrange = (-maximum(abs, trend), maximum(abs, trend)))
    for i in eachindex(run.delta_temperatures), j in eachindex(run.precipitation_scalings)
        text!(ax, run.delta_temperatures[i], run.precipitation_scalings[j];
              text = fmt(trend[i, j]; digits = 3), align = (:center, :center), fontsize = 11)
    end
    Colorbar(fig[1, 2], hm)
    save(joinpath(dir, "perturbation_grid.png"), fig)

    # 5. Where each band's parameters came from, and how far its forcing was extrapolated. Read
    # together: a band that is both mostly-substituted and far above the reanalysis is the one a fit
    # should trust least.
    measured = Float64[]
    for p in run.band_provenance
        warm = get(p, "n_timesteps_above_freezing", 0)
        m = get(p, "glacier_decoupling_factor_n_fitted_above_freezing", 0) +
            get(p, "glacier_decoupling_factor_n_held_above_freezing", 0)
        push!(measured, warm > 0 ? 100 * m / warm : NaN)
    end
    extrap = [get(p, "extrapolation_above_reanalysis", NaN) for p in run.band_provenance]
    fig = Figure(size = (900, 560))
    ax = Axis(fig[1, 1], xlabel = "band center (m)",
              ylabel = "k measured over above-freezing steps (%)",
              title = "$(TILE_NAME): downscaling provenance and lapse extrapolation by band")
    scatter!(ax, centers, measured; color = :steelblue, label = "k measured (%)")
    ax2 = Axis(fig[1, 1], yaxisposition = :right, ylabel = "band above highest reanalysis cell (m)")
    hidespines!(ax2); hidexdecorations!(ax2)
    scatter!(ax2, centers, extrap; color = :firebrick, marker = :utriangle,
             label = "extrapolation (m)")
    hlines!(ax2, [0.0]; color = (:firebrick, 0.4), linestyle = :dash)
    axislegend(ax; position = :lb)
    axislegend(ax2; position = :rb)
    save(joinpath(dir, "parameter_provenance.png"), fig)

    println("\nfigures written to ", abspath(dir))
end

main()
