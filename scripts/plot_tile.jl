# Figures for a written tile, read from its netCDF rather than from a run in memory.
#
# The same five views `gemb_tile_e2e.jl` draws, against a file from the production sweep: per-band
# height change, tile volume change, the decomposition for the modal band, the perturbation grid, and
# the downscaling provenance. Offline — it reads only the tile file, so it needs no forcing, no
# parameters and no credentials.
#
# Run:  julia --project=. scripts/plot_tile.jl [tile_name] [output_dir]
#
# Figures land in `figures/<tile>/` as PNGs.

using GEMB_GlacierSims
using CairoMakie
using Dates
using NCDatasets
using Statistics

const CLIMATE_CACHE = get(ENV, "CLIMATE_CACHE", "/mnt/bylot-r3/data/era5land")
const OUTPUT_DIR = joinpath(CLIMATE_CACHE, "tile_runs")
const PARAMETER_DIR = joinpath(CLIMATE_CACHE, "downscaling_parameters")

# Where each band sits relative to the elevations the fits were regressed over. Colours the parameter
# figures, which is the only place the distinction changes how a value should be read.
const SUPPORT_COLORS = (below = :darkorange, within = :steelblue, above = :firebrick)

const TILE_NAME = length(ARGS) >= 1 ? ARGS[1] : "N62_W152"
const FIGURE_DIR = length(ARGS) >= 2 ? ARGS[2] :
    joinpath(@__DIR__, "..", "figures", TILE_NAME)

# Reference period for the stand-in discharge figure 2 removes. Matches `gemb_tile_e2e.jl`; see
# `reference_discharge_rate` for why it is a reading aid and not a mass budget.
const DISCHARGE_REFERENCE_YEARS = 5

fmt(x; digits = 2) = isfinite(x) ? string(round(x; digits)) : "  --"

# netCDF declares the series with time varying fastest, so the Julia view is
# (time, band, delta_temperature, precipitation_scaling) and the totals drop the band axis.
function main()
    path = joinpath(OUTPUT_DIR, TILE_NAME * ".nc")
    isfile(path) || error("no tile file at $path")
    mkpath(FIGURE_DIR)

    # The file's own `_FillValue` is NaN, so masking to NaN rather than `missing` keeps the series
    # plottable and leaves the gaps where they are.
    NCDataset(path, "r"; maskingvalue = NaN) do ds
        t = DateTime.(ds["time"][:])
        deltas = ds["delta_temperature"][:]
        pscales = ds["precipitation_scaling"][:]
        centers = ds["band_center"][:]
        lower = ds["band_lower"][:]
        upper = ds["band_upper"][:]
        areas = ds["band_area"][:]
        years = (last(t) - first(t)).value / (365.25 * 86_400_000)

        # The baseline of the grid: no temperature offset, unscaled precipitation.
        i_dt = something(findfirst(==(0.0), deltas), 1)
        i_ps = something(findfirst(==(1.0), pscales), 1)

        println("$TILE_NAME: $(length(centers)) bands, $(length(t)) steps, ",
                "$(fmt(sum(areas); digits = 0)) km2 of ice, ",
                "$(fmt(years; digits = 1)) yr, grid $(length(deltas))x$(length(pscales))")
        println("baseline: dT = $(deltas[i_dt]) K, pscale = $(pscales[i_ps])")

        plot_dh_by_band(ds, t, centers, i_dt, i_ps, deltas, pscales)
        plot_volume_change(ds, t, areas, i_dt, i_ps, years)
        plot_dh_decomposition(ds, t, areas, lower, upper, i_dt, i_ps)
        plot_perturbation_grid(ds, deltas, pscales, years)
        plot_provenance(ds, centers)

        # The two applied parameters, each against the axis it actually varies on. Both need the fit
        # cells' elevations, which only the parameter tile carries.
        support = fit_support(TILE_NAME)
        if isnothing(support)
            println("no parameter tile for $TILE_NAME; skipping the parameter figures")
        else
            plot_decoupling_factor_by_band(ds, centers, areas, support)
            plot_lapse_rate(ds, support)
        end
    end

    println("\nfigures written to ", abspath(FIGURE_DIR))
end

# 1. Per-band height change, coloured by elevation. The ablation zone should fall steeply and the
# accumulation zone rise, with the crossover somewhere near the equilibrium line.
function plot_dh_by_band(ds, t, centers, i_dt, i_ps, deltas, pscales)
    dh = ds["dh"][:, :, i_dt, i_ps]
    fig = Figure(size = (900, 560))
    ax = Axis(fig[1, 1], xlabel = "time", ylabel = "surface height change (m)",
              title = "$(TILE_NAME): per-band dh, dT = $(deltas[i_dt]) K, " *
                      "pscale = $(pscales[i_ps])")
    cmap = cgrad(:viridis)
    lo, hi = extrema(centers)
    for i in eachindex(centers)
        frac = hi > lo ? (centers[i] - lo) / (hi - lo) : 0.5
        lines!(ax, t, dh[:, i]; color = cmap[frac], linewidth = 1.5)
    end
    Colorbar(fig[1, 2], colormap = cmap, limits = (lo, hi), label = "band center (m)")
    save(joinpath(FIGURE_DIR, "dh_by_band.png"), fig)
end

# 2. Tile volume change for the baseline, in the unit the altimetry products use.
function plot_volume_change(ds, t, areas, i_dt, i_ps, years)
    dv = ds["total_dv"][:, i_dt, i_ps]
    dv_mass = ds["total_dv_mass"][:, i_dt, i_ps]
    dv_firn = ds["total_dv_firn"][:, i_dt, i_ps]
    fig = Figure(size = (900, 460))
    ax = Axis(fig[1, 1], xlabel = "time", ylabel = "volume change (km³ i.e.)",
              title = "$(TILE_NAME): tile volume change, baseline " *
                      "($(fmt(sum(areas); digits = 0)) km² of ice)")
    lines!(ax, t, dv; color = :black, linewidth = 2, label = "total")
    lines!(ax, t, dv_mass; linewidth = 1.5, label = "mass term")
    lines!(ax, t, dv_firn; linewidth = 1.5, label = "firn term")
    # With a reference discharge removed, so the series does not simply walk off at the accumulation
    # rate. Only defined once the record is long enough to fit one.
    if years >= DISCHARGE_REFERENCE_YEARS
        rate = reference_discharge_rate(dv_mass, t;
                                       reference_years = DISCHARGE_REFERENCE_YEARS)
        lines!(ax, t, discharge_corrected_volume_change(dv, dv_mass, t;
                                                       reference_years = DISCHARGE_REFERENCE_YEARS);
               color = :firebrick, linewidth = 2, linestyle = :dash,
               label = "less reference discharge ($(fmt(rate; digits = 2)) km³/yr)")
    end
    axislegend(ax; position = :lb)
    save(joinpath(FIGURE_DIR, "volume_change.png"), fig)
end

# 3. The decomposition for the highest-area band, which carries the most weight in the total. The
# terms must sum to dh, and the residual is the check that they do.
function plot_dh_decomposition(ds, t, areas, lower, upper, i_dt, i_ps)
    i_band = argmax(areas)
    fig = Figure(size = (900, 460))
    ax = Axis(fig[1, 1], xlabel = "time", ylabel = "height change (m)",
              title = "$(TILE_NAME): dh decomposition, modal band " *
                      "$(lower[i_band])-$(upper[i_band]) m " *
                      "($(fmt(areas[i_band]; digits = 0)) km²)")
    names = ["dh", "dh_mass", "dh_water", "dh_firn"]
    labels = ["dh (total)", "mass", "water storage", "firn compaction"]
    for (k, (name, label)) in enumerate(zip(names, labels))
        lines!(ax, t, ds[name][:, i_band, i_dt, i_ps];
               linewidth = name == "dh" ? 2.5 : 1.5, label = label,
               color = name == "dh" ? :black : Makie.wong_colors()[k])
    end
    axislegend(ax; position = :lb)
    save(joinpath(FIGURE_DIR, "dh_decomposition.png"), fig)
end

# 4. The perturbation grid, which is the shape the fit against altimetry searches. Drawn on
# categorical axes: the grid is clustered near the baseline and reaches far out on both axes, so
# plotting against the values themselves collapses the close rows and hides their cells.
function plot_perturbation_grid(ds, deltas, pscales, years)
    trend = ds["total_dv"][end, :, :] ./ years
    fig = Figure(size = (760, 560))
    ax = Axis(fig[1, 1], xlabel = "temperature offset (K)", ylabel = "precipitation scaling",
              title = "$(TILE_NAME): volume change rate (km³ i.e. / yr)",
              xticks = (eachindex(deltas), string.(deltas)),
              yticks = (eachindex(pscales), string.(pscales)))
    span = maximum(abs, filter(isfinite, trend))
    hm = heatmap!(ax, eachindex(deltas), eachindex(pscales), trend;
                  colormap = :RdBu, colorrange = (-span, span))
    for i in eachindex(deltas), j in eachindex(pscales)
        text!(ax, i, j; text = fmt(trend[i, j]; digits = 2), align = (:center, :center),
              fontsize = 12)
    end
    Colorbar(fig[1, 2], hm)
    save(joinpath(FIGURE_DIR, "perturbation_grid.png"), fig)
end

# 5. Where each band's parameters came from, and how far its forcing was extrapolated. Read
# together: a band that is both mostly-substituted and far above the reanalysis is the one a fit
# should trust least. The `above freezing` restriction is the point — `k` scales
# `max(T - 273.15, 0)`, so below freezing every value gives bit-identical forcing.
function plot_provenance(ds, centers)
    warm = ds["band_n_timesteps_above_freezing"][:]
    fitted = ds["band_glacier_decoupling_factor_n_fitted_above_freezing"][:]
    held = ds["band_glacier_decoupling_factor_n_held_above_freezing"][:]
    measured = [w > 0 ? 100 * (f + h) / w : NaN for (w, f, h) in zip(warm, fitted, held)]
    extrap = ds["band_extrapolation_above_reanalysis"][:]

    fig = Figure(size = (900, 560))
    ax = Axis(fig[1, 1], xlabel = "band center (m)",
              ylabel = "k measured over above-freezing steps (%)",
              title = "$(TILE_NAME): downscaling provenance and lapse extrapolation by band")
    scatter!(ax, centers, measured; color = :steelblue, label = "k measured (%)")
    ax2 = Axis(fig[1, 1], yaxisposition = :right,
               ylabel = "band above highest reanalysis cell (m)")
    hidespines!(ax2); hidexdecorations!(ax2)
    scatter!(ax2, centers, extrap; color = :firebrick, marker = :utriangle,
             label = "extrapolation (m)")
    hlines!(ax2, [0.0]; color = (:firebrick, 0.4), linestyle = :dash)
    axislegend(ax; position = :lb)
    axislegend(ax2; position = :rb)
    save(joinpath(FIGURE_DIR, "parameter_provenance.png"), fig)
end

# Elevation range of the cells the decoupling and lapse fits were regressed over, and the path the
# applied lapse rate series is read from. A band outside this range carries a parameter no cell
# measured at that elevation. `nothing` when the parameter tile is absent, which is the sparse case.
function fit_support(tile)
    path = joinpath(PARAMETER_DIR, tile * ".nc")
    isfile(path) || return nothing
    return NCDataset(path, "r"; maskingvalue = NaN) do ds
        z = filter(isfinite, ds["fit_cell_elevation"][:])
        isempty(z) ? nothing :
            (; lo = minimum(z), hi = maximum(z), n_cells = length(z), path)
    end
end

# Which side of the fit support each band falls on, as keys into `SUPPORT_COLORS`.
band_support(centers, support) =
    [c < support.lo ? :below : c > support.hi ? :above : :within for c in centers]

# Area-proportional markers, so a band holding 0.1 km² of ice cannot draw the eye as hard as one
# holding 190 km².
marker_sizes(areas; smallest = 5, largest = 26) =
    smallest .+ (largest - smallest) .* sqrt.(areas ./ maximum(areas))

# `k` against elevation, which is the axis it varies on, over the value applied and the share of it
# that was measured. Two panels on a shared elevation axis rather than two y-axes on one: the value
# and its trustworthiness are different quantities, and a reader should not have to work out which
# axis a given mark belongs to.
function plot_decoupling_factor_by_band(ds, centers, areas, support)
    k = ds["band_glacier_decoupling_factor_mean"][:]
    warm = ds["band_n_timesteps_above_freezing"][:]
    fitted = ds["band_glacier_decoupling_factor_n_fitted_above_freezing"][:]
    held = ds["band_glacier_decoupling_factor_n_held_above_freezing"][:]
    measured = [w > 0 ? 100 * (f + h) / w : NaN for (w, f, h) in zip(warm, fitted, held)]
    side = band_support(centers, support)
    sizes = marker_sizes(areas)
    total = sum(areas)

    fig = Figure(size = (960, 780))
    ax1 = Axis(fig[1, 1], ylabel = "applied k (1)",
               title = "$(TILE_NAME): decoupling factor by band, against the fit support " *
                       "($(fmt(support.lo; digits = 0))–$(fmt(support.hi; digits = 0)) m, " *
                       "$(support.n_cells) cells)")
    ax2 = Axis(fig[2, 1], xlabel = "band center (m)",
               ylabel = "k measured over above-freezing steps (%)")
    linkxaxes!(ax1, ax2)
    hidexdecorations!(ax1; grid = false)

    for ax in (ax1, ax2)
        vspan!(ax, support.lo, support.hi; color = (:steelblue, 0.08))
        vlines!(ax, [support.lo, support.hi]; color = (:black, 0.35), linestyle = :dash)
    end
    # k = 1 is the identity: no decoupling at all, which is what an unfitted band falls back to.
    hlines!(ax1, [1.0]; color = (:black, 0.3), linestyle = :dot, label = "k = 1 (no decoupling)")

    for key in keys(SUPPORT_COLORS)
        sel = side .== key
        any(sel) || continue
        label = "$(key) support ($(fmt(100 * sum(areas[sel]) / total; digits = 1))% of ice)"
        scatter!(ax1, centers[sel], k[sel]; color = SUPPORT_COLORS[key],
                 markersize = sizes[sel], label = label)
        scatter!(ax2, centers[sel], measured[sel]; color = SUPPORT_COLORS[key],
                 markersize = sizes[sel])
    end
    axislegend(ax1; position = :rt)
    Label(fig[3, 1], "marker size ∝ glacier area in the band; shaded span is the fit support",
          fontsize = 11, color = :gray35)
    rowsize!(fig.layout, 1, Relative(0.52))
    save(joinpath(FIGURE_DIR, "decoupling_factor_by_band.png"), fig)
end

# The lapse rate against time, because it does not vary with elevation: one cross-cell regression per
# timestep serves every band, so the per-band value is the same number in all of them. Drawn as the
# annual median and interquartile range of the fitted series, against the constant the run applied.
function plot_lapse_rate(ds, support)
    applied = ds["band_temperature_lapse_rate"][:]
    spread = maximum(applied) - minimum(applied)

    NCDataset(support.path, "r"; maskingvalue = NaN) do p
        t = DateTime.(p["time"][:])
        lr = p["lapse_rate"][:]
        finite = isfinite.(lr)
        keep = Int[]; med = Float64[]; q25 = Float64[]; q75 = Float64[]
        for y in sort(unique(year.(t)))
            sel = (year.(t) .== y) .& finite
            count(sel) > 0 || continue
            v = lr[sel]
            push!(keep, y); push!(med, median(v))
            push!(q25, quantile(v, 0.25)); push!(q75, quantile(v, 0.75))
        end

        fig = Figure(size = (960, 540))
        ax = Axis(fig[1, 1], xlabel = "year", ylabel = "temperature lapse rate (K/km)",
                  title = "$(TILE_NAME): fitted lapse rate over time — one tile-wide fit per " *
                          "timestep, identical in every band")
        band!(ax, keep, q25, q75; color = (:steelblue, 0.3), label = "interquartile range")
        lines!(ax, keep, med; color = :steelblue, linewidth = 2, label = "annual median")
        hlines!(ax, [applied[1]]; color = :black, linewidth = 2, linestyle = :dash,
                label = "applied to every band ($(fmt(applied[1])) K/km)")
        axislegend(ax; position = :rb)
        Label(fig[2, 1],
              "fitted at $(fmt(100 * count(finite) / length(lr); digits = 1))% of " *
              "$(length(lr)) timesteps; across the $(length(applied)) bands the applied value " *
              "spans $(fmt(spread; digits = 3)) K/km",
              fontsize = 11, color = :gray35)
        save(joinpath(FIGURE_DIR, "lapse_rate.png"), fig)
    end
end

main()
