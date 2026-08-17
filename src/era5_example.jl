# Batch GEMB run over glacier grid cells, forced with ERA5-Land via GEMB_ClimateForcing.jl.
#
# For every glacier grid cell holding at least `cell_area_minimum` of glacier ice, this runs
# GEMB over the full outer product of prescribed temperature deltas and precipitation scalings,
# for the hypsometry bins covering at least `hypsometry_coverage` of the cell's glacier area.
# The per-bin mass fluxes are area-weighted into per-cell mass totals (kg) and written to one
# CF-compliant NetCDF per cell, alongside the final firn profile of every run so the record can
# be extended when new forcing appears without repeating the spinup.
#
# Setup (required before running):
#   1. Install GEMB_ClimateForcing from GitHub:
#      using Pkg
#      Pkg.add(url="https://github.com/alex-s-gardner/GEMB_ClimateForcing.jl")
#   2. Get a CDS API key from: https://cds.climate.copernicus.eu/api-how-to
#   3. Set environment variable: ENV["CDS_API_KEY"] = "your-key-here"
#
# NOTE: This example requires GEMB_ClimateForcing.jl to be installed.
begin
    # cached glacier elevation-class table, will only be built if it doesn't already exist
    climate_model = :era5land

    # --- sweep configuration -----------------------------------------------------------------
    # Prescribed perturbations, run as a full outer product. Note the cost: each cell runs
    # (bins x deltas x scalings) GEMB simulations, each with its own spinup.
    delta_temperatures     = [0.0]        # air temperature offsets (K)
    precipitation_scalings = [1.0]        # precipitation multipliers (1)

    # Fraction of each cell's glacier area the modeled hypsometry bins must cover. Bins are
    # taken largest-area first; the area of the unmodeled remainder is folded into the nearest
    # modeled bin, so the full cell area always contributes to the mass totals.
    hypsometry_coverage = 0.95

    # Skip grid cells holding less than this total glacier area (km²).
    cell_area_minimum = 1.0

    # Temperature lapse rate (K km-1) for the per-bin elevation adjustment. Stated here rather
    # than left to `gemb_glacier_cell`'s default because the up-front check below has to build the
    # same `run_parameters` the run will, and a default that differs between the two would report
    # every existing cell as a parameter change.
    lapse_rate = 6.5

    # Correct the ambient reanalysis air temperature to on-glacier conditions with the Shaw et al.
    # (2025) decoupling factor `k`, weighted by the fraction of the cell ERA5-Land does *not*
    # already treat as ice (`1 - glm`) — the glaciated fraction needs no correction. A cell with no
    # published `k` (RGI regions 05 and 19 are not covered by the dataset) is run on ambient
    # forcing with no correction. `false` disables the lookup; a number prescribes `k` directly.
    #
    # NOTE: not named `glacier_decoupling` — that is the lookup function GEMB_ClimateForcing
    # exports, and a global of the same name would clash with the import in `Main`.
    apply_glacier_decoupling = true

    # Restrict the sweep to the first N qualifying cells (`nothing` or `Inf` runs all of them). Start
    # small: a global sweep is many thousands of cells x the perturbation grid.
    cell_limit = Inf

    # Extending an existing cell file requires that its stored run parameters (every
    # `ModelParameters` field, the hypsometry coverage, the lapse rate) match this sweep's;
    # otherwise the append is refused, because it would splice two different experiments into
    # one record. Set `true` to append across such a change anyway — the file's stored
    # parameters are then overwritten and the pre-seam record no longer reflects them. Changing
    # a model parameter normally means the cells should be rebuilt from scratch instead.
    force_restart = false

    # Display a full `gemb_plot_output` panel for every individual simulation as it finishes —
    # one figure per (bin x delta x scaling), so a single cell can produce dozens. Diagnostic
    # only: keep `cell_limit` small when this is on, and note that it forces the per-cell
    # simulations to run serially (Makie is not thread-safe).
    verbose_plotting = false

    using GEMB
    using Dates
    using Statistics
    using CairoMakie
    using Rasters               # also re-exports DimensionalData (dims, .val)
    using GeoDataFrames
    using GeoParquet            # backend for GeoDataFrames.write(...parquet)
    using GEMB_GlacierSims
    using GEMB_ClimateForcing
    using DimensionalData
    using Logging
    import NCDatasets              # to read the per-cell output files back for plotting

    # Set after the imports above: `DateTime` comes from Dates.
    forcing_time_range = (DateTime(1950, 1, 1), DateTime(2026, 8, 1))

    gemb_elevation_classes_file = joinpath(@__DIR__, "..", "data", "$(climate_model)_glacier_elevation_classes.parquet")
    #output_dir = joinpath(@__DIR__, "..", "data", "gemb_runs", string(climate_model))
    output_dir = joinpath("/Users/gardnera/data/gemb/gemb_runs/run_001", string(climate_model))
    forcing_cache = joinpath(tempdir(), ".cache", "$(climate_model)")

    #disable_logging(Logging.Info)

    # Get CDS API key (automatically reads from ENV or ~/.cdsapirc)
    cds_api_key = GEMB_ClimateForcing.get_cds_api_key()

    glacier_vector_file = get(ENV, "RGI_VECTOR_FILE",
        "/Users/gardnera/data/GlacierOutlines/RGI2000-v7.0-G-global-fix/rgi70_Global.gpkg")
end;

# Build the per-grid-cell glacier elevation-class table. For every chunk-map cell with
# fractional glacier cover above `glacier_cutoff`, this bins the ~30 m Copernicus DEM into a
# 100 m glacier hypsometry (glacier area, km², per elevation bin), attaches the invariant
# fields and chunk id, and returns a GeoParquet-ready DataFrame with a Point geometry column.
# The hypsometry is stored flat as one scalar column per bin (`hyps_<lo>_<hi>`); this keeps the
# Parquet columns 1-D so the cached file reads in ms rather than deserializing a nested list.
if !isfile(gemb_elevation_classes_file)

    @info "Building glacier elevation-class table (this may take several hours)..."

    glacier_polygons = GeoDataFrames.read(glacier_vector_file)

    # Lazy Copernicus 30 m DEM mosaic; cropped per tile and read on demand inside the runfile.
    dem = climate_model_invariant(model = :copernicus_dem_30m)

    # ERA5-Land download chunk map (:geo strategy); its grid defines the output cells.
    geo_chunk_map = climate_chunk_map(climate_model; chunk_strategy=:geo, token=cds_api_key)

    # Invariant fields carried through as columns. `era5_land_invariant` (from GEMB_GlacierSims)
    # reads them and rewraps native 0–359.9°E longitudes to the (-180, 180] grid; the runfile
    # regrids the stack onto its internal glacier grid as needed. Surface elevation is no longer
    # carried here: `climate_forcing` derives it per cell from the ERA5-Land geopotential invariant.
    era5_land_invariants = era5_land_invariant(parameter=(:glm, :lsm, :cl))

    glacier_elevation_classes = gemb_glacier_elevation_class_runfile(
        glacier_polygons, dem, geo_chunk_map, era5_land_invariants;
        elevation_bin_edges = 0:100:10000,
        glacier_cutoff = 0.0,
        oversample_factor = 10,
        dem_fetch_concurrency = 8,
    )

    ## Save glacier_elevation_classes as a GeoParquet file
    mkpath(dirname(gemb_elevation_classes_file))
    GeoDataFrames.write(gemb_elevation_classes_file, glacier_elevation_classes)
    @info "Saved $(nrow(glacier_elevation_classes)) glacier points to $(gemb_elevation_classes_file)"
else
    @info "Loading cached glacier elevation-class table from $(gemb_elevation_classes_file)"
    glacier_elevation_classes = GeoDataFrames.read(gemb_elevation_classes_file)
end;

# Add lat/lon from the Point geometry. Surface elevation is not derived here; `climate_forcing`
# provides the per-cell orthometric elevation in its output metadata.
begin
    glacier_elevation_classes[!,:longitude] = GeoDataFrames.GeoInterface.x.(glacier_elevation_classes.geometry)
    glacier_elevation_classes[!,:latitude] = GeoDataFrames.GeoInterface.y.(glacier_elevation_classes.geometry)
end;

## Sweep every qualifying glacier grid cell

# Model parameters, shared by every run. Monthly output keeps the per-cell files small over a
# 76-year record; `gemb_spinup` overrides the frequency to :last internally, so no separate
# spinup parameters are needed.
mp = initialize_parameters(output_frequency = :daily);

# Cells holding enough glacier ice to be worth running. `glacier_area_total` sums the flat
# hyps_<lo>_<hi> columns; the bin selection itself (largest-area-first to `hypsometry_coverage`,
# with the unmodeled remainder folded into the nearest modeled bin so no area is dropped) happens
# inside `gemb_glacier_cell`, so the ~47,000-cell screen does not pay for it.
begin
    # `view = true` keeps this a SubDataFrame over the cached table rather than copying the
    # ~47,000 rows (and their hyps_* columns) into a second frame.
    qualifying_cells = filter(r -> glacier_area_total(r) >= cell_area_minimum,
                              glacier_elevation_classes; view = true)
    # `view = true` again, so truncating the selection does not materialize the rows either.
    if !(cell_limit === nothing || isinf(cell_limit))
        qualifying_cells = first(qualifying_cells, Int(cell_limit); view = true)
    end
    @info "Cells to run" total=nrow(glacier_elevation_classes) qualifying=nrow(qualifying_cells) runs_per_cell="bins x $(length(delta_temperatures)) x $(length(precipitation_scalings))"
end;

# One NetCDF per cell, named `N52.3_W174.1.nc` by cell center, so a file is traceable to its cell
# and a listing sorts geographically (by hemisphere, then by degree).
#
# The form follows ISO 6709's letter-prefix convention: hemisphere letter, then zero-padded
# magnitude — two integer digits of latitude, three of longitude, which `wrap_lon` bounds at 180.
# The letters carry the sign, so every name is the same length without spending a character on
# '+', and they make `lat`/`lon` prefixes redundant, since N/S can only be a latitude and E/W only
# a longitude. Keeping '-' out of filenames also avoids the corner where shell and `find` argument
# parsing treat a name as an option. The cost is that the degrees no longer `parse` straight out
# of the substring; `parse_cell_lonlat` below is the inverse, so nothing has to re-derive the
# format.
#
# The '_' between the two coordinates is redundant as a delimiter — the E/W letter already marks
# where latitude ends — but it makes the boundary visible without hunting for a letter among
# digits, and it means splitting a name into its two halves does not depend on the zero-padding
# being exactly 2-and-3 digits.
#
# ONE decimal, because that is all the forcing grid has: ERA5-Land is 0.1°, so every cell centre
# in `glacier_elevation_classes` is exact at 1 dp and a second decimal is always '0' — a character
# of noise in every name. That makes the decimal point itself the only punctuation left, and it is
# worth keeping: dropping it would leave the scale implied ('N523' read as tenths), and the
# standard alternative for a point-free name is ISO 6709 degrees-minutes ('N5218'), which is less
# readable than decimal degrees for anyone working in this field.
#
# This ties the name to a 0.1° grid: a finer forcing grid (ERA5 at 0.25° is fine, but a 0.01°
# product would not be) needs another decimal here, or two of its cells would collide on one name.
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
cell_output_path(r) = joinpath(output_dir,
    _degrees_tag(r.latitude, 2, 'N', 'S') * "_" *
    _degrees_tag(wrap_lon(r.longitude), 3, 'E', 'W') * ".nc")

# Inverse of `cell_output_path`: the (lat, lon) a cell file is named for, or `nothing` if the name
# is not one of ours. Rounded to the name's precision, so it identifies the cell but is not the
# exact centre — read the file's attributes for that. Provided so downstream tooling can recover
# the cell from a filename without re-implementing the format and drifting from it.
const _CELL_NAME_REGEX =
    Regex("^([NS])(\\d{2}\\.\\d{$_CELL_NAME_DECIMALS})_([EW])(\\d{3}\\.\\d{$_CELL_NAME_DECIMALS})\\.nc\$")

function parse_cell_lonlat(path::AbstractString)
    m = match(_CELL_NAME_REGEX, basename(path))
    m === nothing && return nothing
    return (latitude = parse(Float64, m[2]) * (m[1] == "S" ? -1 : 1),
            longitude = parse(Float64, m[4]) * (m[3] == "W" ? -1 : 1))
end

# `on_output` hook for `verbose_plotting`: one full GEMB diagnostic panel per simulation, titled
# with the cell and the perturbation it belongs to so the figures stay distinguishable. Built per
# cell so the closure carries that cell's identity; `gemb_plot_output` needs a Makie backend,
# which the `using CairoMakie` above provides.
#
# The decoupling factor is part of the title only when one was applied: a `k=1.0` on every figure
# of an uncorrected sweep is noise, and its absence is what distinguishes ambient forcing from a
# cell the correction happened to leave alone.
verbose_plotter(i, r) = function (output; bin, delta, pscale, decoupling_factor)
    k = decoupling_factor_label(decoupling_factor)
    display(gemb_plot_output(output; datelims = (DateTime(2020, 1, 1), DateTime(2026, 8, 1)),
        title = "GEMB simulation: ($(round(r.latitude, digits = 3))°N, " *
                "$(round(wrap_lon(r.longitude), digits = 3))°E) — " *
                "hyps bin $(round(Int, bin.center)) m, ΔT=$(delta) K, P×$(pscale)" *
                (isempty(k) ? "" : ", $k")))
end

for (i, r) in enumerate(eachrow(qualifying_cells))

    path = cell_output_path(r)

    # One failing cell (a CDS timeout, an infeasible column grid) must not abort the sweep.
    try
        # --- pre-flight on the existing cell file -------------------------------------------
        # Re-running a sweep is the common case (extend the record as forcing is published, or
        # resume an interrupted run), and for most cells the answer is "nothing to do". Deciding
        # that from the file alone — before any network I/O — is what makes a re-run cheap: the
        # forcing download is by far the dominant per-cell cost, and even a cache hit re-reads and
        # re-converts the whole multi-decade series.
        #
        # `read_glacier_cell_status` deliberately does *not* read the firn state; that slab is tens
        # of MB per cell and is only needed once the cell is known to need work.
        status = read_glacier_cell_status(path)

        # `k` is a per-cell property (looked up by position), so it is part of the parameter set
        # the check compares and has to be resolved before it. Resolved here rather than inside
        # `gemb_glacier_cell` — which is why the run below is passed this value instead of
        # `apply_glacier_decoupling` — because the two must agree, or the check would validate a
        # different experiment than the one that runs. `resolve_decoupling_factor` is idempotent,
        # so handing its own output back resolves to itself.
        decoupling_factor = resolve_decoupling_factor(r, apply_glacier_decoupling)

        # The forcing window still to fetch. `nothing` status means a cold start over the whole
        # record; otherwise only forcing after the file's last output time can contribute, so the
        # request starts there — this is the saving, since the fetched window shrinks from decades
        # to whatever has been published since the last run.
        request_range = forcing_time_range

        # Overlap the saved record by this much rather than starting exactly at its last output
        # time. Two reasons, both about the window not being empty: the last *output* time is the
        # end of an accumulation interval and can fall after the last forcing step it consumed, and
        # `climate_forcing` reports a window containing no steps as a bare error rather than as
        # "nothing new". With the overlap the window always contains already-consumed steps,
        # `gemb_glacier_cell` drops everything at or before the saved time, and an unadvanced
        # forcing record surfaces as `ForcingUpToDate` — a benign, quietly skipped outcome.
        restart_overlap = Day(2)

        if status !== nothing
            requested_parameters = run_parameters(mp; coverage = hypsometry_coverage,
                                                  lapse_rate, decoupling_factor)

            # Fail on a parameter change here rather than after the download. The run grid
            # (bin centers, deltas, scalings) is structural and still checked inside
            # `gemb_glacier_cell`, which has the bin selection in hand.
            differences = run_parameter_differences(status.parameters, requested_parameters)
            if !isempty(differences) && !force_restart
                throw(RestartParameterMismatch(differences))
            end

            # Nothing newer than the file has been *asked* for, so there is nothing to fetch and
            # no reason to open the store at all. The forcing record itself may also be shorter
            # than the request, which only becomes visible once it is read — that case surfaces
            # as `ForcingUpToDate` from `gemb_glacier_cell` below.
            if status.time >= last(forcing_time_range)
                @info "Cell already spans the requested forcing; skipping" cell=i path last_time=status.time requested_through=last(forcing_time_range)
                continue
            end

            request_range = (max(first(forcing_time_range), status.time - restart_overlap),
                             last(forcing_time_range))
            @info "Extending existing cell" cell=i path last_time=status.time fetching=request_range
        end

        # An existing file carries the firn state and last time of the previous run; when present
        # the cell resumes from it over only the newer forcing and skips the spinup entirely.
        # Read only now that the cell is known to need work — this is the expensive read.
        restart = status === nothing ? nothing : read_glacier_cell_restart(path)

        # Download the forcing time series for this cell from the Copernicus Climate Data Store;
        # cached locally so re-runs skip the download. The returned stack is self-describing: its
        # metadata carries the cell's absolute (orthometric) surface elevation, which is the
        # reference the per-bin lapse adjustment raises from.
        forcing_data = climate_forcing(climate_model, r.latitude, r.longitude;
                                       time_range = request_range,
                                       token = cds_api_key,
                                       cache_path = forcing_cache)

        # Runs (bins x deltas x scalings) simulations and area-weights the per-bin mass fluxes
        # (kg m-2) by the glacier area attributed to each bin, giving per-cell masses (kg).
        # `gemb_glacier_cell` drops each simulation's output stack as soon as it has the flux
        # vectors, so verbose plotting hooks in there rather than working from `run`. Serial
        # because the hook is called from inside the simulation task and Makie is not
        # thread-safe.
        run = gemb_glacier_cell(r, forcing_data, mp;
                                delta_temperatures, precipitation_scalings,
                                coverage = hypsometry_coverage, lapse_rate,
                                glacier_decoupling = decoupling_factor,
                                restart, force_restart,
                                threaded = !verbose_plotting,
                                on_output = verbose_plotting ? verbose_plotter(i, r) : nothing)

        if restart === nothing
            write_glacier_cell_netcdf(path, run;
                                      institution = "NASA Jet Propulsion Laboratory")
        else
            append_glacier_cell_netcdf(path, run)
        end

        @info "Wrote cell" path bins=length(run.bins) area_km2=sum(run.weights) steps=length(run.time)
    catch e
        e isa InterruptException && rethrow()
        # ERA5-Land is land-only, so a cell whose reanalysis grid point falls on water (common
        # for coastal and island glaciers) has no forcing at all. Those cells are unrunnable
        # rather than failed, so skip them quietly instead of logging an error per cell.
        if e isa ForcingUnavailable
            @info "Skipping cell: no land forcing at this reanalysis grid point" cell=i lat=r.latitude lon=r.longitude
            continue
        end
        # A run-parameter change is a property of this sweep's configuration, not of one cell,
        # so it would fail identically for every remaining cell. Abort rather than log it
        # thousands of times; the message says how to proceed deliberately.
        e isa RestartParameterMismatch && rethrow()
        # Same reasoning for a broken driver: a typo, an undefined name, or a stale session whose
        # loaded GEMB_GlacierSims predates a function used above is a property of the script, not
        # of the cell. Swallowed per cell it turns into thousands of identical "Cell failed"
        # warnings and looks like the *data* is bad — abort on the first one so the real error is
        # what gets read. Requires `using Revise` (or a restart) after editing the package.
        (e isa UndefVarError || e isa MethodError || e isa FieldError) && rethrow()
        # An existing file that already spans the forcing is up to date, not broken; re-running
        # the sweep before new forcing is published hits this for every completed cell.
        if e isa ForcingUpToDate
            @info "Cell already up to date" cell=i path restart_time=e.restart_time
            continue
        end
        @warn "Cell failed; continuing" cell=i path exception=(e, catch_backtrace())
    end
end

# Quick-look plot of the per-cell mass totals for the first cell written: cumulative mass change
# for every prescribed (temperature delta, precipitation scaling) combination.
begin
    files = isdir(output_dir) ? filter(endswith(".nc"), readdir(output_dir; join = true)) : String[]
    if !isempty(files)
        NCDatasets.NCDataset(first(files), "r") do ds
            # `datetime2decyear` takes the whole vector; GEMB and GEMB_ClimateForcing both
            # export it, so qualify which one.
            years = GEMB.datetime2decyear(collect(DateTime, ds["time"][:]))
            fig = Figure(size = (900, 500))
            ax = Axis(fig[1, 1]; xlabel = "year", ylabel = "cumulative mass change (Gt)",
                      title = basename(first(files)))
            for (j, dT) in enumerate(ds["delta_temperature"][:]),
                (k, ps) in enumerate(ds["precipitation_scaling"][:])
                # 1 Gt = 1e12 kg.
                lines!(ax, years, cumsum(ds["mass_change"][:, j, k]) .* 1e-12;
                       label = "ΔT=$(dT) K, P×$(ps)")
            end
            axislegend(ax; position = :lb, framevisible = false)
            display(fig)
        end
    end
end
