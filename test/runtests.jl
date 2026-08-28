using GEMB_GlacierSims
using Test
using DataFrames
using GeoDataFrames
using Dates
using DimensionalData
using Statistics
import GEMB
using GEMB: Z, DimStack, DimArray, initialize_parameters
using GEMB_ClimateForcing: climate_adjust_for_elevation
using NCDatasets

# One-row glacier elevation-class table from bin-center => area (km²) pairs, using the same
# flat `hyps_<lo>_<hi>` column encoding the runfile writes.
function _hyps_row(pairs...)
    cols = Pair{Symbol,Vector{Float64}}[]
    for (center, area) in pairs
        # `_hyps_colnames` is the writer's own naming, so the test table cannot drift from it.
        name = only(GEMB_GlacierSims._hyps_colnames([center - 50, center + 50]))
        push!(cols, name => [Float64(area)])
    end
    return first(eachrow(DataFrame(cols)))
end

# Synthetic profile with distinctive, non-round values so an exact round-trip is meaningful.
function _fake_profile(n::Int, seed::Float64)
    zdim = Z(1:n)
    layers = NamedTuple(v => DimArray([seed + i * 0.123456789012345 + j
                                       for i in 1:n], (zdim,))
                        for (j, v) in enumerate(GEMB_GlacierSims.PROFILE_VARIABLES))
    return DimStack(layers)
end

const _CP_TIME = collect(DateTime(2000, 1, 1):Hour(1):DateTime(2000, 1, 1, 5))

# Synthetic climate forcing shaped exactly like `climate_forcing` output: the seven layers on a `Ti`
# axis, with the metadata keys `forcing_is_complete` and the elevation interval pass reads.
function _cp_forcing(lat, lon, elevation, temperature)
    n = length(temperature)
    ti = Ti(_CP_TIME[1:n])
    const_layers = (pressure_air = 8e4, vapor_pressure = 300.0, wind_speed = 3.0,
                    precipitation = 0.1, shortwave_downward = 100.0, longwave_downward = 250.0)
    layers = merge(
        (temperature_air = DimArray(collect(Float64, temperature), (ti,);
                                    metadata = Dict("units" => "K")),),
        NamedTuple(v => DimArray(fill(Float64(x), n), (ti,)) for (v, x) in pairs(const_layers)))
    return DimStack(layers; metadata = Dict{String,Any}(
        "latitude" => Float64(lat), "longitude" => Float64(lon),
        "elevation" => Float64(elevation), "dataset" => "synthetic",
        "temperature_observation_height" => 2.0, "wind_observation_height" => 10.0))
end

# Multi-cell elevation-class table for the regional derivation. Each cell is
# `(; lon, lat, z, glm, bins)` where `bins` maps a bin center to its glacier area (km²).
function _cp_table(cells)
    df = DataFrame()
    # Every bin the runfile writes, so the table has the full flat schema a real one does and
    # `glacier_hypsometry` decodes it the same way.
    all_names = GEMB_GlacierSims._hyps_colnames(0:100:10_000)
    for c in cells
        d = Dict{Symbol,Any}(:geometry => GeoDataFrames.GeoInterface.Point(c.lon, c.lat),
                             :glm => c.glm, :latitude => Float64(c.lat),
                             :longitude => Float64(c.lon), :chunk_id => 1)
        for nm in all_names
            d[nm] = 0.0
        end
        for (center, area) in c.bins
            d[only(GEMB_GlacierSims._hyps_colnames([center - 50, center + 50]))] = Float64(area)
        end
        push!(df, d; cols = :union)
    end
    return df
end

# A loader closure over `cells`, returning each cell's forcing by its coordinates. `temperature`
# maps a cell to its temperature vector, so a testset controls the physics it wants to recover.
function _cp_loader(cells, temperature)
    # Signature matches `climate_forcing`'s, which is the contract `forcing_loader` promises;
    # everything but the coordinates is ignored here.
    return function (_model, lat, lon; time_range = nothing, token = nothing, cache_path = nothing)
        i = findfirst(c -> c.lat == lat && c.lon == lon, cells)
        i === nothing && throw(ArgumentError("no synthetic cell at ($lat, $lon)"))
        c = cells[i]
        return _cp_forcing(lat, lon, c.z, temperature(c))
    end
end

# Square region polygon, in the (-180, 180] convention a real region file uses.
function _cp_region(x0, y0, x1, y1)
    GI = GeoDataFrames.GeoInterface
    return GI.Polygon([GI.LinearRing([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)])])
end

function _fake_run(; time, deltas, scalings, bins, weights, lengths,
                   decoupling_factor = nothing,
                   parameters = Dict{String,Any}("hypsometry_coverage" => 0.95,
                                                 "temperature_lapse_rate" => 6.5,
                                                 "model_albedo_ice" => 0.48,
                                                 "model_densification_method" => :Arthern,
                                                 "model_shortwave_subsurface_absorption" => false))
    n_bin, n_dt, n_ps = length(bins), length(deltas), length(scalings)
    totals = Dict{Symbol,Array{Float64,3}}()
    for v in GEMB_GlacierSims.CELL_TOTAL_VARIABLES
        totals[v] = [Float64(i) + 100 * j + 10_000 * k
                     for i in eachindex(time), j in 1:n_dt, k in 1:n_ps]
    end
    profiles = Array{Union{Nothing,DimStack}}(nothing, n_bin, n_dt, n_ps)
    for k in 1:n_ps, j in 1:n_dt, i in 1:n_bin
        n = lengths[i]
        n == 0 && continue
        profiles[i, j, k] = _fake_profile(n, 100.0 * i + 10.0 * j + k)
    end
    return GlacierCellRun(
        72.5, -38.25, 1234.5, decoupling_factor, 42, 0.87,
        collect(Float64, deltas), collect(Float64, scalings),
        [(; lo = Int(c) - 50, hi = Int(c) + 50, center = Float64(c), area = w)
         for (c, w) in zip(bins, weights)],
        collect(Float64, weights),
        collect(time), totals, profiles, parameters,
        Dict{String,Any}("spinup_performed" => true, "spinup_cycles" => 37,
                         "spinup_converged" => false,
                         "climatology_window_start" => DateTime(1950, 1, 1),
                         "spinup_convergence_drift_density" => nothing),
    )
end

# A fully-populated `GlacierTileRun` without running GEMB, for the writer and restart tests. Values are
# distinctive and non-round so an exact round-trip is meaningful, and one band is left with no profile
# so the "band produced no output" path is covered.
function _fake_tile_run(; time = collect(DateTime(2000, 1, 1):Month(1):DateTime(2000, 6, 1)),
                        bands = [(1050, 12.5), (1550, 40.25), (2050, 3.0)],
                        deltas = [0.0, 1.0], scalings = [1.0, 1.5])
    n_band, n_dt, n_ps, n_t = length(bands), length(deltas), length(scalings), length(time)
    band_tuples = [(; lo = Int(c) - 50, hi = Int(c) + 50, center = Float64(c),
                    area = a, n_cells = i) for (i, (c, a)) in enumerate(bands)]

    bands_series = Dict{Symbol,Array{Float64,4}}()
    for (j, v) in enumerate((TILE_MASS_VARIABLES..., TILE_HEIGHT_VARIABLES...))
        bands_series[v] = [0.5j + i + 10b + 100d + 1000s
                           for i in 1:n_t, b in 1:n_band, d in 1:n_dt, s in 1:n_ps]
    end
    band_areas = [b.area for b in band_tuples]
    totals = Dict{Symbol,Array{Float64,3}}()
    # Cumulative, as `_tile_totals` produces and as the file documents: the per-band series are per
    # output interval, the tile totals accumulate from the record start.
    for v in TILE_MASS_VARIABLES
        totals[v] = cumsum([tile_mass_total(bands_series[v][i:i, :, d, s], band_areas)[1]
                            for i in 1:n_t, d in 1:n_dt, s in 1:n_ps]; dims = 1)
    end
    totals[:dm] = totals[:precipitation] .- totals[:runoff] .+
                  totals[:evaporation_condensation] .+ totals[:blowing_snow]
    for (name, source) in ((:dv, :dh), (:dv_mass, :dh_mass), (:dv_firn, :dh_firn),
                           (:fac, :firn_air_content))
        totals[name] = [tile_volume_change(bands_series[source][i:i, :, d, s], band_areas)[1]
                        for i in 1:n_t, d in 1:n_dt, s in 1:n_ps]
    end

    profiles = Array{Union{Nothing,DimStack}}(nothing, n_band, n_dt, n_ps)
    for s in 1:n_ps, d in 1:n_dt, b in 1:(n_band - 1)     # last band deliberately has no state
        profiles[b, d, s] = _fake_profile(4 + b, 100.0b + 10.0d + s)
    end

    band_provenance = [Dict{String,Any}(
        "extrapolation_above_reanalysis" => 100.0i - 250.0,
        "temperature_lapse_rate" => 6.2,
        "glacier_decoupling_factor_mean" => 0.4 + 0.05i,
        "glacier_decoupling_factor_n_fit_held" => i - 1,
        "glacier_decoupling_factor_n_fit_in_domain" => 3,
        "n_timesteps_above_freezing" => 4,
        "glacier_decoupling_factor_n_fitted" => 3,
        "glacier_decoupling_factor_n_fitted_above_freezing" => 2,
        "temperature_lapse_rate_n_fitted" => n_t) for i in 1:n_band]
    # One band missing a key the others carry. The writer defines a variable per key seen anywhere, so
    # this band's entry has to read back as absent rather than as a number — the fill is the only thing
    # that distinguishes "not recorded" from "recorded as zero".
    n_band > 1 && delete!(band_provenance[end], "glacier_decoupling_factor_n_fit_held")

    parameters = Dict{String,Any}("downscaling_basis" => "fitted",
                                 "downscaling_min_cells" => 8,
                                 "spinup_window_start" => "1990-01-01T00:00:00",
                                 "spinup_window_stop" => "2019-12-31T00:00:00",
                                 "model_densification_method" => :Arthern,
                                 "model_albedo_ice" => 0.48)

    return GlacierTileRun(
        (-142, 60), "N60_W142.nc", "lat[+60+62]lon[-142-140]",
        (lon_min = -142.0, lon_max = -140.0, lat_min = 60.0, lat_max = 62.0),
        335, 716, band_tuples, collect(Float64, deltas), collect(Float64, scalings),
        collect(time), bands_series, totals, profiles, band_provenance, parameters,
        Dict{String,Any}("spinup_performed" => true, "spinup_cycles" => 37,
                         "spinup_converged" => false,
                         "climatology_window_start" => DateTime(1990, 1, 1),
                         "max_dh_residual" => 1.2e-14),
    )
end

@testset "GEMB_GlacierSims.jl" begin
    @testset "module surface" begin
        @test isdefined(GEMB_GlacierSims, :era5_land_invariant)
        @test isdefined(GEMB_GlacierSims, :gemb_glacier_elevation_class_runfile)
        @test isdefined(GEMB_GlacierSims, :glacier_hypsometry_coverage)
        @test isdefined(GEMB_GlacierSims, :gemb_glacier_cell)
        @test isdefined(GEMB_GlacierSims, :write_glacier_cell_netcdf)
        @test isdefined(GEMB_GlacierSims, :append_glacier_cell_netcdf)
        @test isdefined(GEMB_GlacierSims, :read_glacier_cell_restart)
        @test isdefined(GEMB_GlacierSims, :forcing_is_complete)
        @test isdefined(GEMB_GlacierSims, :ForcingUpToDate)
        @test isdefined(GEMB_GlacierSims, :ForcingUnavailable)
        @test isdefined(GEMB_GlacierSims, :glacier_area_total)
        @test isdefined(GEMB_GlacierSims, :RestartParameterMismatch)
        @test isdefined(GEMB_GlacierSims, :run_parameters)
        @test isdefined(GEMB_GlacierSims, :read_glacier_cell_parameters)
        @test isdefined(GEMB_GlacierSims, :read_glacier_cell_status)
        @test isdefined(GEMB_GlacierSims, :run_parameter_differences)
        @test isdefined(GEMB_GlacierSims, :resolve_decoupling_factor)
        @test isdefined(GEMB_GlacierSims, :cell_decoupling_factor)
        @test isdefined(GEMB_GlacierSims, :decoupling_factor_label)
        @test isdefined(GEMB_GlacierSims, :decoupling_factor_at_elevation)
    end

    @testset "cell_decoupling_factor" begin
        # Reads the vendored Shaw et al. (2025) table shipped with GEMB_ClimateForcing, so this
        # is offline. Haut Glacier d'Arolla (RGI60-11.02810) is ~0.5 km from this point.
        cell(lon, lat, glm...) = first(eachrow(DataFrame(
            :geometry => [GeoDataFrames.GeoInterface.Point(lon, lat)],
            (isempty(glm) ? () : (:glm => [only(glm)],))...)))

        full = cell_decoupling_factor(cell(7.53, 45.97, 0.0))
        @test full.rgi_id == "RGI60-11.02810"
        @test full.distance < 1.0
        # A cell ERA5-Land treats as ice-free gets the published k unweighted.
        @test full.decoupling_factor == full.decoupling_factor_published
        @test 0 < full.decoupling_factor < 1

        k_pub = full.decoupling_factor_published

        # The weighting is linear in the non-glacier fraction and exact at both ends: no glacier
        # in the cell is the full correction, an all-glacier cell is the identity.
        @test cell_decoupling_factor(cell(7.53, 45.97, 1.0)).decoupling_factor == 1.0
        @test cell_decoupling_factor(cell(7.53, 45.97, 0.5)).decoupling_factor ≈
              1 - (1 - k_pub) * 0.5
        @test cell_decoupling_factor(cell(7.53, 45.97, 0.25)).decoupling_factor ≈
              1 - (1 - k_pub) * 0.75

        # A `missing` or `NaN` glm is a real data gap, so it falls back to the uncorrected
        # reanalysis assumption (glm = 0) and gets the full correction.
        @test cell_decoupling_factor(cell(7.53, 45.97, missing)).decoupling_factor == k_pub
        @test cell_decoupling_factor(cell(7.53, 45.97, NaN)).decoupling_factor == k_pub

        # A table with no `:glm` column at all is schema drift, not a data gap: it cannot weight
        # the correction, and silently applying the unweighted factor to every cell in a sweep is
        # the bug this replaces. `scripts/migrate_invariant_colnames.jl` renames `glm_frac`.
        @test_throws ArgumentError cell_decoupling_factor(cell(7.53, 45.97))
        glm_frac_only = first(eachrow(DataFrame(
            geometry = [GeoDataFrames.GeoInterface.Point(7.53, 45.97)], glm_frac = [0.5])))
        @test_throws ArgumentError cell_decoupling_factor(glm_frac_only)

        # Cell longitudes are native 0–359.9°E on the climate grid; the lookup wraps them, so
        # both conventions must find the same glacier.
        @test cell_decoupling_factor(cell(-145.0, 60.5, 0.0)).rgi_id ==
              cell_decoupling_factor(cell(215.0, 60.5, 0.0)).rgi_id

        # No published k is a run-on-ambient-forcing outcome, not a failure: RGI region 19 is
        # absent from the dataset, and open ocean has no glacier within max_distance.
        @test cell_decoupling_factor(cell(0.0, -78.0, 0.0)) === nothing
        @test cell_decoupling_factor(cell(-30.0, 20.0, 0.0)) === nothing

        # `max_distance` bounds the nearest-centroid match, so tightening it past the real
        # distance turns a hit into a no-correction cell.
        @test cell_decoupling_factor(cell(7.53, 45.97, 0.0); max_distance = 0.1) === nothing
    end

    @testset "glacier_decoupling keyword resolution" begin
        cell(glm) = first(eachrow(DataFrame(
            geometry = [GeoDataFrames.GeoInterface.Point(7.53, 45.97)], glm = [glm])))
        resolve = resolve_decoupling_factor

        # `false` skips the lookup entirely; `true` looks k up and weights it by 1 - glm.
        @test resolve(cell(0.0), false) === nothing
        @test resolve(cell(0.0), true) == cell_decoupling_factor(cell(0.0)).decoupling_factor

        # An entirely glaciated cell weights to exactly 1.0 — the identity — so the adjustment
        # is skipped rather than run as a no-op pass over the whole forcing record.
        @test resolve(cell(1.0), true) === nothing

        # A prescribed `k` bypasses both the table and the glm weighting.
        @test resolve(cell(1.0), 0.6) === 0.6

        # A cell with no published k is run on ambient forcing.
        ocean = first(eachrow(DataFrame(
            geometry = [GeoDataFrames.GeoInterface.Point(-30.0, 20.0)], glm = [0.0])))
        @test resolve(ocean, true) === nothing

        @test :glacier_decoupling in Base.kwarg_decl(only(methods(gemb_glacier_cell)))
    end

    @testset "decoupling factor provenance" begin
        time = collect(DateTime(2000, 1, 31):Month(1):DateTime(2000, 3, 31))
        kw = (; time, deltas = [0.0], scalings = [1.0], bins = [1150], weights = [10.3],
              lengths = [12])
        dir = mktempdir()

        # An applied factor is written as a scalar variable, like `forcing_elevation` — it is a
        # property of the cell, not of the perturbation grid.
        applied = joinpath(dir, "applied.nc")
        write_glacier_cell_netcdf(applied, _fake_run(; kw..., decoupling_factor = 0.7742))
        NCDatasets.NCDataset(applied, "r") do ds
            @test ds["glacier_decoupling_factor"][] ≈ 0.7742
            @test ds["glacier_decoupling_factor"].attrib["units"] == "1"
            @test isempty(NCDatasets.dimnames(ds["glacier_decoupling_factor"]))
        end

        # No correction reads back as `missing` (the fill), so a cell left on ambient forcing is
        # distinguishable from one corrected by exactly 1.0 — which the parameter attribute,
        # compared numerically on restart, cannot express.
        ambient = joinpath(dir, "ambient.nc")
        write_glacier_cell_netcdf(ambient, _fake_run(; kw..., decoupling_factor = nothing))
        NCDatasets.NCDataset(ambient, "r") do ds
            @test ismissing(ds["glacier_decoupling_factor"][])
        end

        # The field is on the struct itself, so a caller can read it back off the run.
        @test _fake_run(; kw..., decoupling_factor = 0.8).decoupling_factor == 0.8
        @test _fake_run(; kw...).decoupling_factor === nothing
    end

    @testset "decoupling_factor_label" begin
        # The label a plot header carries. Absent — not "k=1.0" — when no correction was applied:
        # a k on every figure of an uncorrected sweep is noise, and the absence is what
        # distinguishes ambient forcing from a cell the correction happened to leave alone.
        @test decoupling_factor_label(nothing) == ""
        @test decoupling_factor_label(0.7742) == "k=0.774"
        @test decoupling_factor_label(0.8) == "k=0.8"
        # Rounded for display, so a long-tailed k does not widen the header.
        @test decoupling_factor_label(0.77424242) == "k=0.774"
    end

    @testset "glacier_hypsometry_coverage" begin
        row = _hyps_row(1050 => 0.3, 1150 => 10.0, 1250 => 8.0, 1950 => 0.4)

        # The cheap area screen must agree with the full bin decoding it stands in for.
        @test glacier_area_total(row) ≈ 18.7
        @test glacier_area_total(row) ≈
              glacier_hypsometry_coverage(row; coverage = 0.95).total_area

        c = glacier_hypsometry_coverage(row; coverage = 0.95)
        # 10 + 8 = 18 >= 0.95 * 18.7, so only the two big bins are modeled.
        @test [b.center for b in c.modeled] == [1150.0, 1250.0]
        @test c.total_area ≈ 18.7
        # No area is dropped: the reassignment conserves the cell total exactly.
        @test sum(c.weights) ≈ c.total_area
        # 1050 -> 1150 (nearest), 1950 -> 1250 (nearest).
        @test c.weights ≈ [10.3, 8.4]

        # Modeled bins are elevation-sorted even though selection is by descending area.
        @test issorted([b.center for b in c.modeled])

        # coverage = 1 models everything and leaves weights untouched.
        c1 = glacier_hypsometry_coverage(row; coverage = 1.0)
        @test length(c1.modeled) == 4
        @test c1.weights == [b.area for b in c1.modeled]

        # An exactly equidistant unmodeled bin goes to the lower-elevation neighbour.
        tie = glacier_hypsometry_coverage(_hyps_row(1000 => 10.0, 1100 => 0.1, 1200 => 9.0);
                                         coverage = 0.9)
        @test [b.center for b in tie.modeled] == [1000.0, 1200.0]
        @test tie.weights ≈ [10.1, 9.0]

        # Empty and invalid input.
        empty = glacier_hypsometry_coverage(_hyps_row(1000 => 0.0); coverage = 0.95)
        @test isempty(empty.modeled)
        @test empty.total_area == 0.0
        @test_throws ArgumentError glacier_hypsometry_coverage(_hyps_row(1000 => 1.0); coverage = 0)
        @test_throws ArgumentError glacier_hypsometry_coverage(_hyps_row(1000 => 1.0); coverage = 1.5)
    end

    @testset "glacier_area_total and glacier_area_column" begin
        df = DataFrame(
            Symbol(only(GEMB_GlacierSims._hyps_colnames([1000, 1100]))) => [10.0, 1.0, 0.0],
            Symbol(only(GEMB_GlacierSims._hyps_colnames([1100, 1200]))) => [8.0, 2.0, 0.0],
            # A non-hypsometry column must not be summed into a cell's glacier area, which is why
            # the sum tests column names through `_parse_hyps_colname` rather than a `hyps_` prefix.
            :glm => [0.4, 0.6, 0.1])

        # The column form and the row form agree — they are the same screen at two granularities,
        # so a table screened one way and a cell checked the other cannot disagree.
        @test glacier_area_column(df) == [18.0, 3.0, 0.0]
        @test [glacier_area_total(r) for r in eachrow(df)] == [18.0, 3.0, 0.0]

        # With the total cached as a column, both read it instead of summing. This is the point of
        # the column: the screen becomes a scalar read per row rather than a 100-column sum.
        cached = copy(df)
        cached[!, GEMB_GlacierSims.GLACIER_AREA_COLUMN] = [18.0, 3.0, 0.0]
        @test glacier_area_column(cached) == [18.0, 3.0, 0.0]
        @test [glacier_area_total(r) for r in eachrow(cached)] == [18.0, 3.0, 0.0]

        # The cached column is authoritative when present, not merely consistent with the bins —
        # that is what makes it a fast path rather than a redundant check. A table whose cache
        # disagrees returns the cache, so the migration verifies the values it writes.
        stale = copy(df)
        stale[!, GEMB_GlacierSims.GLACIER_AREA_COLUMN] = [99.0, 99.0, 99.0]
        @test glacier_area_column(stale) == [99.0, 99.0, 99.0]
        @test glacier_area_total(first(eachrow(stale))) == 99.0

        # A view screens against its own rows, so `grid_cells_in_region`'s column-wise area screen
        # composes with an already-filtered table.
        sub = view(df, [2, 1], :)
        @test glacier_area_column(sub) == [3.0, 18.0]

        # An Int column (a table built with integer areas) still returns Float64, so the screen
        # comparison and the stored NetCDF weights are the same type either way.
        ints = DataFrame(Symbol(only(GEMB_GlacierSims._hyps_colnames([1000, 1100]))) => [3, 4])
        @test glacier_area_column(ints) == [3.0, 4.0]
        @test glacier_area_column(ints) isa Vector{Float64}
        @test glacier_area_total(first(eachrow(ints))) === 3.0
    end

    @testset "forcing_is_complete" begin
        # ERA5-Land is land-only, so a cell whose grid point falls on water comes back all-NaN
        # with a NaN reference elevation. `gemb_glacier_cell` must reject that before GEMB
        # reports it as an unrelated-looking units assertion.
        tdim = Ti(collect(DateTime(2000, 1, 1):Day(1):DateTime(2000, 1, 10)))
        vars = (:temperature_air, :pressure_air, :vapor_pressure, :wind_speed,
                :precipitation, :shortwave_downward, :longwave_downward)
        good = DimStack(NamedTuple(v => DimArray(fill(260.0, length(tdim)), (tdim,)) for v in vars);
                        metadata = Dict{String,Any}("elevation" => 1200.0))
        @test forcing_is_complete(good)

        water = DimStack(NamedTuple(v => DimArray(fill(NaN, length(tdim)), (tdim,)) for v in vars);
                         metadata = Dict{String,Any}("elevation" => NaN))
        @test !forcing_is_complete(water)

        # A finite elevation but a hole in one variable is still unrunnable.
        holed = DimStack(NamedTuple(v => DimArray(v === :wind_speed ?
                                                  [i == 3 ? NaN : 5.0 for i in 1:length(tdim)] :
                                                  fill(260.0, length(tdim)), (tdim,))
                                    for v in vars);
                         metadata = Dict{String,Any}("elevation" => 1200.0))
        @test !forcing_is_complete(holed)
    end

    @testset "ForcingUpToDate" begin
        e = ForcingUpToDate(DateTime(2004, 12, 31), 1)
        @test e isa Exception
        @test occursin("2004-12-31", sprint(showerror, e))
        @test occursin("1 forcing step", sprint(showerror, e))
    end

    @testset "run_parameters" begin
        mp = initialize_parameters(output_frequency = :monthly)
        p = run_parameters(mp; coverage = 0.95, lapse_rate = 6.5)

        @test p["hypsometry_coverage"] == 0.95
        @test p["temperature_lapse_rate"] == 6.5
        # Every ModelParameters field is captured, prefixed, except what gemb derives itself.
        @test p["model_output_frequency"] === :monthly
        @test p["model_albedo_ice"] == mp.albedo_ice
        @test p["model_densification_method"] === mp.densification_method
        @test !haskey(p, "model_dt_divisors")
        for field in GEMB.DERIVED_PARAMETERS
            @test !haskey(p, "model_" * string(field))
        end
        @test length(p) == length(propertynames(mp)) - length(GEMB.DERIVED_PARAMETERS) + 3

        # A monthly cycle is stored as the 12-element vector, not flattened to a scalar or a
        # string: NetCDF holds a numeric attribute vector natively, so the restart check compares
        # it elementwise. Given as `Int`s it must land as the same parameter as the float form,
        # or a cycle written one way and requested the other reads as a parameter change.
        monthly = round.(6.5 .+ 0.5 .* sin.(2π .* (1:12) ./ 12), digits = 3)
        pm = run_parameters(mp; coverage = 0.95, lapse_rate = monthly)
        @test pm["temperature_lapse_rate"] == monthly
        @test run_parameters(mp; coverage = 0.95,
                             lapse_rate = collect(1:12))["temperature_lapse_rate"] ==
              collect(Float64, 1:12)

        # A per-timestep rate is refused at the point the caller can act on it. Its length tracks
        # the forcing record, so it could not be recorded as a parameter a continuation matches —
        # every append would read as a parameter change with no way to store it that does not.
        err = try
            run_parameters(mp; coverage = 0.95, lapse_rate = fill(6.5, 100))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("12-element monthly", err.msg)
        @test occursin("continuation", err.msg)

        # No decoupling is recorded as the identity, not as an absent key, so switching the
        # correction on or off is a visible parameter change on restart.
        @test p["glacier_decoupling_factor"] == 1.0
        @test run_parameters(mp; coverage = 0.95, lapse_rate = 6.5,
                             decoupling_factor = 0.8)["glacier_decoupling_factor"] == 0.8

        # Provenance that legitimately changes between a run and its continuation is not part of
        # the parameter set, so extending a record never trips the check on it.
        @test !any(k -> startswith(k, "spinup") || startswith(k, "climatology"), keys(p))

        # A changed setting is visible as a changed parameter.
        p2 = run_parameters(initialize_parameters(output_frequency = :monthly, albedo_ice = 0.4);
                            coverage = 0.95, lapse_rate = 6.5)
        @test p2["model_albedo_ice"] == 0.4
    end

    @testset "run_parameter_differences" begin
        # Compared in the NetCDF-encoded form, so a stored `"Arthern"` matches a requested
        # `:Arthern` and a stored `"false"` matches a requested `false`.
        saved = Dict{String,Any}("model_albedo_ice" => 0.48,
                                 "model_densification_method" => "Arthern",
                                 "model_shortwave_subsurface_absorption" => "false")
        @test isempty(run_parameter_differences(saved,
            Dict{String,Any}("model_albedo_ice" => 0.48,
                             "model_densification_method" => :Arthern,
                             "model_shortwave_subsurface_absorption" => false)))

        # Only keys present in both are compared: a file written by an older `run_parameters`
        # carries fewer of them, which is a gap rather than a disagreement.
        @test isempty(run_parameter_differences(saved, Dict{String,Any}("brand_new_key" => 1.0)))

        d = run_parameter_differences(saved, Dict{String,Any}("model_albedo_ice" => 0.40,
                                                             "model_densification_method" => :Ligtenberg))
        @test d["model_albedo_ice"] == (0.48, 0.40)
        @test d["model_densification_method"] == ("Arthern", :Ligtenberg)

        # A monthly lapse-rate cycle compares elementwise, so a stored integer cycle matches the
        # float form and a single changed month is a disagreement. Without vector encoding these
        # would compare by print formatting, where `Any[...]` and `Float64[...]` differ while the
        # values do not.
        cycle = collect(Float64, 1:12)
        @test isempty(run_parameter_differences(
            Dict{String,Any}("temperature_lapse_rate" => cycle),
            Dict{String,Any}("temperature_lapse_rate" => collect(1:12))))
        changed = copy(cycle); changed[7] = 9.0
        @test haskey(run_parameter_differences(
            Dict{String,Any}("temperature_lapse_rate" => cycle),
            Dict{String,Any}("temperature_lapse_rate" => changed)), "temperature_lapse_rate")

        # A scalar and a monthly cycle are different configurations even when every month equals
        # the scalar — one is a request for a constant rate, the other for a seasonal one.
        @test haskey(run_parameter_differences(
            Dict{String,Any}("temperature_lapse_rate" => 6.5),
            Dict{String,Any}("temperature_lapse_rate" => fill(6.5, 12))), "temperature_lapse_rate")
    end

    @testset "read_glacier_cell_status" begin
        # The cheap pre-flight read: everything `read_glacier_cell_restart` returns except the
        # firn state, so a driver can decide whether a cell needs any work before downloading
        # forcing or paging in the restart slab.
        t = collect(DateTime(2000, 1, 31):Month(1):DateTime(2000, 6, 30))
        dir = mktempdir()
        path = joinpath(dir, "cell.nc")
        write_glacier_cell_netcdf(path, _fake_run(; time = t, deltas = [0.0, 1.0],
                                                 scalings = [1.0], bins = [1150],
                                                 weights = [10.3], lengths = [12]))

        status = read_glacier_cell_status(path)
        @test status.time == last(t)
        @test status.bin_centers == [1150.0]
        @test status.delta_temperatures == [0.0, 1.0]
        @test status.precipitation_scalings == [1.0]
        @test status.parameters["hypsometry_coverage"] == 0.95
        @test !haskey(status, :profiles)

        # The fields it shares with the restart reader must agree, or the pre-flight check would
        # validate something other than what the run then checks.
        restart = read_glacier_cell_restart(path)
        for k in (:time, :bin_centers, :delta_temperatures, :precipitation_scalings, :parameters)
            @test getproperty(status, k) == getproperty(restart, k)
        end

        # A cold start is `nothing` rather than an error, so a driver can fall through to it.
        @test read_glacier_cell_status(joinpath(dir, "absent.nc")) === nothing
    end

    @testset "restart parameter validation" begin
        t1 = collect(DateTime(2000, 1, 31):Month(1):DateTime(2000, 6, 30))
        t2 = collect(DateTime(2000, 7, 31):Month(1):DateTime(2000, 12, 31))
        kw = (; deltas = [0.0], scalings = [1.0], bins = [1150], weights = [10.3],
              lengths = [12])
        dir = mktempdir()
        path = joinpath(dir, "cell.nc")
        write_glacier_cell_netcdf(path, _fake_run(; time = t1, kw...))

        # The stored parameters round-trip, named by the file's own roster.
        stored = read_glacier_cell_parameters(path)
        @test stored["hypsometry_coverage"] == 0.95
        # NetCDF attributes hold no Symbol or Bool, so those come back encoded.
        @test stored["model_densification_method"] == "Arthern"
        @test stored["model_shortwave_subsurface_absorption"] == "false"
        @test read_glacier_cell_parameters(joinpath(dir, "absent.nc")) === nothing

        # A monthly lapse-rate cycle survives the file as numbers, so the restart check compares
        # it to the requested cycle elementwise. This is the round trip the in-memory
        # `run_parameter_differences` tests assume: without it the cycle would come back
        # stringified and every continuation would read as a parameter change.
        cycle_path = joinpath(dir, "cycle.nc")
        cycle = round.(6.5 .+ 0.5 .* sin.(2π .* (1:12) ./ 12), digits = 3)
        cycle_params = run_parameters(initialize_parameters(); coverage = 0.95,
                                     lapse_rate = cycle)
        write_glacier_cell_netcdf(cycle_path,
                                  _fake_run(; time = t1, kw..., parameters = cycle_params))
        cycle_restart = read_glacier_cell_restart(cycle_path)
        @test cycle_restart.parameters["temperature_lapse_rate"] == cycle
        @test isempty(run_parameter_differences(cycle_restart.parameters, cycle_params))
        @test GEMB_GlacierSims._validate_restart_parameters(
            cycle_restart, cycle_params; force_restart = false) === nothing

        restart = read_glacier_cell_restart(path)
        @test restart.parameters["model_albedo_ice"] == 0.48

        # An identical parameter set validates silently. Keys the file does not carry are
        # reported as unverified, not as a mismatch, so the full set can be passed as-is.
        matching = run_parameters(initialize_parameters(); coverage = 0.95, lapse_rate = 6.5)
        @test GEMB_GlacierSims._validate_restart_parameters(
            restart, matching; force_restart = false) === nothing

        # A changed parameter is refused, and the exception names the offender and both values.
        changed = Dict{String,Any}("model_albedo_ice" => 0.40)
        err = try
            GEMB_GlacierSims._validate_restart_parameters(restart, changed; force_restart = false)
        catch e
            e
        end
        @test err isa RestartParameterMismatch
        @test haskey(err.differences, "model_albedo_ice")
        @test err.differences["model_albedo_ice"] == (0.48, 0.40)
        msg = sprint(showerror, err)
        @test occursin("model_albedo_ice", msg)
        @test occursin("force_restart = true", msg)

        # force_restart waives it.
        @test GEMB_GlacierSims._validate_restart_parameters(restart, changed;
                                                            force_restart = true) === nothing

        # A file carrying no roster cannot be verified, but is not refused.
        @test GEMB_GlacierSims._validate_restart_parameters(
            (; parameters = Dict{String,Any}()), changed; force_restart = false) === nothing

        # Appending refreshes the stored parameters (the forced case must not leave the file
        # advertising settings the new segment was not run with) and extends `history`.
        forced = _fake_run(; time = t2, kw...,
                           parameters = merge(stored, Dict{String,Any}("model_albedo_ice" => 0.40)))
        append_glacier_cell_netcdf(path, forced)
        NCDatasets.NCDataset(path, "r") do ds
            @test ds.attrib["model_albedo_ice"] == 0.40
            @test occursin("extended through", ds.attrib["history"])
            @test occursin("created by", ds.attrib["history"])
        end
    end

    @testset "PROFILE_VARIABLES is the complete GEMB state" begin
        # The roster is what the restart group stores, so a layer missing from it is silently
        # dropped on write and then rejected inside `gemb` on continuation. It is an alias of
        # `GEMB.RESTART_LAYERS`, and this pins that alias against a real initialized profile —
        # so the assertion still fails here if GEMB's constant and GEMB's state diverge, rather
        # than at the first restart months later.
        n = 10
        t = collect(DateTime(2000, 1, 1):Day(1):DateTime(2000, 1, n))
        cf = GEMB.initialize_forcing(t, fill(250.0, n), fill(85000.0, n), fill(3.0, n),
                                     fill(5.0, n), fill(0.0, n), fill(180.0, n), fill(50.0, n);
                                     temperature_observation_height = 2.0,
                                     wind_observation_height = 10.0)
        profile = GEMB.initialize_profile(initialize_parameters(), cf)

        # Set equality both ways: no state layer is unsaved, and nothing is listed that a
        # profile does not actually carry.
        @test Set(GEMB_GlacierSims.PROFILE_VARIABLES) == Set(keys(profile))

        # `age` specifically — the column's only clock, and the layer this roster was missing.
        @test :age in GEMB_GlacierSims.PROFILE_VARIABLES

        # Every listed layer has CF attributes, which `_write_restart_group!` writes unguarded.
        for v in GEMB_GlacierSims.PROFILE_VARIABLES
            @test haskey(GEMB.cf_attributes(v; time_axis = false), "units")
        end
    end

    @testset "restart group predating a state layer" begin
        # Files written before `age` joined the roster have no `age` variable. Continuing one
        # would hand `gemb` an incomplete profile; the read must say so and name the remedy.
        dir = mktempdir()
        path = joinpath(dir, "cell.nc")
        write_glacier_cell_netcdf(path, _fake_run(; time = collect(DateTime(2000, 1, 31):Month(1):DateTime(2000, 3, 31)),
                                                  deltas = [0.0], scalings = [1.0],
                                                  bins = [1150], weights = [10.3],
                                                  lengths = [12]))
        # Drop `age` to emulate an old file. NCDatasets cannot delete a variable, so rewrite
        # the restart group without it by copying the file through a fresh dataset.
        old = joinpath(dir, "old.nc")
        NCDatasets.NCDataset(path, "r") do src
            NCDatasets.NCDataset(old, "c") do dst
                # All dims fixed-length, including `time`: an unlimited dimension starts empty
                # in the copy, so a whole-array assignment would not match. The read path only
                # needs `time`'s length and last value, not its extensibility.
                for (d, l) in src.dim
                    NCDatasets.defDim(dst, d, l)
                end
                for (k, v) in src.attrib
                    dst.attrib[k] = v
                end
                for name in keys(src)
                    v = NCDatasets.defVar(dst, name, eltype(src[name].var),
                                          NCDatasets.dimnames(src[name]))
                    v.var[:] = src[name].var[:]
                end
                g_src = src.group["restart"]
                g_dst = NCDatasets.defGroup(dst, "restart")
                for name in keys(g_src)
                    name == "age" && continue          # the layer the old writer never stored
                    v = NCDatasets.defVar(g_dst, name, eltype(g_src[name].var),
                                          NCDatasets.dimnames(g_src[name]))
                    v.var[:] = g_src[name].var[:]
                end
            end
        end

        err = try
            read_glacier_cell_restart(old)
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("age", err.msg)
        @test occursin("cannot be continued", err.msg)
        @test occursin("Delete the file", err.msg)

        # A file written by the current writer reads back fine.
        @test read_glacier_cell_restart(path) !== nothing
    end

    @testset "NetCDF write and restart round-trip" begin
        time = DateTime(2000, 1, 31):Month(1):DateTime(2000, 6, 30)
        run = _fake_run(; time = collect(time), deltas = [0.0, 1.5],
                        scalings = [1.0], bins = [1150, 1250, 1350],
                        weights = [10.3, 8.4, 1.0],
                        lengths = [12, 9, 0])   # last bin produced no output

        dir = mktempdir()
        path = joinpath(dir, "cell.nc")
        write_glacier_cell_netcdf(path, run; institution = "JPL")
        @test isfile(path)

        NCDatasets.NCDataset(path, "r") do ds
            @test ds.attrib["Conventions"] == "CF-1.11"
            @test ds.attrib["institution"] == "JPL"
            @test haskey(ds.attrib, "history")
            # Non-string provenance values are encoded, `nothing` is omitted.
            @test ds.attrib["spinup_converged"] == "false"
            @test ds.attrib["climatology_window_start"] == "1950-01-01T00:00:00"
            @test !haskey(ds.attrib, "spinup_convergence_drift_density")

            @test size(ds["melt"]) == (length(time), 2, 1)
            @test ds["melt"].attrib["units"] == "kg"
            # The area weighting makes these masses, not per-area amounts, so the CF
            # amount-per-area standard names must not be carried through.
            @test !haskey(ds["melt"].attrib, "standard_name")
            @test ds["mass_change"].attrib["units"] == "kg"
            @test occursin("precipitation - runoff", ds["mass_change"].attrib["comment"])

            @test collect(ds["bin_weight"][:]) ≈ run.weights
            @test collect(ds["bin_center"][:]) == [1150.0, 1250.0, 1350.0]
            @test ds["latitude"][] ≈ 72.5
            @test collect(ds["melt"][:, :, :]) ≈ run.totals[:melt]

            # Time encodes and decodes losslessly.
            @test GEMB_GlacierSims._nc_decode_time(ds["time"][1]) == first(time)
            @test GEMB_GlacierSims._nc_decode_time(ds["time"][end]) == last(time)

            g = ds.group["restart"]
            @test collect(g["valid_layers"][:, 1, 1]) == Int32[12, 9, 0]
            # Ragged columns are padded to the layer dimension.
            @test size(g["dz"]) == (12, 3, 2, 1)
            @test ismissing(g["dz"][10, 2, 1, 1])
        end

        restart = read_glacier_cell_restart(path)
        @test restart.time == last(time)
        @test restart.bin_centers == [1150.0, 1250.0, 1350.0]
        @test restart.delta_temperatures == [0.0, 1.5]
        @test restart.precipitation_scalings == [1.0]

        # Every run with output round-trips; the empty bin has no saved state.
        @test length(restart.profiles) == 4
        @test !haskey(restart.profiles, (3, 1, 1))
        for key in [(1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1)]
            saved = restart.profiles[key]
            original = run.profiles[key...]
            @test keys(saved) == keys(original)
            @test length(saved[:dz]) == length(original[:dz])
            for v in GEMB_GlacierSims.PROFILE_VARIABLES
                # `dz` in particular must be bitwise exact: `gemb` pins the column geometry
                # from it and asserts the grid is feasible.
                @test collect(saved[v]) == collect(original[v])
            end
        end

        # The spinup a column descends from comes back on the profile's *stack* metadata, which is
        # where `gemb` reads provenance from. Without it a continuation would report
        # `spinup_performed => false` and no climatology window, and the appended file would claim
        # a record that was spun up never was.
        for key in [(1, 1, 1), (2, 2, 1)]
            pm = DimensionalData.metadata(restart.profiles[key])
            @test pm["spinup_cycles"] == 37
            @test pm["climatology_window_start"] == "1950-01-01T00:00:00"
            # Encoded form, matching the file — re-encoding it is idempotent, so the window a
            # record was spun up on reads identically however many times it is appended to.
            @test pm["spinup_converged"] == "false"
            # Only provenance is restored: the run parameters are checked separately and a
            # `latitude` on the profile would be a claim `gemb` does not make about a column.
            @test !haskey(pm, "hypsometry_coverage")
            @test !haskey(pm, "latitude")
        end

        # And it survives a second hop: provenance read off a file and re-attached to a profile is
        # what `gemb` copies onto the next output, which the appender then writes back.
        @test GEMB_GlacierSims._stack_provenance(first(values(restart.profiles)))["spinup_cycles"] == 37

        # The file-level provenance is returned alongside the profiles, because the climatology
        # window is needed exactly when a bin has *no* profile to read it from.
        @test restart.provenance["climatology_window_start"] == "1950-01-01T00:00:00"

        @test read_glacier_cell_restart(joinpath(dir, "absent.nc")) === nothing
    end

    @testset "spinup window inheritance" begin
        # A cold start derives the window from the record: the first 30 complete years.
        t = collect(DateTime(1960, 1, 1):Day(1):DateTime(1994, 12, 31))
        cold = DimStack((x = DimArray(zeros(length(t)), (Ti(t),)),))
        @test GEMB_GlacierSims._default_spinup_window(cold) ==
              (DateTime(1960, 1, 1), DateTime(1989, 12, 31))

        # A continuation inherits the window its record was spun up on, parsed back from the
        # strings NetCDF stored (it has no date attribute type). This is the whole point: a
        # restart does not re-spin up, so a fallback bin must use the file's climatology rather
        # than the continuation's own few years of fetched forcing.
        saved = (; provenance = Dict{String,Any}(
            "climatology_window_start" => "1960-01-01T00:00:00",
            "climatology_window_stop" => "1989-12-31T00:00:00"))
        @test GEMB_GlacierSims._restart_spinup_window(saved) ==
              (DateTime(1960, 1, 1), DateTime(1989, 12, 31))

        # Read from the restart, not from a saved profile — on a single-bin cell the fallback case
        # is also the case where `profiles` is empty, so a profile-sourced window would vanish
        # precisely when it is needed.
        @test GEMB_GlacierSims._restart_spinup_window(
            (; saved..., profiles = Dict{Tuple{Int,Int,Int},DimStack}())) ==
              (DateTime(1960, 1, 1), DateTime(1989, 12, 31))

        # A file predating provenance, or one whose window will not parse, yields `nothing` so the
        # caller derives a window instead of failing.
        @test GEMB_GlacierSims._restart_spinup_window((; provenance = Dict{String,Any}())) === nothing
        @test GEMB_GlacierSims._restart_spinup_window((; parameters = Dict{String,Any}())) === nothing
        @test GEMB_GlacierSims._restart_spinup_window((; provenance = Dict{String,Any}(
            "climatology_window_start" => "not a date",
            "climatology_window_stop" => "1989-12-31T00:00:00"))) === nothing
    end

    @testset "SpinupWindowUnavailable" begin
        # A window whose years were never fetched must say so. Silently averaging whatever the
        # window does select would spin one bin up on a different climate than the rest of the
        # record — the failure this exception exists to make visible.
        t = collect(DateTime(1995, 1, 1):Day(1):DateTime(1999, 12, 31))
        cf = DimStack((x = DimArray(zeros(length(t)), (Ti(t),)),))
        window = (DateTime(1960, 1, 1), DateTime(1989, 12, 31))
        err = try
            GEMB_GlacierSims._spinup_climatology(cf, window)
            nothing
        catch e
            e
        end
        @test err isa SpinupWindowUnavailable
        @test err.window == window
        @test err.n_selected == 0
        @test err.extent == (first(t), last(t))
        # The message names the window to fetch, since that is the remedy.
        @test occursin("1960-01-01T00:00:00", sprint(showerror, err))

        # A partial year is refused too, and this is why the test is a span rather than a count:
        # `forcing_climatology` calls whichever years hold the most steps "complete", so six
        # months of daily data would average into a 182-step "climatological year" without
        # complaint.
        partial = (DateTime(1995, 1, 1), DateTime(1995, 6, 30))
        @test_throws SpinupWindowUnavailable GEMB_GlacierSims._spinup_climatology(cf, partial)
    end

    @testset "NetCDF append" begin
        t1 = collect(DateTime(2000, 1, 31):Month(1):DateTime(2000, 6, 30))
        t2 = collect(DateTime(2000, 7, 31):Month(1):DateTime(2000, 12, 31))
        kw = (; deltas = [0.0, 1.5], scalings = [1.0], bins = [1150, 1250],
              weights = [10.3, 8.4], lengths = [12, 9])
        run1 = _fake_run(; time = t1, kw...)
        run2 = _fake_run(; time = t2, kw...)

        dir = mktempdir()
        path = joinpath(dir, "cell.nc")
        write_glacier_cell_netcdf(path, run1)
        before = NCDatasets.NCDataset(path, "r") do ds
            collect(ds["melt"][:, :, :])
        end

        append_glacier_cell_netcdf(path, run2)

        NCDatasets.NCDataset(path, "r") do ds
            @test length(ds["time"]) == length(t1) + length(t2)
            times = GEMB_GlacierSims._nc_decode_time.(collect(ds["time"][:]))
            @test times == vcat(t1, t2)
            @test issorted(times) && allunique(times)
            # The pre-existing slab is untouched.
            @test collect(ds["melt"][1:length(t1), :, :]) == before
            @test collect(ds["melt"][(length(t1)+1):end, :, :]) ≈ run2.totals[:melt]
        end

        # The restart state is overwritten, not appended, and now points at the new end time.
        r = read_glacier_cell_restart(path)
        @test r.time == last(t2)
        @test collect(r.profiles[(1, 1, 1)][:dz]) == collect(run2.profiles[1, 1, 1][:dz])

        # Appending an overlapping or non-advancing record is refused.
        @test_throws ArgumentError append_glacier_cell_netcdf(path, run2)
        @test_throws ArgumentError append_glacier_cell_netcdf(path, run1)

        # A run grid that disagrees with the file is refused.
        other = _fake_run(; time = collect(DateTime(2001, 1, 31):Month(1):DateTime(2001, 3, 31)),
                          deltas = [0.0, 2.5], scalings = [1.0], bins = [1150, 1250],
                          weights = [10.3, 8.4], lengths = [12, 9])
        @test_throws ArgumentError append_glacier_cell_netcdf(path, other)
    end

    @testset "threaded kwarg" begin
        # The threaded path is accepted and documented. Whether it *agrees* with the serial path
        # cannot be checked here (it needs real GEMB runs, so forcing, so network); that is what
        # `scripts/check_threaded_equivalence.jl` asserts, bit-for-bit, on a real cell.
        m = only(methods(gemb_glacier_cell))
        @test :threaded in Base.kwarg_decl(m)
    end

    @testset "monthly lapse rate reaches the forcing" begin
        # `gemb_glacier_cell` forwards `lapse_rate` to `forcing_at_elevation` untouched, so a
        # monthly cycle must lapse each step by its own month's rate. Checked on the forcing
        # rather than through a run — the latter needs real GEMB, so network — and against a
        # January-only fixture, where the cycle's first entry is the only one that applies.
        f = _cp_forcing(45.0, 7.0, 1000.0, fill(260.0, 6))
        cycle = collect(Float64, 1:12)
        raise(lr) = climate_adjust_for_elevation(f, 500.0; lapse_rate = lr)
        @test all(≈(260.0 - cycle[1] * 0.5), raise(cycle)[:temperature_air])

        # And the scalar path is unchanged: a constant cycle equals the scalar of that value.
        @test raise(fill(6.5, 12))[:temperature_air] == raise(6.5)[:temperature_air]

        # The signature no longer pins `lapse_rate` to a scalar, which is what let the monthly
        # form through in the first place.
        @test :lapse_rate in Base.kwarg_decl(only(methods(gemb_glacier_cell)))
    end

    @testset "grid_cells_in_region" begin
        GI = GeoDataFrames.GeoInterface

        # Three cells: one inside the square, one outside, one on its boundary.
        table = _cp_table([(lon = 10.0, lat = 46.0, z = 1000.0, glm = 0.2, bins = [(1050, 5.0)]),
                           (lon = 20.0, lat = 46.0, z = 1000.0, glm = 0.2, bins = [(1050, 5.0)]),
                           (lon = 11.0, lat = 46.0, z = 1000.0, glm = 0.2, bins = [(1050, 5.0)])])
        sel = grid_cells_in_region(table, _cp_region(9.5, 45.5, 11.5, 46.5))
        # A view, not a copy: the real table is ~47k rows x ~100 hyps columns.
        @test sel isa SubDataFrame
        @test sort(collect(sel.longitude)) == [10.0, 11.0]

        # `GO.contains` is strictly interior, so a cell centered exactly on the region's edge is
        # excluded. Worth pinning: it decides which side of a shared border a cell lands on when a
        # domain is tiled into adjacent regions, so no cell is derived twice.
        @test sort(collect(grid_cells_in_region(table,
                                                _cp_region(9.5, 45.5, 11.0, 46.5)).longitude)) ==
              [10.0]

        # No shape assumption: a MultiPolygon and a plain vector of polygons both work, and a cell
        # is kept if it falls in *any* of the parts.
        parts = [_cp_region(9.5, 45.5, 10.5, 46.5), _cp_region(19.5, 45.5, 20.5, 46.5)]
        @test sort(collect(grid_cells_in_region(table, parts).longitude)) == [10.0, 20.0]
        @test sort(collect(grid_cells_in_region(table, GI.MultiPolygon(parts)).longitude)) ==
              [10.0, 20.0]

        # A region table (what `GeoDataFrames.read` of a region file gives) is accepted too.
        @test nrow(grid_cells_in_region(table, DataFrame(geometry = parts))) == 2

        # The wrap trap: cell centers are stored in native 0-359.9°E, region polygons in
        # (-180, 180]. 350°E is -10°, and must be found by a polygon around -10.
        west = _cp_table([(lon = 350.0, lat = 46.0, z = 1000.0, glm = 0.2, bins = [(1050, 5.0)])])
        @test nrow(grid_cells_in_region(west, _cp_region(-10.5, 45.5, -9.5, 46.5))) == 1

        # `area_minimum` screens on total glacier area before the geometric test.
        @test nrow(grid_cells_in_region(table, _cp_region(9.5, 45.5, 11.5, 46.5);
                                        area_minimum = 6.0)) == 0

        # A non-geometry argument, and an antimeridian-spanning region, both throw rather than
        # silently selecting the wrong cells.
        @test_throws ArgumentError grid_cells_in_region(table, "not a polygon")
        @test_throws ArgumentError grid_cells_in_region(table, [nothing, nothing])
        @test_throws ArgumentError grid_cells_in_region(table, DataFrame(geometry = []))
        @test_throws ArgumentError grid_cells_in_region(table, _cp_region(-170.0, 45.0, 170.0, 46.0))
    end

    @testset "derive_downscaling_parameters" begin
        # Four cells spanning 1000-2500 m with a range of glacier fractions, one populated
        # elevation interval each, inside one square region.
        cells = [(lon = 10.0, lat = 46.0, z = 1000.0, glm = 0.0,  bins = [(1050, 10.0)]),
                 (lon = 10.1, lat = 46.0, z = 1500.0, glm = 0.5,  bins = [(1550, 10.0)]),
                 (lon = 10.2, lat = 46.0, z = 2000.0, glm = 1.0,  bins = [(2050, 10.0)]),
                 (lon = 10.3, lat = 46.0, z = 2500.0, glm = 0.25, bins = [(2550, 10.0)])]
        table = _cp_table(cells)
        region = _cp_region(9.5, 45.5, 10.5, 46.5)
        time_range = (_CP_TIME[1], _CP_TIME[end])

        k_true = 0.7
        gamma_true = 5.0        # K/km, ambient
        # Ambient temperature at sea level, high enough that every cell stays above melting: the
        # estimator assumes the ambient excess is linear in elevation, and it is only linear while
        # no cell clips at the melting point.
        t0 = 292.0
        ambient(c) = t0 - gamma_true * c.z / 1000.0
        # The decoupling ERA5-Land has *already* applied: a glm=1 cell it runs as ice is fully
        # decoupled, a glm=0 cell it runs as bare ground is left ambient.
        function observed(c)
            a = ambient(c)
            k_applied = 1 - c.glm * (1 - k_true)
            return fill(a + (k_applied - 1) * max(a - 273.15, 0.0), length(_CP_TIME))
        end

        p = derive_downscaling_parameters(:synthetic, time_range, table, region;
                                      token = nothing, cache_path = nothing,
                                      min_cells = 4,
                                      forcing_loader = _cp_loader(cells, observed),
                                      elevation_interval_batch = 2)

        @testset "decoupling factor" begin
            # The forward model is inverted exactly, so the fit recovers the truth to roundoff, at
            # every timestep — no aggregation involved.
            @test all(k -> isapprox(k, k_true, atol = 1e-9), p.decoupling.decoupling_factor)
            @test all(≈(1.0), filter(!isnan, p.decoupling.r2))
            @test all(p.decoupling.n_cells .== 4)
            # A `DimStack` on the shared time axis, which is what makes downstream grouping a
            # `groupby` rather than a keyword on this function.
            @test p.decoupling isa DimStack
            @test collect(dims(p.decoupling, Ti)) == p.time
            @test DimensionalData.metadata(p.decoupling)["reference_elevation"] ≈ 1750.0
            @test DimensionalData.metadata(p.decoupling)["n_fitted"] == length(_CP_TIME)
            # The four coefficients come back so `k` is re-evaluable at any elevation.
            @test all(isfinite, p.decoupling.coef_alpha)
            @test all(isfinite, p.decoupling.coef_delta)
            # `ambient_excess` is the diagnostic the fit's precision hinges on: the ice-free excess
            # at `z̄`, which here is `t0 - Γ·z̄/1000 - 273.15`.
            @test all(≈(t0 - gamma_true * 1.75 - 273.15), p.decoupling.ambient_excess)
        end

        @testset "lapse rate" begin
            # The ON-GLACIER lapse rate, which is not the ambient one. Above melting the decoupling
            # maps T -> 273.15 + k(T - 273.15), so it compresses the profile and the on-glacier
            # slope is k*Γ.
            @test all(g -> isapprox(g, k_true * gamma_true, atol = 1e-8), p.lapse_rate.lapse_rate)
            @test p.lapse_rate isa DimStack
            @test collect(dims(p.lapse_rate, Ti)) == p.time
            @test DimensionalData.metadata(p.lapse_rate)["n_fitted"] == length(_CP_TIME)
            # Elevation spread is the fit diagnostic: the population sd of 1000..2500 m.
            @test all(s -> isapprox(s, 559.0169943749474, atol = 1e-9),
                      p.lapse_rate.elevation_spread)
        end

        @testset "downstream grouping is a groupby" begin
            # The documented aggregation path, replacing the `monthly`/`scalar`/`quantiles` that used
            # to be precomputed here. The fixture is six hours of one January day, so the month
            # grouping is one group and the hour grouping is six.
            by_month = groupby(p.lapse_rate, Ti => month)
            @test length(by_month) == 1
            @test Statistics.median(filter(isfinite, only(by_month).lapse_rate)) ≈
                  k_true * gamma_true atol = 1e-8
            by_hour = groupby(p.decoupling, Ti => hour)
            @test length(by_hour) == length(_CP_TIME)
            @test all(g -> isapprox(Statistics.median(filter(isfinite, g.decoupling_factor)),
                                    k_true, atol = 1e-9), by_hour)
        end

        @testset "lapse rate uncertainty" begin
            # The three extra accumulator moments must describe the *same* regression `derive_lapse_rate`
            # reports, or the error belongs to a different slope than the one beside it. Checked against
            # a from-scratch regression on the reconstructed corrected temperatures — no sums involved.
            un = p.lapse_rate_uncertainty
            @test issetequal(keys(un), (:lapse_rate_stderr, :lapse_rate_r2, :residual_sd))
            @test all(isfinite, un.lapse_rate_stderr)
            @test all(>=(0), un.lapse_rate_stderr)          # an error has no direction
            @test all(x -> 0 <= x <= 1, un.lapse_rate_r2)

            zs = [c.z for c in cells]
            gs = [c.glm for c in cells]
            kk = collect(p.decoupling.decoupling_factor)[1]
            a, b = collect(p.decoupling.coef_alpha)[1], collect(p.decoupling.coef_beta)[1]
            usable = isfinite(kk) && 0 < kk <= 1
            raw = [observed(c)[1] for c in cells]
            corrected = usable ?
                [raw[i] + (kk - 1) * (1 - gs[i]) * (a + b * zs[i]) for i in eachindex(raw)] : copy(raw)
            Szz = sum((zs .- mean(zs)) .^ 2)
            SzT = sum((zs .- mean(zs)) .* (corrected .- mean(corrected)))
            STT = sum((corrected .- mean(corrected)) .^ 2)
            rss = STT - SzT^2 / Szz
            # This fixture is noiseless, so the corrected temperatures fall exactly on a line and the
            # reference residual is identically zero. A one-pass moment accumulator cannot reproduce
            # that exactly — `ΣT² - (ΣT)²/m` is a difference of nearly equal numbers — so the
            # assertion is against the accumulator's precision floor. Accumulating the second moment
            # about the melting point rather than about absolute zero is what keeps that floor at
            # ~1e-7 K instead of ~5e-6 K; the realistic case, where the residual is a real 0.1-1 K, is
            # checked below and agrees tightly.
            @test rss ≈ 0 atol = 1e-12
            @test collect(un.residual_sd)[1] ≈ 0 atol = 1e-6
            @test collect(un.lapse_rate_stderr)[1] ≈ 0 atol = 1e-6

            # The decomposition that makes this worth having: leverage the tile always has, times
            # scatter it has at this instant. `elevation_spread` is a tile constant, the scatter is not.
            @test collect(un.lapse_rate_stderr)[1] ≈
                  1000 * collect(un.residual_sd)[1] / sqrt(Szz) rtol = 1e-10
            @test length(unique(collect(p.lapse_rate.elevation_spread))) == 1

            @test p.provenance["n_lapse_rate_fitted"] == length(_CP_TIME)

            # The realistic case: cells that do *not* fall exactly on a line, which is what the
            # diagnostic exists to measure. Here the sums must reproduce a from-scratch regression
            # tightly, not just to a floor.
            scatter = [0.0, 0.4, -0.3, 0.25, -0.15, 0.05][1:length(cells)]
            noisy(c) = (i = findfirst(x -> x.lat == c.lat && x.lon == c.lon, cells);
                        observed(c) .+ scatter[i])
            pn = derive_downscaling_parameters(:synthetic, time_range, table, region;
                                               token = nothing, cache_path = nothing, min_cells = 4,
                                               forcing_loader = _cp_loader(cells, noisy))
            kn = collect(pn.decoupling.decoupling_factor)[1]
            an, bn = collect(pn.decoupling.coef_alpha)[1], collect(pn.decoupling.coef_beta)[1]
            rawn = [noisy(c)[1] for c in cells]
            corrn = (isfinite(kn) && 0 < kn <= 1) ?
                [rawn[i] + (kn - 1) * (1 - gs[i]) * (an + bn * zs[i]) for i in eachindex(rawn)] :
                copy(rawn)
            SzTn = sum((zs .- mean(zs)) .* (corrn .- mean(corrn)))
            STTn = sum((corrn .- mean(corrn)) .^ 2)
            rssn = STTn - SzTn^2 / Szz
            @test rssn > 1e-6                       # genuinely scattered, so the check has teeth
            @test collect(pn.lapse_rate_uncertainty.residual_sd)[1] ≈
                  sqrt(rssn / (length(zs) - 2)) rtol = 1e-6
            @test collect(pn.lapse_rate_uncertainty.lapse_rate_stderr)[1] ≈
                  1000 * sqrt(rssn / (length(zs) - 2)) / sqrt(Szz) rtol = 1e-6
            @test collect(pn.lapse_rate_uncertainty.lapse_rate_r2)[1] ≈
                  1 - rssn / STTn rtol = 1e-6
            # And the slope the error belongs to is still upstream's, unchanged by any of this.
            @test collect(pn.lapse_rate.lapse_rate)[1] ≈ -1000 * SzTn / Szz rtol = 1e-8
        end

        @testset "elevation interval forcing" begin
            intervals = collect(p.elevation_interval_forcing)
            @test length(intervals) == 4
            # No `length` on the iterator: an interval whose donor cells have no usable forcing cannot
            # be emitted, and which those are is only known once the forcing has been read.
            @test Base.IteratorSize(typeof(p.elevation_interval_forcing)) == Base.SizeUnknown()
            # Intervals ascend, and each is centered on its bin.
            @test [iv.center for iv in intervals] == [1050.0, 1550.0, 2050.0, 2550.0]
            @test [iv.lo for iv in intervals] == [1000, 1500, 2000, 2500]
            @test [iv.n_cells for iv in intervals] == [1, 1, 1, 1]

            # Area conservation: no glacier area is dropped or double-counted.
            @test sum(iv.area for iv in intervals) ≈
                  sum(glacier_area_total(r) for r in eachrow(p.grid_cells))

            for iv in intervals
                # `metadata["elevation"] == center` is the contract `gemb_glacier_cell` enforces:
                # the stack is already *at* the interval, so `forcing_at_elevation(f, 0)` is
                # identity.
                @test DimensionalData.metadata(iv.forcing)["elevation"] == iv.center
                @test forcing_is_complete(iv.forcing)
                @test issetequal(keys(iv.forcing), GEMB_GlacierSims._FORCING_VARIABLES)
                @test collect(dims(iv.forcing, Ti)) == p.time
                # Cell-specific metadata would be a lie on a regional average.
                @test !haskey(DimensionalData.metadata(iv.forcing), "latitude")
                @test DimensionalData.metadata(iv.forcing)["glacier_area"] == iv.area
                @test DimensionalData.metadata(iv.forcing)["elevation_interval_lower"] ==
                      Float64(iv.lo)
                @test DimensionalData.metadata(iv.forcing)["elevation_interval_upper"] ==
                      Float64(iv.hi)
                # Every timestep of this fixture is measured, so every applied value is a fit and
                # nothing was substituted — the source counts exist precisely so a substituted
                # interval is distinguishable from a measured one.
                m = DimensionalData.metadata(iv.forcing)
                n = length(_CP_TIME)
                @test m["glacier_decoupling_factor_n_fitted"] == n
                @test m["temperature_lapse_rate_n_fitted"] == n
                for source in (:held, :climatology, :prior, :ambient)
                    @test m["glacier_decoupling_factor_n_$source"] == 0
                    @test m["temperature_lapse_rate_n_$source"] == 0
                end
                # The fixture is warm at every timestep, so `k` mattered throughout and the
                # above-freezing counts equal the unrestricted ones.
                @test m["n_timesteps_above_freezing"] == n
                @test m["glacier_decoupling_factor_n_fitted_above_freezing"] == n
                @test m["glacier_decoupling_factor_n_fit_in_domain"] == n
                @test m["glacier_decoupling_factor_n_fit_held"] == 0
                # Each interval here sits 50 m above its own single contributing cell, so every one
                # reports the same small extrapolation. Measured against the cells that actually
                # feed the interval, not the region's highest, so the batching cannot change it.
                @test m["extrapolation_above_reanalysis"] == 50.0
            end

            # Interval temperature decreases with elevation.
            @test issorted([mean(iv.forcing[:temperature_air]) for iv in intervals], rev = true)

            # Each interval here draws on exactly one cell whose own elevation is 50 m below the
            # center, so its temperature is that cell's forcing lapsed 50 m and then given the
            # decoupling it still needs — a value that can be written out by hand.
            for (iv, c) in zip(intervals, cells)
                lapsed = observed(c)[1] - k_true * gamma_true * (iv.center - c.z) / 1000.0
                still_needed = 1 - (1 - k_true) * (1 - c.glm)
                expected = lapsed + (still_needed - 1) * max(lapsed - 273.15, 0.0)
                @test mean(iv.forcing[:temperature_air]) ≈ expected atol = 1e-8
                # Precipitation is area-averaged unadjusted: the downstream sweep solves for its
                # correction factor, which is the whole point of producing this forcing.
                @test all(≈(0.1), iv.forcing[:precipitation])
            end
        end

        @testset "laziness" begin
            # `elevation_interval_batch` trades passes over the warm cache against peak memory, and
            # must not change the answer. `0` means one pass holding every interval.
            p0 = derive_downscaling_parameters(:synthetic, time_range, table, region;
                                           token = nothing, cache_path = nothing,
                                           min_cells = 4,
                                           forcing_loader = _cp_loader(cells, observed),
                                           elevation_interval_batch = 0)
            b0 = collect(p0.elevation_interval_forcing)
            b2 = collect(p.elevation_interval_forcing)
            @test [x.center for x in b0] == [x.center for x in b2]
            @test [x.area for x in b0] == [x.area for x in b2]
            @test all(isequal(x.forcing[:temperature_air], y.forcing[:temperature_air])
                      for (x, y) in zip(b0, b2))

            # A counting loader proves the laziness is real rather than incidental.
            loads = Ref(0)
            counting = let inner = _cp_loader(cells, observed)
                (args...; kwargs...) -> (loads[] += 1; inner(args...; kwargs...))
            end
            pc = derive_downscaling_parameters(:synthetic, time_range, table, region;
                                           token = nothing, cache_path = nothing,
                                           min_cells = 4,
                                           forcing_loader = counting,
                                           elevation_interval_batch = 2)
            # The fit pass loads all 4 cells. No interval has been requested yet, so nothing more.
            @test loads[] == 4
            first(pc.elevation_interval_forcing)
            # Asking for one interval accumulates its whole batch, and no more than that. Each cell
            # here holds ice in exactly one interval, so a 2-interval batch loads only the 2 cells
            # that contribute to it — a cell with no area in the current batch is skipped *before*
            # its forcing is fetched, which is what keeps a region of mostly-irrelevant cells cheap.
            @test loads[] == 4 + 2
            # Iteration is stateless, so a fresh pass re-accumulates from the first batch rather
            # than resuming: two batches of two intervals, two contributing cells each.
            loads[] = 0
            collect(pc.elevation_interval_forcing)
            @test loads[] == 2 * 2
        end

        @testset "provenance" begin
            @test p.provenance["n_grid_cells_in_region"] == 4
            @test p.provenance["n_grid_cells_used"] == 4
            @test p.provenance["n_elevation_intervals"] == 4
            @test p.provenance["adjustment_order"] == "lapse_then_decouple"
            # How much of each fit is measurement rather than absence — the count, not a verdict.
            @test p.provenance["n_timesteps"] == length(_CP_TIME)
            @test p.provenance["n_decoupling_factor_fitted"] == length(_CP_TIME)
            @test p.provenance["n_lapse_rate_fitted"] == length(_CP_TIME)
            # Area-weighted mean of equal-area cells at 1000..2500 m.
            @test p.provenance["mean_elevation"] ≈ 1750.0
        end

        @testset "unusable cells" begin
            # ERA5-Land is land-only, so a cell center can land on water and come back all-NaN.
            # Those cells are skipped, not fatal — the sweep's `ForcingUnavailable` policy.
            wet(c) = c.z == 1500.0 ? fill(NaN, length(_CP_TIME)) : observed(c)
            pw = derive_downscaling_parameters(:synthetic, time_range, table, region;
                                           token = nothing, cache_path = nothing,
                                           min_cells = 3,
                                           forcing_loader = _cp_loader(cells, wet),
                                           elevation_interval_batch = 0)
            @test nrow(pw.grid_cells) == 3
            # It is the 1500 m cell that was dropped, not an arbitrary one.
            @test 10.1 ∉ pw.grid_cells.longitude
            @test pw.provenance["n_grid_cells_used"] == 3
            # The dropped cell's interval goes with it, so area still balances over what remains.
            @test sum(iv.area for iv in pw.elevation_interval_forcing) ≈
                  sum(glacier_area_total(r) for r in eachrow(pw.grid_cells))

            # Every cell unusable is an error, not an empty result — and a *typed* one, distinct from
            # the `ArgumentError` a bad argument raises. A tiled sweep hits this legitimately (a
            # region of island glaciers whose every centre is on water) and has to be able to record
            # the region as unrunnable rather than treat it as a bug, which matching on an error
            # message could not do safely.
            @test_throws RegionForcingUnavailable derive_downscaling_parameters(
                :synthetic, time_range, table, region; token = nothing, cache_path = nothing,
                min_cells = 4,
                forcing_loader = _cp_loader(cells, _ -> fill(NaN, length(_CP_TIME))))
            # The count of cells that were tried is carried, so a log line can say how big the
            # unrunnable region was without re-selecting it.
            e = try
                derive_downscaling_parameters(
                    :synthetic, time_range, table, region; token = nothing, cache_path = nothing,
                    min_cells = 4,
                    forcing_loader = _cp_loader(cells, _ -> fill(NaN, length(_CP_TIME))))
                nothing
            catch err
                err
            end
            @test e isa RegionForcingUnavailable && e.n_cells == 4
            @test occursin("land-only", sprint(showerror, e))

            # A region containing no cells at all, likewise.
            @test_throws ArgumentError derive_downscaling_parameters(
                :synthetic, time_range, table, _cp_region(0.0, 0.0, 1.0, 1.0);
                token = nothing, cache_path = nothing,
                min_cells = 4,
                forcing_loader = _cp_loader(cells, observed))
        end

        @testset "unmeasurable timesteps report NaN" begin
            # `NaN` is the whole contract of the fits: it says "the forcing carried no information
            # here", which is a different claim from any substituted number. Four parameters need
            # four cells, and the `glm x z` interaction needs spread in both regressors.
            thin = cells[1:3]
            pt = derive_downscaling_parameters(:synthetic, time_range, _cp_table(thin), region;
                                           token = nothing, cache_path = nothing,
                                           min_cells = 3,
                                           forcing_loader = _cp_loader(thin, observed))
            @test all(isnan, pt.decoupling.decoupling_factor)
            @test pt.provenance["n_decoupling_factor_fitted"] == 0
            # A lapse rate is still measurable — a two-parameter slope needs only two cells — but it
            # is no longer the on-glacier one: with `k` unmeasurable the fit treats it as the
            # identity, so no decoupling is undone and the slope is taken on the still-partly-damped
            # temperatures. That reads steeper than ambient here, because the damping grows with
            # `glm` and `glm` happens to rise with elevation across these three cells. Which is
            # exactly why the raw series reports `NaN` for `k` instead of a number: the lapse rate is
            # only interpretable alongside the `k` that was available when it was fitted.
            @test all(isfinite, pt.lapse_rate.lapse_rate)
            zs = [c.z for c in thin]
            raw = [observed(c)[1] for c in thin]
            expected = -1000.0 * (sum((zs .- mean(zs)) .* (raw .- mean(raw))) /
                                  sum((zs .- mean(zs)) .^ 2))
            @test all(g -> isapprox(g, expected, atol = 1e-8), pt.lapse_rate.lapse_rate)
            @test expected > gamma_true

            # A region only marginally above melting cannot measure `k` either, and this is the case
            # real forcing actually produces: `k` is a *ratio* of fitted excesses, so a small
            # denominator fits noise. Found on real Alpine winter forcing, where a 0.1 K excess
            # "fitted" k = 0.2. A ~0.2 K excess is below `_MIN_AMBIENT_EXCESS`.
            marginal(c) = fill(273.35 - 0.05 * (c.z - 1000.0) / 1000.0, length(_CP_TIME))
            pm = derive_downscaling_parameters(:synthetic, time_range, table, region;
                                           token = nothing, cache_path = nothing,
                                           min_cells = 4,
                                           forcing_loader = _cp_loader(cells, marginal))
            @test all(isnan, pm.decoupling.decoupling_factor)
            @test all(isnan, pm.decoupling.ambient_excess)
            # The genuine melt-season case is well above the threshold and still fits.
            @test all(isfinite, p.decoupling.decoupling_factor)

            # All cells below melting: the warm excess is identically zero, so it carries no
            # information about `k` at all. The lapse rate is still measurable there, and with no
            # decoupling to undo it is simply the ambient one.
            frozen(c) = fill(250.0 - gamma_true * c.z / 1000.0, length(_CP_TIME))
            pf = derive_downscaling_parameters(:synthetic, time_range, table, region;
                                           token = nothing, cache_path = nothing,
                                           min_cells = 4,
                                           forcing_loader = _cp_loader(cells, frozen))
            @test all(isnan, pf.decoupling.decoupling_factor)
            @test all(g -> isapprox(g, gamma_true, atol = 1e-8), pf.lapse_rate.lapse_rate)

            # Every cell at one elevation: no slope is measurable however many cells there are.
            flat = [(; c..., z = 1500.0) for c in cells]
            pl = derive_downscaling_parameters(:synthetic, time_range, _cp_table(flat), region;
                                           token = nothing, cache_path = nothing,
                                           min_cells = 4,
                                           forcing_loader = _cp_loader(flat, observed))
            @test all(isnan, pl.lapse_rate.lapse_rate)
            @test all(isnan, pl.lapse_rate.elevation_spread)
            @test pl.provenance["n_lapse_rate_fitted"] == 0

            # The interval forcing still builds, because `resolve_downscaling` substitutes — and it
            # says which tier it reached rather than substituting silently. Without a substitution a
            # single `NaN` would make the interval fail `forcing_is_complete` and a sweep would skip
            # it like an ocean cell. With no accepted fit anywhere there is no climatology either, so
            # both series fall to their last tier: the lapse rate to its prior, and `k` — which has no
            # prior here — to the identity.
            for iv in pl.elevation_interval_forcing
                @test forcing_is_complete(iv.forcing)
                m = DimensionalData.metadata(iv.forcing)
                @test m["temperature_lapse_rate_n_prior"] == length(_CP_TIME)
                @test m["temperature_lapse_rate_n_fitted"] == 0
                @test m["temperature_lapse_rate_n_climatology"] == 0
                @test m["temperature_lapse_rate"] ≈ 6.5
                @test m["glacier_decoupling_factor_n_ambient"] == length(_CP_TIME)
                @test m["glacier_decoupling_factor_mean"] == 1.0   # the bit-exact no-op
            end

            # Priors are honoured, which is the point of the keywords: a region with no usable fit can
            # be given a published or regional value instead of the identity, and the label then says
            # the value is external evidence rather than this region's forcing.
            pprior = derive_downscaling_parameters(:synthetic, time_range, _cp_table(flat), region;
                                              token = nothing, cache_path = nothing,
                                              min_cells = 4,
                                              forcing_loader = _cp_loader(flat, observed),
                                              lapse_rate_prior = 7.25,
                                              decoupling_factor_prior = 0.85)
            for iv in pprior.elevation_interval_forcing
                m = DimensionalData.metadata(iv.forcing)
                @test m["temperature_lapse_rate"] ≈ 7.25
                @test m["glacier_decoupling_factor_mean"] ≈ 0.85
                @test m["glacier_decoupling_factor_n_prior"] == length(_CP_TIME)
                @test m["glacier_decoupling_factor_n_ambient"] == 0
            end
            # A monthly prior cycle is accepted too, which is the form the published regional lapse
            # rates take.
            pmonthly = derive_downscaling_parameters(:synthetic, time_range, _cp_table(flat), region;
                                              token = nothing, cache_path = nothing,
                                              min_cells = 4,
                                              forcing_loader = _cp_loader(flat, observed),
                                              lapse_rate_prior = collect(1.0:12.0))
            # `_CP_TIME` is entirely in January, so the cycle's first entry is the one applied.
            @test DimensionalData.metadata(
                first(pmonthly.elevation_interval_forcing).forcing)["temperature_lapse_rate"] ≈ 1.0

            # ...and validated, since they are applied: an out-of-domain prior would be rejected
            # downstream no differently from an out-of-domain fit, but with nothing to inspect.
            @test_throws ArgumentError derive_downscaling_parameters(
                :synthetic, time_range, table, region; token = nothing, cache_path = nothing,
                min_cells = 4,
                forcing_loader = _cp_loader(cells, observed), lapse_rate_prior = 99.0)
            @test_throws ArgumentError derive_downscaling_parameters(
                :synthetic, time_range, table, region; token = nothing, cache_path = nothing,
                min_cells = 4,
                forcing_loader = _cp_loader(cells, observed), decoupling_factor_prior = 1.5)
            @test_throws ArgumentError derive_downscaling_parameters(
                :synthetic, time_range, table, region; token = nothing, cache_path = nothing,
                min_cells = 4,
                forcing_loader = _cp_loader(cells, observed), lapse_rate_prior = [1.0, 2.0])
        end

        @testset "the fits report raw; the application resolves" begin
            # Here the glaciated cells are *warmer* than ambient, so the fit exceeds 1 — a value
            # `climate_adjust_for_glacier` would reject. The fit reports it anyway: that number is
            # evidence about the fit, and clamping it to 1.0 in the report would silently convert a
            # diagnostic into a plausible-looking measurement.
            function inverted(c)
                a = ambient(c)
                return fill(a + c.glm * 0.5 * max(a - 273.15, 0.0), length(_CP_TIME))
            end
            pi_ = derive_downscaling_parameters(
                :synthetic, time_range, table, region; token = nothing, cache_path = nothing,
                min_cells = 4,
                forcing_loader = _cp_loader(cells, inverted))
            @test all(>(1.0), pi_.decoupling.decoupling_factor)

            # ...but the interval forcing, which *applies* it, is in domain and finite. An
            # out-of-domain fit is **excluded** rather than clamped: dragging 1.5 to the nearest bound
            # would report the identity as though it had been measured. With every fit out of domain
            # there is nothing left to build a climatology from either, so the series resolves to its
            # last tier and the label says so. This is the assertion that policy moved rather than
            # vanished.
            for iv in pi_.elevation_interval_forcing
                @test forcing_is_complete(iv.forcing)
                @test all(k -> 0 < k <= 1, iv.decoupling_factor)
                m = DimensionalData.metadata(iv.forcing)
                @test m["glacier_decoupling_factor_n_ambient"] == length(_CP_TIME)
                @test m["glacier_decoupling_factor_n_fitted"] == 0
                # No fit at this elevation was in domain, which is the count that says the tier was
                # reached because the fits failed rather than because they were absent.
                @test m["glacier_decoupling_factor_n_fit_in_domain"] == 0
            end

            # A prior displaces the identity where the fits are unusable, and is labelled as external
            # evidence rather than as a measurement.
            pip = derive_downscaling_parameters(
                :synthetic, time_range, table, region; token = nothing, cache_path = nothing,
                min_cells = 4,
                forcing_loader = _cp_loader(cells, inverted),
                decoupling_factor_prior = 0.7)
            iv1 = first(pip.elevation_interval_forcing)
            @test all(≈(0.7), iv1.decoupling_factor)
            @test DimensionalData.metadata(
                iv1.forcing)["glacier_decoupling_factor_n_prior"] == length(_CP_TIME)

            # The clamp keyword is gone, not silently ignored: there is nothing left to opt out of.
            @test_throws MethodError derive_downscaling_parameters(
                :synthetic, time_range, table, region; token = nothing, cache_path = nothing,
                min_cells = 4,
                forcing_loader = _cp_loader(cells, inverted),
                clamp_to_valid_domain = false)

            # The lapse fit *consumes* `k`, and an out-of-domain one corrupts it: the correction is
            # proportional to `k - 1` and weighted by `(1 - glm)`, so a `k` of 1.5 scales every cell
            # by an arbitrary amount. Because `glm` covaries with elevation, that lands as a smooth
            # false gradient rather than as scatter, so it enters the slope in full and no estimator —
            # robust or otherwise — can detect it from the slope series. On real Wrangell forcing this
            # took a 7.0 K/km slope to 46.1 K/km and broke the downstream longwave ceiling.
            #
            # So the correction is applied only where `k` lands inside `(0, 1]`. Here it does not, so
            # the slope must be the *ambient* one — bit-exactly what the uncorrected temperatures
            # give, which for this fixture is `gamma_true` scaled by the fixture's own warming.
            @test all(isfinite, pi_.lapse_rate.lapse_rate)
            zs = [c.z for c in cells]
            raw = [inverted(c)[1] for c in cells]
            ambient_slope = -1000.0 * (sum((zs .- mean(zs)) .* (raw .- mean(raw))) /
                                       sum((zs .- mean(zs)) .^ 2))
            @test all(g -> isapprox(g, ambient_slope, atol = 1e-9),
                      pi_.lapse_rate.lapse_rate)
            # And the raw `k` is still reported out of domain — the screen guards the lapse fit, it
            # does not sanitize the decoupling series.
            @test all(>(1.0), pi_.decoupling.decoupling_factor)

            # Where `k` is valid the correction *is* applied, so the two paths are distinguishable
            # rather than the screen being a no-op: the `observed` fixture fits `k` in domain and its
            # slope is the on-glacier one, which differs from its own ambient slope.
            raw_obs = [observed(c)[1] for c in cells]
            ambient_obs = -1000.0 * (sum((zs .- mean(zs)) .* (raw_obs .- mean(raw_obs))) /
                                     sum((zs .- mean(zs)) .^ 2))
            @test all(k -> 0 < k <= 1, p.decoupling.decoupling_factor)
            @test !isapprox(first(p.lapse_rate.lapse_rate), ambient_obs, atol = 1e-3)
        end

        # The two sub-testsets that used to sit here — the per-timestep spread being reported rather
        # than collapsed to a constant, and `decoupling_factor_at_elevation`'s re-evaluation and
        # hold-out guard — test the fits themselves, which now live in GEMB_ClimateForcing. They
        # moved with them, to that package's test/test_downscaling_parameters.jl, where they run
        # against a hand-built accumulator instead of a synthetic elevation-class table. What stays
        # below is what this package still owns: orchestration, interval aggregation, laziness,
        # provenance, and argument validation.

        @testset "argument validation" begin
            @test_throws ArgumentError derive_downscaling_parameters(
                :synthetic, time_range, table, region; token = nothing, cache_path = nothing,
                forcing_loader = _cp_loader(cells, observed), elevation_interval_batch = -1)
            @test_throws ArgumentError derive_downscaling_parameters(
                :synthetic, time_range, table, region; token = nothing, cache_path = nothing,
                forcing_loader = _cp_loader(cells, observed), min_cells = 1)

            # The keywords that encoded policy in the fits are gone, not silently ignored.
            @test_throws MethodError derive_downscaling_parameters(
                :synthetic, time_range, table, region; token = nothing, cache_path = nothing,
                min_cells = 4,
                forcing_loader = _cp_loader(cells, observed),
                decoupling_factor_bounds = (0.2, 1.0))

            # The `:glm` column is what the decoupling fit regresses against, so a table built
            # before the current invariant column naming must fail loudly.
            stale = select(table, Not(:glm))
            stale.glm_frac = [c.glm for c in cells]
            @test_throws ArgumentError derive_downscaling_parameters(
                :synthetic, time_range, stale, region; token = nothing, cache_path = nothing,
                min_cells = 4,
                forcing_loader = _cp_loader(cells, observed))
        end
    end

    @testset "downscaling_tiles" begin
        # Four cells in one 2° tile, one in the next tile east, one far away.
        cells = [(lon = 10.0, lat = 46.0, z = 1000.0, glm = 0.0, bins = [(1050, 10.0)]),
                 (lon = 10.1, lat = 46.0, z = 1500.0, glm = 0.5, bins = [(1550, 10.0)]),
                 (lon = 11.9, lat = 46.0, z = 2000.0, glm = 1.0, bins = [(2050, 10.0)]),
                 (lon = 12.1, lat = 46.0, z = 2500.0, glm = 0.25, bins = [(2550, 10.0)]),
                 (lon = 40.0, lat = 20.0, z = 1000.0, glm = 0.1, bins = [(1050, 3.0)])]
        table = _cp_table(cells)

        tiles = downscaling_tiles(table; tile_size = 2, buffer = 1)
        by_index = Dict(t.index => t for t in tiles)
        @test sort(collect(keys(by_index))) == [(10, 46), (12, 46), (40, 20)]

        # The cores partition the table: every row in exactly one, none in two. This is the property
        # the arithmetic assignment exists for, and the reason the tiling does not go through
        # `grid_cells_in_region` — see the `_cp_region` boundary test above, where a cell centered on
        # a region edge is (correctly, for that function) excluded from both sides.
        core_rows = vcat([parentindices(t.core)[1] for t in tiles]...)
        @test length(core_rows) == nrow(table)
        @test allunique(core_rows)
        @test sort(core_rows) == collect(1:nrow(table))

        # A cell exactly on a tile edge lands in the tile it is the *lower* edge of, not in neither.
        @test tile_index(10.0, 46.0) == (10, 46)
        @test tile_index(12.0, 46.0) == (12, 46)

        # Buffered is a superset of core and reaches across the tile boundary: the 11.9 cell is 1°
        # from the 12-tile's western edge, so it fits that tile too.
        for t in tiles
            @test issubset(Set(parentindices(t.core)[1]), Set(parentindices(t.buffered)[1]))
        end
        # The 12-tile spans 12-14°, so a 1° buffer reaches back to 11.0: it picks up the 11.9 cell
        # from its western neighbour but not the 10.1 one, which is 2.9° from its centre.
        @test sort(collect(by_index[(12, 46)].buffered.longitude)) == [11.9, 12.1]
        @test sort(collect(by_index[(10, 46)].buffered.longitude)) == [10.0, 10.1, 11.9, 12.1]
        # ...and the far cell's tile sees only itself, buffer or no buffer.
        @test nrow(by_index[(40, 20)].buffered) == 1

        # Views, not copies: the real table is ~47k rows x ~100 hyps columns per tile selection.
        @test all(t -> t.core isa SubDataFrame && t.buffered isa SubDataFrame, tiles)
        # Row order preserved inside a selection, so a chunk_id-sorted table stays sorted and a
        # streaming pass over a tile keeps its Zarr cache locality.
        @test all(t -> issorted(parentindices(t.buffered)[1]), tiles)

        # The wrap trap, as for `grid_cells_in_region`: centers are stored native 0-359.9°E and the
        # tiling is on (-180, 180], so 350°E is -10° and belongs to the tile at -10.
        west = _cp_table([(lon = 350.0, lat = 46.0, z = 1000.0, glm = 0.2, bins = [(1050, 5.0)])])
        @test only(downscaling_tiles(west; tile_size = 2, buffer = 1)).index == (-10, 46)

        # `area_minimum` screens before tiling, so a screened-out cell is in no tile at all — which
        # keeps the partition a partition of the *qualifying* rows.
        @test sum(nrow(t.core) for t in downscaling_tiles(table; area_minimum = 6.0)) == 4
        @test isempty(downscaling_tiles(table; area_minimum = 1e6))

        # The table must carry the columns the forcing loader is keyed on, and say so here rather
        # than failing tens of thousands of cell loads into a sweep.
        @test_throws ArgumentError downscaling_tiles(select(table, Not(:latitude)))
        @test_throws ArgumentError downscaling_tiles(table; order = :nonsense)
        @test_throws ArgumentError downscaling_tiles(table; buffer = -1)

        @testset "antimeridian" begin
            # The case that rules out a polygon per tile: a tile against the seam has a buffer that
            # crosses it, and a region spanning ±180 is exactly what `grid_cells_in_region` refuses.
            am = _cp_table([(lon = 179.5, lat = 60.0, z = 1000.0, glm = 0.2, bins = [(1050, 5.0)]),
                            (lon = 180.5, lat = 60.0, z = 1200.0, glm = 0.2, bins = [(1250, 5.0)]),
                            (lon = 179.9, lat = 60.0, z = 1400.0, glm = 0.2, bins = [(1450, 5.0)])])
            tiles = downscaling_tiles(am; tile_size = 2, buffer = 1)
            by = Dict(t.index => t for t in tiles)
            # 180.5°E wraps to -179.5, which is west of the seam; the other two are east of it. So
            # the three cells split across the two tiles that share the antimeridian.
            @test sort(collect(keys(by))) == [(-180, 60), (178, 60)]
            @test nrow(by[(-180, 60)].core) == 1
            @test nrow(by[(178, 60)].core) == 2
            # Each tile's buffer reaches across the seam and picks up the other's cells. Without the
            # modular longitude distance these would be ~359° apart and neither would see the other.
            @test nrow(by[(-180, 60)].buffered) == 3
            @test nrow(by[(178, 60)].buffered) == 3

            # No tile index at +180: tile edges land exactly on ±180 (tile_size divides 360), and a
            # cell exactly on the seam folds onto the -180 tile rather than creating a duplicate
            # column of tiles there.
            @test all(t -> t.index[1] != 180, tiles)
            @test tile_index(180.0, 60.0) == (-180, 60)
            @test tile_index(-179.9, 60.0) == (-180, 60)
            @test tile_index(179.9, 60.0) == (178, 60)

            # The recorded buffered bounds are the honest span, not a wrapped one — a reader must be
            # able to see that this window crosses the seam.
            @test by[(-180, 60)].buffered_bounds.lon_min == -181.0
            @test by[(178, 60)].buffered_bounds.lon_max == 181.0

            # And it is not special-cased to one tile: at a coarse tile size several tiles have
            # seam-crossing buffers, and the partition still holds.
            coarse = downscaling_tiles(am; tile_size = 10, buffer = 1)
            @test sum(nrow(t.core) for t in coarse) == nrow(am)
        end

        @testset "grid options" begin
            spread = _cp_table([(lon = 10.0 + 0.5i, lat = 46.0 + 0.5j, z = 1000.0 + 100i,
                                 glm = 0.1, bins = [(1050 + 100i, 4.0)])
                                for i in 0:5, j in 0:5] |> vec)
            for tile_size in (1, 2, 5), buffer in (0, 1, 2)
                tiles = downscaling_tiles(spread; tile_size, buffer)
                rows = vcat([parentindices(t.core)[1] for t in tiles]...)
                # Total and disjoint at every grid setting, which is what makes the setting safe to
                # change: no combination silently drops or duplicates a cell.
                @test sort(rows) == collect(1:nrow(spread))
                @test all(t -> issubset(Set(parentindices(t.core)[1]),
                                        Set(parentindices(t.buffered)[1])), tiles)
                # A zero buffer is legal and degenerate: the fits then see the tile alone.
                buffer == 0 &&
                    @test all(t -> parentindices(t.core)[1] == parentindices(t.buffered)[1], tiles)
            end
            # Finer tiles mean more of them; coarser mean fewer.
            @test length(downscaling_tiles(spread; tile_size = 1)) >
                  length(downscaling_tiles(spread; tile_size = 2)) >
                  length(downscaling_tiles(spread; tile_size = 5))

            # A tile size that does not divide 360 would make the column of tiles against +180
            # narrower than every other while still reporting this `tile_size`; a fractional one
            # would let two tiles collide on one file name. Both refuse.
            @test_throws ArgumentError downscaling_tiles(spread; tile_size = 7)
            @test_throws ArgumentError downscaling_tiles(spread; tile_size = 2.5)
            @test_throws ArgumentError downscaling_tiles(spread; tile_size = 0)
            # An integer-valued float is the same grid, so it is accepted.
            @test length(downscaling_tiles(spread; tile_size = 2.0)) ==
                  length(downscaling_tiles(spread; tile_size = 2))
        end
    end

    @testset "is_caller_error" begin
        # The predicate every per-cell `catch` in the package rethrows on. Pinned because it is the
        # thing standing between one code bug and thousands of warnings that read like bad data.
        @test GEMB_GlacierSims.is_caller_error(MethodError(sin, (1,)))
        @test GEMB_GlacierSims.is_caller_error(UndefVarError(:nope))
        # Not caller errors: these are the network and data failures the per-cell skip exists for.
        @test !GEMB_GlacierSims.is_caller_error(ErrorException("CDS timeout"))
        @test !GEMB_GlacierSims.is_caller_error(ArgumentError("bad argument"))

        # `FieldError` is Julia 1.12+, and this package supports 1.11 — naming it unconditionally is
        # itself an `UndefVarError` on 1.11, i.e. the handler becomes the bug it exists to catch.
        # (Caught by CI on 1.11 after passing locally on 1.12.) So the type list is built from what
        # the running version actually defines.
        @test (:FieldError in nameof.(GEMB_GlacierSims._CALLER_ERROR_TYPES)) ==
              isdefined(Base, :FieldError)
        @test all(T -> T <: Exception, GEMB_GlacierSims._CALLER_ERROR_TYPES)
    end

    @testset "output file naming" begin
        # Round-trips, including the extremes of the padding.
        for index in ((-142, 60), (-180, -80), (178, 82), (0, 0), (10, -2))
            @test parse_tile_index(tile_output_name(index)) == index
        end
        @test tile_output_name((-142, 60)) == "N60_W142.nc"
        @test tile_output_name((178, -80)) == "S80_E178.nc"

        @test cell_output_name(52.3, -174.1) == "N52.3_W174.1.nc"
        rt = parse_cell_lonlat(cell_output_name(52.3, -174.1))
        @test rt.latitude == 52.3 && rt.longitude == -174.1
        # Native longitudes name the same file as their wrapped equivalents.
        @test cell_output_name(46.0, 350.0) == cell_output_name(46.0, -10.0)
        # Rounding carries into the whole degrees rather than producing "52.10".
        @test cell_output_name(52.97, 0.0) == "N53.0_E000.0.nc"

        # The two name forms cannot be confused, which is what lets either be parsed without knowing
        # which produced it: a cell name always carries its 0.1° decimal, a tile name never does.
        @test parse_tile_index(cell_output_name(52.3, -174.1)) === nothing
        @test parse_cell_lonlat(tile_output_name((-142, 60))) === nothing
        @test parse_cell_lonlat("not-ours.nc") === nothing
        @test parse_tile_index("not-ours.nc") === nothing
    end

    @testset "downscaling tile NetCDF" begin
        # The same four synthetic cells the fits are validated against above, so the file's contents
        # can be checked against a fit whose truth is known exactly.
        cells = [(lon = 10.0, lat = 46.0, z = 1000.0, glm = 0.0, bins = [(1050, 10.0)]),
                 (lon = 10.1, lat = 46.0, z = 1500.0, glm = 0.5, bins = [(1550, 10.0)]),
                 (lon = 10.2, lat = 46.0, z = 2000.0, glm = 1.0, bins = [(2050, 10.0)]),
                 (lon = 10.3, lat = 46.0, z = 2500.0, glm = 0.25, bins = [(2550, 10.0)])]
        table = _cp_table(cells)
        time_range = (_CP_TIME[1], _CP_TIME[end])
        k_true = 0.7
        ambient(c) = 292.0 - 5.0 * c.z / 1000.0
        function observed(c)
            a = ambient(c)
            k_applied = 1 - c.glm * (1 - k_true)
            return fill(a + (k_applied - 1) * max(a - 273.15, 0.0), length(_CP_TIME))
        end

        tile = only(downscaling_tiles(table; tile_size = 2, buffer = 1))
        p = derive_downscaling_parameters(:synthetic, time_range, tile.buffered;
                                         token = nothing, cache_path = nothing, min_cells = 4,
                                         forcing_loader = _cp_loader(cells, observed),
                                         elevation_interval_batch = 0)

        dir = mktempdir()
        path = joinpath(dir, tile.name)
        write_downscaling_tile_netcdf(path, tile, p; climate_model = :synthetic, time_range,
                                      tile_size = 2, buffer = 1, min_cells = 4)
        r = read_downscaling_tile(path)

        @test r.time == p.time
        @test collect(r.lapse_rate.lapse_rate) ≈ collect(p.lapse_rate.lapse_rate) atol = 1e-3
        @test collect(r.decoupling.decoupling_factor) ≈
              collect(p.decoupling.decoupling_factor) atol = 1e-6

        # THE reason the coefficients are stored, and stored in Float64: `k` is a function of
        # elevation (the fit carries a glm x z interaction), so a file that stored only the series
        # would pin it to the reference elevation — while the glacier sits mostly above that. Storing
        # the coefficients has to make `k(z)` reproducible *exactly*, including far above where it was
        # fitted, which is where a Float32 beta would visibly drift.
        for z in (1000.0, 1750.0, 5000.0, 8000.0)
            from_file = collect(decoupling_factor_at_elevation(r.decoupling, z))
            in_memory = collect(decoupling_factor_at_elevation(p.decoupling, z))
            @test all(((a, b),) -> (isnan(a) && isnan(b)) || a == b,
                      zip(from_file, in_memory))
        end
        @test all(c -> collect(r.decoupling[c]) == collect(p.decoupling[c]),
                  (:coef_alpha, :coef_beta, :coef_gamma, :coef_delta))

        # The joint screen, as a variable rather than a comment a reader may skip.
        @test r.fit_usable ==
              [isfinite(k) && 0 < k <= 1 for k in collect(p.decoupling.decoupling_factor)]

        # Two cell sets, answering two different questions: what the parameters apply to (the tile's
        # core, its share of the global partition) and what they were fitted from (the buffered
        # selection that had usable forcing).
        @test nrow(r.cells) == nrow(tile.core)
        @test sort(r.cells.longitude) == sort([wrap_lon(c.lon) for c in cells])
        @test nrow(r.fit_cells) == nrow(p.grid_cells)
        @test r.fit_cells.elevation == collect(p.cell_elevations)
        @test r.fit_cells.glacier_area == collect(p.cell_areas)

        # What a consumer needs to reproduce `k(z)` without this package.
        for key in ("min_ambient_excess", "decoupling_reference_temperature",
                    "decoupling_factor_formula", "reference_elevation", "tile_size", "tile_buffer",
                    "tile_assignment", "longitude_convention", "n_cells_core", "n_cells_used",
                    "requested_time_coverage_start", "downscaling_parameters")
            @test haskey(r.attributes, key)
        end
        @test r.attributes["tile_size"] == 2.0
        @test r.attributes["tile_buffer"] == 1.0
        @test r.attributes["sparse"] == "false"

        status = read_downscaling_tile_status(path)
        @test status.sparse == false
        @test status.n_timesteps == length(_CP_TIME)
        @test status.n_cells_core == nrow(tile.core)
        @test status.time_first == first(_CP_TIME) && status.time_last == last(_CP_TIME)
        @test read_downscaling_tile_status(joinpath(dir, "N00_E000.nc")) === nothing

        @testset "NaN survives Float32 and deflate" begin
            # An unmeasurable timestep must read back as unmeasurable. Every cell at one elevation
            # leaves the slope unidentifiable, so the lapse fit is NaN throughout — and a fill-value
            # misconfiguration would turn that into a number, which is the one thing a diagnostic
            # must not do.
            flat = [(lon = c.lon, lat = c.lat, z = 1500.0, glm = c.glm, bins = c.bins)
                    for c in cells]
            ft = _cp_table(flat)
            ftile = only(downscaling_tiles(ft; tile_size = 2, buffer = 1))
            pf = derive_downscaling_parameters(:synthetic, time_range, ftile.buffered;
                                              token = nothing, cache_path = nothing, min_cells = 4,
                                              forcing_loader = _cp_loader(flat, observed),
                                              elevation_interval_batch = 0)
            fpath = joinpath(dir, "flat.nc")
            write_downscaling_tile_netcdf(fpath, ftile, pf; climate_model = :synthetic, time_range,
                                          tile_size = 2, buffer = 1, min_cells = 4)
            rf = read_downscaling_tile(fpath)
            @test all(isnan, collect(pf.lapse_rate.lapse_rate))
            @test all(isnan, collect(rf.lapse_rate.lapse_rate))
            @test !any(rf.fit_usable)
        end

        @testset "elevation interval forcing" begin
            ivs = collect(p.elevation_interval_forcing)
            ipath = joinpath(dir, "intervals.nc")
            write_downscaling_tile_netcdf(ipath, tile, p; climate_model = :synthetic, time_range,
                                          tile_size = 2, buffer = 1, min_cells = 4,
                                          elevation_intervals = ivs)
            ri = read_downscaling_tile(ipath)

            @test ri.intervals !== nothing
            @test length(ri.intervals) == length(ivs)
            @test [x.center for x in ri.intervals] == [x.center for x in ivs]
            # No ice is dropped between the cells and the intervals.
            @test sum(x.area for x in ri.intervals) ≈
                  sum(glacier_area_total(row) for row in eachrow(p.grid_cells))
            # Physics that has to survive the round-trip: higher intervals are colder, and the
            # applied factor is inside the domain its consumer validates against.
            @test issorted([mean(x.forcing[:temperature_air]) for x in ri.intervals], rev = true)
            @test all(x -> all(v -> 0 < v <= 1, x.decoupling_factor), ri.intervals)
            for j in eachindex(ivs)
                @test collect(ri.intervals[j].forcing[:temperature_air]) ≈
                      collect(ivs[j].forcing[:temperature_air]) atol = 1e-2
            end
            # Recorded as a stored setting, so a sweep asked for intervals does not reuse a
            # fits-only file (and vice versa).
            @test read_downscaling_tile_status(ipath).parameters["elevation_interval_forcing"] ==
                  "true"
            @test read_downscaling_tile_status(path).parameters["elevation_interval_forcing"] ==
                  "false"

            # The provenance counts survive the round trip, which is what makes a stored interval
            # judgeable without re-deriving it. The per-timestep source labels they summarize are not
            # stored, so the counts are the whole record.
            for j in eachindex(ivs)
                stored = DimensionalData.metadata(ri.intervals[j].forcing)
                live = DimensionalData.metadata(ivs[j].forcing)
                for key in ("glacier_decoupling_factor_n_fit_held",
                            "glacier_decoupling_factor_n_fit_in_domain",
                            "n_timesteps_above_freezing")
                    @test stored[key] == live[key]
                end
                for source in DOWNSCALING_SOURCES
                    @test stored["glacier_decoupling_factor_n_$source"] ==
                          live["glacier_decoupling_factor_n_$source"]
                    @test stored["glacier_decoupling_factor_n_$(source)_above_freezing"] ==
                          live["glacier_decoupling_factor_n_$(source)_above_freezing"]
                    @test stored["temperature_lapse_rate_n_$source"] ==
                          live["temperature_lapse_rate_n_$source"]
                end
            end

            # A very small applied `k` must survive the write still inside the domain it came from.
            #
            # `k`'s domain is the half-open `(0, 1]`, so a stored value has to read back *strictly*
            # above zero. Float32's smallest subnormal is ~1e-45, so storing the applied series at
            # `precision` would flatten anything below that to exactly 0.0 and the file would read
            # back outside the domain `climate_adjust_for_glacier` validates — which is why this one
            # variable is Float64 whatever `precision` says.
            @test Float32(nextfloat(0.0)) == 0.0f0            # the underflow this guards against
            tiny = [(; lo = 1000, hi = 1100, center = 1050.0, area = 1.0, n_cells = 1,
                     decoupling_factor = fill(nextfloat(0.0), length(_CP_TIME)),
                     decoupling_factor_source = fill(Int8(1), length(_CP_TIME)),
                     forcing = first(ivs).forcing)]
            cpath = joinpath(dir, "tiny_factor.nc")
            write_downscaling_tile_netcdf(cpath, tile, p; climate_model = :synthetic, time_range,
                                          tile_size = 2, buffer = 1, min_cells = 4,
                                          elevation_intervals = tiny, precision = Float32)
            back = only(read_downscaling_tile(cpath).intervals).decoupling_factor
            @test all(v -> 0 < v <= 1, back)
            @test back == fill(nextfloat(0.0), length(_CP_TIME))
        end

        @testset "sparse tile" begin
            spath = joinpath(dir, "sparse.nc")
            write_sparse_downscaling_tile_netcdf(spath, tile; climate_model = :synthetic, time_range,
                                                 tile_size = 2, buffer = 1, min_cells = 8,
                                                 reason = "below_min_cells")
            st = read_downscaling_tile_status(spath)
            @test st.sparse == true
            @test st.sparse_reason == "below_min_cells"
            # No time axis at all: a tile below `min_cells` never loads forcing, so its native
            # timestep is genuinely unknown rather than empty-by-accident.
            @test st.n_timesteps == 0
            # ...so coverage is judged from the requested window instead, which is what keeps a
            # sparse tile skippable on a re-run rather than re-derived forever.
            @test st.time_first == first(time_range) && st.time_last == last(time_range)
            # The cells are still listed. A file is written at all so that "missing" unambiguously
            # means "not yet run", and so every cell on Earth stays accounted for.
            @test st.n_cells_core == nrow(tile.core)
            rs = read_downscaling_tile(spath)
            @test nrow(rs.cells) == nrow(tile.core)
            @test isempty(rs.time)
            @test rs.intervals === nothing
        end
    end

    @testset "derive_downscaling_parameter_tiles" begin
        # Two fittable tiles and one holding a single cell, so one sweep exercises all three outcomes.
        many = vcat([(lon = 10.0 + 0.1i, lat = 46.0, z = 1000.0 + 500i, glm = 0.05i,
                      bins = [(1050 + 500i, 10.0)]) for i in 0:3],
                    [(lon = 20.0 + 0.1i, lat = 46.0, z = 1200.0 + 400i, glm = 0.1i,
                      bins = [(1250 + 400i, 8.0)]) for i in 0:3],
                    [(lon = 30.0, lat = 46.0, z = 1500.0, glm = 0.3, bins = [(1550, 5.0)])])
        mt = _cp_table(many)
        time_range = (_CP_TIME[1], _CP_TIME[end])
        obs(c) = (a = 292.0 - 5.0 * c.z / 1000.0;
                  fill(a + ((1 - c.glm * (1 - 0.7)) - 1) * max(a - 273.15, 0.0), length(_CP_TIME)))

        dir = mktempdir()
        s = derive_downscaling_parameter_tiles(:synthetic, time_range, mt, dir;
                                              token = nothing, cache_path = nothing,
                                              tile_size = 2, buffer = 1, min_cells = 4,
                                              forcing_loader = _cp_loader(many, obs))
        @test nrow(s) == 3
        @test sort(s.status) == ["sparse", "written", "written"]
        @test all(isfile(joinpath(dir, n)) for n in s.name)
        # The lone-cell tile is recorded rather than skipped, so its cells stay accounted for.
        sparse_row = only(eachrow(s[s.status .== "sparse", :]))
        @test sparse_row.sparse_reason == "below_min_cells"
        @test sparse_row.n_cells_core == 1

        # The point->tile mapping: one row per cell, every cell present exactly once. This is what
        # makes "parameters for every cell" a join rather than a scan of 819 files.
        index = DataFrame(GEMB_GlacierSims.Parquet2.Dataset(joinpath(dir, "tiles_index.parquet")))
        @test nrow(index) == nrow(mt)
        @test sort(index.row) == collect(1:nrow(mt))
        @test Set(index.tile_name) == Set(s.name)
        @test isfile(joinpath(dir, "tiles_summary.parquet"))

        @testset "resumability" begin
            # A re-run over the same window must decide from metadata alone. Counted, because the
            # forcing pass is the whole cost of a sweep: a re-run that reads even one cell's forcing
            # per tile would make resuming a global sweep as expensive as starting it.
            calls = Ref(0)
            counting = function (m, lat, lon; kwargs...)
                calls[] += 1
                return _cp_loader(many, obs)(m, lat, lon; kwargs...)
            end
            again = derive_downscaling_parameter_tiles(:synthetic, time_range, mt, dir;
                                                      token = nothing, cache_path = nothing,
                                                      tile_size = 2, buffer = 1, min_cells = 4,
                                                      forcing_loader = counting)
            @test all(==("skipped"), again.status)
            @test calls[] == 0

            # Any change to what the file means re-derives it. The file name records only the tile
            # corner, so without this a changed setting would silently read a tile that means
            # something else.
            for kw in ((; min_cells = 5), (; buffer = 2), (; area_minimum = 6.0),
                       (; retain_elevation_interval_forcing = true), (; force = true))
                calls[] = 0
                out = derive_downscaling_parameter_tiles(:synthetic, time_range, mt, dir;
                                                        token = nothing, cache_path = nothing,
                                                        tile_size = 2, buffer = 1, min_cells = 4,
                                                        forcing_loader = counting, kw...)
                # Nothing is skipped: the stored settings no longer match the request.
                @test all(!=("skipped"), out.status)
                # Forcing is re-read wherever a fit is still attempted. Not asserted for
                # `min_cells = 5`, where every tile of this small fixture falls below the threshold
                # and is (correctly) written sparse without loading anything at all.
                haskey(kw, :min_cells) || @test calls[] > 0
            end

            # A wider window is not covered by the stored one, so it re-derives too — there is no
            # append for a pooled cross-cell regression.
            calls[] = 0
            wider = derive_downscaling_parameter_tiles(
                :synthetic, (first(_CP_TIME) - Day(1), last(_CP_TIME)), mt, dir;
                token = nothing, cache_path = nothing, tile_size = 2, buffer = 1, min_cells = 4,
                forcing_loader = counting)
            @test any(==("written"), wider.status)
        end

        @testset "failure isolation" begin
            # A tile whose every cell is on water is unrunnable, not failed: it is recorded as sparse
            # while its neighbours are still written, which is how a global sweep survives coastal
            # and island ice.
            d = mktempdir()
            nanloader = function (m, lat, lon; kwargs...)
                c = many[findfirst(x -> x.lat == lat && x.lon == lon, many)]
                return 10.0 <= lon <= 10.3 ?
                       _cp_forcing(lat, lon, c.z, fill(NaN, length(_CP_TIME))) :
                       _cp_loader(many, obs)(m, lat, lon; kwargs...)
            end
            out = derive_downscaling_parameter_tiles(:synthetic, time_range, mt, d;
                                                     token = nothing, cache_path = nothing,
                                                     tile_size = 2, buffer = 1, min_cells = 4,
                                                     forcing_loader = nanloader)
            row = only(eachrow(out[out.name .== "N46_E010.nc", :]))
            @test row.status == "sparse"
            @test row.sparse_reason == "no_usable_forcing"
            @test only(out[out.name .== "N46_E020.nc", :].status) == "written"

            # An ordinary per-tile failure is logged and skipped, not fatal.
            d2 = mktempdir()
            boom = function (m, lat, lon; kwargs...)
                20.0 <= lon <= 20.3 && error("synthetic forcing store failure")
                return _cp_loader(many, obs)(m, lat, lon; kwargs...)
            end
            out2 = derive_downscaling_parameter_tiles(:synthetic, time_range, mt, d2;
                                                      token = nothing, cache_path = nothing,
                                                      tile_size = 2, buffer = 1, min_cells = 4,
                                                      forcing_loader = boom)
            @test only(out2[out2.name .== "N46_E010.nc", :].status) == "written"
            @test only(out2[out2.name .== "N46_E020.nc", :].status) != "written"

            # But a broken *caller* must not be swallowed. A typo or a moved signature fails
            # identically for every tile, and per-tile handling would turn one bug into hundreds of
            # warnings that read like bad data — so it aborts on the first one instead.
            d3 = mktempdir()
            @test_throws MethodError derive_downscaling_parameter_tiles(
                :synthetic, time_range, mt, d3; token = nothing, cache_path = nothing,
                tile_size = 2, buffer = 1, min_cells = 4,
                forcing_loader = (m, lat, lon; kwargs...) -> throw(MethodError(sin, (1,))))
        end

        @testset "a view of a larger table" begin
            # Sweeping a region of the global table means passing a `SubDataFrame`, which is a
            # different case than it looks: the tile `core`/`buffered` views are built over the frame
            # handed in, so `parentindices` on them resolves to the *ultimate* parent and returns
            # indices into a frame with different row numbering. Using those to index the passed frame
            # is an out-of-bounds error (or, worse, silently the wrong rows). Caught on a real
            # regional sweep, hence `core_rows`.
            padded = vcat(_cp_table([(lon = 100.0, lat = -60.0, z = 900.0, glm = 0.1,
                                      bins = [(950, 2.0)])]), mt)
            sub = view(padded, 2:nrow(padded), :)
            @test all(t -> t.core_rows == parentindices(t.core)[1] .- 0,
                      downscaling_tiles(mt))          # equal when the input is not itself a view
            d = mktempdir()
            out = derive_downscaling_parameter_tiles(:synthetic, time_range, sub, d;
                                                    token = nothing, cache_path = nothing,
                                                    tile_size = 2, buffer = 1, min_cells = 4,
                                                    forcing_loader = _cp_loader(many, obs))
            @test nrow(out) == 3
            index = DataFrame(GEMB_GlacierSims.Parquet2.Dataset(joinpath(d, "tiles_index.parquet")))
            @test nrow(index) == nrow(sub)
            # The recorded coordinates are the subset's own rows, not the parent's — the padding cell
            # at (100, -60) must not appear.
            @test !any(≈(100.0), index.longitude)
            @test sort(index.row) == collect(1:nrow(sub))
        end

        @testset "tile_limit" begin
            d = mktempdir()
            out = derive_downscaling_parameter_tiles(:synthetic, time_range, mt, d;
                                                    token = nothing, cache_path = nothing,
                                                    tile_size = 2, buffer = 1, min_cells = 4,
                                                    tile_limit = 1,
                                                    forcing_loader = _cp_loader(many, obs))
            @test nrow(out) == 1
            # The index still covers every cell: it comes from the tiling, not from what was swept,
            # so an interrupted or truncated sweep still leaves a complete mapping behind.
            index = DataFrame(GEMB_GlacierSims.Parquet2.Dataset(joinpath(d, "tiles_index.parquet")))
            @test nrow(index) == nrow(mt)
        end
    end

    @testset "CachedForcingLoader" begin
        calls = Ref(0)
        base = function (model, lat, lon; time_range = nothing, kwargs...)
            calls[] += 1
            return (model, lat, lon, time_range)
        end
        tr = (DateTime(2018, 1, 1), DateTime(2020, 1, 1))

        c = CachedForcingLoader(base; capacity = 4)
        @test c(:era5land, 60.0, -142.0; time_range = tr) == (:era5land, 60.0, -142.0, tr)
        @test c(:era5land, 60.0, -142.0; time_range = tr) == (:era5land, 60.0, -142.0, tr)
        # The point of the whole file: a repeated cell costs one underlying load, not two. At the
        # 2°/1° tiling each cell is requested ~3.62 times over a global sweep.
        @test calls[] == 1
        r = forcing_cache_report(c)
        @test (r.hits, r.misses, r.requests) == (1, 1, 2)
        @test r.hit_rate == 0.5

        # Bounded: past capacity the least recently used are dropped, and an evicted cell reloads.
        for i in 1:6
            c(:era5land, 60.0 + i, -142.0; time_range = tr)
        end
        r = forcing_cache_report(c)
        @test r.cached_cells <= 4
        @test r.evictions > 0
        @test calls[] == 7

        # A different window is different data under the same key. Silently returning the wrong one
        # would be invisible, so it is refused.
        @test_throws "one cache cannot serve two of them" c(:era5land, 60.0, -142.0;
                                                            time_range = (DateTime(1990, 1, 1),
                                                                          DateTime(1991, 1, 1)))

        # A throwing load is not cached, so a transient store error is retried rather than remembered
        # as a permanent failure.
        boom = Ref(0)
        flaky = function (model, lat, lon; time_range = nothing, kwargs...)
            boom[] += 1
            boom[] == 1 && error("transient")
            return :ok
        end
        c2 = CachedForcingLoader(flaky; capacity = 2)
        @test_throws ErrorException c2(:era5land, 1.0, 2.0; time_range = tr)
        @test c2(:era5land, 1.0, 2.0; time_range = tr) == :ok
        @test boom[] == 2

        # Capacity from a memory budget depends on the record length, which is the easy thing to
        # forget: the same budget that holds thousands of cells of a two-year window holds ~100 of
        # the full record.
        @test forcing_cache_capacity(4 * 2^30, 17_521) > 4_000
        @test forcing_cache_capacity(4 * 2^30, 666_000) < 200
        @test forcing_cache_capacity(1, 10_000) == 1        # floored at one, never zero
        @test_throws ArgumentError CachedForcingLoader(base; capacity = 0)
        @test_throws ArgumentError forcing_cache_capacity(0, 100)
        @test_throws ArgumentError forcing_cache_capacity(100, 0)
    end

    @testset "volume change" begin
        # A synthetic `gemb` output built so the identity holds exactly by construction: `ice_flux` is
        # derived *from* the mass, water and firn terms, which is what makes the closure a test of the
        # decomposition rather than of the fixture.
        nt, nz, rho = 6, 4, 917.0
        ti = Ti(collect(DateTime(2000, 1, 1):Month(1):DateTime(2000, 6, 1)))
        zd = Z(1:nz)
        dz = [0.5 + 0.01i + 0.002j for j in 1:nz, i in 1:nt]
        density = [400.0 + 120.0 * (j - 1) + 5.0i for j in 1:nz, i in 1:nt]
        density[nz, :] .= rho                       # base at ice density, as the datum requires
        water = [j == 2 ? 3.0sin(i) + 3.0 : 0.0 for j in 1:nz, i in 1:nt]
        precip, ec = fill(40.0, nt), fill(-1.5, nt)
        runoff = [0.0, 0.0, 5.0, 30.0, 12.0, 0.0]

        fac = [GEMB.firn_air_content(view(dz, :, i), view(density, :, i), rho) for i in 1:nt]
        stored = [sum(view(water, :, i)) for i in 1:nt]
        smb = cumsum(precip .+ ec .- runoff)
        dh_true = [(smb[i] - smb[1] - (stored[i] - stored[1])) / rho + (fac[i] - fac[1])
                   for i in 1:nt]
        flux = -diff(vcat(0.0, dh_true))            # so -cumsum(flux) == dh_true

        out = DimStack((ice_flux = DimArray(flux, (ti,)),
                        precipitation = DimArray(precip, (ti,)),
                        evaporation_condensation = DimArray(ec, (ti,)),
                        runoff = DimArray(runoff, (ti,)),
                        dz = DimArray(dz, (zd, ti)),
                        density = DimArray(density, (zd, ti)),
                        water = DimArray(water, (zd, ti)));
                       metadata = Dict{String,Any}("density_ice" => rho))

        dh = surface_height_change(out)
        @test dh ≈ -cumsum(flux)
        c = height_change_components(out)
        # Every term is an anomaly against the first output time, so all six start at zero and the
        # four physical ones sum to the total. This is the check every dh in a tile file carries.
        @test c.total ≈ dh .- dh[1]
        @test all(iszero, (c.total[1], c.mass[1], c.water[1], c.firn[1], c.strain[1]))
        @test c.total ≈ c.mass .+ c.water .+ c.firn .+ c.strain .+ c.residual
        @test maximum(abs, c.residual) < 1e-12
        # `strain_thinning` is absent here, which is what a run at the default
        # `horizontal_strain_rate` produces — treated as zero rather than as a missing layer.
        @test all(iszero, c.strain)

        # The datum assumes the material below the model base is already ice. A base short of ice
        # density means the height series under-counts compaction, silently, so it is checkable.
        @test column_reaches_ice_density(out)
        shallow_density = copy(density)
        shallow_density[nz, nt] = 800.0
        shallow = DimStack((ice_flux = DimArray(flux, (ti,)),
                            dz = DimArray(dz, (zd, ti)),
                            density = DimArray(shallow_density, (zd, ti)),
                            water = DimArray(water, (zd, ti)));
                           metadata = Dict{String,Any}("density_ice" => rho))
        @test !column_reaches_ice_density(shallow)
        @test column_reaches_ice_density(shallow; tolerance = 200.0)
        @test_throws ArgumentError column_reaches_ice_density(out; tolerance = -1)

        # A stack with no `density_ice` cannot convert between mass and thickness at all, so it says
        # so rather than assuming a density.
        @test_throws ArgumentError height_change_components(
            DimStack((ice_flux = DimArray(flux, (ti,)),)))
        @test_throws ArgumentError surface_height_change(
            DimStack((melt = DimArray(precip, (ti,)),)))

        @testset "area weighting and units" begin
            areas = [10.0, 25.0, 4.0]
            dhb = [-1.0 -0.5 0.2; -2.0 -1.0 0.4]
            # A metre over a square kilometre is a thousandth of a cubic kilometre.
            @test tile_volume_change(dhb, areas) ≈
                  [(10 * -1.0 + 25 * -0.5 + 4 * 0.2) / 1000,
                   (10 * -2.0 + 25 * -1.0 + 4 * 0.4) / 1000]
            # kg m-2 over km² into Gt: times 1e6 m²/km², divided by 1e12 kg/Gt.
            @test tile_mass_total([-100.0 -50.0 20.0], areas) ≈
                  [(10 * -100.0 + 25 * -50.0 + 4 * 20.0) * 1e-6]
            @test mie2cubickm(areas) ≈ sum(areas) / 1000
            # A zero-area band contributes nothing and cannot make the total `NaN`, which matters
            # because an unrunnable band is a normal outcome.
            @test tile_volume_change([1.0 NaN], [2.0, 0.0]) ≈ [2.0 / 1000]
            @test_throws DimensionMismatch tile_volume_change(dhb, [1.0, 2.0])
        end
    end

    @testset "geotile_id" begin
        # The exact strings the 2° geotile products on disk use, so a join needs no translation.
        @test geotile_id((-142, 60), 2) == "lat[+60+62]lon[-142-140]"
        @test geotile_id((28, 0), 2) == "lat[+00+02]lon[+028+030]"
        @test geotile_id((36, -2), 2) == "lat[-02+00]lon[+036+038]"
        @test geotile_id((-180, -90), 2) == "lat[-90-88]lon[-180-178]"
        @test geotile_id((-24, 82), 2) == "lat[+82+84]lon[-024-022]"
        # Bounds and index forms agree, so a tile's own `bounds` and its index name the same geotile.
        @test geotile_id(tile_bounds((-142, 60); tile_size = 2)) == geotile_id((-142, 60), 2)
        # Zero carries an explicit `+`, which is what the ids on disk do.
        @test geotile_id((0, 0), 2) == "lat[+00+02]lon[+000+002]"
    end

    @testset "resolve_downscaling" begin
        # A fit fixture built directly rather than derived, so each acceptance test can put one
        # timestep into one state. Twelve hourly steps in January, so the month-and-hour climatology
        # has one populated cell per step.
        rt = collect(DateTime(2001, 1, 1):Hour(1):DateTime(2001, 1, 1, 11))
        n = length(rt)
        function _fit(; k, lapse, cells = fill(20, n), spread = fill(900.0, n),
                      alpha = fill(8.0, n), beta = fill(-0.001, n),
                      gamma = fill(-3.0, n), delta = fill(0.0004, n))
            ti = Ti(rt)
            dec = DimStack((decoupling_factor = DimArray(collect(Float64, k), (ti,)),
                            n_cells = DimArray(fill(20, n), (ti,)),
                            r2 = DimArray(fill(0.99, n), (ti,)),
                            ambient_excess = DimArray(fill(5.0, n), (ti,)),
                            coef_alpha = DimArray(collect(Float64, alpha), (ti,)),
                            coef_beta = DimArray(collect(Float64, beta), (ti,)),
                            coef_gamma = DimArray(collect(Float64, gamma), (ti,)),
                            coef_delta = DimArray(collect(Float64, delta), (ti,))))
            lr = DimStack((lapse_rate = DimArray(collect(Float64, lapse), (ti,)),
                           n_cells = DimArray(collect(Int, cells), (ti,)),
                           elevation_spread = DimArray(collect(Float64, spread), (ti,))))
            return (; time = rt, decoupling = dec, lapse_rate = lr)
        end
        bands = [(; lo = 1000, hi = 1100, center = 1050.0),
                 (; lo = 2000, hi = 2100, center = 2050.0)]

        @testset "a good fit is used as measured" begin
            a = resolve_downscaling(_fit(k = fill(0.7, n), lapse = fill(6.0, n)), bands, rt;
                                    basis = :fitted)
            @test a.lapse_rate == fill(6.0, n)
            @test downscaling_source_counts(a.lapse_rate_source)[:fitted] == n
            @test length(a.bands) == 2
            # `k` is re-evaluated at each band's own centre, so the two bands differ — the fit carries
            # a `glm x z` interaction and the bands are the glacier, not the reanalysis surface.
            @test a.bands[1].decoupling_factor != a.bands[2].decoupling_factor
            @test all(v -> 0 < v <= 1, a.bands[1].decoupling_factor)
            @test downscaling_source_counts(a.bands[1].decoupling_factor_source)[:fitted] == n
        end

        @testset "the joint screen against k, not fit_usable" begin
            # `derive_lapse_rate` consumes `k`, and a `k` outside `(0, 1]` corrupts that timestep's
            # slope through a correction proportional to `k - 1`. So it is dropped...
            k = fill(0.7, n); k[3] = 2.5
            a = resolve_downscaling(_fit(; k, lapse = fill(6.0, n)), bands, rt; basis = :fitted)
            @test a.lapse_rate_source[3] != GEMB_GlacierSims._SOURCE_FITTED
            @test downscaling_source_counts(a.lapse_rate_source)[:fitted] == n - 1

            # ...but a `NaN` `k` is NOT a rejection: the fit treats it as `k = 1`, the exact no-op, so
            # that timestep's slope is the ambient one and is sound. This is the difference between
            # screening on the stored `fit_usable` flag (which is 0 for both cases) and screening on
            # what actually corrupts the slope — on a real tile, 17,159 kept fits against 4,777.
            k2 = fill(0.7, n); k2[3] = NaN
            a2 = resolve_downscaling(_fit(k = k2, lapse = fill(6.0, n)), bands, rt; basis = :fitted)
            @test downscaling_source_counts(a2.lapse_rate_source)[:fitted] == n
        end

        @testset "each rejection reaches the next tier" begin
            # Too few cells, too little elevation spread, and a physically absent slope: three
            # independent reasons a finite fit is still not evidence. Each falls to the climatology of
            # the accepted ones, which here is the median of the good timesteps.
            lapse = fill(6.0, n)
            cells = fill(20, n); cells[2] = 3
            spread = fill(900.0, n); spread[4] = 10.0
            lapse[6] = 40.0                          # inside the validator's range, physically absent
            a = resolve_downscaling(_fit(k = fill(0.7, n), lapse = lapse; cells, spread), bands, rt;
                                    basis = :fitted)
            for i in (2, 4, 6)
                @test a.lapse_rate_source[i] == GEMB_GlacierSims._SOURCE_CLIMATOLOGY
                @test a.lapse_rate[i] ≈ 6.0          # the median of the accepted fits
            end
            @test downscaling_source_counts(a.lapse_rate_source)[:fitted] == n - 3

            # With nothing accepted at all there is no climatology either, so the prior is reached.
            none = resolve_downscaling(_fit(k = fill(0.7, n), lapse = fill(NaN, n)), bands, rt;
                                       basis = :fitted, lapse_rate_prior = 7.5)
            @test none.lapse_rate == fill(7.5, n)
            @test downscaling_source_counts(none.lapse_rate_source)[:prior] == n
        end

        @testset "an out-of-domain k is excluded, not clamped" begin
            # A band's `k` is re-evaluated from the four coefficients, not read off the stored
            # `decoupling_factor` series, so it is the coefficients that decide whether a band's fit
            # is in domain. A positive `gamma` puts the decoupled excess *above* the ambient one,
            # i.e. `k > 1`: the glaciated cells fitted warmer than the ice-free ones.
            #
            # Clamping such a value to the nearest bound would report the identity as though it had
            # been measured. Excluding it and taking the median of the fits that *are* in domain is
            # both a better estimate and honestly labelled.
            gamma = fill(-3.0, n); gamma[5] = 3.0
            a = resolve_downscaling(_fit(k = fill(0.7, n), lapse = fill(6.0, n); gamma), bands, rt;
                                    basis = :fitted)
            # The label is the assertion, not the value: here the other eleven timesteps are identical,
            # so their median happens to equal them and the substitution is invisible numerically.
            # That is exactly why the provenance is recorded separately from the series.
            @test a.bands[1].decoupling_factor_source[5] ==
                  GEMB_GlacierSims._SOURCE_CLIMATOLOGY
            @test all(i -> a.bands[1].decoupling_factor_source[i] ==
                           GEMB_GlacierSims._SOURCE_FITTED, [1, 2, 3, 4, 6, 7])
            @test all(v -> 0 < v <= 1, a.bands[1].decoupling_factor)
            @test a.bands[1].n_fit_in_domain == n - 1

            # With every fit out of domain and no prior, the identity is reached — and is labelled
            # `:ambient`, which is a different claim from `:fitted` at the same numeric value.
            allbad = resolve_downscaling(_fit(k = fill(0.7, n), lapse = fill(6.0, n),
                                              gamma = fill(3.0, n)), bands, rt; basis = :fitted)
            @test allbad.bands[1].decoupling_factor == fill(1.0, n)
            @test downscaling_source_counts(allbad.bands[1].decoupling_factor_source)[:ambient] == n
            @test allbad.bands[1].n_fit_in_domain == 0
            # A prior displaces the identity, and says so.
            withprior = resolve_downscaling(_fit(k = fill(0.7, n), lapse = fill(6.0, n),
                                                 gamma = fill(3.0, n)), bands, rt;
                                            basis = :fitted, decoupling_factor_prior = 0.6)
            @test withprior.bands[1].decoupling_factor == fill(0.6, n)
            @test downscaling_source_counts(
                withprior.bands[1].decoupling_factor_source)[:prior] == n
        end

        @testset "a held k is labelled, not substituted" begin
            # Above the elevation where the tile's ambient warm excess `alpha + beta*z` falls below the
            # fit's threshold, `decoupling_factor_at_elevation` holds the evaluation elevation down
            # rather than reverting to 1 — a step in the vertical profile is read downstream as a
            # mass-balance signal. A held value is still a fit, so it is used; the label records that
            # it is not a fit at this band's own elevation.
            #
            # `2.0 - 0.001z` crosses 0.5 K at 1500 m, so the 2050 m band is held and the 1050 m band
            # is not. `gamma`/`delta` are set so the held evaluation still lands inside `(0, 1]` —
            # a held value that is out of domain would be excluded like any other, and this testset is
            # about the held-but-usable case.
            held = _fit(k = fill(0.7, n), lapse = fill(6.0, n),
                        alpha = fill(2.0, n), beta = fill(-0.001, n),
                        gamma = fill(-0.3, n), delta = fill(0.0001, n))
            a = resolve_downscaling(held, [(; lo = 1000, hi = 1100, center = 1050.0),
                                           (; lo = 2000, hi = 2100, center = 2050.0)], rt;
                                    basis = :fitted)
            @test a.bands[1].n_fit_held == 0
            @test a.bands[2].n_fit_held == n
            @test downscaling_source_counts(a.bands[1].decoupling_factor_source)[:fitted] == n
            @test downscaling_source_counts(a.bands[2].decoupling_factor_source)[:held] == n
            @test all(v -> 0 < v <= 1, a.bands[2].decoupling_factor)
        end

        @testset "the climatology basis applies to any window" begin
            fit = _fit(k = fill(0.7, n), lapse = fill(6.0, n))
            # A run window the fits do not cover, which is the normal case: the fits are a pooled
            # cross-cell regression and cannot be appended to, so a decades-long run over a two-year
            # fit window has to reduce them.
            long = collect(DateTime(1990, 1, 1):Hour(1):DateTime(1990, 1, 2, 5))
            a = resolve_downscaling(fit, bands, long; basis = :climatology)
            @test length(a.lapse_rate) == length(long)
            @test a.lapse_rate ≈ fill(6.0, length(long))
            # Nothing is `:fitted` under this basis — every value is a climatology by construction, and
            # the label distinguishes a populated month-hour cell from one that fell to a prior.
            counts = downscaling_source_counts(a.lapse_rate_source)
            @test counts[:fitted] == 0
            @test counts[:climatology] == length(long)
            # `:fitted` refuses the same mismatch rather than silently reverting to the climatology.
            @test_throws ArgumentError resolve_downscaling(fit, bands, long; basis = :fitted)

            # A month with no accepted fit falls through to the prior, not to another month's median —
            # this fixture has January only, so a July run window is entirely prior.
            july = collect(DateTime(1990, 7, 1):Hour(1):DateTime(1990, 7, 1, 5))
            # (The whole-record median is still reached first, which is the intended third tier.)
            aj = resolve_downscaling(fit, bands, july; basis = :climatology)
            @test aj.lapse_rate ≈ fill(6.0, length(july))
            @test downscaling_source_counts(aj.lapse_rate_source)[:climatology] == length(july)
        end

        @testset "argument validation" begin
            fit = _fit(k = fill(0.7, n), lapse = fill(6.0, n))
            @test_throws ArgumentError resolve_downscaling(fit, bands, rt; basis = :nonsense)
            @test_throws ArgumentError resolve_downscaling(fit, bands, DateTime[])
            @test_throws ArgumentError resolve_downscaling(fit, bands, rt; spread_minimum = -1)
            @test_throws ArgumentError resolve_downscaling(fit, bands, rt;
                                                           lapse_rate_window = (10.0, 5.0))
            # A window wider than the validator's would accept a fit that is then rejected where it is
            # applied, which is the one failure mode the acceptance test exists to prevent.
            @test_throws ArgumentError resolve_downscaling(fit, bands, rt;
                                                           lapse_rate_window = (-50.0, 50.0))
            # Priors are applied, so they must be applicable.
            @test_throws ArgumentError resolve_downscaling(fit, bands, rt; lapse_rate_prior = 99.0)
            @test_throws ArgumentError resolve_downscaling(fit, bands, rt;
                                                           decoupling_factor_prior = 1.5)
            @test_throws ArgumentError resolve_downscaling(fit, bands, rt;
                                                           decoupling_factor_prior = 0.0)
            @test_throws ArgumentError resolve_downscaling(fit, bands, rt;
                                                           lapse_rate_prior = [1.0, 2.0])
            # No bands is not an error: a tile can carry cells with no populated hypsometry, and the
            # lapse rate is still resolvable for it.
            empty_bands = resolve_downscaling(fit, typeof(bands)(), rt)
            @test isempty(empty_bands.bands)
            @test length(empty_bands.lapse_rate) == n
        end

        @testset "source counts and masking" begin
            a = resolve_downscaling(_fit(k = fill(0.7, n), lapse = fill(6.0, n)), bands, rt;
                                    basis = :fitted)
            counts = downscaling_source_counts(a.lapse_rate_source)
            # Every source is a key even when unused, so two reports are comparable without either
            # having to know which tiers the other reached.
            @test issetequal(keys(counts), DOWNSCALING_SOURCES)
            @test sum(values(counts)) == n
            # The mask is the honest denominator for `k`: below freezing every value gives identical
            # forcing, so an unmasked count describes the fit rather than the run.
            mask = [i <= 4 for i in 1:n]
            @test sum(values(downscaling_source_counts(a.lapse_rate_source, mask))) == 4
            @test_throws DimensionMismatch downscaling_source_counts(a.lapse_rate_source, [true])
        end
    end

    @testset "hypsometry_intervals" begin
        cells = [(lon = 10.0, lat = 46.0, z = 1000.0, glm = 0.0, bins = [(1050, 10.0), (1550, 2.0)]),
                 (lon = 10.1, lat = 46.0, z = 1500.0, glm = 0.5, bins = [(1550, 5.0), (2050, 1.0)])]
        iv = hypsometry_intervals(_cp_table(cells))
        # The union of populated bins, ascending, each centred on its bin.
        @test [(x.lo, x.hi) for x in iv] == [(1000, 1100), (1500, 1600), (2000, 2100)]
        @test [x.center for x in iv] == [1050.0, 1550.0, 2050.0]
        # Which bins are populated, not how much area is in them — the area a band carries is
        # accumulated over the cells whose forcing was usable, and a total summed here would disagree
        # with it whenever a cell drops out.
        @test !any(hasproperty(x, :area) for x in iv)
        @test isempty(hypsometry_intervals(_cp_table([(lon = 0.0, lat = 0.0, z = 1.0, glm = 0.0,
                                                       bins = Tuple{Int,Float64}[])])))
    end

    @testset "tile run NetCDF" begin
        run = _fake_tile_run()
        dir = mktempdir()
        path = write_glacier_tile_netcdf(joinpath(dir, run.name), run;
                                       institution = "JPL", references = "none")

        @testset "status reads metadata only" begin
            st = read_glacier_tile_status(path)
            @test st.geotile_id == run.geotile_id
            @test st.band_centers == [b.center for b in run.bands]
            @test st.delta_temperatures == run.delta_temperatures
            @test st.precipitation_scalings == run.precipitation_scalings
            @test st.time == last(run.time)
            # The downscaling policy is a run parameter, because two runs of one tile that resolved
            # their lapse rate differently are different experiments.
            @test st.parameters["downscaling_basis"] == "fitted"
            @test haskey(st.parameters, "model_densification_method")
            @test read_glacier_tile_status(joinpath(dir, "absent.nc")) === nothing
        end

        @testset "series and totals round trip" begin
            NCDatasets.NCDataset(path, "r") do ds
                # Band-resolved series and tile totals are both stored, at both resolutions: the
                # bands are what altimetry dh is binned at, the totals are what a regional sum uses.
                @test size(ds["dh"]) == (length(run.time), length(run.bands),
                                         length(run.delta_temperatures),
                                         length(run.precipitation_scalings))
                @test ds["dh"].attrib["units"] == "m"
                @test size(ds["total_dv"]) == (length(run.time), length(run.delta_temperatures),
                                               length(run.precipitation_scalings))
                @test ds["total_dv"].attrib["units"] == "km3"
                @test ds["total_dm"].attrib["units"] == "Gt"
                # A per-band flux and its tile total share a name, so the total is prefixed; both must
                # be present and distinguishable.
                @test ds["melt"].attrib["units"] == "kg m-2"
                @test ds["total_melt"].attrib["units"] == "Gt"
                # The two are per-interval and cumulative respectively, which the file has to say or a
                # reader compares a record-long volume change against one month's melt.
                @test ds["total_melt"].attrib["cell_methods"] == "area: sum time: sum"
                @test ds["total_fac"].attrib["cell_methods"] == "area: sum time: mean"
                @test occursin("CUMULATIVE", ds.attrib["total_accumulation_comment"])
                @test collect(ds["total_dv"][:, :, :]) ≈ run.totals[:dv]
                @test collect(ds["band_area"][:]) == [b.area for b in run.bands]
                @test ds["glacier_area_total"][] ≈ sum(b.area for b in run.bands)
                @test ds["mie2cubickm"][] ≈ mie2cubickm([b.area for b in run.bands])
                # Both identifiers, so a join against the 2° products needs no filename parsing.
                @test ds.attrib["geotile_id"] == run.geotile_id
                @test ds.attrib["tile_name"] == run.name
                @test haskey(ds.attrib, "height_change_identity")
                # The per-band parameter provenance is a variable, not an attribute: `k` is resolved
                # at each band's own centre, so it varies with the band.
                @test collect(ds["band_extrapolation_above_reanalysis"][:]) ==
                      [p["extrapolation_above_reanalysis"] for p in run.band_provenance]
                @test ds["band_extrapolation_above_reanalysis"].attrib["units"] == "m"
                # A key one band did not record must read back as absent, not as zero: the writer
                # defines a variable per key seen on *any* band, so the fill is what carries "this
                # band has no value" through to a reader.
                held = ds["band_glacier_decoupling_factor_n_fit_held"][:]
                @test ismissing(held[end])
                @test held[1] == run.band_provenance[1]["glacier_decoupling_factor_n_fit_held"]
            end
        end

        @testset "restart round trip" begin
            r = read_glacier_tile_restart(path)
            @test length(r.profiles) == count(!isnothing, run.profiles)
            @test r.band_centers == [b.center for b in run.bands]
            @test r.time == last(run.time)
            for (key, profile) in r.profiles
                saved = run.profiles[key...]
                @test issetequal(keys(profile), GEMB_GlacierSims.PROFILE_VARIABLES)
                # Bit-exact, not approximate: `gemb` pins the column depth and cell count from the
                # restored `dz`, so a rounded column is a different grid and cannot be resumed.
                @test collect(profile[:dz]) == collect(saved[:dz])
                @test collect(profile[:density]) == collect(saved[:density])
            end
            # The spinup provenance is restored onto each profile, so a continuation reports the
            # spinup it inherited rather than none at all.
            @test DimensionalData.metadata(first(values(r.profiles)))["spinup_cycles"] == 37
            @test read_glacier_tile_restart(joinpath(dir, "absent.nc")) === nothing
        end

        @testset "append" begin
            later = _fake_tile_run(; time = collect(DateTime(2001, 1, 1):Month(1):DateTime(2001, 3, 1)))
            append_glacier_tile_netcdf(path, later)
            st = read_glacier_tile_status(path)
            @test st.n_timesteps == length(run.time) + length(later.time)
            @test st.time == last(later.time)
            # An overlapping or out-of-order continuation would duplicate output, so it is refused.
            @test_throws ArgumentError append_glacier_tile_netcdf(path, run)
            # A changed run grid means the arrays do not line up at all, so it cannot be reconciled.
            wider = _fake_tile_run(; time = collect(DateTime(2002, 1, 1):Month(1):DateTime(2002, 3, 1)),
                                   deltas = [0.0, 1.0, 2.0])
            @test_throws ArgumentError append_glacier_tile_netcdf(path, wider)
        end
    end

    # Note: end-to-end simulation tests need CDS API credentials and network access to the
    # Copernicus DEM, so the runfile builder, `gemb_glacier_cell` and `gemb_glacier_tile` are
    # exercised by `src/era5_example.jl`, `scripts/derive_downscaling_parameters_e2e.jl` and
    # `scripts/gemb_tile_e2e.jl` rather than here.
end
