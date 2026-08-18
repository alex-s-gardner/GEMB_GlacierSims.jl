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

        @testset "elevation interval forcing" begin
            intervals = collect(p.elevation_interval_forcing)
            @test length(intervals) == length(p.elevation_interval_forcing) == 4
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
                # Every timestep of this fixture is measured, so nothing was filled or clamped —
                # the counts exist precisely so a substituted interval is distinguishable.
                m = DimensionalData.metadata(iv.forcing)
                @test m["glacier_decoupling_factor_n_filled"] == 0
                @test m["glacier_decoupling_factor_n_clamped"] == 0
                @test m["temperature_lapse_rate_n_filled"] == 0
                @test m["temperature_lapse_rate_n_clamped"] == 0
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

            # Every cell unusable is an error, not an empty result.
            @test_throws ArgumentError derive_downscaling_parameters(
                :synthetic, time_range, table, region; token = nothing, cache_path = nothing,
                min_cells = 4,
                forcing_loader = _cp_loader(cells, _ -> fill(NaN, length(_CP_TIME))))

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

            # The interval forcing still builds, because it fills — and the fill is visible in the
            # metadata rather than silent. Without it a single `NaN` would make the interval fail
            # `forcing_is_complete` and a sweep would skip it like an ocean cell.
            for iv in pl.elevation_interval_forcing
                @test forcing_is_complete(iv.forcing)
                m = DimensionalData.metadata(iv.forcing)
                @test m["temperature_lapse_rate_n_filled"] == length(_CP_TIME)
                @test m["temperature_lapse_rate"] ≈ 6.5
                @test m["glacier_decoupling_factor_n_filled"] == length(_CP_TIME)
                @test m["glacier_decoupling_factor_mean"] == 1.0   # the bit-exact no-op
            end

            # Non-default fills are honoured, which is the point of the keywords: a region with no
            # usable fit can be given a published or regional value instead of the identity.
            pfill = derive_downscaling_parameters(:synthetic, time_range, _cp_table(flat), region;
                                              token = nothing, cache_path = nothing,
                                              min_cells = 4,
                                              forcing_loader = _cp_loader(flat, observed),
                                              lapse_rate_fill = 7.25,
                                              decoupling_factor_fill = 0.85)
            for iv in pfill.elevation_interval_forcing
                @test DimensionalData.metadata(iv.forcing)["temperature_lapse_rate"] ≈ 7.25
                @test DimensionalData.metadata(
                    iv.forcing)["glacier_decoupling_factor_mean"] ≈ 0.85
            end
            # ...and validated, since they are applied: an out-of-domain fill would be rejected
            # downstream no differently from an out-of-domain fit, but with nothing to inspect.
            @test_throws ArgumentError derive_downscaling_parameters(
                :synthetic, time_range, table, region; token = nothing, cache_path = nothing,
                min_cells = 4,
                forcing_loader = _cp_loader(cells, observed), lapse_rate_fill = 99.0)
            @test_throws ArgumentError derive_downscaling_parameters(
                :synthetic, time_range, table, region; token = nothing, cache_path = nothing,
                min_cells = 4,
                forcing_loader = _cp_loader(cells, observed), decoupling_factor_fill = 1.5)
        end

        @testset "the fits report raw; only the application clamps" begin
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

            # ...but the interval forcing, which *applies* it, is in domain and finite, and says how
            # many timesteps it had to clamp. This is the assertion that policy moved rather than
            # vanished.
            for iv in pi_.elevation_interval_forcing
                @test forcing_is_complete(iv.forcing)
                @test all(k -> 0 < k <= 1, iv.decoupling_factor)
                @test DimensionalData.metadata(
                    iv.forcing)["glacier_decoupling_factor_n_clamped"] == length(_CP_TIME)
            end

            # Opting out leaves the raw fit in the applied series too, which is what the keyword is
            # for — inspecting what would have been applied.
            pu = derive_downscaling_parameters(
                :synthetic, time_range, table, region; token = nothing, cache_path = nothing,
                min_cells = 4,
                forcing_loader = _cp_loader(cells, inverted),
                clamp_to_valid_domain = false)
            iv1 = first(pu.elevation_interval_forcing)
            @test all(>(1.0), iv1.decoupling_factor)
            @test DimensionalData.metadata(
                iv1.forcing)["glacier_decoupling_factor_n_clamped"] == 0

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

    # Note: end-to-end simulation tests need CDS API credentials and network access to the
    # Copernicus DEM, so the runfile builder and `gemb_glacier_cell` are exercised by
    # `src/era5_example.jl` rather than here.
end
