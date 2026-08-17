using GEMB_GlacierSims
using Test
using DataFrames
using GeoDataFrames
using Dates
using DimensionalData
import GEMB
using GEMB: Z, DimStack, DimArray, initialize_parameters
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
        @test full.k == full.k_published
        @test 0 < full.k < 1

        k_pub = full.k_published

        # The weighting is linear in the non-glacier fraction and exact at both ends: no glacier
        # in the cell is the full correction, an all-glacier cell is the identity.
        @test cell_decoupling_factor(cell(7.53, 45.97, 1.0)).k == 1.0
        @test cell_decoupling_factor(cell(7.53, 45.97, 0.5)).k ≈ 1 - (1 - k_pub) * 0.5
        @test cell_decoupling_factor(cell(7.53, 45.97, 0.25)).k ≈ 1 - (1 - k_pub) * 0.75

        # No `glm` at all — a missing value, or a table without the column — is the uncorrected
        # reanalysis assumption (glm = 0), hence the full correction rather than a skip.
        @test cell_decoupling_factor(cell(7.53, 45.97, missing)).k == k_pub
        @test cell_decoupling_factor(cell(7.53, 45.97, NaN)).k == k_pub
        @test cell_decoupling_factor(cell(7.53, 45.97)).k == k_pub

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
        @test resolve(cell(0.0), true) == cell_decoupling_factor(cell(0.0)).k

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
        @test length(p) == length(propertynames(mp)) - 1 + 3

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
        # dropped on write and then a bare `FieldError` inside `gemb` on continuation. Pin it
        # against a real initialized profile rather than a hand-copied list, so a state layer
        # added in GEMB.jl fails here instead of at the first restart months later.
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

        @test read_glacier_cell_restart(joinpath(dir, "absent.nc")) === nothing
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

    # Note: end-to-end simulation tests need CDS API credentials and network access to the
    # Copernicus DEM, so the runfile builder and `gemb_glacier_cell` are exercised by
    # `src/era5_example.jl` rather than here.
end
