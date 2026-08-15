using GEMB_GlacierSims
using Test
using DataFrames
using Dates
using DimensionalData
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
        72.5, -38.25, 1234.5, 42, 0.87,
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
        @test length(p) == length(propertynames(mp)) - 1 + 2

        # Provenance that legitimately changes between a run and its continuation is not part of
        # the parameter set, so extending a record never trips the check on it.
        @test !any(k -> startswith(k, "spinup") || startswith(k, "climatology"), keys(p))

        # A changed setting is visible as a changed parameter.
        p2 = run_parameters(initialize_parameters(output_frequency = :monthly, albedo_ice = 0.4);
                            coverage = 0.95, lapse_rate = 6.5)
        @test p2["model_albedo_ice"] == 0.4
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

    # Note: end-to-end simulation tests need CDS API credentials and network access to the
    # Copernicus DEM, so the runfile builder and `gemb_glacier_cell` are exercised by
    # `src/era5_example.jl` rather than here.
end
