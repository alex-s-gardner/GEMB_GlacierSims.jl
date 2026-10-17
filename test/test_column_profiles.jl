# Column profiles for the WGMS density test: the recorder, the profile file, and the sweep's setting.
# Included from `runtests.jl` after `test_physics_members.jl`; uses `_tiny_tile_run` and
# `_tiny_band_forcing` from `test_pipeline_audit.jl` and `_cp_table` from `runtests.jl`.
@testset "column profiles" begin
    @testset "the recorder keeps the top of the column at each date, for its bands only" begin
        forcing = _tiny_band_forcing(; temperature = 263.15, precipitation = 0.2, shortwave = 150.0,
                                     longwave = 230.0)
        dates = [DateTime(2001, 3, 15), DateTime(2002, 4, 1)]
        recorder = ColumnProfileRecorder([1150.0], dates; mass_max = 300.0)
        outputs = Dict{Float64,Any}()
        capture = function (output; band, delta, pscale)
            outputs[band.center] = output
            recorder(output; band, delta, pscale)
        end
        run = _tiny_tile_run([0.48, 0.48]; forcing, core_bins = [(1050, 10.0), (1150, 10.0)],
                             on_output = capture)
        @test collect(keys(recorder.profiles)) == [(1150.0, 0.0, 1.0)]
        for (d, c) in zip(dates, recorder.profiles[(1150.0, 0.0, 1.0)])
            times = collect(dims(outputs[1150.0], Ti))
            @test c.time == times[argmin(abs.(Dates.value.(times .- d)))]
            # Down to the first layer at which 300 kg m-2 is reached, and no further.
            mass = cumsum(c.dz .* c.density)
            @test mass[end] >= 300.0
            @test length(mass) == 1 || mass[end - 1] < 300.0
            @test all(c.density .> 0) && all(c.age .>= 0)
            direct = column_profile_top(outputs[1150.0], d; mass_max = 300.0)
            @test direct == c
        end
        @test column_profile_top(outputs[1150.0], DateTime(1990, 1, 1)) === nothing

        path = joinpath(mktempdir(), "N46_E010_column_profiles.nc")
        write_column_profiles_netcdf(path, recorder; tile_name = "N46_E010", delta_temperatures = [0.0],
                                     precipitation_scalings = [1.0])
        back = read_column_profiles(path)
        @test back.tile_name == "N46_E010"
        @test back.dates == dates && back.band_centers == [1150.0]
        for (d, c) in zip(dates, recorder.profiles[(1150.0, 0.0, 1.0)])
            b = back.profile(d, 1150.0, 0.0, 1.0)
            @test b.dz ≈ c.dz && b.density ≈ c.density && b.age ≈ c.age
            @test abs(b.time - c.time) < Second(1)
        end
        @test_throws "no profile slot" back.profile(dates[1], 1050.0, 0.0, 1.0)
        # A node the recorder never saw is refused rather than written as empty.
        @test_throws "no column profile for band 1150.0 m at ΔT 1.0" write_column_profiles_netcdf(
            path, recorder; tile_name = "N46_E010", delta_temperatures = [0.0, 1.0],
            precipitation_scalings = [1.0])
        @test_throws "mass_max must be positive" ColumnProfileRecorder([1150.0], dates; mass_max = 0)
    end

    @testset "the sweep's setting" begin
        repo = normpath(joinpath(@__DIR__, ".."))
        load(env...) = withenv("PHYSICS_MEMBER" => nothing, "BACKGROUND_BLACK_CARBON" => nothing,
                               "BACKGROUND_BLACK_CARBON_OPTICS" => nothing,
                               "EFFECTIVE_LOADING_TABLE" => nothing, env...) do
            m = Module()
            Base.eval(m, :(using Dates))
            Base.include(m, joinpath(repo, "scripts", "run_settings.jl"))
            m
        end
        call(m, name, args...) = Base.invokelatest(() -> getglobal(m, name)(args...))
        off = load("COLUMN_PROFILES" => nothing)
        @test Base.invokelatest(getglobal, off, :COLUMN_PROFILES) === nothing
        intervals = [(; lo = 2000, hi = 2100, center = 2050.0), (; lo = 2100, hi = 2200, center = 2150.0)]
        @test call(off, :tile_column_profiles, "N46_E010", intervals) === nothing
        table = joinpath(mktempdir(), "records.csv")
        write(table, "tile,date,z\nN46_E010,2005-05-01,2120\nN46_E010,2006-04-28,2125\n" *
                     "N46_E010,2007-05-02,2600\nN60_W140,2005-05-01,1500\n")
        on = load("COLUMN_PROFILES" => table)
        r = call(on, :tile_column_profiles, "N46_E010", intervals)
        @test r.band_centers == [2150.0]
        @test r.dates == [DateTime(2005, 5, 1), DateTime(2006, 4, 28)]
        @test call(on, :tile_column_profiles, "N78_E016", intervals) === nothing
        @test_throws "COLUMN_PROFILES=/nonexistent.csv does not exist" load("COLUMN_PROFILES" => "/nonexistent.csv")
        bad = joinpath(mktempdir(), "bad.csv")
        write(bad, "tile,date\nN46_E010,2005-05-01\n")
        @test_throws "lacks column(s) z" load("COLUMN_PROFILES" => bad)
        # The sweep records them through the run's callback, beside the tile file, and rebuilds
        # rather than appends a tile that records them.
        text = read(joinpath(repo, "scripts", "gemb_tile_sweep.jl"), String)
        @test occursin("profiles = tile_column_profiles(name, intervals)", text)
        @test occursin("on_output = profiles", text)
        @test occursin("joinpath(OUTPUT_DIR, \"column_profiles\")", text)
        @test occursin("disposition === :extendable && profiles !== nothing", text)
    end
end
