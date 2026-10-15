# Launch mechanics of the sweep scripts: the run's time range against the cached forcing record, exit
# statuses, claims, relaunch dispositions and file completeness. Included from `runtests.jl`.

import Zarr

const _REPO = normpath(joinpath(@__DIR__, ".."))

# A disk cache shaped like the one `climate_forcing` fills: one consolidated Zarr group per ERA5-Land
# variable group, each with an hourly `time` axis in hours since 1950-01-01 and a (lon, lat, time)
# variable. `hours[group]` overrides one group's axis.
function _fixture_era5_cache(hours; overrides = Dict{String,Vector{Float64}}())
    cache = mktempdir()
    for group in keys(GEMB_ClimateForcing.ERA5_LAND_STORES)
        h = get(overrides, group, hours)
        store = Zarr.DirectoryStore(joinpath(cache, group))
        g = Zarr.zgroup(store, "")
        t = Zarr.zcreate(Float64, g, "time", length(h); chunks = (4,),
                         attrs = Dict("units" => "hours since 1950-01-01"))
        t[:] = h
        Zarr.zcreate(Float32, g, "var", 3, 2, length(h); chunks = (3, 2, 4))
        Zarr.consolidate_metadata(store)
    end
    return cache
end

@testset "run launch: the cached forcing record bounds the run" begin
    # 0 .. 9 hours after the epoch: the record runs 1950-01-01T00 to 1950-01-01T09.
    cache = _fixture_era5_cache(collect(0.0:9.0))
    span = era5_land_record_span(cache)
    @test span == (; start = DateTime(1950, 1, 1), stop = DateTime(1950, 1, 1, 9))

    # The end defaults to the last cached step; a year means its 1 January; ISO date-times pass.
    r = run_time_range(String[], span)
    @test r.time_range == (DateTime(1950, 1, 1), DateTime(1950, 1, 1, 9))
    @test r.source == "start default, end of the cached record"
    @test run_time_range(["1950", "1950-01-01T05:00:00"], span).time_range ==
          (DateTime(1950, 1, 1), DateTime(1950, 1, 1, 5))
    @test run_time_range(["1950-01-01T02:00:00"], span).time_range ==
          (DateTime(1950, 1, 1, 2), DateTime(1950, 1, 1, 9))
    # A run past the record is refused, naming the record, before any tile is claimed.
    @test_throws "reaches outside the cached ERA5-Land record, which runs from 1950-01-01T00:00:00 to 1950-01-01T09:00:00" run_time_range(["1950", "2027"], span)
    @test_throws "reaches outside" run_time_range(["1949"], span)
    @test_throws "not before its end" run_time_range(["1950-01-01T09:00:00"], span)
    @test_throws "neither a year nor an ISO date-time" run_time_range(["1950", "soon"], span)

    # A cache the loader would refuse is refused here too.
    @test_throws "no ERA5-Land disk cache" era5_land_record_span(joinpath(cache, "absent"))
    rm(joinpath(cache, "sfc-wind"); recursive = true)
    @test_throws "has no consolidated metadata" era5_land_record_span(cache)
    short = _fixture_era5_cache(collect(0.0:9.0);
                                overrides = Dict("sfc-wind" => collect(0.0:7.0)))
    @test_throws "disagree on the record" era5_land_record_span(short)
    holed = _fixture_era5_cache(collect(0.0:9.0))
    rm(joinpath(holed, "sfc-radiation-heat", "time", "1"))       # the chunk holding hours 4-7
    @test_throws "cannot be read in full" era5_land_record_span(holed)
    unordered = _fixture_era5_cache([0.0, 1.0, 3.0, 2.0])
    @test_throws "not strictly increasing in whole hours" era5_land_record_span(unordered)

    # The sweep script itself, in a process of its own: its settings pass reads the record, and a
    # request past it stops the script before anything is claimed. The record ends at 1950-01-01T09.
    climate = mktempdir()
    mv(_fixture_era5_cache(collect(0.0:9.0)), joinpath(climate, "cache"))
    script = joinpath(_REPO, "scripts", "gemb_tile_sweep.jl")
    sweep(args...) = addenv(`$(Base.julia_cmd()) --project=$_REPO --startup-file=no $script $args`,
                            "CLIMATE_CACHE" => climate, "SHOW_SETTINGS" => "1")
    out = IOBuffer()
    @test success(pipeline(sweep("1950"); stdout = out, stderr = devnull))
    @test occursin("1950-01-01 .. 1950-01-01T09:00", String(take!(out)))
    @test occursin("end of the cached record", String(read(sweep(), String)))
    err = IOBuffer()
    @test !success(pipeline(sweep("1950", "2027"); stdout = devnull, stderr = err))
    @test occursin("reaches outside the cached ERA5-Land record", String(take!(err)))
end

# A stand-in for `julia` that a launcher can drive without loading anything: the settings pass
# succeeds, `-e` prints `STUB_INVARIANT_TARGET` (the invariant path a launcher asks for), and each
# worker writes the log line its block is told to (`STUB_FAIL_BLOCK` writes a tile failure) and exits
# with `STUB_EXIT_<block>`, default 0.
function _stub_julia(dir)
    path = joinpath(dir, "julia")
    write(path, """
        #!/usr/bin/env bash
        for a in "\$@"; do
            [ "\$a" = "-e" ] && { printf '%s' "\$STUB_INVARIANT_TARGET"; exit 0; }
        done
        [ "\${SHOW_SETTINGS:-0}" = "1" ] && exit 0
        block="\${TILE_BLOCK:-\${NATIVE_BLOCK:-1}}"
        if [ "\$block" = "\${STUB_FAIL_BLOCK:-0}" ]; then
            echo "┌ Error: Tile failed; continuing"
        else
            echo "┌ Info: Wrote tile"
        fi
        code_var="STUB_EXIT_\$block"
        exit "\${!code_var:-0}"
        """)
    chmod(path, 0o755)
    return path
end

@testset "run launch: a launcher with a failing tile exits non-zero" begin
    if Sys.which("bash") === nothing
        @test_skip "bash is not available"
    else
        dir = mktempdir()
        climate = mktempdir()
        mkpath(joinpath(climate, "downscaling_parameters"))
        invariant = joinpath(climate, "invariant", "era5_land")
        mkpath(invariant)
        write(joinpath(invariant, "geo_1279l4_0.1x0.1.grib2_v4_unpack.nc"), "stub")
        target = joinpath(mktempdir(), "pkg", "data", "invariant", "era5_land",
                          "geo_1279l4_0.1x0.1.grib2_v4_unpack.nc")
        env = ("JULIA" => _stub_julia(dir), "CLIMATE_CACHE" => climate, "TILE_BLOCKS" => "2",
               "STUB_INVARIANT_TARGET" => target, "OUTPUT_DIR" => joinpath(climate, "out"))
        launch(extra...) = addenv(`bash $(joinpath(_REPO, "scripts", "run_tile_sweep.sh"))`,
                                  env..., extra...)
        run_quiet(cmd) = (err = IOBuffer(); ok = success(pipeline(cmd; stdout = devnull,
                                                                  stderr = err));
                          (ok, String(take!(err))))
        @test first(run_quiet(launch()))
        # Block 2 fails one tile and, as a worker does, exits 1: the launcher exits 1 with the count.
        ok, err = run_quiet(launch("STUB_FAIL_BLOCK" => "2", "STUB_EXIT_2" => "1"))
        @test !ok
        @test occursin("1 tile(s) failed, 1 block(s) exited non-zero", err)
        # A failed tile in the logs fails the launch even if every block exited 0.
        ok, err = run_quiet(launch("STUB_FAIL_BLOCK" => "1"))
        @test !ok
        @test occursin("1 tile(s) failed, 0 block(s) exited non-zero", err)
    end
end

@testset "run launch: a heartbeat keeps a long-running claim" begin
    claims = mktempdir()
    output = joinpath(mktempdir(), "N00_E000.nc")              # never written: the tile is running
    # Without a heartbeat the claim would be stale 2 s in, before the owner's 3 s of work end.
    stale = Millisecond(2000)
    @test claim_tile!(claims, "N00_E000.nc", output; stale_after = stale)
    beat = joinpath(claims, "N00_E000.nc", "heartbeat")
    hb = start_claim_heartbeat(claims, "N00_E000.nc"; interval = Millisecond(100))
    # The owner computes for 3 s without yielding to any other task, as a tile run does, while a
    # second worker tries to take the tile. The heartbeat keeps advancing, so the tile stays claimed.
    seen = Set{Float64}()
    stolen = false
    t_end = time() + 3.0
    x = 0.0
    while time() < t_end
        x += sum(rand(1000))
        push!(seen, mtime(beat))
        stolen |= claim_tile!(claims, "N00_E000.nc", output; stale_after = stale)
    end
    # About 30 beats in 3 s at a 0.1 s interval; a heartbeat held up by the computation would show one.
    @test length(seen) >= 10
    @test !stolen
    # The owner stops beating, as a crashed owner does: the claim holds until its heartbeat is older
    # than the bound, and is reclaimed after.
    stop_claim_heartbeat!(hb)
    @test !claim_tile!(claims, "N00_E000.nc", output; stale_after = stale)
    sleep(2.5)
    @test claim_tile!(claims, "N00_E000.nc", output; stale_after = stale)

    # The heartbeat process ends with its owner: one started for a process that has already exited
    # stops on its own, and stopping it then reports that the claim was not kept alive.
    owner = run(`sleep 0.2`; wait = false)
    orphan_claims = mktempdir()
    @test claim_tile!(orphan_claims, "t", output)
    orphan = start_claim_heartbeat(orphan_claims, "t"; interval = Millisecond(100),
                                   owner = getpid(owner))
    wait(owner)
    sleep(0.5)
    @test !process_running(orphan.process)
    @test_throws "stopped while its owner ran" stop_claim_heartbeat!(orphan)
    @test with_claim_heartbeat(() -> 42, claims, "N00_E000.nc"; interval = Millisecond(100)) == 42

    # A claim with no heartbeat file is judged by the age of its directory.
    claims = mktempdir()
    mkdir(joinpath(claims, "t"))
    @test !claim_tile!(claims, "t", "/nonexistent"; stale_after = Hour(1))
    sleep(0.05)
    @test claim_tile!(claims, "t", "/nonexistent"; stale_after = Millisecond(10))
    @test isfile(joinpath(claims, "t", "heartbeat"))        # the reclaimer's own heartbeat
    # A finished tile is never reclaimed, and reclaiming can be turned off.
    done = touch(joinpath(mktempdir(), "done.nc"))
    @test !claim_tile!(claims, "t", done; stale_after = Millisecond(1))
    @test !claim_tile!(claims, "t", "/nonexistent"; stale_after = Hour(0))
    @test_throws "no claim for absent" start_claim_heartbeat(claims, "absent")
end

@testset "run launch: relaunch disposition" begin
    # Spec 10. A tile whose hypsometry holds ice at 1050, 1550 and 2050 m, whose file stores the first
    # two bands and records 2050 m as dropped: the file accounts for every interval, so a relaunch with
    # the same settings skips it.
    run = _fake_tile_run(; bands = [(1050, 12.5), (1550, 40.25)])
    merge!(run.provenance, Dict{String,Any}("dropped_band_centers" => [2050.0],
                                            "dropped_band_areas" => [0.5],
                                            "dropped_band_causes" => [2.0],
                                            "hypsometry_area_km2" => 53.25,
                                            "lost_area_km2" => 0.5))
    path = write_glacier_tile_netcdf(joinpath(mktempdir(), "t.nc"), run)
    status = read_glacier_tile_status(path)
    intervals = [(; center = c) for c in (1050.0, 1550.0, 2050.0)]
    grid = (; delta_temperatures = run.delta_temperatures,
            precipitation_scalings = run.precipitation_scalings, output_frequency = :monthly)
    disposition(s = status; requested = run.parameters, run_time = run.time, ivs = intervals, kw...) =
        tile_run_disposition(s, requested, run_time, ivs; grid..., kw...)
    @test disposition() === :current
    # The record has grown by more than one output period: append.
    @test disposition(; run_time = [run.time; last(run.time) + Day(40)]) === :extendable
    # An interval holding ice that the file neither stores nor records as dropped: a different
    # hypsometry, so the file is rebuilt.
    @test disposition(; ivs = [intervals; (; center = 2550.0)]) === :rebuild
    # A stored band the request does not hold ice in.
    @test disposition(; ivs = intervals[2:3]) === :rebuild
    # A different experiment.
    @test disposition(; requested = merge(run.parameters, Dict{String,Any}("model_albedo_ice" => 0.5))) ===
          :rebuild
    @test tile_run_disposition(status, run.parameters, run.time, intervals; grid...,
                               delta_temperatures = [0.0]) === :rebuild
    # A loss above the rule is rebuilt (and refused again), and so is a file that does not record its
    # accounting at all.
    @test disposition(; lost_area_fraction_maximum = 0.005) === :rebuild
    @test disposition(merge(status, (; lost_area_km2 = missing))) === :rebuild
    @test disposition(nothing) === :rebuild
    @test output_period_bound(:weekly) == Day(7)
    @test output_period_bound(:all) === nothing
    @test_throws "no output period is known for output_frequency = :hourly" output_period_bound(:hourly)
end

@testset "run launch: partial files are rebuilt" begin
    # Spec 11. Every writer builds the file beside its path and renames it into place when whole, with
    # the `file_complete` mark set last. A truncated file, or one without the mark, is rebuilt; it does
    # not throw before `FORCE` is consulted and is never taken as current.
    run = _fake_tile_run()
    merge!(run.provenance, Dict{String,Any}("hypsometry_area_km2" => 55.75, "lost_area_km2" => 0.0))
    intervals = [(; center = b.center) for b in run.bands]
    grid = (; delta_temperatures = run.delta_temperatures,
            precipitation_scalings = run.precipitation_scalings, output_frequency = :monthly)
    disposition(path) = tile_run_disposition(read_glacier_tile_status(path), run.parameters,
                                             run.time, intervals; grid...)
    dir = mktempdir()
    path = write_glacier_tile_netcdf(joinpath(dir, "t.nc"), run)
    @test readdir(dir) == ["t.nc"]                         # no temporary file left behind
    @test read_glacier_tile_status(path).complete
    @test disposition(path) === :current

    truncated = joinpath(dir, "truncated.nc")
    write(truncated, read(path)[1:(filesize(path) ÷ 2)])
    status = read_glacier_tile_status(truncated)
    @test !status.complete
    @test startswith(status.reason, "unreadable")
    @test disposition(truncated) === :rebuild

    unmarked = cp(path, joinpath(dir, "unmarked.nc"))
    NCDatasets.NCDataset(ds -> delete!(ds.attrib, "file_complete"), unmarked, "a")
    @test read_glacier_tile_status(unmarked) == (; complete = false, reason = "no file_complete mark")
    @test disposition(unmarked) === :rebuild

    # An append that fails part-way, after the time axis and the series were extended, leaves the file
    # as it was: here the continuation's columns are deeper than the file's layer dimension.
    deeper = _fake_tile_run(; time = [last(run.time) + Month(1)])
    deeper.profiles[1, 1, 1] = _fake_profile(50, 1.0)
    before = read(path)
    @test_throws "continuation profiles are deeper" append_glacier_tile_netcdf(path, deeper)
    @test read(path) == before
    @test !isfile(path * ".part")
    # And a write that fails part-way leaves the previous file in place.
    broken = _fake_tile_run()
    broken.profiles[1, 1, 1] = DimStack((dz = DimArray([1.0], (Z(1:1),)),))  # lacks most state layers
    @test_throws Exception write_glacier_tile_netcdf(path, broken)
    @test read(path) == before
    @test !isfile(path * ".part")
    @test disposition(path) === :current
    # A successful append keeps the mark.
    later = _fake_tile_run(; time = [last(run.time) + Month(1)])
    append_glacier_tile_netcdf(path, later)
    @test read_glacier_tile_status(path).complete
    @test read_glacier_tile_status(path).n_timesteps == length(run.time) + 1
end

@testset "run launch: every launcher restores the invariant cache" begin
    if Sys.which("bash") === nothing
        @test_skip "bash is not available"
    else
        file = "geo_1279l4_0.1x0.1.grib2_v4_unpack.nc"
        for launcher in ("run_tile_sweep.sh", "run_native_sweep.sh", "run_derive_parameters.sh")
            climate = mktempdir()
            mkpath(joinpath(climate, "downscaling_parameters"))
            mkpath(joinpath(climate, "invariant", "era5_land"))
            write(joinpath(climate, "invariant", "era5_land", file), "preserved")
            target = joinpath(mktempdir(), "data", "invariant", "era5_land", file)
            env = ("JULIA" => _stub_julia(mktempdir()), "CLIMATE_CACHE" => climate,
                   "TILE_BLOCKS" => "1", "NATIVE_BLOCKS" => "1", "STUB_INVARIANT_TARGET" => target,
                   "OUTPUT_DIR" => joinpath(climate, "out"))
            launch() = addenv(`bash $(joinpath(_REPO, "scripts", launcher))`, env...)
            # The preserved copy is put in place, once, before any worker starts.
            @test success(pipeline(launch(); stdout = devnull, stderr = devnull))
            @test read(target, String) == "preserved"
            @test !isfile(target * ".part")
            # With no preserved copy and a download that produces nothing, nothing is launched.
            rm(target)
            rm(joinpath(climate, "invariant"); recursive = true)
            err = IOBuffer()
            @test !success(pipeline(launch(); stdout = devnull, stderr = err))
            @test occursin("nothing launched", String(take!(err)))
        end
    end
end
