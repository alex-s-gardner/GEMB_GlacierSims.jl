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
# succeeds, `-e` prints the package directory a launcher asks for, and each worker writes the log
# line its block is told to (`STUB_FAIL_BLOCK` writes a tile failure) and exits with
# `STUB_EXIT_<block>`, default 0.
function _stub_julia(dir)
    path = joinpath(dir, "julia")
    write(path, """
        #!/usr/bin/env bash
        for a in "\$@"; do
            [ "\$a" = "-e" ] && { printf '%s' "\$STUB_PKGDIR"; exit 0; }
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
        pkgdir_stub = mktempdir()
        invariant = joinpath(climate, "invariant", "era5_land")
        mkpath(invariant)
        write(joinpath(invariant, "geo_1279l4_0.1x0.1.grib2_v4_unpack.nc"), "stub")
        env = ("JULIA" => _stub_julia(dir), "CLIMATE_CACHE" => climate, "TILE_BLOCKS" => "2",
               "STUB_PKGDIR" => pkgdir_stub, "OUTPUT_DIR" => joinpath(climate, "out"))
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
