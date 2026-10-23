# Launch mechanics of the sweep scripts: the run's time range against the cached forcing record, exit
# statuses, claims, relaunch dispositions and file completeness. Included from `runtests.jl`.

import Zarr
import TOML

const _REPO = normpath(joinpath(@__DIR__, ".."))

# A disk cache laid out as `climate_forcing` leaves the production cache. `Zarr.CachingStore` stores
# only the keys that were read, so each variable group holds its consolidated `.zmetadata` and the
# chunks read, and no `.zgroup`, `.zarray` or `.zattrs` files. `climate_forcing` reads the time axis
# from `sfc-2m-temperature` only, so only that group holds time chunks. Time is Int64 hours since
# 1970-01-01 with no fill value, as in the ARCO stores; each group carries its two (lon, lat, time)
# variables. `steps[group]` gives one group other than the temperature group another record length;
# `time_chunk` is the time axis's chunk length.
const _FIXTURE_VARIABLES = Dict("sfc-2m-temperature" => ("t2m", "d2m"),
                                "sfc-pressure-precipitation" => ("sp", "tp"),
                                "sfc-wind" => ("u10", "v10"), "sfc-radiation-heat" => ("ssrd", "strd"))
function _fixture_era5_cache(times; steps = Dict{String,Int}(), time_chunk = 1000)
    hours = [Dates.value(t - DateTime(1970, 1, 1)) ÷ 3_600_000 for t in times]
    cache = mktempdir()
    for (group, variables) in _FIXTURE_VARIABLES
        dir = joinpath(cache, group)
        store = Zarr.DirectoryStore(dir)
        g = Zarr.zgroup(store, "")
        n = group == "sfc-2m-temperature" ? length(hours) : get(steps, group, length(hours))
        t = Zarr.zcreate(Int64, g, "time", n; chunks = (time_chunk,), fill_value = nothing,
                         attrs = Dict("units" => "hours since 1970-01-01"))
        group == "sfc-2m-temperature" && (t[:] = hours)
        for v in variables
            Zarr.zcreate(Float32, g, v, 3, 2, n; chunks = (3, 2, time_chunk), fill_value = NaN32)
        end
        Zarr.consolidate_metadata(store)
        for (root, _, files) in walkdir(dir; topdown = false)
            foreach(f -> f in (".zgroup", ".zarray", ".zattrs") && rm(joinpath(root, f)), files)
            isempty(readdir(root)) && rm(root)
        end
    end
    return cache
end

# The production record: hourly from 1950-01-02T00, where ERA5-Land starts, to 2026-08-14T23, where the
# cache on bylot ends.
const _PRODUCTION_RECORD = collect(DateTime(1950, 1, 2):Hour(1):DateTime(2026, 8, 14, 23))
_production_fixture() = _fixture_era5_cache(_PRODUCTION_RECORD; time_chunk = 100_000)

# Every file under `dir` with its size and modification time.
_tree_state(dir) = sort!([(joinpath(root, f), filesize(joinpath(root, f)), mtime(joinpath(root, f)))
                          for (root, _, files) in walkdir(dir) for f in files])

@testset "run launch: the cached forcing record bounds the run" begin
    # The record runs hourly from 1950-01-02T00, where ERA5-Land starts, to 1951-01-01T05.
    record = collect(DateTime(1950, 1, 2):Hour(1):DateTime(1951, 1, 1, 5))
    cache = _fixture_era5_cache(record)
    @test !isfile(joinpath(cache, "sfc-2m-temperature", ".zgroup"))
    @test !isdir(joinpath(cache, "sfc-wind", "time"))
    before = _tree_state(cache)
    span = era5_land_record_span(cache)
    @test span == (; start = DateTime(1950, 1, 2), stop = DateTime(1951, 1, 1, 5))
    @test _tree_state(cache) == before                           # nothing written to the cache

    # Each end defaults to the cached record's; a year means its 1 January; ISO date-times pass;
    # `record-start` and `record-end` name the record's own ends.
    r = run_time_range(String[], span)
    @test r.time_range == (DateTime(1950, 1, 2), DateTime(1951, 1, 1, 5))
    @test r.source == "start of the cached record, end of the cached record"
    @test run_time_range(["1951"], span).time_range ==
          (DateTime(1951, 1, 1), DateTime(1951, 1, 1, 5))
    @test run_time_range(["record-start", "1951"], span).time_range ==
          (DateTime(1950, 1, 2), DateTime(1951, 1, 1))
    @test run_time_range(["record-start", "1951"], span).source == "start of the cached record, end from ARGS"
    @test run_time_range(["1950-06-01T02:00:00", "record-end"], span).time_range ==
          (DateTime(1950, 6, 1, 2), DateTime(1951, 1, 1, 5))
    # A year whose 1 January precedes the record start is refused, naming the record start.
    @test_throws "the run starts at 1950-01-01T00:00:00, before the cached ERA5-Land record, which starts at 1950-01-02T00:00:00; start at or after it, or pass `record-start`" run_time_range(["1950"], span)
    @test_throws "before the cached ERA5-Land record" run_time_range(["1949", "1951"], span)
    # A run past the record is refused, naming the record, before any tile is claimed.
    @test_throws "reaches outside the cached ERA5-Land record, which runs from 1950-01-02T00:00:00 to 1951-01-01T05:00:00" run_time_range(["record-start", "2027"], span)
    @test_throws "not before its end" run_time_range(["1951-01-01T05:00:00"], span)
    @test_throws "neither a year, an ISO date-time nor `record-start`" run_time_range(["soon"], span)
    @test_throws "neither a year, an ISO date-time nor `record-end`" run_time_range(["1951", "record-start"], span)

    # A cache the loader would refuse is refused here too.
    @test_throws "no ERA5-Land disk cache" era5_land_record_span(joinpath(cache, "absent"))
    rm(joinpath(cache, "sfc-wind"); recursive = true)
    @test_throws "has no consolidated metadata" era5_land_record_span(cache)
    short = _fixture_era5_cache(record; steps = Dict("sfc-wind" => length(record) - 2))
    @test_throws "record different record lengths" era5_land_record_span(short)
    holed = _fixture_era5_cache(record)
    rm(joinpath(holed, "sfc-2m-temperature", "time", "1"))      # the chunk holding steps 1001-2000
    @test_throws "cannot be read in full" era5_land_record_span(holed)
    t0 = DateTime(1950, 1, 2)
    unordered = _fixture_era5_cache([t0, t0 + Hour(1), t0 + Hour(3), t0 + Hour(2)])
    @test_throws "not strictly increasing in whole hours" era5_land_record_span(unordered)

    # The sweep script itself, in a process of its own, on the production record: its settings pass
    # reads the record, defaults to the whole of it, and refuses a start before it or an end after it
    # before anything is claimed.
    climate = mktempdir()
    mv(_production_fixture(), joinpath(climate, "cache"))
    script = joinpath(_REPO, "scripts", "gemb_tile_sweep.jl")
    sweep(args...) = addenv(`$(Base.julia_cmd()) --project=$_REPO --startup-file=no $script $args`,
                            "CLIMATE_CACHE" => climate, "SHOW_SETTINGS" => "1")
    out = String(read(sweep(), String))
    @test occursin("1950-01-02 .. 2026-08-14T23:00", out)
    @test occursin("start of the cached record, end of the cached record", out)
    @test occursin(r"spinup climatology +1951-01-01 \.\. 1980-12-31T23:00", out)
    # A later start moves the default spinup window to the whole years the run fetches.
    out = String(read(sweep("1960"), String))
    @test occursin("1960-01-01 .. 2026-08-14T23:00", out)
    @test occursin(r"spinup climatology +1960-01-01 \.\. 1980-12-31T23:00", out)
    for (cmd, message) in ((sweep("1950"), "before the cached ERA5-Land record"),
                           (sweep("record-start", "2027"), "reaches outside the cached ERA5-Land record"),
                           (addenv(sweep(), "SPINUP_CLIMATOLOGY_START" => "1950"),
                            "the spinup climatology window starts at 1950-01-01T00:00:00, before the " *
                            "cached ERA5-Land record, which starts at 1950-01-02T00:00:00"),
                           (addenv(sweep(), "SPINUP_CLIMATOLOGY_STOP" => "1980-12-31T00:00:00"),
                            "must end at the last step of 31 December of a year (1980-12-31T23:00:00"),
                           (addenv(sweep("1960"), "SPINUP_CLIMATOLOGY_START" => "1955"),
                            "before the run, which starts at 1960-01-01T00:00:00"),
                           (sweep("2000"), "the spinup climatology window defaults to the whole " *
                            "calendar years inside 2000-01-01T00:00:00 .. 1980-12-31T23:00:00"))
        err = IOBuffer()
        @test !success(pipeline(cmd; stdout = devnull, stderr = err))
        @test occursin(message, String(take!(err)))
    end
end

@testset "run launch: the spinup window is the whole calendar years it uses" begin
    # GEMB's climatology keeps only complete calendar years, so the window a run records must name
    # whole years: the production record starts 1950-01-02T00, so its first whole year is 1951.
    record = (; start = DateTime(1950, 1, 2), stop = DateTime(2026, 8, 14, 23))
    full = (record.start, record.stop)
    window(start = nothing, stop = nothing, time_range = full; minimum_years = 3) =
        spinup_window_range(start, stop, record, time_range; last_year = 1980, minimum_years)
    @test window() == (DateTime(1951, 1, 1), DateTime(1980, 12, 31, 23))
    # The default follows the run: its own whole years inside the record and through 1980.
    @test window(nothing, nothing, (DateTime(1960, 1, 1), record.stop)) ==
          (DateTime(1960, 1, 1), DateTime(1980, 12, 31, 23))
    @test window(nothing, nothing, (DateTime(1960, 6, 1), record.stop)) ==
          (DateTime(1961, 1, 1), DateTime(1980, 12, 31, 23))
    @test window(nothing, nothing, (record.start, DateTime(1970, 6, 30))) ==
          (DateTime(1951, 1, 1), DateTime(1969, 12, 31, 23))
    @test window(nothing, nothing, (DateTime(1960, 1, 1), DateTime(1964, 6, 30))) ==
          (DateTime(1960, 1, 1), DateTime(1963, 12, 31, 23))
    @test_throws "the spinup climatology window defaults to the whole calendar years inside 2000-01-01T00:00:00 .. 1980-12-31T23:00:00 (the cached ERA5-Land record and the run, through 31 December 1980), and that interval holds none" window(
        nothing, nothing, (DateTime(2000, 1, 1), record.stop))
    @test_throws "inside 1950-01-02T00:00:00 .. 1950-12-31T00:00:00" window(
        nothing, nothing, (record.start, DateTime(1950, 12, 31)))
    # Explicit ends: years, or ISO date-times on a calendar-year boundary.
    @test window("1960", "1970") == (DateTime(1960, 1, 1), DateTime(1970, 12, 31, 23))
    @test window("1951-01-01T00:00:00", "1980-12-31T23:00:00") ==
          (DateTime(1951, 1, 1), DateTime(1980, 12, 31, 23))
    @test window("1960") == (DateTime(1960, 1, 1), DateTime(1980, 12, 31, 23))
    @test_throws "must start on 1 January, 00:00 of a year, not at 1950-01-02T00:00:00" window(
        "1950-01-02T00:00:00")
    @test_throws "must end at the last step of 31 December of a year (1980-12-31T23:00:00 for this 1 hour forcing), not at 1980-12-31T00:00:00" window(
        nothing, "1980-12-31T00:00:00")
    # Outside the record or the run, naming the bound.
    @test_throws "the spinup climatology window starts at 1950-01-01T00:00:00, before the cached ERA5-Land record, which starts at 1950-01-02T00:00:00" window("1950")
    @test_throws "ends at 2026-12-31T23:00:00, after the cached ERA5-Land record, which ends at 2026-08-14T23:00:00" window(
        "2000", "2026")
    @test_throws "starts at 1955-01-01T00:00:00, before the run, which starts at 1960-01-01T00:00:00" window(
        "1955", nothing, (DateTime(1960, 1, 1), record.stop))
    @test_throws "ends at 1975-12-31T23:00:00, after the run, which ends at 1970-06-30T00:00:00" window(
        nothing, "1975", (record.start, DateTime(1970, 6, 30)))
    @test_throws "holds 2 whole calendar year(s); the spinup climatology needs at least 3" window(
        "1951", "1952")
    @test_throws "not before its end" window("1960", "1959")
    @test_throws "neither a year nor an ISO date-time" window(nothing, "soon")

    # The native run's window is computed by the same call as the sweep's, and its reference check
    # accepts the pair through a tile file the sweep wrote. The previous default window does not match.
    settings = Module()
    Base.eval(settings, :(using Dates))
    Base.include(settings, joinpath(_REPO, "scripts", "run_settings.jl"))
    for script in ("gemb_tile_sweep.jl", "gemb_native_run.jl")
        @test occursin("last_year = SPINUP_CLIMATOLOGY_LAST_YEAR",
                       read(joinpath(_REPO, "scripts", script), String))
    end
    sweep_window = spinup_window_range(nothing, nothing, record, full;
                                       last_year = settings.SPINUP_CLIMATOLOGY_LAST_YEAR)
    native_window = spinup_window_range(nothing, nothing, record, full;
                                        last_year = settings.SPINUP_CLIMATOLOGY_LAST_YEAR)
    as_parameters(w) = Dict{String,Any}("spinup_window_start" => string(w[1]),
                                        "spinup_window_stop" => string(w[2]))
    run = _fake_tile_run()
    merge!(run.parameters, as_parameters(sweep_window))
    reference = read_glacier_tile_status(write_glacier_tile_netcdf(joinpath(mktempdir(), "t.nc"), run))
    @test reference.parameters["spinup_window_start"] == "1951-01-01T00:00:00"
    @test reference.parameters["spinup_window_stop"] == "1980-12-31T23:00:00"
    @test isempty(run_parameter_differences(reference.parameters,
                                            merge(run.parameters, as_parameters(native_window))))
    @test !isempty(run_parameter_differences(reference.parameters, merge(run.parameters,
        as_parameters((DateTime(1950, 1, 2), DateTime(1980, 12, 31))))))
end

# Whether parsing produced an error or an incomplete expression anywhere.
_parse_failed(ex) = ex isa Expr && (ex.head in (:error, :incomplete) || any(_parse_failed, ex.args))

# The right-hand side of the first assignment to `name` (`const` or not) anywhere in `ex`.
function _assigned(ex, name)
    ex isa Expr || return nothing
    ex.head === :(=) && ex.args[1] === name && return ex.args[2]
    for a in ex.args
        found = _assigned(a, name)
        found === nothing || return found
    end
    return nothing
end

@testset "run launch: example scripts request forcing from the cached record's start" begin
    # Each script computes its forcing request from the cached record, as the run scripts do. The
    # expression it assigns is evaluated against the production-shaped fixture and must pass the
    # loader's own checks of a request against a cached record.
    cache = _production_fixture()
    epoch = DateTime(1970, 1, 1)
    hours = [Dates.value(t - epoch) ÷ 3_600_000 for t in _PRODUCTION_RECORD]
    for (file, cache_name, range_name, stop) in
            (("scripts/alaska_sweep_timing.jl", :FORCING_CACHE, :FORCING_TIME_RANGE, DateTime(2026, 8, 1)),
             ("scripts/check_threaded_equivalence.jl", :FORCING_CACHE, :FORCING_TIME_RANGE, DateTime(1985, 1, 1)),
             ("src/era5_example.jl", :forcing_cache, :forcing_time_range, DateTime(2026, 8, 1)))
        source = read(joinpath(_REPO, file), String)
        ast = Meta.parseall(source; filename = file)
        @test !_parse_failed(ast)
        @test !occursin("DateTime(1950, 1, 1)", source)
        rhs = _assigned(ast, range_name)
        @test rhs !== nothing
        sandbox = Module()
        Base.eval(sandbox, :(using GEMB_GlacierSims, Dates))
        Base.eval(sandbox, :(const $cache_name = $cache))
        request = Base.eval(sandbox, rhs)
        @test request == (DateTime(1950, 1, 2), stop)
        @test GEMB_ClimateForcing._record_slice(hours, epoch, request) ==
              1:findlast(<=(Dates.value(stop - epoch) ÷ 3_600_000), hours)
        @test GEMB_ClimateForcing._check_cached_record(cache, hours, epoch, request, ()) === nothing
    end
end

# A stand-in for `julia` that a launcher can drive without loading anything: the settings pass
# succeeds, `-e` prints `STUB_INVARIANT_TARGET` (the invariant path a launcher asks for), and each
# worker writes the log line its block is told to (`STUB_FAIL_BLOCK` writes a tile failure) and exits
# with `STUB_EXIT_<block>`, default 0. Every call other than `-e` appends its arguments as one line to
# `STUB_ARGS_LOG` when that is set.
function _stub_julia(dir)
    path = joinpath(dir, "julia")
    write(path, """
        #!/usr/bin/env bash
        for a in "\$@"; do
            [ "\$a" = "-e" ] && { printf '%s' "\$STUB_INVARIANT_TARGET"; exit 0; }
        done
        [ -n "\${STUB_ARGS_LOG:-}" ] && echo "\$*" >> "\$STUB_ARGS_LOG"
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
            calls = joinpath(mktempdir(), "calls")
            env = ("JULIA" => _stub_julia(mktempdir()), "CLIMATE_CACHE" => climate,
                   "TILE_BLOCKS" => "1", "NATIVE_BLOCKS" => "1", "STUB_INVARIANT_TARGET" => target,
                   "OUTPUT_DIR" => joinpath(climate, "out"), "STUB_ARGS_LOG" => calls)
            launch() = addenv(`bash $(joinpath(_REPO, "scripts", launcher))`, env...)
            # The preserved copy is put in place, once, before any worker starts.
            @test success(pipeline(launch(); stdout = devnull, stderr = devnull))
            @test read(target, String) == "preserved"
            @test !isfile(target * ".part")
            # Without arguments the settings pass and the worker start at the cached record's start.
            @test length(readlines(calls)) == 2
            @test all(endswith(" record-start"), readlines(calls))
            # With no preserved copy and a download that produces nothing, nothing is launched.
            rm(target)
            rm(joinpath(climate, "invariant"); recursive = true)
            err = IOBuffer()
            @test !success(pipeline(launch(); stdout = devnull, stderr = err))
            @test occursin("nothing launched", String(take!(err)))
        end
    end
end

@testset "run launch: native corrections come from the reference member and its sweep" begin
    # The reference member GlobalGlacierAnalysis.jl declares, read from its source.
    gga = mktempdir()
    mkpath(joinpath(gga, "src"))
    binned = mktempdir()
    member = joinpath(binned, "glacier_rgi7_dh_cop30_v2_cc_nmad5_v01_filled_ac_p6_aligned.jld2")
    write(joinpath(gga, "src", "utilities_project.jl"), """
        const δice = 917;
        const reference_ensemble_file = "$member";
        function project_products(; project_id = :v01) end
        """)
    @test gga_reference_ensemble_file(gga) == member
    # Its `gemb_calibration` output, named as GlobalGlacierAnalysis.jl names it.
    @test_throws "no `gemb_calibration` output for the reference ensemble member" reference_forcing_corrections(member)
    fit_name = "glacier_rgi7_dh_cop30_v2_cc_nmad5_v01_filled_ac_p6_synthesized_gembfit.arrow"
    beside = touch(joinpath(binned, fit_name))
    @test reference_forcing_corrections(member) == beside
    # A physics member's fit is in that member's folder beside it, under the same name.
    @test_throws "no `gemb_calibration` output" reference_forcing_corrections(
        member; physics_member = "B-kfit-table")
    fit = joinpath(binned, "gemb_physics", "B-kfit-table", fit_name)
    mkpath(dirname(fit))
    touch(fit)
    @test reference_forcing_corrections(member; physics_member = "B-kfit-table") == fit
    @test_throws "is not an `…_aligned.jld2` file" reference_forcing_corrections(joinpath(binned, "x.jld2"))
    @test_throws "no GlobalGlacierAnalysis.jl source" gga_reference_ensemble_file(mktempdir())
    write(joinpath(gga, "src", "utilities_project.jl"), "const reference_ensemble_file = joinpath(\"a\", \"b\")\n")
    @test_throws "not a string literal" gga_reference_ensemble_file(gga)
    write(joinpath(gga, "src", "utilities_project.jl"), "const x = 1\n")
    @test_throws "declares `reference_ensemble_file` 0 times" gga_reference_ensemble_file(gga)

    # The provenance of the fit: the directory of tile runs it read, and their source trees.
    sweep = mktempdir()
    other_sweep = mktempdir()
    trees = Dict("software_gemb_tree" => "a"^40, "software_gemb_climateforcing_tree" => "b"^40,
                 "software_gemb_glaciersims_tree" => "c"^40)
    reference_parameters = Dict{String,Any}(trees..., "software_gemb_version" => "2.0.0",
                                            "model_albedo_ice" => 0.48)
    provenance = forcing_corrections_provenance_path(fit)
    @test provenance == replace(fit, ".arrow" => "_provenance.toml")
    record(dir, parameters) = open(provenance, "w") do io
        TOML.print(io, Dict("tile_dir" => dir, "tile_run_parameters" => parameters))
    end
    @test_throws "carry no tile-run provenance" assert_corrections_fitted_on(fit, sweep)
    record(sweep, Dict(trees..., "software_gemb_version" => "2.0.0"))
    @test assert_corrections_fitted_on(fit, sweep) === nothing
    @test assert_corrections_fitted_on(fit, sweep, reference_parameters) === nothing
    # The same directory through a link is the same sweep.
    link = joinpath(mktempdir(), "link")
    symlink(sweep, link)
    @test assert_corrections_fitted_on(fit, link, reference_parameters) === nothing
    # Fitted on another directory of tile runs.
    @test_throws "not on the reference sweep" assert_corrections_fitted_on(fit, other_sweep)
    # Fitted on tile runs from other code.
    @test_throws "software_gemb_tree: fitted on" assert_corrections_fitted_on(
        fit, sweep, merge(reference_parameters, Dict("software_gemb_tree" => "d"^40)))
    @test_throws "reference nothing" assert_corrections_fitted_on(
        fit, sweep, delete!(copy(reference_parameters), "software_gemb_version"))
    # A reference sweep run as a physics member needs a fit that records the member.
    members = Dict("physics_member" => "B-kfit-table", "physics_grain" => "B",
                   "physics_band_decoupling" => "kfit", "physics_impurity" => "table")
    @test_throws "was run as physics member \"B-kfit-table\", but $provenance does not record physics_band_decoupling, physics_grain, physics_impurity, physics_member" assert_corrections_fitted_on(
        fit, sweep, merge(reference_parameters, members))
    record(sweep, Dict(trees..., "software_gemb_version" => "2.0.0", members...))
    @test assert_corrections_fitted_on(fit, sweep, merge(reference_parameters, members)) === nothing
    @test_throws "physics_member: fitted on \"B-kfit-table\", reference \"C-ks35-table\"" assert_corrections_fitted_on(
        fit, sweep, merge(reference_parameters, members, Dict("physics_member" => "C-ks35-table")))
    # A record without the source trees cannot show which code the fit saw.
    record(sweep, Dict("software_gemb_version" => "2.0.0"))
    @test_throws "does not record software_gemb_tree" assert_corrections_fitted_on(fit, sweep)

    # The tile sweep writes, and the native run reads, the same directory by default: the physics
    # member's under `TILE_RUN_DIR`. The launcher agrees with the scripts.
    settings = Module()
    Base.eval(settings, :(using Dates))
    withenv("PHYSICS_MEMBER" => nothing) do
        Base.include(settings, joinpath(_REPO, "scripts", "run_settings.jl"))
    end
    @test settings.TILE_RUN_DIR == "tile_runs_v2"
    for script in ("gemb_tile_sweep.jl", "gemb_native_run.jl")
        @test occursin("tile_run_dir(CLIMATE_CACHE)",
                       read(joinpath(_REPO, "scripts", script), String))
    end
    @test occursin("OUTPUT_DIR=\"\${OUTPUT_DIR:-\$CLIMATE_CACHE/$(settings.TILE_RUN_DIR)/\$PHYSICS_MEMBER}\"",
                   read(joinpath(_REPO, "scripts", "run_tile_sweep.sh"), String))

    # The native script's settings pass takes the corrections from the reference member, and stops when
    # the member's GlobalGlacierAnalysis.jl checkout cannot be found.
    write(joinpath(gga, "src", "utilities_project.jl"), "const reference_ensemble_file = \"$member\"\n")
    climate = mktempdir()
    mv(_production_fixture(), joinpath(climate, "cache"))
    native(gga_dir) = addenv(`$(Base.julia_cmd()) --project=$_REPO --startup-file=no
                              $(joinpath(_REPO, "scripts", "gemb_native_run.jl"))`,
                             "CLIMATE_CACHE" => climate, "SHOW_SETTINGS" => "1", "GGA_PROJECT" => gga_dir)
    out = read(native(gga), String)
    @test occursin("1950-01-02 .. 2026-08-14T23:00", out)       # the whole cached record
    @test occursin(r"spinup climatology +1951-01-01 \.\. 1980-12-31T23:00", out)
    @test occursin(fit, out)
    @test occursin(joinpath(climate, "tile_runs_v2", "B-kfit-table"), out)
    @test occursin(r"physics member +B-kfit-table", out)
    err = IOBuffer()
    @test !success(pipeline(native(mktempdir()); stdout = devnull, stderr = err))
    @test occursin("no GlobalGlacierAnalysis.jl source", String(take!(err)))
end

@testset "run launch: a native tile over the lost-ice rule fails every launch" begin
    if Sys.which("bash") === nothing
        @test_skip "bash is not available"
    else
        # A tile whose native run left more than 1 % of its ice unmodelled keeps its
        # `.lost_ice_exceeded` file, so a launch in which no worker ran it still exits non-zero.
        climate = mktempdir()
        mkpath(joinpath(climate, "downscaling_parameters"))
        mkpath(joinpath(climate, "invariant", "era5_land"))
        write(joinpath(climate, "invariant", "era5_land", "geo_1279l4_0.1x0.1.grib2_v4_unpack.nc"), "x")
        output = joinpath(climate, "native_runs")
        env = ("JULIA" => _stub_julia(mktempdir()), "CLIMATE_CACHE" => climate, "NATIVE_BLOCKS" => "1",
               "STUB_INVARIANT_TARGET" => joinpath(mktempdir(), "geo_1279l4_0.1x0.1.grib2_v4_unpack.nc"),
               "OUTPUT_DIR" => output)
        launch() = addenv(`bash $(joinpath(_REPO, "scripts", "run_native_sweep.sh"))`, env...)
        @test success(pipeline(launch(); stdout = devnull, stderr = devnull))
        mkpath(joinpath(output, "N60_W142"))
        write(joinpath(output, "N60_W142", "N60_W142.lost_ice_exceeded"), "message")
        err = IOBuffer()
        @test !success(pipeline(launch(); stdout = devnull, stderr = err))
        @test occursin("1 tile(s) over the lost-ice rule", String(take!(err)))
    end
end
