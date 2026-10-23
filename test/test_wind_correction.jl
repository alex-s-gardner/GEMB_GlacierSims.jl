# The slope-dependent wind correction: the function and its coefficients, the band slope read from the
# elevation-class table, its application to the band forcing, the `ws` physics member, and the run
# parameters that record it. Included from `runtests.jl` after `test_physics_members.jl`; uses
# `_cp_table` from `runtests.jl`, `_audit_forcing`, `_flat_applied` from `test_pipeline_audit.jl`.

@testset "slope wind correction" begin
    @testset "the function" begin
        c = SLOPE_WIND_COEFFICIENTS
        @test c == (; m0 = 0.398, m1 = 0.168, theta_floor = 0.1, u_min_a = 0.0, u_min_b = 0.0)
        m(θ) = exp(0.398) * max(θ, 0.1)^0.168
        for θ in (0.0, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
            @test slope_wind_multiplier(θ) == m(θ)
        end
        # The published anchor values: 1.0 at 0.1°, 1.5 at 1°, 2.2 at 10°.
        @test round(slope_wind_multiplier(0.1); digits = 2) == 1.01
        @test round(slope_wind_multiplier(1.0); digits = 2) == 1.49
        @test round(slope_wind_multiplier(10.0); digits = 2) == 2.19
        # Below the floor the slope makes no difference.
        @test slope_wind_multiplier(0.0) == slope_wind_multiplier(0.1) == slope_wind_multiplier(0.03)
        @test issorted([slope_wind_multiplier(θ) for θ in 0.1:0.1:20])
        # With the fitted coefficients the floor is zero, so the correction is the multiplier alone.
        @test slope_wind_floor(5.0) == 0.0
        @test slope_wind_correction(4.0, 2.0) == m(2.0) * 4.0
        @test slope_wind_correction(0.0, 2.0) == 0.0
        # A floor, when given, holds the calm hours up and leaves the rest.
        f = (; c..., u_min_a = 1.0, u_min_b = 0.5)
        @test slope_wind_floor(1.0; coefficients = f) == 1.0
        @test slope_wind_floor(exp(1.0); coefficients = f) ≈ 1.5
        @test slope_wind_floor(0.01; coefficients = f) == max(1.0 + 0.5 * log(0.1), 0.0)
        @test slope_wind_correction(0.1, 1.0; coefficients = f) == 1.0
        @test slope_wind_correction(5.0, 1.0; coefficients = f) == exp(0.398) * 5.0
        # Negative floors clamp at zero.
        @test slope_wind_floor(0.1; coefficients = (; c..., u_min_a = -1.0)) == 0.0
        @test_throws "surface slope must be finite and ≥ 0°, got -1.0" slope_wind_multiplier(-1.0)
        @test_throws "surface slope must be finite and ≥ 0°, got NaN" slope_wind_multiplier(NaN)
    end

    @testset "the rule" begin
        @test wind_correction_rule((; rule = :scaling)) == (; rule = :scaling)
        r = wind_correction_rule((; rule = :slope, band_slope = [1, 2.5]); n_bands = 2)
        @test r.band_slope == [1.0, 2.5] && r.coefficients == SLOPE_WIND_COEFFICIENTS
        c2 = (; SLOPE_WIND_COEFFICIENTS..., m1 = 0.2)
        @test wind_correction_rule((; rule = :slope, band_slope = [1.0], coefficients = c2)).coefficients == c2
        @test_throws "wind_correction rule must be one of (:scaling, :slope), got :zone" wind_correction_rule((; rule = :zone))
        @test_throws "wind_correction needs a `rule`" wind_correction_rule((; band_slope = [1.0]))
        @test_throws "wind_correction rule :slope takes the fields" wind_correction_rule((; rule = :slope))
        @test_throws "wind_correction rule :scaling takes the fields" wind_correction_rule((; rule = :scaling, band_slope = [1.0]))
        @test_throws "gives 1 band slopes for 2 bands" wind_correction_rule((; rule = :slope, band_slope = [1.0]); n_bands = 2)
        @test_throws "band slope must be finite and ≥ 0°, got NaN for band 2" wind_correction_rule(
            (; rule = :slope, band_slope = [1.0, NaN]))
        @test_throws "slope wind coefficients must be" wind_correction_rule(
            (; rule = :slope, band_slope = [1.0], coefficients = (; m0 = 0.4)))
        @test_throws "theta_floor must be > 0°" wind_correction_rule(
            (; rule = :slope, band_slope = [1.0], coefficients = (; SLOPE_WIND_COEFFICIENTS..., theta_floor = 0.0)))
    end

    @testset "band slope from the table" begin
        cells = _cp_table([(lon = 10.0, lat = 46.0, z = 1000.0, glm = 0.2, bins = [(1050, 3.0), (1150, 1.0)]),
                           (lon = 10.1, lat = 46.0, z = 1000.0, glm = 0.2, bins = [(1050, 1.0)])])
        intervals = hypsometry_intervals(cells)
        @test_throws "the elevation-class table has no band slope column slope2km_01000_01100" tile_interval_slopes(cells, intervals)
        for nm in GEMB_GlacierSims._hyps_colnames(0:100:10_000)
            cells[!, Symbol(replace(string(nm), "hyps" => "slope2km"))] = fill(NaN, nrow(cells))
        end
        cells.slope2km_01000_01100 = [2.0, 6.0]
        cells.slope2km_01100_01200 = [4.0, NaN]       # the second cell holds no ice here
        @test tile_interval_slopes(cells, intervals) == [(3 * 2.0 + 1 * 6.0) / 4, 4.0]
        # A cell with ice but no finite slope is left out of the mean; a band with none left throws.
        cells.slope2km_01000_01100 = [2.0, NaN]
        @test tile_interval_slopes(cells, intervals)[1] == 2.0
        cells.slope2km_01100_01200 = [NaN, NaN]
        @test_throws "no cell with ice in band 1100-1200 m has a finite slope2km_01100_01200" tile_interval_slopes(cells, intervals)
        @test_throws "band 1200-1300 m holds no ice in these cells" tile_interval_slopes(
            cells, [(; lo = 1200, hi = 1300, center = 1250.0)])
    end

    @testset "applied to the band wind, and nothing else" begin
        T_ref = 273.15
        z = collect(range(300.0, 3000.0; length = 8))
        days = [DateTime(2003, 7, d, h) for d in 1:10 for h in (0, 12)]
        a = [16.0 + 6.0 * sin(0.9i) for i in eachindex(days)]
        own_bin(zi) = 100 * floor(Int, zi / 100) + 50
        cells = [(lon = 10.0 + 0.01i, lat = 46.0, z = z[i], glm = 0.3, bins = [(own_bin(z[i]), 1.0)])
                 for i in eachindex(z)]
        table = _cp_table(cells)
        loader = function (_model, lat, lon; kwargs...)
            c = cells[findfirst(c -> c.lat == lat && c.lon == lon, cells)]
            fd = _audit_forcing(days, lat, lon, c.z, [T_ref + ai - 0.006 * c.z for ai in a])
            fd[:wind_speed] .= [0.2 + 3.0 * abs(sin(0.7i)) + c.z / 1000 for i in eachindex(days)]
            fd
        end
        p = derive_downscaling_parameters(:synthetic, (first(days), last(days)), table;
                                          token = nothing, cache_path = nothing, forcing_loader = loader)
        fit = (; time = p.time, p.decoupling, p.lapse_rate, p.lapse_rate_uncertainty)
        intervals = hypsometry_intervals(table)
        θ = collect(range(0.05, 12.0; length = length(intervals)))
        resolved(; kw...) = resolve_downscaling(fit, intervals, days; basis = :fitted, kw...)
        bands(applied) = collect(elevation_interval_forcing(
            table, applied; climate_model = :synthetic, time_range = (first(days), last(days)),
            forcing_loader = loader))
        layers(b) = Dict(v => collect(b.forcing[v]) for v in keys(b.forcing))
        none = bands(resolved())
        scal = bands(resolved(; wind_correction = (; rule = :scaling)))
        slope = bands(resolved(; wind_correction = (; rule = :slope, band_slope = θ)))
        floored = (; SLOPE_WIND_COEFFICIENTS..., u_min_a = 2.0)
        slopef = bands(resolved(; wind_correction = (; rule = :slope, band_slope = θ, coefficients = floored)))
        @test length(none) == length(scal) == length(slope) == length(intervals) > 1
        # `:scaling` is the default, bit for bit, metadata included, and records no new setting.
        @test resolved().settings == resolved(; wind_correction = (; rule = :scaling)).settings
        @test !any(startswith("downscaling_wind_correction"), keys(resolved().settings))
        for (i, (b0, b1, b2, b3)) in enumerate(zip(none, scal, slope, slopef))
            @test layers(b1) == layers(b0)
            @test DimensionalData.metadata(b1.forcing) == DimensionalData.metadata(b0.forcing)
            @test !haskey(DimensionalData.metadata(b0.forcing), "wind_speed_correction")
            # The band's own slope sets its multiplier, and only the wind changes.
            u0 = collect(b0.forcing.wind_speed)
            @test collect(b2.forcing.wind_speed) == slope_wind_multiplier(θ[i]) .* u0
            @test all(layers(b2)[v] == layers(b0)[v] for v in keys(b0.forcing) if v !== :wind_speed)
            md = DimensionalData.metadata(b2.forcing)
            @test md["wind_speed_correction"] == "slope" && md["surface_slope_2km"] == θ[i]
            @test md["wind_speed_slope_multiplier"] == slope_wind_multiplier(θ[i])
            @test md["wind_speed_mean"] ≈ slope_wind_multiplier(θ[i]) * DimensionalData.metadata(b0.forcing)["wind_speed_mean"]
            @test md["wind_speed_scaling"] == 1.0
            # A floor holds every hour at or above it.
            @test collect(b3.forcing.wind_speed) == max.(slope_wind_multiplier(θ[i]) .* u0, 2.0)
        end
        # The floor binds somewhere, so the check above is not vacuous.
        @test any(b -> any(<(2.0), slope_wind_multiplier(θ[1]) .* collect(b.forcing.wind_speed)), none[1:1])
        s = resolved(; wind_correction = (; rule = :slope, band_slope = θ)).settings
        @test s["downscaling_wind_correction"] == "slope"
        @test s["downscaling_wind_correction_coefficients"] == [0.398, 0.168, 0.1, 0.0, 0.0]
        @test s["downscaling_wind_correction_coefficient_names"] == "m0 m1 theta_floor u_min_a u_min_b"
        @test s["downscaling_wind_correction_band_slope"] == θ
        @test s["downscaling_wind_scaling"] == 1.0
        @test_throws "wind_correction :slope replaces wind_scaling, which must be 1; got 2.0" resolved(
            ; wind_scaling = 2.0, wind_correction = (; rule = :slope, band_slope = θ))
        @test_throws "gives 2 band slopes for $(length(intervals)) bands" resolved(
            ; wind_correction = (; rule = :slope, band_slope = [1.0, 2.0]))
    end

    @testset "the ws member" begin
        ws = physics_member("C-kfit-table-ws")
        @test ws.wind_correction === :slope && ws.wind_scaling === 1.0
        @test ws.model == physics_member("C-kfit-table").model
        @test occursin("m(2 km slope)", ws.description)
        @test physics_member("C-kfit-table").wind_correction === :scaling
        @test physics_member("C-kfit-table-w2").wind_correction === :scaling
        @test physics_member_parameters(ws)["physics_wind_correction"] == "slope"
        @test !haskey(physics_member_parameters(physics_member("C-kfit-table-w2")), "physics_wind_correction")
        @test_throws "or the slope correction \"ws\"" physics_member("C-kfit-table-wx")

        time = collect(DateTime(2000, 1, 1):Hour(3):DateTime(2000, 1, 2))
        window = (DateTime(2000, 1, 1), DateTime(2000, 12, 31, 21))
        cells = _cp_table([(lon = 10.0, lat = 46.0, z = 3000.0, glm = 1.0, bins = [(3050, 1.0)])])
        bbc = background_black_carbon(cells; basis = :bc_dust, optics = :snicar_adv3)
        flat = _flat_applied(time, [(1000, 1100)])
        withslope = GEMB_GlacierSims.AppliedDownscaling(
            flat.time, flat.lapse_rate, flat.lapse_rate_source, flat.bands, flat.basis,
            merge(flat.settings, Dict{String,Any}("downscaling_wind_correction" => "slope",
                                                  "downscaling_wind_correction_coefficients" => [0.398, 0.168, 0.1, 0.0, 0.0],
                                                  "downscaling_wind_correction_band_slope" => [3.0])))
        mp = physics_model_parameters(ws)
        p = tile_run_parameters(mp, withslope; spinup_window = window, background_black_carbon = bbc,
                                physics_member = ws)
        @test p["physics_wind_correction"] == "slope" && p["downscaling_wind_correction"] == "slope"
        @test p["downscaling_wind_correction_band_slope"] == [3.0]
        @test_throws "physics member C-kfit-table-ws corrects the band wind by slope, but the downscaling was resolved with wind_correction = scaling" tile_run_parameters(
            mp, flat; spinup_window = window, background_black_carbon = bbc, physics_member = ws)
        m = physics_member("C-kfit-table")
        @test_throws "physics member C-kfit-table corrects the band wind by scaling, but the downscaling was resolved with wind_correction = slope" tile_run_parameters(
            mp, withslope; spinup_window = window, background_black_carbon = bbc, physics_member = m)
        p0 = tile_run_parameters(mp, flat; spinup_window = window, background_black_carbon = bbc, physics_member = m)
        @test !any(startswith("downscaling_wind_correction"), keys(p0)) && !haskey(p0, "physics_wind_correction")
        d = run_parameter_differences(p0, p)
        @test haskey(d, "physics_member")
    end

    @testset "the scripts' wind correction" begin
        repo = normpath(joinpath(@__DIR__, ".."))
        load(env...) = withenv(env...) do
            mod = Module()
            Base.eval(mod, :(using Dates))
            Base.include(mod, joinpath(repo, "scripts", "run_settings.jl"))
            mod
        end
        call(mod, name, args...) = Base.invokelatest(() -> getglobal(mod, name)(args...))
        clear = ("BACKGROUND_BLACK_CARBON" => nothing, "BACKGROUND_BLACK_CARBON_OPTICS" => nothing,
                 "EFFECTIVE_LOADING_TABLE" => nothing)
        cells = _cp_table([(lon = 6.9, lat = 45.9, z = 3000.0, glm = 1.0, bins = [(3050, 2.0)])])
        cells.slope2km_03000_03100 = [7.5]
        for nm in GEMB_GlacierSims._hyps_colnames(0:100:10_000)
            s = Symbol(replace(string(nm), "hyps" => "slope2km"))
            hasproperty(cells, s) || (cells[!, s] = [NaN])
        end
        tile = (; core = cells)
        intervals = hypsometry_intervals(cells)
        ws = load(clear..., "PHYSICS_MEMBER" => "B-kfit-table-ws")
        @test call(ws, :tile_wind_correction, tile, intervals) == (; rule = :slope, band_slope = [7.5])
        w2 = load(clear..., "PHYSICS_MEMBER" => "B-kfit-table-w2")
        @test call(w2, :tile_wind_correction, tile, intervals) == (; rule = :scaling)
        for script in ("gemb_tile_sweep.jl", "gemb_native_run.jl")
            text = read(joinpath(repo, "scripts", script), String)
            @test count("wind_correction = wind_rule", text) == 2
            @test count("wind_rule = tile_wind_correction(tile, intervals)", text) == 1
        end
    end
end
