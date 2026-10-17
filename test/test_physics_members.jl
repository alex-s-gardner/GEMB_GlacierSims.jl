# Physics members: the member registry, the band decoupling rule, the effective loading, the run
# parameters that name a member and refuse a mislabelled run, and the scripts' member setting.
# Included from `runtests.jl` after `test_background_black_carbon.jl`; uses `_flat_applied`,
# `_tiny_tile_run` and `_tiny_band_forcing` from `test_pipeline_audit.jl` and `_cp_table` from
# `runtests.jl`.

# A regional effective-loading table with the columns `effective_black_carbon` reads, plus a `source`
# column as the fit's output carries.
function _effective_table(rows; variant = "C0")
    path = joinpath(mktempdir(), "effective.csv")
    open(path, "w") do io
        println(io, "rgi_o1region,black_carbon_snow,variant,source")
        for (region, value) in rows
            v = variant isa AbstractDict ? variant[region] : variant
            println(io, "\"$region\",$value,\"$v\",\"modis_fit\"")
        end
    end
    return path
end

# Twelve hourly steps of a fit fixture whose band `k` at 1050 m is 0.4 at every step: with
# `gamma = -3`, `delta = 0` and the ambient excess `alpha + beta·z` = 5 K at that elevation,
# `k = 1 + gamma/(alpha + beta·z) = 1 − 3/5`.
function _physics_fit()
    rt = collect(DateTime(2001, 1, 1):Hour(1):DateTime(2001, 1, 1, 11))
    n = length(rt)
    ti = Ti(rt)
    dec = DimStack((decoupling_factor = DimArray(fill(0.4, n), (ti,)),
                    n_cells = DimArray(fill(20, n), (ti,)),
                    r2 = DimArray(fill(0.99, n), (ti,)),
                    ambient_excess = DimArray(fill(5.0, n), (ti,)),
                    coef_alpha = DimArray(fill(5.0 + 0.001 * 1050, n), (ti,)),
                    coef_beta = DimArray(fill(-0.001, n), (ti,)),
                    coef_gamma = DimArray(fill(-3.0, n), (ti,)),
                    coef_delta = DimArray(fill(0.0, n), (ti,))))
    lr = DimStack((lapse_rate = DimArray(fill(6.0, n), (ti,)),
                   n_cells = DimArray(fill(20, n), (ti,)),
                   elevation_spread = DimArray(fill(900.0, n), (ti,))))
    return (; time = rt, decoupling = dec, lapse_rate = lr)
end

@testset "physics members" begin
    @testset "the registry" begin
        m = physics_member("C-ks35-table")
        @test m.id == "C-ks35-table"
        @test (m.grain, m.band_decoupling, m.impurity) === (:C, :ks35, :table)
        @test m.model == (; grain_growth_method = :FlannerZender2006,
                          new_snow_method = :vanKampenhout2017, densification_method = :Crocus,
                          destructive_metamorphism = true)
        @test m.band_rule == (; rule = :scaled, scale = 3.5)
        @test m.impurity_level.basis === :bc_dust && m.impurity_level.optics === :snicar_adv3
        # Every combination of levels is a member, and every level fixes the same fields.
        for g in keys(PHYSICS_GRAIN_VARIANTS), k in keys(PHYSICS_BAND_DECOUPLING),
            i in keys(PHYSICS_IMPURITY)
            @test physics_member("$g-$k-$i").id == "$g-$k-$i"
        end
        @test allequal(keys(v.model) for v in PHYSICS_GRAIN_VARIANTS)
        @test_throws "a physics member is \"<grain>-<band decoupling>-<impurity>\"" physics_member("C-ks35")
        @test_throws "unknown grain variant \"D\" in physics member \"D-kfit-table\"; one of A, B, C" physics_member(
            "D-kfit-table")
        @test_throws "unknown band decoupling level \"ks9\"" physics_member("A-ks9-table")
        @test_throws "unknown impurity level \"soot\"" physics_member("A-kfit-soot")
    end

    @testset "model parameters under a member" begin
        for g in keys(PHYSICS_GRAIN_VARIANTS)
            m = physics_member("$g-kfit-bc0")
            mp = physics_model_parameters(m; output_frequency = :monthly)
            @test mp.output_frequency === :monthly
            for (field, value) in pairs(m.model)
                @test getproperty(mp, field) == value
            end
        end
        # A is the GEMB default snow configuration.
        default = initialize_parameters()
        @test all(getproperty(default, f) == v for (f, v) in pairs(PHYSICS_GRAIN_VARIANTS.A.model))
        @test_throws "physics member B-kfit-bc0 sets grain_growth_method; remove it from the keywords" physics_model_parameters(
            physics_member("B-kfit-bc0"); grain_growth_method = :hybrid)
    end

    @testset "band decoupling rule" begin
        @test band_decoupling_rule((; rule = :fitted)) == (; rule = :fitted, scale = "none",
                                                           constant = "none")
        @test band_decoupling_rule((; rule = :scaled, scale = 2)) ==
              (; rule = :scaled, scale = 2.0, constant = "none")
        @test band_decoupling_rule((; rule = :constant, constant = 1)) ==
              (; rule = :constant, scale = "none", constant = 1.0)
        @test_throws "band_decoupling rule must be one of (:fitted, :scaled, :constant), got :shaw" band_decoupling_rule(
            (; rule = :shaw))
        @test_throws "band_decoupling needs a `rule`" band_decoupling_rule((; scale = 2.0))
        @test_throws "band_decoupling rule :scaled takes the fields [:rule, :scale], got [:rule]" band_decoupling_rule(
            (; rule = :scaled))
        @test_throws "takes the fields [:rule], got [:rule, :scale]" band_decoupling_rule(
            (; rule = :fitted, scale = 2.0))
        @test_throws "scale must be positive and finite, got 0.0" band_decoupling_rule(
            (; rule = :scaled, scale = 0.0))
        @test_throws "scale must be positive and finite, got Inf" band_decoupling_rule(
            (; rule = :scaled, scale = Inf))
        @test_throws "constant must be in (0.0, 1.0], got 1.2" band_decoupling_rule(
            (; rule = :constant, constant = 1.2))
        @test_throws "constant must be in (0.0, 1.0], got 0.0" band_decoupling_rule(
            (; rule = :constant, constant = 0.0))
    end

    @testset "resolve_downscaling applies the rule to band k only" begin
        fit = _physics_fit()
        rt = fit.time
        n = length(rt)
        bands = [(; lo = 1000, hi = 1100, center = 1050.0)]
        base = resolve_downscaling(fit, bands, rt; basis = :fitted)
        k = base.bands[1].decoupling_factor
        @test k ≈ fill(0.4, n)
        @test base.settings["downscaling_band_decoupling_rule"] == "fitted"
        @test base.settings["downscaling_band_decoupling_scale"] == "none"
        @test base.settings["downscaling_band_decoupling_constant"] == "none"

        for s in (2.2, 3.5)
            a = resolve_downscaling(fit, bands, rt; basis = :fitted,
                                    band_decoupling = (; rule = :scaled, scale = s))
            @test a.bands[1].decoupling_factor == min.(1.0, s .* k)
            # 2.2 × 0.4 = 0.88 stays below the cap; 3.5 × 0.4 = 1.4 is capped at 1.
            @test all(a.bands[1].decoupling_factor .≈ (s == 2.2 ? 0.88 : 1.0))
            @test a.bands[1].decoupling_factor_source == base.bands[1].decoupling_factor_source
            @test a.settings["downscaling_band_decoupling_rule"] == "scaled"
            @test a.settings["downscaling_band_decoupling_scale"] == s
            @test a.settings["downscaling_band_decoupling_constant"] == "none"
            # Donor conversion keeps its own resolution.
            @test a.donor_decoupling_factor_at(400.0) == base.donor_decoupling_factor_at(400.0)
        end
        c = resolve_downscaling(fit, bands, rt; basis = :fitted,
                                band_decoupling = (; rule = :constant, constant = 0.77))
        @test c.bands[1].decoupling_factor == fill(0.77, n)
        @test downscaling_source_counts(c.bands[1].decoupling_factor_source)[:prior] == n
        @test c.bands[1].n_fit_in_domain == base.bands[1].n_fit_in_domain
        @test c.settings["downscaling_band_decoupling_constant"] == 0.77
        # Two resolutions under different rules are different experiments.
        mp = initialize_parameters()
        window = (DateTime(2001, 1, 1), DateTime(2001, 12, 31, 23))
        @test !isempty(run_parameter_differences(tile_run_parameters(mp, base; spinup_window = window),
                                                 tile_run_parameters(mp, c; spinup_window = window)))
        @test_throws "band_decoupling rule must be one of" resolve_downscaling(
            fit, bands, rt; band_decoupling = (; rule = :shaw))
    end

    @testset "effective loading" begin
        cells = _cp_table([(lon = 10.0, lat = 46.0, z = 3000.0, glm = 1.0, bins = [(3050, 1.0)]),
                           (lon = 6.9, lat = 45.9, z = 3000.0, glm = 1.0, bins = [(3050, 3.0)])])
        path = _effective_table([("11", 0.012), ("01", 0.5)])
        e = effective_black_carbon(cells, path)
        @test e.black_carbon_snow == 0.012
        @test e.variant == "C0"
        @test e.region_shares == ["11" => 1.0]
        @test e.table_file == "effective.csv"
        @test e.table_sha256 == bytes2hex(open(GEMB_GlacierSims.SHA.sha256, path))
        @test e.cap == Inf
        # Glacier-area weighting across regions: one cell in Alaska (region 01) with 1 km², one in
        # Central Europe (11) with 3 km².
        two = _cp_table([(lon = -140.0, lat = 60.3, z = 1500.0, glm = 1.0, bins = [(1550, 1.0)]),
                         (lon = 6.9, lat = 45.9, z = 3000.0, glm = 1.0, bins = [(3050, 3.0)])])
        w = effective_black_carbon(two, path)
        @test w.region_shares == ["11" => 0.75, "01" => 0.25]
        @test w.black_carbon_snow ≈ 0.75 * 0.012 + 0.25 * 0.5
        # The cap applies to every region before weighting.
        capped = effective_black_carbon(two, path; cap = 0.045)
        @test capped.region_values == ["11" => 0.012, "01" => 0.045]
        @test capped.black_carbon_snow ≈ 0.75 * 0.012 + 0.25 * 0.045
        @test_throws "the cap on an effective loading must be positive" effective_black_carbon(
            cells, path; cap = 0)
        @test_throws "no effective-loading table at" effective_black_carbon(cells, "/nonexistent.csv")
        @test_throws "has 0 rows for RGI region 11" effective_black_carbon(
            cells, _effective_table([("01", 0.5)]))
        @test_throws "has 2 rows for RGI region 11" effective_black_carbon(
            cells, _effective_table([("11", 0.5), ("11", 0.4)]))
        @test_throws "gives region 11 a loading of 2.5; it must be in [0, 2.0] ppmw" effective_black_carbon(
            cells, _effective_table([("11", 2.5)]))
        @test_throws "holds variants" effective_black_carbon(
            cells, _effective_table([("11", 0.01), ("01", 0.5)];
                                    variant = Dict("11" => "C0", "01" => "B0")))
        bad = joinpath(mktempdir(), "bad.csv")
        write(bad, "rgi_o1region,variant\n\"11\",\"C0\"\n")
        @test_throws "lacks column(s) black_carbon_snow" effective_black_carbon(cells, bad)
    end

    @testset "run parameters name the member and refuse a mislabelled run" begin
        time = collect(DateTime(2000, 1, 1):Hour(3):DateTime(2000, 1, 2))
        window = (DateTime(2000, 1, 1), DateTime(2000, 12, 31, 21))
        cells = _cp_table([(lon = 10.0, lat = 46.0, z = 3000.0, glm = 1.0, bins = [(3050, 1.0)])])
        bbc = background_black_carbon(cells; basis = :bc_dust, optics = :snicar_adv3)
        # `_flat_applied` resolves no rule, which is the fitted one.
        flat = _flat_applied(time, [(1000, 1100)])
        scaled = GEMB_GlacierSims.AppliedDownscaling(
            flat.time, flat.lapse_rate, flat.lapse_rate_source, flat.bands, flat.basis,
            Dict{String,Any}("downscaling_band_decoupling_rule" => "scaled",
                             "downscaling_band_decoupling_scale" => 3.5,
                             "downscaling_band_decoupling_constant" => "none"))

        m = physics_member("C-ks35-table")
        mp = physics_model_parameters(m)
        p = tile_run_parameters(mp, scaled; spinup_window = window, background_black_carbon = bbc,
                                physics_member = m)
        @test p["physics_member"] == "C-ks35-table"
        @test p["physics_grain"] == "C"
        @test p["physics_band_decoupling"] == "ks35"
        @test p["physics_impurity"] == "table"
        @test p["model_grain_growth_method"] === :FlannerZender2006
        @test p["model_black_carbon_snow"] == bbc.black_carbon_snow
        # Without a member, no `physics_*` key is written.
        plain = tile_run_parameters(mp, scaled; spinup_window = window, background_black_carbon = bbc)
        @test !any(startswith("physics_"), keys(plain))
        # Two members are two experiments.
        mA = physics_member("A-ks35-table")
        pA = tile_run_parameters(physics_model_parameters(mA), scaled; spinup_window = window,
                                 background_black_carbon = bbc, physics_member = mA)
        d = run_parameter_differences(p, pA)
        @test haskey(d, "physics_member") && haskey(d, "physics_grain")
        @test haskey(d, "model_grain_growth_method")

        # A run whose settings are not the member's is refused, naming what disagrees.
        @test_throws "physics member C-ks35-table sets grain_growth_method = :FlannerZender2006, but the run's ModelParameters has :hybrid" tile_run_parameters(
            initialize_parameters(), scaled; spinup_window = window, background_black_carbon = bbc,
            physics_member = m)
        @test_throws "applies band decoupling (rule = \"scaled\", scale = 3.5, constant = \"none\"), but the downscaling was resolved with (rule = \"fitted\"" tile_run_parameters(
            mp, flat; spinup_window = window, background_black_carbon = bbc, physics_member = m)
        @test_throws "needs the background loading" tile_run_parameters(
            mp, scaled; spinup_window = window, physics_member = m)
        @test_throws "with basis :bc_dust and optics :snicar_adv3, but the loading given has basis :bc and optics :snicar_adv3" tile_run_parameters(
            mp, scaled; spinup_window = window, physics_member = m,
            background_black_carbon = background_black_carbon(cells; basis = :bc))
        m0 = physics_member("C-ks35-bc0")
        @test_throws "runs without a snow impurity loading, but one was given" tile_run_parameters(
            mp, scaled; spinup_window = window, background_black_carbon = bbc, physics_member = m0)
        @test tile_run_parameters(mp, scaled; spinup_window = window,
                                  physics_member = m0)["model_black_carbon_snow"] == 0.0

        # The capped effective loading, fitted for the member's own grain variant.
        mv = physics_member("C-ks35-cvis")
        table = _effective_table([("11", 0.3)])
        ebc = effective_black_carbon(cells, table; cap = 0.045)
        q = tile_run_parameters(mp, scaled; spinup_window = window, effective_black_carbon = ebc,
                                physics_member = mv)
        @test q["model_black_carbon_snow"] == 0.045
        @test q["effective_black_carbon_cap"] == 0.045
        @test q["effective_black_carbon_variant"] == "C0"
        @test q["effective_black_carbon_region_values"] == "11:0.045"
        @test q["effective_black_carbon_table_sha256"] == ebc.table_sha256
        @test !isempty(run_parameter_differences(p, q))
        @test_throws "needs an effective loading" tile_run_parameters(
            mp, scaled; spinup_window = window, background_black_carbon = bbc, physics_member = mv)
        @test_throws "caps the effective loading at 0.045 ppmw, but the loading given is capped at Inf" tile_run_parameters(
            mp, scaled; spinup_window = window, physics_member = mv,
            effective_black_carbon = effective_black_carbon(cells, table))
        @test_throws "has grain variant C, but the effective loading was fitted for variant B0" tile_run_parameters(
            mp, scaled; spinup_window = window, physics_member = mv,
            effective_black_carbon = effective_black_carbon(cells, _effective_table([("11", 0.3)];
                                                                                   variant = "B0");
                                                            cap = 0.045))
        @test_throws "give a background loading or an effective loading, not both" tile_run_parameters(
            mp, scaled; spinup_window = window, background_black_carbon = bbc,
            effective_black_carbon = ebc)
    end

    @testset "a tile run records its member and fails before any GEMB run when mislabelled" begin
        forcing = _tiny_band_forcing()
        m = physics_member("A-kfit-bc0")
        run = _tiny_tile_run([0.48]; forcing, physics_member = m)
        @test run.parameters["physics_member"] == "A-kfit-bc0"
        written = write_glacier_tile_netcdf(joinpath(mktempdir(), "t.nc"), run)
        status = read_glacier_tile_status(written)
        @test status.parameters["physics_member"] == "A-kfit-bc0"
        @test status.parameters["physics_band_decoupling"] == "kfit"
        # The `_tiny_tile_run` forcing is `:hybrid`; member B would need Flanner & Zender.
        calls = Ref(0)
        @test_throws "physics member B-kfit-bc0 sets grain_growth_method" _tiny_tile_run(
            [0.48]; forcing, physics_member = physics_member("B-kfit-bc0"),
            on_output = (o; kw...) -> (calls[] += 1))
        @test calls[] == 0
    end

    @testset "the scripts' member setting" begin
        repo = normpath(joinpath(@__DIR__, ".."))
        load(env...) = withenv(env...) do
            mod = Module()
            Base.eval(mod, :(using Dates))
            Base.include(mod, joinpath(repo, "scripts", "run_settings.jl"))
            mod
        end
        get_(mod, name) = Base.invokelatest(getglobal, mod, name)
        call(mod, name, args...; kw...) = Base.invokelatest(() -> getglobal(mod, name)(args...; kw...))
        cells = _cp_table([(lon = 6.9, lat = 45.9, z = 3000.0, glm = 1.0, bins = [(3050, 1.0)])])
        clear = ("PHYSICS_MEMBER" => nothing, "BACKGROUND_BLACK_CARBON" => nothing,
                 "BACKGROUND_BLACK_CARBON_OPTICS" => nothing, "EFFECTIVE_LOADING_TABLE" => nothing)

        # The default is Flanner & Zender grain growth with the production impurity table and the
        # fitted k.
        d = load(clear...)
        @test get_(d, :DEFAULT_PHYSICS_MEMBER) == "B-kfit-table"
        @test get_(d, :PHYSICS_MEMBER).id == "B-kfit-table"
        @test get_(d, :BACKGROUND_BLACK_CARBON) === :bc_dust
        @test get_(d, :BACKGROUND_BLACK_CARBON_OPTICS) === :snicar_adv3
        @test call(d, :tile_background_black_carbon, cells) ==
              background_black_carbon("11"; basis = :bc_dust, optics = :snicar_adv3)
        @test call(d, :tile_effective_black_carbon, cells) === nothing
        @test call(d, :tile_run_dir, "/cache") == "/cache/tile_runs_v2/B-kfit-table"
        # The launcher carries the same default member and directory.
        sh = read(joinpath(repo, "scripts", "run_tile_sweep.sh"), String)
        @test occursin("PHYSICS_MEMBER=\"\${PHYSICS_MEMBER:-$(get_(d, :DEFAULT_PHYSICS_MEMBER))}\"", sh)
        @test occursin("OUTPUT_DIR=\"\${OUTPUT_DIR:-\$CLIMATE_CACHE/$(get_(d, :TILE_RUN_DIR))/\$PHYSICS_MEMBER}\"", sh)
        native_sh = read(joinpath(repo, "scripts", "run_native_sweep.sh"), String)
        @test occursin("PHYSICS_MEMBER=\"\${PHYSICS_MEMBER:-$(get_(d, :DEFAULT_PHYSICS_MEMBER))}\"",
                       native_sh)

        # A member sets the model parameters, the band rule and the loading.
        c = load(clear..., "PHYSICS_MEMBER" => "C-ks35-bc0")
        mp = call(c, :physics_model_parameters; output_frequency = :weekly)
        @test mp.grain_growth_method === :FlannerZender2006 && mp.output_frequency === :weekly
        @test get_(c, :PHYSICS_MEMBER).band_rule == (; rule = :scaled, scale = 3.5)
        @test get_(c, :BACKGROUND_BLACK_CARBON) === :off
        @test call(c, :tile_background_black_carbon, cells) === nothing
        @test call(c, :tile_run_dir, "/cache") == "/cache/tile_runs_v2/C-ks35-bc0"
        # An `export` that agrees is accepted; one that disagrees stops the script.
        @test get_(load(clear..., "PHYSICS_MEMBER" => "C-ks35-table",
                        "BACKGROUND_BLACK_CARBON" => "bc_dust"), :BACKGROUND_BLACK_CARBON) === :bc_dust
        @test_throws "BACKGROUND_BLACK_CARBON=bc disagrees with physics member B-kfit-table, whose impurity level table sets bc_dust" load(
            clear..., "BACKGROUND_BLACK_CARBON" => "bc")
        @test_throws "BACKGROUND_BLACK_CARBON_OPTICS=dang2015 disagrees" load(
            clear..., "BACKGROUND_BLACK_CARBON_OPTICS" => "dang2015")
        @test_throws "BACKGROUND_BLACK_CARBON=bc_dust disagrees with physics member A-kfit-bc0" load(
            clear..., "PHYSICS_MEMBER" => "A-kfit-bc0", "BACKGROUND_BLACK_CARBON" => "bc_dust")
        @test_throws "unknown impurity level \"dust\"" load(clear..., "PHYSICS_MEMBER" => "A-kfit-dust")

        # A capped member reads its table, and only a capped member may be given one.
        table = _effective_table([("11", 0.3)])
        v = load(clear..., "PHYSICS_MEMBER" => "C-ks35-cvis", "EFFECTIVE_LOADING_TABLE" => table)
        @test get_(v, :BACKGROUND_BLACK_CARBON) === :off
        @test call(v, :tile_effective_black_carbon, cells).black_carbon_snow == 0.045
        @test_throws "takes an effective loading; set EFFECTIVE_LOADING_TABLE" load(
            clear..., "PHYSICS_MEMBER" => "C-ks35-cvis")
        @test_throws "EFFECTIVE_LOADING_TABLE=/nonexistent.csv does not exist" load(
            clear..., "PHYSICS_MEMBER" => "C-ks35-cvis", "EFFECTIVE_LOADING_TABLE" => "/nonexistent.csv")
        @test_throws "EFFECTIVE_LOADING_TABLE is set, but physics member B-kfit-table takes no effective loading" load(
            clear..., "EFFECTIVE_LOADING_TABLE" => table)

        # Both scripts run as the member: model parameters, band rule and both loadings.
        for script in ("gemb_tile_sweep.jl", "gemb_native_run.jl")
            text = read(joinpath(repo, "scripts", script), String)
            @test occursin("mp = physics_model_parameters(output_frequency = OUTPUT_FREQUENCY)", text)
            @test !occursin("GEMB.initialize_parameters", text)
            @test count("band_decoupling = PHYSICS_MEMBER.band_rule", text) == 2
            @test count("effective_black_carbon = effective", text) == 2
            @test count(r"physics_member = PHYSICS_MEMBER[,)\s]", text) == 2
            @test occursin("tile_run_dir(CLIMATE_CACHE)", text)
        end
    end
end
