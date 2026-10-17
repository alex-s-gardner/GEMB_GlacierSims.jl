# Physics members: the member registry, the band decoupling rules, the Shaw et al. (2025) glacier
# bands, the glacier-fraction form of the decoupling fit, the effective loading, the run parameters
# that name a member and refuse a mislabelled run, and the scripts' member setting.
# Included from `runtests.jl` after `test_background_black_carbon.jl`; uses `_flat_applied`,
# `_tiny_tile_run` and `_tiny_band_forcing` from `test_pipeline_audit.jl` and `_cp_table` from
# `runtests.jl`, and `_audit_forcing` from `test_pipeline_audit.jl`.

# A Shaw glacier band table in the vendored format, from `(rgi_id, lat, lon, k, [(band_lo, area)])`.
function _shaw_table(glaciers)
    path = joinpath(mktempdir(), "shaw.csv.gz")
    open(GEMB_GlacierSims.CodecZlib.GzipCompressorStream, path, "w") do io
        println(io, "# test table")
        println(io, "rgi_id,lat,lon,k,area_km2,band_lo,band_area_km2")
        for (id, lat, lon, k, bands) in glaciers
            println(io, join((id, lat, lon, k, sum(last, bands), join(first.(bands), ';'),
                              join(last.(bands), ';')), ','))
        end
    end
    return path
end

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
        m = physics_member("C-kshaw-table")
        @test m.id == "C-kshaw-table"
        @test (m.grain, m.band_decoupling, m.impurity) === (:C, :kshaw, :table)
        @test m.model == (; grain_growth_method = :FlannerZender2006,
                          new_snow_method = :vanKampenhout2017, densification_method = :Crocus,
                          destructive_metamorphism = true, blowing_snow_method = :none,
                          drift_fetch_length = 10.0)
        # D1 is C with Keenan et al. (2021) wind packing at a 10 m fetch, and nothing else.
        d1 = physics_member("D1-kfit-table")
        @test d1.grain === :D1
        @test d1.model == merge(m.model, (; blowing_snow_method = :Keenan2021, drift_fetch_length = 10.0))
        @test m.model.blowing_snow_method === :none
        @test m.band_rule == (; rule = :shaw)
        @test physics_member("B-kfit-table").band_rule == (; rule = :fitted)
        @test m.impurity_level.basis === :bc_dust && m.impurity_level.optics === :snicar_adv3
        # Every combination of levels is a member, and every level fixes the same fields.
        for g in keys(PHYSICS_GRAIN_VARIANTS), k in keys(PHYSICS_BAND_DECOUPLING),
            i in keys(PHYSICS_IMPURITY)
            @test physics_member("$g-$k-$i").id == "$g-$k-$i"
        end
        @test allequal(keys(v.model) for v in PHYSICS_GRAIN_VARIANTS)
        @test_throws "a physics member is \"<grain>-<band decoupling>-<impurity>\"" physics_member("C-kfit")
        @test_throws "unknown grain variant \"D\" in physics member \"D-kfit-table\"; one of A, B, C, D1" physics_member(
            "D-kfit-table")
        @test_throws "unknown band decoupling level \"ks35\" in physics member \"A-ks35-table\"; one of kfit, kshaw" physics_member(
            "A-ks35-table")
        @test_throws "unknown impurity level \"soot\"" physics_member("A-kfit-soot")

        # The wind level: a fourth token w<factor>, in one spelling; without it the factor is 1.
        @test physics_member("B-kfit-table").wind_scaling === 1.0
        w2 = physics_member("B-kfit-table-w2")
        @test w2.wind_scaling === 2.0 && w2.id == "B-kfit-table-w2"
        @test (w2.grain, w2.band_decoupling, w2.impurity) === (:B, :kfit, :table)
        @test w2.model == physics_member("B-kfit-table").model
        @test physics_member("D1-kshaw-table-w1.5").wind_scaling === 1.5
        @test physics_member("B-kfit-table-w1").wind_scaling === 1.0
        @test physics_member_parameters(w2)["physics_wind_scaling"] === 2.0
        @test physics_member_parameters(physics_member("B-kfit-table"))["physics_wind_scaling"] === 1.0
        @test_throws "must be a wind factor w<factor>" physics_member("B-kfit-table-x2")
        @test_throws "write the wind factor of physics member \"B-kfit-table-w2.0\" as \"w2\"" physics_member(
            "B-kfit-table-w2.0")
        @test_throws "must be > 0" physics_member("B-kfit-table-w0")
        @test_throws "a physics member is" physics_member("B-kfit-table-w2-w3")
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
        @test_throws "physics member D1-kfit-table sets blowing_snow_method" physics_model_parameters(
            physics_member("D1-kfit-table"); blowing_snow_method = :none)
        mp = physics_model_parameters(physics_member("D1-kfit-table"))
        @test mp.blowing_snow_method === :Keenan2021 && mp.drift_fetch_length == 10.0
    end

    @testset "band decoupling rule" begin
        @test band_decoupling_rule((; rule = :fitted)) == (; rule = :fitted, band_k = "none")
        @test band_decoupling_rule((; rule = :shaw, band_k = [0.5, 1])) ==
              (; rule = :shaw, band_k = [0.5, 1.0])
        @test band_decoupling_rule((; rule = :shaw, band_k = [0.5]); n_bands = 1).band_k == [0.5]
        @test_throws "band_decoupling rule must be one of (:fitted, :shaw), got :scaled" band_decoupling_rule(
            (; rule = :scaled, scale = 3.5))
        @test_throws "band_decoupling needs a `rule`" band_decoupling_rule((; band_k = [0.5]))
        @test_throws "band_decoupling rule :shaw takes the fields [:band_k, :rule], got [:rule]" band_decoupling_rule(
            (; rule = :shaw))
        @test_throws "takes the fields [:rule], got [:band_k, :rule]" band_decoupling_rule(
            (; rule = :fitted, band_k = [0.5]))
        @test_throws "band_decoupling :shaw gives 1 band k for 2 bands" band_decoupling_rule(
            (; rule = :shaw, band_k = [0.5]); n_bands = 2)
        @test_throws "band k must be in (0.0, 1.0], got 0.0 for band 2" band_decoupling_rule(
            (; rule = :shaw, band_k = [0.5, 0.0]))
        @test_throws "got NaN for band 1" band_decoupling_rule((; rule = :shaw, band_k = [NaN]))
    end

    @testset "resolve_downscaling under the Shaw rule" begin
        fit = _physics_fit()
        rt = fit.time
        n = length(rt)
        bands = [(; lo = 1000, hi = 1100, center = 1050.0), (; lo = 1100, hi = 1200, center = 1150.0)]
        base = resolve_downscaling(fit, bands, rt; basis = :fitted)
        @test base.bands[1].decoupling_factor ≈ fill(0.4, n)
        @test base.settings["downscaling_band_decoupling_rule"] == "fitted"
        @test base.settings["downscaling_band_decoupling_band_k"] == "none"
        @test base.settings["downscaling_donor_conversion"] == "fitted"
        @test base.settings["downscaling_glacier_fraction_form"] == "fraction"
        @test base.donor_decoupling_factor_at(400.0) != ones(n)

        shaw = resolve_downscaling(fit, bands, rt; basis = :fitted,
                                   band_decoupling = (; rule = :shaw, band_k = [0.7, 0.9]))
        @test shaw.bands[1].decoupling_factor == fill(0.7, n)
        @test shaw.bands[2].decoupling_factor == fill(0.9, n)
        @test downscaling_source_counts(shaw.bands[1].decoupling_factor_source)[:prior] == n
        @test shaw.bands[1].n_fit_in_domain == base.bands[1].n_fit_in_domain
        @test shaw.settings["downscaling_band_decoupling_rule"] == "shaw"
        @test shaw.settings["downscaling_band_decoupling_band_k"] == [0.7, 0.9]
        @test shaw.settings["downscaling_donor_conversion"] == "none"
        # No donor is converted: k = 1 at every elevation, the identity of the conversion.
        @test shaw.donor_decoupling_factor_at(400.0) == ones(n)
        @test shaw.donor_decoupling_factor_at(3000.0) == ones(n)
        # The lapse rate does not depend on the rule.
        @test shaw.lapse_rate == base.lapse_rate
        mp = initialize_parameters()
        window = (DateTime(2001, 1, 1), DateTime(2001, 12, 31, 23))
        d = run_parameter_differences(tile_run_parameters(mp, base; spinup_window = window),
                                      tile_run_parameters(mp, shaw; spinup_window = window))
        @test issubset(["downscaling_band_decoupling_rule", "downscaling_band_decoupling_band_k",
                        "downscaling_donor_conversion"], keys(d))
        @test_throws "band_decoupling :shaw gives 1 band k for 2 bands" resolve_downscaling(
            fit, bands, rt; band_decoupling = (; rule = :shaw, band_k = [0.7]))
        # The glacier-fraction form of the fits is carried into the settings.
        ind = resolve_downscaling(merge(fit, (; glacier_fraction_form = :indicator)), bands, rt;
                                  basis = :fitted)
        @test ind.settings["downscaling_glacier_fraction_form"] == "indicator"
    end

    @testset "the Shaw framework bypasses the donor conversion, and only there" begin
        # A reduced form of the exact-model region of the climatology audit: cells at 300-3000 m with
        # glm 0 to 1 whose warm excess is damped by [1 - glm (1 - 0.7)].
        T_ref = 273.15
        z = collect(range(300.0, 3000.0; length = 20))
        glm = [clamp((zi - 300) / 2700, 0.0, 1.0) for zi in z]
        days = [DateTime(2003, 7, d, h) for d in 1:31 for h in (12, 15)]
        a = [16.0 + 6.0 * sin(0.9i) for i in eachindex(days)]
        T_obs(zi, g, ai) = (e = ai - 0.006 * zi; e > 0 ? T_ref + (1 - g * 0.3) * e : T_ref + e)
        own_bin(zi) = 100 * floor(Int, zi / 100) + 50
        cells = [(lon = 10.0 + 0.01i, lat = 46.0, z = z[i], glm = glm[i], bins = [(own_bin(z[i]), 1.0)])
                 for i in eachindex(z)]
        table = _cp_table(cells)
        loader = function (_model, lat, lon; kwargs...)
            c = cells[findfirst(c -> c.lat == lat && c.lon == lon, cells)]
            _audit_forcing(days, lat, lon, c.z, [T_obs(c.z, c.glm, a[i]) for i in eachindex(days)])
        end
        p = derive_downscaling_parameters(:synthetic, (first(days), last(days)), table;
                                          token = nothing, cache_path = nothing, forcing_loader = loader)
        fit = (; time = p.time, p.decoupling, p.lapse_rate, p.lapse_rate_uncertainty)
        band_forcing(rows, rule) = only(collect(elevation_interval_forcing(
            rows, resolve_downscaling(fit, hypsometry_intervals(rows), days; basis = :fitted,
                                      band_decoupling = rule);
            climate_model = :synthetic, time_range = (first(days), last(days)), forcing_loader = loader)))
        top = table[[20], :]
        center = only(hypsometry_intervals(top)).center
        lapse = resolve_downscaling(fit, hypsometry_intervals(top), days; basis = :fitted).lapse_rate
        fitted = band_forcing(top, (; rule = :fitted))
        raw = band_forcing(top, (; rule = :shaw, band_k = [1.0]))
        shaw = band_forcing(top, (; rule = :shaw, band_k = [0.6]))
        # With k = 1 the Shaw band is the glm-1 cell's own damped series lapsed to the band, unconverted.
        T_cell = [T_obs(z[20], 1.0, a[i]) for i in eachindex(days)]
        Ta = collect(raw.forcing.temperature_air)
        @test Ta ≈ T_cell .- lapse .* (center - z[20]) ./ 1000 atol = 1e-9
        # The Shaw band k then acts on the raw band ambient's warm excess.
        @test collect(shaw.forcing.temperature_air) ≈ Ta .+ (0.6 - 1) .* max.(Ta .- T_ref, 0) atol = 1e-9
        # The fitted rule converts the cell to the ambient and damps that once, by the fitted k of
        # about 0.7; Shaw's k of 0.7 damps the already damped series again, so it is colder on every
        # warm hour.
        warm = T_cell .> T_ref + 1
        @test any(warm)
        shaw07 = band_forcing(top, (; rule = :shaw, band_k = [0.7]))
        @test all(collect(shaw07.forcing.temperature_air)[warm] .<
                  collect(fitted.forcing.temperature_air)[warm] .- 0.1)
        # A glm-0 cell has nothing to convert, so with k = 1 it is its own lapsed series either way.
        low = table[[1], :]
        f0 = band_forcing(low, (; rule = :shaw, band_k = [1.0]))
        c0 = only(hypsometry_intervals(low)).center
        @test collect(f0.forcing.temperature_air) ≈
              [T_obs(z[1], 0.0, a[i]) for i in eachindex(days)] .- lapse .* (c0 - z[1]) ./ 1000 atol = 1e-9
    end

    @testset "the wind level scales the band wind and nothing else" begin
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
            # A wind that varies in time and between cells, so a scaling cannot pass by coincidence.
            fd[:wind_speed] .= [2.0 + 0.37 * sin(0.7i) + c.z / 1000 for i in eachindex(days)]
            fd
        end
        p = derive_downscaling_parameters(:synthetic, (first(days), last(days)), table;
                                          token = nothing, cache_path = nothing, forcing_loader = loader)
        fit = (; time = p.time, p.decoupling, p.lapse_rate, p.lapse_rate_uncertainty)
        intervals = hypsometry_intervals(table)
        bands(; kw...) = collect(elevation_interval_forcing(
            table, resolve_downscaling(fit, intervals, days; basis = :fitted, kw...);
            climate_model = :synthetic, time_range = (first(days), last(days)), forcing_loader = loader))
        layers(b) = Dict(v => collect(b.forcing[v]) for v in keys(b.forcing))
        none, w1, w2 = bands(), bands(; wind_scaling = 1.0), bands(; wind_scaling = 2.0)
        @test length(none) == length(w1) == length(w2) == length(intervals) > 1
        for (b0, b1, b2) in zip(none, w1, w2)
            # w1 is the band without a wind level, bit for bit, metadata included.
            @test layers(b1) == layers(b0)
            @test DimensionalData.metadata(b1.forcing) == DimensionalData.metadata(b0.forcing)
            # w2 doubles the wind exactly and leaves every other layer as it was.
            @test collect(b2.forcing.wind_speed) == 2 .* collect(b0.forcing.wind_speed)
            @test all(layers(b2)[v] == layers(b0)[v] for v in keys(b0.forcing) if v !== :wind_speed)
            @test DimensionalData.metadata(b2.forcing)["wind_speed_scaling"] == 2.0
            @test DimensionalData.metadata(b2.forcing)["wind_speed_mean"] ==
                  2 * DimensionalData.metadata(b0.forcing)["wind_speed_mean"]
        end
        @test resolve_downscaling(fit, intervals, days; basis = :fitted, wind_scaling = 2.0).settings["downscaling_wind_scaling"] === 2.0
        @test resolve_downscaling(fit, intervals, days; basis = :fitted).settings["downscaling_wind_scaling"] === 1.0
        for bad in (0.0, -1.0, Inf, NaN)
            @test_throws "wind_scaling must be a positive finite factor" resolve_downscaling(
                fit, intervals, days; basis = :fitted, wind_scaling = bad)
        end
    end

    @testset "Shaw glacier bands: the area-weighted prior and band k" begin
        t = shaw_glacier_bands()
        @test length(t.rgi_id) == 186_757
        @test issorted(t.rgi_id)
        @test all(i -> isapprox(sum(t.band_area_km2[i]), t.area_km2[i]; rtol = 1e-4), eachindex(t.rgi_id))
        # The table's glacier-area weighted mean, which the fill value rounds.
        @test sum(t.area_km2 .* t.k) / sum(t.area_km2) ≈ SHAW_FILL_DECOUPLING_FACTOR atol = 0.005

        # Two glaciers in the tile, one north of it, one on its western edge (inside) and one on its
        # eastern edge (outside: it belongs to the next tile).
        path = _shaw_table([("RGI60-11.00001", 46.5, 10.5, 0.6, [(3000, 1.0), (3100, 3.0)]),
                            ("RGI60-11.00002", 46.2, 11.0, 0.9, [(3100, 1.0)]),
                            ("RGI60-11.00003", 48.5, 10.5, 0.3, [(3000, 5.0)]),
                            ("RGI60-11.00004", 46.1, 10.0, 0.8, [(2900, 2.0)]),
                            ("RGI60-11.00005", 46.1, 12.0, 0.2, [(3000, 9.0)])])
        tab = shaw_glacier_bands(path)
        @test tab.sha256 == bytes2hex(open(GEMB_GlacierSims.SHA.sha256, path))
        bounds = (lon_min = 10.0, lon_max = 12.0, lat_min = 46.0, lat_max = 48.0)
        @test shaw_decoupling_prior(bounds; table = tab) ≈ (4 * 0.6 + 1 * 0.9 + 2 * 0.8) / 7
        @test shaw_decoupling_prior((lon_min = -40.0, lon_max = -38.0, lat_min = 64.0, lat_max = 66.0);
                                    table = tab) === nothing
        # Per band: 2900 m all Shaw (0.8); 3000 m 1 km² of Shaw (0.6) in 4 km² of ice, the rest at the
        # fill; 3100 m Shaw area (3 + 1) beyond the band's 3 km², so all Shaw; 3200 m no Shaw.
        intervals = [(; lo = 2900, hi = 3000, center = 2950.0), (; lo = 3000, hi = 3100, center = 3050.0),
                     (; lo = 3100, hi = 3200, center = 3150.0), (; lo = 3200, hi = 3300, center = 3250.0)]
        r = shaw_band_decoupling(bounds, intervals, [2.0, 4.0, 3.0, 1.5]; table = tab)
        @test r.band_k ≈ [0.8, (0.6 + 3 * 0.77) / 4, (3 * 0.6 + 0.9) / 4, 0.77]
        @test r.covered_fraction ≈ [1.0, 0.25, 1.0, 0.0]
        @test r.shaw_area_km2 ≈ [2.0, 1.0, 4.0, 0.0]
        @test shaw_band_decoupling(bounds, intervals, [2.0, 4.0, 3.0, 1.5]; table = tab,
                                   fill_k = 1.0).band_k[4] == 1.0
        @test_throws "needs 100 m bands" shaw_band_decoupling(
            bounds, [(; lo = 3000, hi = 3050, center = 3025.0)], [1.0]; table = tab)
        @test_throws DimensionMismatch shaw_band_decoupling(bounds, intervals, [1.0]; table = tab)
        @test_throws "no Shaw glacier band table at" shaw_glacier_bands("/nonexistent.csv.gz")
        bad = _shaw_table([("RGI60-11.00001", 46.5, 10.5, 1.2, [(3000, 1.0)])])
        @test_throws "invalid row" shaw_glacier_bands(bad)
        # A tile's band areas from its hypsometry.
        cells = _cp_table([(lon = 10.5, lat = 46.5, z = 3000.0, glm = 1.0, bins = [(3050, 1.5), (3150, 2.0)]),
                           (lon = 10.6, lat = 46.5, z = 3000.0, glm = 1.0, bins = [(3050, 0.5)])])
        @test tile_interval_areas(cells, hypsometry_intervals(cells)) ≈ [2.0, 2.0]
    end

    @testset "the glacier-fraction form of the decoupling fit" begin
        @test decoupling_glacier_fraction(0.3, :fraction) == 0.3
        @test decoupling_glacier_fraction(0.49, :indicator) == 0.0
        @test decoupling_glacier_fraction(GLACIER_INDICATOR_THRESHOLD, :indicator) == 1.0
        @test_throws "glacier fraction form must be one of (:fraction, :indicator), got :step" decoupling_glacier_fraction(
            0.5, :step)
        # A region whose cells above glm 0.5 are damped as a whole and those below not at all: the
        # indicator form fits it exactly, and the derivation converts its donors with the same form.
        T_ref = 273.15
        z = collect(range(300.0, 3000.0; length = 24))
        glm = [clamp((zi - 300) / 2700 + 0.2 * sin(2i), 0.0, 1.0) for (i, zi) in enumerate(z)]
        days = [DateTime(2003, 7, d, h) for d in 1:31 for h in (12, 15)]
        a = [16.0 + 6.0 * sin(0.9i) for i in eachindex(days)]
        k = 0.6
        T_obs(zi, g, ai) = (e = ai - 0.006 * zi; e > 0 ? T_ref + (1 - (g >= 0.5) * (1 - k)) * e : T_ref + e)
        own_bin(zi) = 100 * floor(Int, zi / 100) + 50
        cells = [(lon = 10.0 + 0.01i, lat = 46.0, z = z[i], glm = glm[i], bins = [(own_bin(z[i]), 1.0)])
                 for i in eachindex(z)]
        table = _cp_table(cells)
        loader = function (_model, lat, lon; kwargs...)
            c = cells[findfirst(c -> c.lat == lat && c.lon == lon, cells)]
            _audit_forcing(days, lat, lon, c.z, [T_obs(c.z, c.glm, a[i]) for i in eachindex(days)])
        end
        derive(form) = derive_downscaling_parameters(:synthetic, (first(days), last(days)), table;
                                                     token = nothing, cache_path = nothing,
                                                     forcing_loader = loader, glacier_fraction_form = form)
        pin = derive(:indicator)
        pfr = derive(:fraction)
        @test pin.glacier_fraction_form === :indicator
        @test pin.provenance["glacier_fraction_form"] == "indicator"
        @test pin.applied.settings["downscaling_glacier_fraction_form"] == "indicator"
        # Every fitted step recovers k (a step with too few warm cells is not fitted), and the
        # fraction form does not.
        kin = collect(pin.decoupling.decoupling_factor)
        fitted_steps = isfinite.(kin)
        @test count(fitted_steps) > length(days) / 2
        @test kin[fitted_steps] ≈ fill(k, count(fitted_steps)) atol = 1e-9
        kfr = collect(pfr.decoupling.decoupling_factor)
        @test maximum(abs.(kfr[isfinite.(kfr)] .- k)) > 0.01
        # On the fitted steps every band of the indicator derivation is the true decoupled band,
        # from every donor glm: the donors are converted with the same form.
        for b in collect(pin.elevation_interval_forcing)
            truth = [(e = a[i] - 0.006 * b.center; e > 0 ? T_ref + k * e : T_ref + e) for i in eachindex(days)]
            @test maximum(abs.(collect(b.forcing.temperature_air)[fitted_steps] .- truth[fitted_steps])) <= 1e-9
        end
        @test_throws "glacier_fraction_form must be one of" derive(:step)

        # The form is a derivation setting: recorded in the file, read back, and compared on a re-run.
        tile = only(downscaling_tiles(table; tile_size = 2, buffer = 1))
        path = write_sparse_downscaling_tile_netcdf(joinpath(mktempdir(), tile.name), tile;
                                                    climate_model = :synthetic,
                                                    time_range = (first(days), last(days)),
                                                    tile_size = 2, buffer = 1, reason = "test",
                                                    glacier_fraction_form = :indicator)
        @test read_downscaling_tile(path).glacier_fraction_form === :indicator
        status = read_downscaling_tile_status(path)
        @test status.parameters["glacier_fraction_form"] == "indicator"
        # Its stored settings with a covering window: current under the form it was derived with only.
        covering = (; status.parameters, time_first = first(days), time_last = last(days))
        current(form) = GEMB_GlacierSims._tile_is_current(
            covering, (first(days), last(days)); climate_model = :synthetic, tile_size = 2, buffer = 1,
            min_cells = GEMB_GlacierSims._MIN_CELLS_DEFAULT, area_minimum = 0.0,
            retain_elevation_interval_forcing = false, glacier_fraction_form = form)
        @test current(:indicator)
        @test !current(:fraction)
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
        # `_flat_applied` resolves no rule, which is the fitted one on the converted ambient.
        flat = _flat_applied(time, [(1000, 1100)])
        shawapplied = GEMB_GlacierSims.AppliedDownscaling(
            flat.time, flat.lapse_rate, flat.lapse_rate_source, flat.bands, flat.basis,
            Dict{String,Any}("downscaling_band_decoupling_rule" => "shaw",
                             "downscaling_band_decoupling_band_k" => [0.8],
                             "downscaling_donor_conversion" => "none"))

        m = physics_member("C-kshaw-table")
        mp = physics_model_parameters(m)
        p = tile_run_parameters(mp, shawapplied; spinup_window = window, background_black_carbon = bbc,
                                physics_member = m)
        @test p["physics_member"] == "C-kshaw-table"
        @test p["physics_grain"] == "C"
        @test p["physics_band_decoupling"] == "kshaw"
        @test p["physics_impurity"] == "table"
        @test p["model_grain_growth_method"] === :FlannerZender2006
        @test p["model_black_carbon_snow"] == bbc.black_carbon_snow
        # Without a member, no `physics_*` key is written.
        plain = tile_run_parameters(mp, shawapplied; spinup_window = window, background_black_carbon = bbc)
        @test !any(startswith("physics_"), keys(plain))
        # Two members are two experiments.
        mA = physics_member("A-kshaw-table")
        pA = tile_run_parameters(physics_model_parameters(mA), shawapplied; spinup_window = window,
                                 background_black_carbon = bbc, physics_member = mA)
        d = run_parameter_differences(p, pA)
        @test haskey(d, "physics_member") && haskey(d, "physics_grain")
        @test haskey(d, "model_grain_growth_method")

        # A run whose settings are not the member's is refused, naming what disagrees.
        @test_throws "physics member C-kshaw-table sets grain_growth_method = :FlannerZender2006, but the run's ModelParameters has :hybrid" tile_run_parameters(
            initialize_parameters(), shawapplied; spinup_window = window, background_black_carbon = bbc,
            physics_member = m)
        @test_throws "applies band decoupling (rule = \"shaw\", donor_conversion = \"none\"), but the downscaling was resolved with (rule = \"fitted\", donor_conversion = \"fitted\")" tile_run_parameters(
            mp, flat; spinup_window = window, background_black_carbon = bbc, physics_member = m)
        @test_throws "needs the background loading" tile_run_parameters(
            mp, shawapplied; spinup_window = window, physics_member = m)
        @test_throws "with basis :bc_dust and optics :snicar_adv3, but the loading given has basis :bc and optics :snicar_adv3" tile_run_parameters(
            mp, shawapplied; spinup_window = window, physics_member = m,
            background_black_carbon = background_black_carbon(cells; basis = :bc))
        m0 = physics_member("C-kshaw-bc0")
        @test_throws "runs without a snow impurity loading, but one was given" tile_run_parameters(
            mp, shawapplied; spinup_window = window, background_black_carbon = bbc, physics_member = m0)
        @test tile_run_parameters(mp, shawapplied; spinup_window = window,
                                  physics_member = m0)["model_black_carbon_snow"] == 0.0

        # The wind level must be the one the downscaling was resolved with.
        windapplied(w) = GEMB_GlacierSims.AppliedDownscaling(
            shawapplied.time, shawapplied.lapse_rate, shawapplied.lapse_rate_source, shawapplied.bands,
            shawapplied.basis, merge(shawapplied.settings, Dict{String,Any}("downscaling_wind_scaling" => w)))
        mw = physics_member("C-kshaw-table-w2")
        pw = tile_run_parameters(mp, windapplied(2.0); spinup_window = window, background_black_carbon = bbc,
                                 physics_member = mw)
        @test pw["physics_wind_scaling"] === 2.0 && pw["downscaling_wind_scaling"] === 2.0
        @test_throws "physics member C-kshaw-table-w2 scales the band wind by 2.0, but the downscaling was resolved with wind_scaling = 1.0" tile_run_parameters(
            mp, windapplied(1.0); spinup_window = window, background_black_carbon = bbc, physics_member = mw)
        @test_throws "physics member C-kshaw-table scales the band wind by 1.0, but the downscaling was resolved with wind_scaling = 2.0" tile_run_parameters(
            mp, windapplied(2.0); spinup_window = window, background_black_carbon = bbc, physics_member = m)
        @test haskey(run_parameter_differences(p, pw), "physics_wind_scaling")

        # The capped effective loading, fitted for the member's own grain variant.
        mv = physics_member("C-kfit-cvis")
        table = _effective_table([("11", 0.3)])
        ebc = effective_black_carbon(cells, table; cap = 0.045)
        q = tile_run_parameters(mp, flat; spinup_window = window, effective_black_carbon = ebc,
                                physics_member = mv)
        @test q["model_black_carbon_snow"] == 0.045
        @test q["effective_black_carbon_cap"] == 0.045
        @test q["effective_black_carbon_variant"] == "C0"
        @test q["effective_black_carbon_region_values"] == "11:0.045"
        @test q["effective_black_carbon_table_sha256"] == ebc.table_sha256
        @test !isempty(run_parameter_differences(p, q))
        @test_throws "needs an effective loading" tile_run_parameters(
            mp, flat; spinup_window = window, background_black_carbon = bbc, physics_member = mv)
        @test_throws "caps the effective loading at 0.045 ppmw, but the loading given is capped at Inf" tile_run_parameters(
            mp, flat; spinup_window = window, physics_member = mv,
            effective_black_carbon = effective_black_carbon(cells, table))
        @test_throws "has grain variant C, but the effective loading was fitted for variant B0" tile_run_parameters(
            mp, flat; spinup_window = window, physics_member = mv,
            effective_black_carbon = effective_black_carbon(cells, _effective_table([("11", 0.3)];
                                                                                   variant = "B0");
                                                            cap = 0.045))
        @test_throws "give a background loading or an effective loading, not both" tile_run_parameters(
            mp, shawapplied; spinup_window = window, background_black_carbon = bbc,
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
        c = load(clear..., "PHYSICS_MEMBER" => "C-kshaw-bc0")
        mp = call(c, :physics_model_parameters; output_frequency = :weekly)
        @test mp.grain_growth_method === :FlannerZender2006 && mp.output_frequency === :weekly
        @test get_(c, :PHYSICS_MEMBER).band_rule == (; rule = :shaw)
        @test get_(c, :BACKGROUND_BLACK_CARBON) === :off
        @test call(c, :tile_background_black_carbon, cells) === nothing
        @test call(c, :tile_run_dir, "/cache") == "/cache/tile_runs_v2/C-kshaw-bc0"
        # A Shaw member fills its rule with the tile's band k; the fitted member does not.
        tile = (; core = cells, bounds = (lon_min = 6.0, lon_max = 8.0, lat_min = 44.0, lat_max = 46.0))
        iv = hypsometry_intervals(cells)
        r = call(c, :tile_band_decoupling, tile, iv)
        @test r.rule === :shaw
        @test r.band_k == shaw_band_decoupling(tile.bounds, iv, tile_interval_areas(cells, iv)).band_k
        @test call(d, :tile_band_decoupling, tile, iv) == (; rule = :fitted)
        @test call(d, :tile_decoupling_factor_prior, tile) == shaw_decoupling_prior(tile.bounds)
        # An `export` that agrees is accepted; one that disagrees stops the script.
        @test get_(load(clear..., "PHYSICS_MEMBER" => "C-kshaw-table",
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
        v = load(clear..., "PHYSICS_MEMBER" => "C-kfit-cvis", "EFFECTIVE_LOADING_TABLE" => table)
        @test get_(v, :BACKGROUND_BLACK_CARBON) === :off
        @test call(v, :tile_effective_black_carbon, cells).black_carbon_snow == 0.045
        @test_throws "takes an effective loading; set EFFECTIVE_LOADING_TABLE" load(
            clear..., "PHYSICS_MEMBER" => "C-kfit-cvis")
        @test_throws "EFFECTIVE_LOADING_TABLE=/nonexistent.csv does not exist" load(
            clear..., "PHYSICS_MEMBER" => "C-kfit-cvis", "EFFECTIVE_LOADING_TABLE" => "/nonexistent.csv")
        @test_throws "EFFECTIVE_LOADING_TABLE is set, but physics member B-kfit-table takes no effective loading" load(
            clear..., "EFFECTIVE_LOADING_TABLE" => table)

        # Both scripts run as the member: model parameters, band rule and both loadings.
        for script in ("gemb_tile_sweep.jl", "gemb_native_run.jl")
            text = read(joinpath(repo, "scripts", script), String)
            @test occursin("mp = physics_model_parameters(output_frequency = OUTPUT_FREQUENCY)", text)
            @test !occursin("GEMB.initialize_parameters", text)
            @test count("band_decoupling = band_rule", text) == 2
            @test count("band_rule = tile_band_decoupling(tile, intervals)", text) == 1
            @test occursin("tile_decoupling_factor_prior(tile)", text)
            @test !occursin("decoupling_factor_prior(tile.core)", text)
            @test count("effective_black_carbon = effective", text) == 2
            @test count(r"physics_member = PHYSICS_MEMBER[,)\s]", text) == 2
            @test occursin("tile_run_dir(CLIMATE_CACHE)", text)
        end
    end
end
