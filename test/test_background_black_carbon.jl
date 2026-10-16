# Regional background black carbon: the Dang et al. (2015) dust conversion, the vendored table and
# region outlines, the tile lookup, the run parameters, and a GEMB run whose albedo sees the value.
# Included from `runtests.jl` after `test_pipeline_audit.jl`, whose `_tiny_tile_run` and
# `_tiny_band_forcing` fixtures the run testset uses.

# A table file with `rows` replacing the matching regions of the vendored one, for the validation
# tests. Rows are written as the vendored file writes them: comma-separated, text fields quoted.
function _bbc_table_with(edit::Function)
    lines = readlines(BACKGROUND_BLACK_CARBON_TABLE)
    path = joinpath(mktempdir(), "bbc.csv")
    write(path, join(edit(lines), "\n") * "\n")
    return path
end

@testset "background black carbon" begin
    @testset "dust BC-equivalent (Dang et al. 2015 Eqs. 9-10, Table 5)" begin
        # f = 159 + 23.94 x + 4.30 x², x = log10(C_dust/ppm); C_BC,eq = C_dust / f.
        @test dust_bc_equivalent(1.0) ≈ 1000 / 159                              # x = 0
        @test dust_bc_equivalent(10.0) ≈ 10_000 / (159 + 23.94 + 4.30)          # x = 1
        @test dust_bc_equivalent(100.0) ≈ 100_000 / (159 + 2 * 23.94 + 4 * 4.30) # x = 2
        # f is held at the ends of its fitted range, 0.1-100 ppm.
        @test dust_bc_equivalent(1000.0) ≈ 1_000_000 / (159 + 2 * 23.94 + 4 * 4.30)
        @test dust_bc_equivalent(0.01) ≈ 10 / (159 - 23.94 + 4.30)
        @test dust_bc_equivalent(0.0) == 0.0
        @test_throws DomainError dust_bc_equivalent(-1.0)
    end

    @testset "region outlines" begin
        # One point well inside each of several regions, in both longitude conventions.
        @test rgi_o1_region(-134.4, 58.6) == "01"         # Juneau Icefield
        @test rgi_o1_region(360 - 134.4, 58.6) == "01"
        @test rgi_o1_region(-121.8, 46.9) == "02"         # Mount Rainier
        @test rgi_o1_region(-49.5, 67.1) == "05"          # Kangerlussuaq margin
        @test rgi_o1_region(15.6, 78.2) == "07"           # Svalbard
        @test rgi_o1_region(6.86, 45.83) == "11"          # Mont Blanc
        @test rgi_o1_region(42.44, 43.35) == "12"         # Elbrus
        @test rgi_o1_region(86.81, 43.11) == "13"         # Urumqi Glacier No. 1
        @test rgi_o1_region(76.5, 35.7) == "14"           # Baltoro, Karakoram
        @test rgi_o1_region(86.92, 27.99) == "15"         # Everest
        @test rgi_o1_region(-77.6, -9.1) == "16"          # Cordillera Blanca
        @test rgi_o1_region(-73.5, -49.0) == "17"         # Southern Patagonian Icefield
        @test rgi_o1_region(170.1, -43.5) == "18"         # Southern Alps
        @test rgi_o1_region(0.0, -80.0) == "20"           # Antarctic plateau
        @test_throws "lies in no RGI 7.0 first-order region" rgi_o1_region(-30.0, 40.0)
    end

    @testset "tile region is the one with most glacier area" begin
        # Two Alpine cells (2 km² in all) and one Caucasus cell with 3 km²: the Caucasus wins on
        # area although it has fewer cells. Longitudes are native 0-360, as the table stores them.
        cells = _cp_table([(lon = 6.9, lat = 45.9, z = 3000.0, glm = 1.0, bins = [(3050, 1.0)]),
                           (lon = 7.0, lat = 45.9, z = 3000.0, glm = 1.0, bins = [(3050, 1.0)]),
                           (lon = 42.4, lat = 43.3, z = 3500.0, glm = 1.0, bins = [(3550, 3.0)])])
        @test tile_rgi_o1_region(cells) == "12"
        @test tile_rgi_o1_region(cells[1:2, :]) == "11"
        @test background_black_carbon(cells[1:2, :]) == background_black_carbon("11")
        # A tie goes to the lower code.
        tie = _cp_table([(lon = 6.9, lat = 45.9, z = 3000.0, glm = 1.0, bins = [(3050, 3.0)]),
                         (lon = 42.4, lat = 43.3, z = 3500.0, glm = 1.0, bins = [(3550, 3.0)])])
        @test tile_rgi_o1_region(tie) == "11"
        empty = _cp_table([(lon = 6.9, lat = 45.9, z = 3000.0, glm = 1.0, bins = [(3050, 0.0)])])
        @test_throws "hold no glacier area" tile_rgi_o1_region(empty)
    end

    @testset "lookup" begin
        table = background_black_carbon_table()
        @test sort(filter(r -> length(r) == 2, table.rgi_o1region)) == [lpad(i, 2, '0') for i in 1:20]
        @test "GrIS" in table.rgi_o1region && "AIS" in table.rgi_o1region
        for row in eachrow(table)
            length(row.rgi_o1region) == 2 || continue
            bc = background_black_carbon(row.rgi_o1region)
            # Under `:bc` the loading is the measured BC alone, ng g⁻¹ → ppmw.
            @test bc.black_carbon_snow == row.bc_ng_g / 1000
            @test bc.dust_bc_equivalent_ng_g == 0.0
            @test bc.basis === :bc
            bd = background_black_carbon(row.rgi_o1region; basis = :bc_dust)
            # Dust mass through Dang's f, else the optically measured non-BC BC-equivalent.
            eq = row.dust_ug_g isa Float64 ? dust_bc_equivalent(row.dust_ug_g) :
                 row.nonbc_bc_equivalent_ng_g isa Float64 ? row.nonbc_bc_equivalent_ng_g : 0.0
            @test bd.dust_bc_equivalent_ng_g == eq
            @test bd.black_carbon_snow ≈ (row.bc_ng_g + eq) / 1000
            # Inside the range GEMB validates.
            @test 0 <= bd.black_carbon_snow <= 2
        end
        @test background_black_carbon(7) == background_black_carbon("07")
        @test_throws "basis must be one of" background_black_carbon("07"; basis = :dust)
        @test_throws "RGI 7.0 o1 region" background_black_carbon("GrIS")
        @test_throws "RGI 7.0 o1 region" background_black_carbon("21")
    end

    @testset "a region without a measurement takes its proxy's value" begin
        table = background_black_carbon_table()
        proxied = filter(r -> !isempty(r.proxy_region), table)
        @test nrow(proxied) > 0
        for row in eachrow(proxied)
            length(row.rgi_o1region) == 2 || continue
            proxy = table[findfirst(==(row.proxy_region), table.rgi_o1region), :]
            bbc = background_black_carbon(row.rgi_o1region)
            @test bbc.proxy_region == row.proxy_region
            @test bbc.bc_ng_g == proxy.bc_ng_g
            # The source says which measurement stands in.
            @test occursin(proxy.bc_source, bbc.source)
        end
        measured = filter(r -> isempty(r.proxy_region) && length(r.rgi_o1region) == 2, table)
        @test background_black_carbon(first(measured.rgi_o1region)).proxy_region == "none"
    end

    @testset "table validation" begin
        lines = readlines(BACKGROUND_BLACK_CARBON_TABLE)
        header = first(lines)
        drop07 = _bbc_table_with(l -> filter(x -> !startswith(x, "07,"), l))
        @test_throws "0 rows for RGI region 07" background_black_carbon_table(drop07)
        noheader = _bbc_table_with(l -> [replace(l[1], "bc_source" => "bc_ref"); l[2:end]])
        @test_throws "lacks column(s) bc_source" background_black_carbon_table(noheader)
        # Range out of order: bc_low above bc for region 07.
        function swap_low(l)
            map(l) do x
                startswith(x, "07,") || return x
                f = split(x, ',')
                f[4] = string(parse(Float64, f[3]) + 1)
                f[5] = string(parse(Float64, f[3]) + 2)
                join(f, ',')
            end
        end
        @test_throws "needs 0 ≤ bc_low ≤ bc ≤ bc_high" background_black_carbon_table(
            _bbc_table_with(swap_low))
        @test occursin("rgi_o1region", header)
    end

    @testset "run parameters record the loading" begin
        time = collect(DateTime(2000, 1, 1):Hour(3):DateTime(2000, 1, 2))
        applied = _flat_applied(time, [(1000, 1100)])
        mp = initialize_parameters()
        window = (DateTime(2000, 1, 1), DateTime(2000, 12, 31, 21))
        plain = tile_run_parameters(mp, applied; spinup_window = window)
        @test plain["model_black_carbon_snow"] == 0.0
        @test !any(startswith("background_black_carbon"), keys(plain))
        bbc = background_black_carbon("11")
        p = tile_run_parameters(mp, applied; spinup_window = window, background_black_carbon = bbc)
        @test p["model_black_carbon_snow"] == bbc.black_carbon_snow
        @test p["background_black_carbon_region"] == "11"
        @test p["background_black_carbon_basis"] == "bc"
        @test p["background_black_carbon_proxy_region"] == bbc.proxy_region
        @test p["background_black_carbon_bc_ng_g"] == bbc.bc_ng_g
        @test p["background_black_carbon_dust_bc_equivalent_ng_g"] == 0.0
        @test p["background_black_carbon_source"] == bbc.source
        # The two experiments differ, so a restart of one from the other is refused.
        @test !isempty(run_parameter_differences(plain, p))
    end

    @testset "the scripts' setting" begin
        repo = normpath(joinpath(@__DIR__, ".."))
        load(value) = withenv("BACKGROUND_BLACK_CARBON" => value) do
            m = Module()
            Base.eval(m, :(using Dates))
            Base.include(m, joinpath(repo, "scripts", "run_settings.jl"))
            m
        end
        # The module's bindings are newer than this testset's world, hence `invokelatest`.
        setting(m) = Base.invokelatest(getglobal, m, :BACKGROUND_BLACK_CARBON)
        loading(m, cells) =
            Base.invokelatest(() -> getglobal(m, :tile_background_black_carbon)(cells))
        cells = _cp_table([(lon = 6.9, lat = 45.9, z = 3000.0, glm = 1.0, bins = [(3050, 1.0)])])
        off = load(nothing)
        @test setting(off) === :off
        @test loading(off, cells) === nothing
        for basis in BACKGROUND_BLACK_CARBON_BASES
            m = load(string(basis))
            @test setting(m) === basis
            @test loading(m, cells) == background_black_carbon("11"; basis)
        end
        @test_throws "BACKGROUND_BLACK_CARBON must be off, bc or bc_dust" load("dust")
        # Both scripts pass the tile's loading to the run parameters and to the run.
        for script in ("gemb_tile_sweep.jl", "gemb_native_run.jl")
            text = read(joinpath(repo, "scripts", script), String)
            @test count("background_black_carbon = black_carbon", text) == 2
            @test occursin("black_carbon = tile_background_black_carbon(tile.core)", text)
        end
    end

    @testset "forcing impurity metadata" begin
        mp = GEMB_GlacierSims._with_black_carbon_snow(initialize_parameters(), 0.02)
        f = _tiny_band_forcing(; years = 1)
        cf = GEMB.initialize_forcing(GEMB_GlacierSims._with_impurity_metadata(f, mp))
        @test all(==(0.02), cf.black_carbon_snow)
        @test all(==(mp.black_carbon_ice), cf.black_carbon_ice)
        # GEMB refuses a forcing whose layer differs from the parameter, which is why the tile run
        # sets the metadata at all.
        @test_throws ArgumentError GEMB.gemb(GEMB.initialize_profile(mp, GEMB.initialize_forcing(f)),
                                             GEMB.initialize_forcing(f), mp)
        conflicting = DimensionalData.rebuild(f; metadata = merge(DimensionalData.metadata(f),
                                                  Dict{String,Any}("black_carbon_snow" => 0.5)))
        @test_throws "carries black_carbon_snow = 0.5" GEMB_GlacierSims._with_impurity_metadata(
            conflicting, mp)
    end

    @testset "a tile run reaches the albedo" begin
        # One band of cold, snowy forcing: the surface stays snow, so the Gardner & Sharp (2010)
        # snow albedo is what the shortwave sees all year, and the soot term is all that differs
        # between the two runs.
        forcing = _tiny_band_forcing(; years = 3, temperature = 263.15, precipitation = 0.2,
                                     shortwave = 300.0, longwave = 220.0)
        albedo = Dict{Float64,Vector{Float64}}()
        capture(c) = (output; kw...) -> (albedo[c] = collect(output[:albedo_broadband]))
        bbc = background_black_carbon("11")
        clean = _tiny_tile_run([0.48]; forcing, on_output = capture(0.0))
        dirty = _tiny_tile_run([0.48]; forcing, on_output = capture(bbc.black_carbon_snow),
                               background_black_carbon = bbc)
        @test dirty.parameters["model_black_carbon_snow"] == bbc.black_carbon_snow
        @test dirty.parameters["background_black_carbon_region"] == "11"
        @test clean.parameters["model_black_carbon_snow"] == 0.0
        a0, a1 = albedo[0.0], albedo[bbc.black_carbon_snow]
        @test length(a0) == length(a1)
        # Darker at every output step, by an amount inside Gardner & Sharp's (2010) eq. 8 soot term
        # at this loading, -dα_c = c^0.55 / (0.16 + 0.6 S^0.5 + 1.8 c^0.6 S^-0.25), between the
        # finest grains of the fit (r = 0.05 mm) and the coarsest (r = 5 mm). The zenith and cloud
        # terms scale it by less than a factor of two either way.
        @test all(a1 .< a0)
        c = bbc.black_carbon_snow
        soot(r_mm) = (S = 3 / (0.091 * r_mm); c^0.55 / (0.16 + 0.6 * S^0.5 + 1.8 * c^0.6 * S^-0.25))
        @test minimum(a0 .- a1) > 0.5 * soot(0.05)
        @test maximum(a0 .- a1) < 1.5 * soot(5.0)
    end
end
