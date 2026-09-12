using Test
using GEMB_ClimateForcing
using DimensionalData
using Statistics

const GH = GEMB_ClimateForcing

# A minimal stand-in for what `bare_ice_albedo` returns: the layers the reduction reads, over a
# `Dim{:cell}` axis. Built here rather than by calling `bare_ice_albedo` so these tests need no
# pooled table and no network.
function _hyps_stack(albedo, n_valid = fill(100, length(albedo)),
                     in_product = trues(length(albedo)); sky = :bsa)
    cell = Dim{:cell}(1:length(albedo))
    return DimStack(NamedTuple((
        Symbol("albedo_", sky) => DimArray(collect(albedo), (cell,)),
        Symbol("n_valid_", sky) => DimArray(collect(n_valid), (cell,)),
        :in_product => DimArray(collect(in_product), (cell,)),
    )))
end

@testset "bare_ice_albedo_hyps" begin

    @testset "recovers a known linear albedo(z)" begin
        # 0.30 at 0 m rising by 0.05 per 100 m, i.e. 0.5 per km. Elevations span several bins so
        # the fit is well conditioned, and every bin holds enough cells to be `:observed`.
        z = collect(500.0:10.0:1500.0)
        a = 0.30 .+ 0.0005 .* z
        f = bare_ice_albedo_hyps(_hyps_stack(a), z, 0:100:2000)

        @test f.fit.slope_per_km ≈ 0.5 rtol = 1e-8
        @test f.fit.intercept ≈ 0.30 rtol = 1e-8
        # An exact line has no residual, so the slope's standard error is zero and r2 is one — to
        # within the rounding of sums over 1e6-scale squared elevations, seven orders of magnitude
        # below the slope itself.
        @test f.fit.slope_stderr ≈ 0 atol = 1e-7
        @test f.fit.r2 ≈ 1 rtol = 1e-9
        @test f.fit.n == length(z)
        @test f.fit.elevation_range == (500.0, 1500.0)

        # `slope_per_km` really is per km, not per metre: 900 m of relief moves albedo by 0.45.
        # Both bins are inside the observed span, so both are `:observed`.
        @test f(1450) - f(550) ≈ 0.45 rtol = 0.02
    end

    @testset "observed bins are the mean of their own cells" begin
        # Two cells in the 1000-1100 bin, three in 1100-1200.
        z = [1010.0, 1090.0, 1110.0, 1150.0, 1190.0]
        a = [0.40, 0.50, 0.20, 0.30, 0.40]
        f = bare_ice_albedo_hyps(_hyps_stack(a), z, 1000:100:1300; min_cells = 3)

        # 1100-1200 has 3 cells, so it reports its own mean.
        @test bare_ice_albedo_source(f, 1150) === :observed
        @test f(1150) ≈ mean([0.20, 0.30, 0.40])
        # 1000-1100 has only 2, so it is fit-filled — but its observed mean is still recorded.
        @test bare_ice_albedo_source(f, 1050) === :fit
        @test f.observed[1] ≈ mean([0.40, 0.50])
        @test f(1050) != f.observed[1]

        @test f.n_cells == [2, 3, 0]
        @test f.n_valid == [200, 300, 0]
    end

    @testset "the fit sees cells, not bin means" begin
        # 1000-1100 holds one cell, 1100-1200 holds nine. Fitting the two bin means would weight
        # them equally; fitting the cells does not, so the slope follows the nine.
        z = vcat([1050.0], fill(1150.0, 8), [1190.0])
        a = vcat([0.90], fill(0.30, 8), [0.28])
        f = bare_ice_albedo_hyps(_hyps_stack(a), z, 1000:100:1300; min_cells = 1)
        # The bin-mean regression would give (0.29 - 0.90) / 0.1 km = -6.1 per km. Over the cells
        # the single high point is one of ten, so the slope is far shallower.
        @test f.fit.slope_per_km > -6.0
        @test f.fit.n == 10
    end

    @testset "drops missing, NaN and off-product cells, and counts why" begin
        z = [1050.0, 1150.0, 1250.0, 1350.0, 1450.0, 1550.0]
        a = [0.40, NaN32, 0.30, 0.35, 0.45, 0.50]
        in_product = [true, true, false, true, true, true]
        # `missing` elevation (a DEM hole) and a non-finite one are both unusable.
        e = Union{Missing,Float64}[1050.0, 1150.0, 1250.0, missing, NaN, 1550.0]
        f = bare_ice_albedo_hyps(_hyps_stack(a, fill(10, 6), in_product), e, 1000:100:1700)

        @test f.n_input == 6
        @test f.n_no_albedo == 2      # the NaN albedo and the off-product cell
        @test f.n_no_elevation == 2   # the missing and the NaN elevation
        @test f.n_used == 2
        @test sum(f.n_cells) == 2
        # An off-product cell is not glacier at all, so its bin holds nothing.
        @test f.n_cells[3] == 0
    end

    @testset "accepts Rasters.extract output shapes" begin
        z = collect(1000.0:50.0:1500.0)
        a = 0.30 .+ 0.0002 .* z
        want = bare_ice_albedo_hyps(_hyps_stack(a), z, 1000:100:1600)

        # `extract(...; geometry = false)`: the value field carries the raster's own name, which
        # for the Copernicus DEM is empty and so cannot be requested by name.
        bare = [NamedTuple{(Symbol(""),)}((Float32(v),)) for v in z]
        @test bare_ice_albedo_hyps(_hyps_stack(a), bare, 1000:100:1600).albedo ≈
              want.albedo nans = true
        # `extract(...)` with the geometry kept: skip `:geometry`, take the next field.
        withgeom = [(; geometry = (0.0, 0.0), var"" = Float32(v)) for v in z]
        @test bare_ice_albedo_hyps(_hyps_stack(a), withgeom, 1000:100:1600).albedo ≈
              want.albedo nans = true
    end

    @testset "degenerate input gives NaN, never a zero slope" begin
        # Two cells cannot support a slope and its standard error.
        f2 = bare_ice_albedo_hyps(_hyps_stack([0.3, 0.4]), [1000.0, 1200.0], 1000:100:1300)
        @test isnan(f2.fit.slope_per_km)
        @test isnan(f2.fit.slope_stderr)
        @test f2.fit.n == 0

        # Every cell at one elevation: no slope is identifiable however many cells there are.
        fflat = bare_ice_albedo_hyps(_hyps_stack(fill(0.4, 20)), fill(1150.0, 20), 1000:100:1300)
        @test isnan(fflat.fit.slope_per_km)
        # The populated bin still reports its own observations; only the fit is unavailable, so
        # the empty bins are `:none` rather than fit-filled.
        @test bare_ice_albedo_source(fflat, 1150) === :observed
        @test bare_ice_albedo_source(fflat, 1050) === :none
        @test isnan(fflat(1050))

        # Constant albedo over spread elevation: the slope is exactly zero and known to be, so it
        # is reported as zero. r2 is NaN because the total sum of squares is zero.
        fconst = bare_ice_albedo_hyps(_hyps_stack(fill(0.4, 20)),
                                      collect(1000.0:25.0:1475.0), 1000:100:1600)
        @test fconst.fit.slope_per_km ≈ 0 atol = 1e-12
        @test isnan(fconst.fit.r2)
    end

    @testset "no extrapolation past the observed elevations by default" begin
        # A steep fit over a narrow span would, extended across the whole 0-10000 m binning,
        # return albedos far outside 0-1.
        z = collect(2000.0:25.0:2500.0)
        a = 0.65 .- 0.0002 .* (z .- 2000)
        f = bare_ice_albedo_hyps(_hyps_stack(a), z, 0:100:10000)

        resolved = filter(!isnan, f.albedo)
        @test !isempty(resolved)
        @test all(0 .<= resolved .<= 1)
        # Nothing is resolved a kilometre below or above the glacier.
        @test isnan(f(1000))
        @test isnan(f(3500))
        @test bare_ice_albedo_source(f, 1000) === :none

        # `extrapolate` reaches past the observed span, and reaches further with a larger value.
        g = bare_ice_albedo_hyps(_hyps_stack(a), z, 0:100:10000; extrapolate = 300)
        @test count(!isnan, g.albedo) > count(!isnan, f.albedo)
        @test bare_ice_albedo_source(g, 1800) === :fit
    end

    @testset "extrapolate = :hold freezes the fit at the sampled edges" begin
        # A steep profile over a narrow span. Held outward it must stay inside what the fit covers
        # over the sampled range, which the fit extended past the data cannot promise.
        z = collect(2000.0:25.0:2500.0)
        a = 0.65 .- 0.0002 .* (z .- 2000)
        f = bare_ice_albedo_hyps(_hyps_stack(a), z, 0:100:10000; extrapolate = :hold)

        obs = findall(==(:observed), f.sources)
        @test !isempty(obs)
        # The held value is the FIT at the lowest / highest observed elevation, not that end bin's
        # own mean: the fit sees every observation, an end bin only what fell in it.
        slope = f.fit.slope_per_km / 1000
        lo_val = f.fit.intercept + slope * f.fit.elevation_range[1]
        hi_val = f.fit.intercept + slope * f.fit.elevation_range[2]

        @test all(isfinite, f.albedo)
        @test all(min(lo_val, hi_val) - 1e-12 .<= f.albedo .<= max(lo_val, hi_val) + 1e-12)
        @test all(0 .<= f.albedo .<= 1)

        # Below the span, the fit at the lowest observed elevation — at any distance.
        @test f(0) ≈ lo_val
        @test f(1000) ≈ lo_val
        @test f(1900) ≈ lo_val
        @test bare_ice_albedo_source(f, 1000) === :hold
        # Above it, the fit at the highest observed elevation.
        @test f(3000) ≈ hi_val
        @test f(9950) ≈ hi_val
        @test bare_ice_albedo_source(f, 9950) === :hold
        # Inside the span nothing changes.
        @test bare_ice_albedo_source(f, 2250) === :observed
        @test f(2250) == bare_ice_albedo_hyps(_hyps_stack(a), z, 0:100:10000)(2250)

        # No observed bin, so there is no edge to freeze at: unresolved, not a propagated NaN.
        none = bare_ice_albedo_hyps([0.4, 0.5], [1000.0, 1200.0], 0:100:3000;
                                    min_cells = 3, extrapolate = :hold)
        @test all(==(:none), none.sources)
        @test all(isnan, none.albedo)

        # An interior bin under min_cells is still fit-filled: :hold governs the ends only.
        zi = vcat(fill(1050.0, 5), [1150.0], fill(1250.0, 5))
        ai = vcat(fill(0.30, 5), [0.9], fill(0.40, 5))
        g = bare_ice_albedo_hyps(ai, zi, 1000:100:1300; min_cells = 3, extrapolate = :hold)
        @test bare_ice_albedo_source(g, 1150) === :fit
    end

    @testset "z outside the bins, and the exclusive top edge" begin
        # Cells reach into the topmost bin, so it is observed and the exclusive top edge is the
        # only thing that can leave a value just below 2000 m unresolved.
        z = collect(1000.0:25.0:1975.0)
        a = fill(0.4, length(z))
        f = bare_ice_albedo_hyps(_hyps_stack(a), z, 1000:100:2000)

        @test isnan(f(999))
        @test isnan(f(-5))
        @test isnan(f(NaN))
        @test isnan(f(Inf))
        # `[lo, hi)`: the very top edge belongs to no bin.
        @test isnan(f(2000))
        @test !isnan(f(1999))
        # The lowest edge is inclusive.
        @test !isnan(f(1000))
        @test bare_ice_albedo_source(f, 2000) === :none
    end

    @testset "broadcasting and the table view" begin
        z = collect(1000.0:25.0:1500.0)
        a = 0.30 .+ 0.0002 .* z
        f = bare_ice_albedo_hyps(_hyps_stack(a), z, 1000:100:1600)

        got = f.([1050, 1250, 1450])
        @test length(got) == 3
        @test got == [f(1050), f(1250), f(1450)]

        t = bare_ice_albedo_table(f)
        @test length(t.lo) == length(f.albedo)
        @test t.hi .- t.lo == fill(100.0, length(t.lo))
        @test t.center ≈ (t.lo .+ t.hi) ./ 2
        # Every column is the same length, which is what makes it a valid column table.
        @test allequal(length(getproperty(t, k)) for k in propertynames(t))
        # A bin's row agrees with evaluating at its centre.
        @test all(i -> isequal(f(t.center[i]), t.albedo[i]), eachindex(t.center))
    end

    @testset "white-sky selects the other layer" begin
        z = collect(1000.0:25.0:1500.0)
        cell = Dim{:cell}(1:length(z))
        st = DimStack((albedo_bsa = DimArray(fill(0.40, length(z)), (cell,)),
                       albedo_wsa = DimArray(fill(0.55, length(z)), (cell,)),
                       in_product = DimArray(trues(length(z)), (cell,))))
        @test bare_ice_albedo_hyps(st, z, 1000:100:1600; sky = :bsa)(1250) ≈ 0.40
        @test bare_ice_albedo_hyps(st, z, 1000:100:1600; sky = :wsa)(1250) ≈ 0.55
        # No `n_valid_*` companion, so the retrieval count is zero rather than an error.
        @test all(==(0), bare_ice_albedo_hyps(st, z, 1000:100:1600).n_valid)
    end

    @testset "a plain albedo vector needs no stack" begin
        z = collect(1000.0:25.0:1500.0)
        a = 0.30 .+ 0.0002 .* z
        f = bare_ice_albedo_hyps(a, z, 1000:100:1600)
        @test f.n_used == length(z)
        @test f.fit.slope_per_km ≈ 0.2 rtol = 1e-8
        # No counts are available from a bare vector, so none are claimed.
        @test all(==(0), f.n_valid)

        # Supplied explicitly, they are binned like the stack's own.
        counts = fill(7, length(z))
        g = bare_ice_albedo_hyps(a, z, 1000:100:1600; n_valid = counts)
        @test g.n_valid == 7 .* g.n_cells
        @test sum(g.n_valid) == 7 * g.n_used
        @test g.albedo ≈ f.albedo nans = true      # counts do not alter the statistic

        # The keyword overrides what a stack carries, and must line up with the cells.
        st = _hyps_stack(a, fill(100, length(z)))
        @test sum(bare_ice_albedo_hyps(st, z, 1000:100:1600).n_valid) == 100 * length(z)
        @test sum(bare_ice_albedo_hyps(st, z, 1000:100:1600; n_valid = counts).n_valid) ==
              7 * length(z)
        @test_throws "one-to-one and in the same order" bare_ice_albedo_hyps(
            a, z, 1000:100:1600; n_valid = counts[1:end - 1])
    end

    @testset "input validation explains the problem" begin
        z = collect(1000.0:25.0:1500.0)
        a = fill(0.4, length(z))

        @test_throws "sky must be :bsa (black-sky) or :wsa (white-sky)" bare_ice_albedo_hyps(
            _hyps_stack(a), z; sky = :blue)
        @test_throws "min_cells must be >= 1" bare_ice_albedo_hyps(
            _hyps_stack(a), z; min_cells = 0)
        @test_throws "extrapolate must be :hold, or a finite non-negative distance in metres" bare_ice_albedo_hyps(
            _hyps_stack(a), z; extrapolate = -100)
        @test_throws "extrapolate as a symbol must be :hold" bare_ice_albedo_hyps(
            _hyps_stack(a), z; extrapolate = :clamp)
        @test_throws "they must be one-to-one and in the same order" bare_ice_albedo_hyps(
            _hyps_stack(a), z[1:end - 1])
        @test_throws "bins needs at least two edges" bare_ice_albedo_hyps(
            _hyps_stack(a), z, [1000.0])
        @test_throws "bins must increase strictly" bare_ice_albedo_hyps(
            _hyps_stack(a), z, [0.0, 500.0, 400.0, 1000.0])
        # A repeated edge is a zero-width bin, which `searchsortedlast` cannot resolve.
        @test_throws "bins must increase strictly" bare_ice_albedo_hyps(
            _hyps_stack(a), z, [0.0, 500.0, 500.0, 1000.0])
        # A stack without the requested albedo layer names the layers it does have.
        cell = Dim{:cell}(1:length(z))
        @test_throws "no :albedo_bsa layer in the albedo input" bare_ice_albedo_hyps(
            DimStack((wrong = DimArray(a, (cell,)),)), z)
    end

    @testset "show reports coverage and provenance" begin
        z = collect(2000.0:25.0:2500.0)
        a = 0.65 .- 0.0002 .* (z .- 2000)
        f = bare_ice_albedo_hyps(_hyps_stack(a), z, 0:100:10000)
        s = sprint(show, MIME"text/plain"(), f)
        @test occursin("sky = :bsa", s)
        @test occursin("observed", s)
        @test occursin("per km", s)
        # The unresolved bins are reported, not hidden: most of a 0-10000 m binning is empty here.
        @test occursin("unresolved", s)

        # A struct with no fit at all still prints.
        fflat = bare_ice_albedo_hyps(_hyps_stack(fill(0.4, 20)), fill(1150.0, 20), 1000:100:1300)
        @test occursin("none identifiable", sprint(show, MIME"text/plain"(), fflat))
    end
end
