# Check that `gemb_glacier_cell(...; threaded=true)` reproduces the serial result exactly.
#
# The threaded path must be a pure speedup: same totals, same restart profiles, same time axis.
# Bit-for-bit is the right bar here, not approximate — the reduction is ordered, so any
# difference would mean state is leaking between concurrent simulations rather than rounding.
#
# Run with more than one thread or it proves nothing:
#   julia -t 6 --project scripts/check_threaded_equivalence.jl

using Dates
using GeoDataFrames
using GEMB
using GEMB_GlacierSims
using GEMB_ClimateForcing
using DimensionalData
using Rasters

Threads.nthreads() == 1 &&
    @warn "Running with 1 thread; the threaded path will not actually run concurrently"

const CLASSES_FILE = joinpath(@__DIR__, "..", "data",
                              "era5land_glacier_elevation_classes.parquet")
# Shared persistent forcing cache: the ERA5-Land chunks are tens of GB and expensive to re-fetch, so
# they live off `tempdir()` and survive a reboot. Override with `ENV["CLIMATE_CACHE"]` elsewhere.
const CLIMATE_CACHE = get(ENV, "CLIMATE_CACHE", "/mnt/bylot-r3/data/era5land")
const FORCING_CACHE = joinpath(CLIMATE_CACHE, "cache")

classes = GeoDataFrames.read(CLASSES_FILE)
classes[!, :longitude] = GeoDataFrames.GeoInterface.x.(classes.geometry)
classes[!, :latitude]  = GeoDataFrames.GeoInterface.y.(classes.geometry)
rows = collect(eachrow(classes))

# A small Alaska cell: enough bins that threading is exercised, small enough to run twice.
i = findfirst(i -> -170 <= wrap_lon(rows[i].longitude) <= -129 &&
                   54 <= rows[i].latitude <= 72 &&
                   glacier_area_total(rows[i]) >= 1.0, eachindex(rows))
r = rows[i]

# Short forcing window: this test is about equivalence, not about a realistic record.
forcing = climate_forcing(:era5land, r.latitude, r.longitude;
                          time_range = (DateTime(1950, 1, 1), DateTime(1985, 1, 1)),
                          token = GEMB_ClimateForcing.get_cds_api_key(),
                          cache_path = FORCING_CACHE)

mp = initialize_parameters(output_frequency = :monthly)
kw = (; delta_temperatures = [-1.0, 0.0, 1.0],
        precipitation_scalings = [0.5, 1.0, 2.0],
        coverage = 0.95)

bins = length(glacier_hypsometry_coverage(r; coverage = 0.95).modeled)
@info "Cell under test" cell=i bins sims=bins*9 threads=Threads.nthreads()

t0 = time(); serial   = gemb_glacier_cell(r, forcing, mp; kw..., threaded = false); t_serial = time() - t0
t0 = time(); threaded = gemb_glacier_cell(r, forcing, mp; kw..., threaded = true);  t_threaded = time() - t0

# --- compare ---------------------------------------------------------------------------------
const failures = String[]
function report(name, pass)
    pass || push!(failures, name)
    println((pass ? "  PASS  " : "  FAIL  ") * name)
    return pass
end

report("time axis identical", serial.time == threaded.time)
report("bins identical", serial.bins == threaded.bins)
report("weights identical", serial.weights == threaded.weights)

for v in CELL_TOTAL_VARIABLES
    a, b = serial.totals[v], threaded.totals[v]
    same = a == b                     # bit-for-bit, not isapprox
    report("totals[$(v)] bit-for-bit identical", same)
    if !same
        d = maximum(abs.(a .- b))
        rel = d / max(maximum(abs.(a)), eps())
        println("          max abs diff = $(d)  (relative $(rel))")
    end
end

# Restart profiles must match too, or a resumed run would diverge from a serial one.
prof_same = let same = true
    for idx in eachindex(serial.profiles)
        ps, pt = serial.profiles[idx], threaded.profiles[idx]
        if ps === nothing || pt === nothing
            same &= (ps === nothing) == (pt === nothing)
        else
            for v in PROFILE_VARIABLES
                same &= collect(ps[v]) == collect(pt[v])
            end
        end
    end
    same
end
report("all restart profiles bit-for-bit identical", prof_same)

println()
println("serial   : $(round(t_serial, digits=1)) s")
println("threaded : $(round(t_threaded, digits=1)) s   ($(round(t_serial/t_threaded, digits=2))x on $(Threads.nthreads()) threads)")
println()
if isempty(failures)
    println("RESULT: threaded == serial (bit-for-bit)")
else
    println("RESULT: MISMATCH — threaded path is not equivalent:")
    foreach(f -> println("  - ", f), failures)
    exit(1)
end
