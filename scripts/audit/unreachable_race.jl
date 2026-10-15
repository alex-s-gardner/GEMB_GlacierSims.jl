# Finding: the per-band `unreachable` flags of the elevation-interval pass are a BitVector written from
# `Threads.@threads`. A BitVector packs 64 flags into one word, so concurrent writes lose flags, and a
# band the validator refused is then reported as a band with no forcing.
#
# One cell holds ice in 64 bands, all of which fail the forcing validator (a 200 K/km lapse rate); the
# pass is repeated 100 times and the number of bands reported as unreachable is tallied.
#
# Run from the repository root with threads:  julia -t 8 --project=<env> scripts/audit/unreachable_race.jl
# Expected on feat/native-resolution-run:  some repetitions report fewer than 64 (flags lost)
# Expected on fix/pipeline-audit:          every repetition reports 64
include(joinpath(@__DIR__, "common.jl"))
using Logging, Test

Threads.nthreads() > 1 || @warn "run with -t 8: the race needs several threads"
n = 3
time = collect(DateTime(2000, 1, 1):Hour(1):DateTime(2000, 1, 1, n - 1))
bins = [(1050 + 100i, 1.0) for i in 0:63]
table = one_cell_table(; bins)
applied = flat_applied(time, [(c - 50, c + 50) for (c, _) in bins]; lapse_rate = 200.0)
loader = (m, lat, lon; kw...) -> synthetic_forcing(time; temperature = 270.0, elevation = 1000.0)
counts = map(1:100) do _
    logs, _ = Test.collect_test_logs() do
        collect(elevation_interval_forcing(table, applied; climate_model = :synthetic,
                                           time_range = (first(time), last(time)), forcing_loader = loader))
    end
    hit = filter(l -> occursin("no donor cell could be adjusted", l.message), logs)
    isempty(hit) ? 0 : only(hit).kwargs[:n_dropped]
end
println("threads = ", Threads.nthreads())
println("bands reported unreachable per repetition: min ", minimum(counts), ", max ", maximum(counts),
        "; repetitions short of 64: ", count(<(64), counts), " of 100")
