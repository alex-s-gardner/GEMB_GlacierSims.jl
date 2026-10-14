# Finding: a tile run's dh includes the first output interval, while dh_mass, dh_water and dh_firn
# are anomalies at the first output time, so dh != dh_mass + dh_water + dh_firn + dh_residual.
#
# Runs GEMB for two 3-year bands on a constant, melting synthetic forcing (about a minute) and reports
# the largest violation of the stored identity and the size of dh[1].
#
# Run from the repository root:  julia --project=<env> scripts/audit/height_components_closure.jl
# Expected on feat/native-resolution-run:  closure error equals |dh[1]| (about 1 m here), NOT CLOSED
# Expected on fix/pipeline-audit:          closure error < 1e-12 m, closed
include(joinpath(@__DIR__, "common.jl"))

run = tiny_tile_run(synthetic_forcing(three_years_3hourly()), [0.3, 0.5])
s = run.bands_series
gap = maximum(abs, s[:dh] .- s[:dh_mass] .- s[:dh_water] .- s[:dh_firn] .- s[:dh_residual])
println("max |dh - (dh_mass + dh_water + dh_firn + dh_residual)| = ", gap, " m",
        gap < 1e-12 ? "  (closed)" : "  (NOT CLOSED)")
println("dh at the first output time (per band)                = ", s[:dh][1, :, 1, 1], " m")
println("max |dh_residual|                                       = ", maximum(abs, s[:dh_residual]), " m")
