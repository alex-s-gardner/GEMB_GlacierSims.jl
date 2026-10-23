# Finding: output files recorded no package versions, and `run_parameter_differences` compared only
# keys both sides carry, so a run under different code passed as the same experiment: a native run
# from the longwave-fixed code (`native_runs_lwfix`) compares equal to the old-physics reference sweep.
#
# 1. Compares the stored run parameters of a reference tile file and a cell file (read only).
# 2. Compares the reference against itself plus one key it lacks, as a file written by code that
#    records `software_*` versions would.
#
# Run from the repository root:
#   julia --project=<env> scripts/audit/parameter_differences.jl <reference_tile.nc> <cell.nc>
# e.g. $CLIMATE_CACHE/tile_runs_merged/S50_E068.nc and $CLIMATE_CACHE/native_runs_lwfix/S50_E068/S49.4_E068.9.nc
# Expected on feat/native-resolution-run:  1. no differences (the code change is invisible);
#                                          2. no differences (the one-sided key is ignored)
# Expected on fix/pipeline-audit:          1. no differences between two old files, which carry no
#                                          versions (files written by this branch carry them);
#                                          2. the one-sided key is reported, with `nothing` for the
#                                          side that lacks it
using GEMB_GlacierSims
const NCDatasets = GEMB_GlacierSims.NCDatasets

read_parameters(p) = NCDatasets.NCDataset(GEMB_GlacierSims._read_run_parameters, p)
ref, cell = read_parameters(ARGS[1]), read_parameters(ARGS[2])
println("software_* keys in the reference:     ", sort!(filter(startswith("software_"), collect(keys(ref)))))
println("1. reference vs cell differences:     ", run_parameter_differences(ref, cell))
newer = merge(ref, Dict{String,Any}("software_gemb_tree" => "0123456789abcdef0123456789abcdef01234567"))
println("2. reference vs reference + one key:  ", run_parameter_differences(ref, newer))
