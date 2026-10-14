# Finding: in the merged tile runs, spinups unconverged at the 100-year ceiling cover a large share of
# glacier area, and carry a residual firn-air drift into the transient.
#
# Scans every tile file of a merged sweep directory (read only) and reports, at three perturbation
# points, the glacier-area fraction of unconverged columns and their area-weighted firn-air drift.
#
# Run from the repository root:
#   julia --project=<env> scripts/audit/spinup_unconverged_outputs.jl [tile_dir]
# tile_dir defaults to $CLIMATE_CACHE/tile_runs_merged. Any branch reads these files.
# Expected on the 100-year sweep: about 11% of area unconverged at (0 K, pscale 1) and 41% at -3 K.
using GEMB_GlacierSims
const NCDatasets = GEMB_GlacierSims.NCDatasets

root = length(ARGS) >= 1 ? ARGS[1] :
       joinpath(get(ENV, "CLIMATE_CACHE", "/mnt/bylot-r3/data/era5land"), "tile_runs_merged")
files = sort(filter(endswith(".nc"), readdir(root)))
NCDatasets.NCDataset(joinpath(root, first(files))) do ds
    global deltas = collect(ds["delta_temperature"][:]); global scalings = collect(ds["precipitation_scaling"][:])
end
for (dT, ps) in ((0.0, 1.0), (-3.0, 1.0), (3.0, 1.0))
    idt, ips = findfirst(==(dT), deltas), findfirst(==(ps), scalings)
    (idt === nothing || ips === nothing) && continue
    total = unconverged = drift_area = 0.0
    n = 0
    for f in files
        NCDatasets.NCDataset(joinpath(root, f)) do ds
            area = Float64.(ds["band_area"][:])
            converged = ds["spinup_converged"][:, idt, ips]
            # The last cycle's drift as a fraction of the tolerance, times the 1 cm/yr tolerance.
            drift = 0.01 .* ds["spinup_final_drift_density"][:, idt, ips] ./
                    ds["spinup_convergence_drift_density"][:, idt, ips]
            for b in eachindex(area)
                total += area[b]
                (ismissing(converged[b]) || converged[b] == 1) && continue
                unconverged += area[b]; n += 1
                ismissing(drift[b]) || (drift_area += area[b] * abs(drift[b]))
            end
        end
    end
    println("ΔT = $dT, pscale = $ps: $n unconverged columns, ", round(100 * unconverged / total; digits = 1),
            "% of glacier area, area-weighted |drift| ", round(drift_area / max(unconverged, eps()); sigdigits = 3),
            " m/yr over them")
end
