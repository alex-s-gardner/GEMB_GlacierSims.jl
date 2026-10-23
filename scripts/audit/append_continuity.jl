# Finding: `append_glacier_tile_netcdf` restarts cumulative series at zero at the seam.
#
# A tile file of two monthly steps with dh = total_dv = [-1, -2] is extended by a continuation with the
# same values. The continuation's series start at its own seam, so a continuous record reads
# [-1, -2, -3, -4]; written as they are, the record reads [-1, -2, -1, -2].
#
# Run from the repository root:  julia --project=<env> scripts/audit/append_continuity.jl
# Expected on feat/native-resolution-run:  dh after append = [-1.0, -2.0, -1.0, -2.0]  (DISCONTINUOUS)
# Expected on fix/pipeline-audit:          dh after append = [-1.0, -2.0, -3.0, -4.0]  (continuous)
include(joinpath(@__DIR__, "common.jl"))
import NCDatasets

path = joinpath(mktempdir(), "tile.nc")
write_glacier_tile_netcdf(path, minimal_tile_run([DateTime(2000, 1, 31), DateTime(2000, 2, 29)]); precision = Float64)
append_glacier_tile_netcdf(path, minimal_tile_run([DateTime(2000, 3, 31), DateTime(2000, 4, 30)]))
NCDatasets.NCDataset(path) do ds
    dh = collect(Float64, ds["dh"][:, 1, 1, 1])
    dv = collect(Float64, ds["total_dv"][:, 1, 1])
    println("dh after append       = ", dh, dh == [-1.0, -2.0, -3.0, -4.0] ? "  (continuous)" : "  (DISCONTINUOUS)")
    println("total_dv after append = ", dv)
end
