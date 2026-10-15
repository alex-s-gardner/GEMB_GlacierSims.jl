# Finding: the hypsometry builder drops a glacier pixel whose elevation lies outside the bin edges
# (here 10 m below the geoid, as at a tidewater front), so a cell's bins sum to less than its glacier
# area.
#
# Two fully glaciated 0.1° cells at 46-46.1°N on a flat 1050 m DEM, one pixel set to -10 m.
#
# Run from the repository root:  julia --project=<env> scripts/audit/hypsometry_drop.jl
# Expected on feat/native-resolution-run:  cell 1 short of the spherical box by one pixel's area;
#                                          lowest bin 0
# Expected on fix/pipeline-audit:          both cells equal the box (to 1e-3); the pixel is in the
#                                          lowest bin
include(joinpath(@__DIR__, "common.jl"))
const Rs = GS.Rasters

xs, ys = 9.905:0.01:11.095, 45.905:0.01:47.095
dem = Rs.Raster(fill(1050.0f0, length(xs), length(ys)), (X(xs), Y(ys)); crs = Rs.EPSG(4326))
dem[findfirst(≈(10.055), collect(xs)), findfirst(≈(46.055), collect(ys))] = -10.0f0
glaciers = Rs.Raster(zeros(Float32, 10, 10), (X(10.05:0.1:10.95), Y(46.05:0.1:46.95)))
geoms = [GI.Polygon([GI.LinearRing([(9.5, 45.5), (11.5, 45.5), (11.5, 47.5), (9.5, 47.5), (9.5, 45.5)])])]
gtree = GS.SortTileRecursiveTree.STRtree(geoms)
gp = (; geometry = [GI.Point(10.05, 46.05), GI.Point(10.15, 46.05)],
      index = [CartesianIndex(1, 1), CartesianIndex(2, 1)])
edges = 0:100:10000
H = zeros(2, length(edges) - 1)
GS.tile_hypsometry!(H, gp, 1:2, dem, geoms, gtree, glaciers, (0.1, 0.1), edges, length(edges) - 1,
                    Base.Semaphore(1))
box = 6371.0088^2 * deg2rad(0.1) * (sind(46.1) - sind(46.0))
println("spherical box area        = ", round(box; digits = 4), " km2")
println("cell 1 (with -10 m pixel) = ", round(sum(H[1, :]); digits = 4), " km2; lowest bin ",
        round(H[1, 1]; digits = 4))
println("cell 2                    = ", round(sum(H[2, :]); digits = 4), " km2")
