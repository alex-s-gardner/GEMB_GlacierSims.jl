# The Copernicus DEM read from a local mirror of the copernicus-dem-30m bucket
# (`copernicus_dem_archive`), against the network loader reading the same files from its tile cache.

using Test
using GEMB_GlacierSims
using Rasters
using Rasters.Lookups: ForwardOrdered, ReverseOrdered, Regular, Intervals, Start
import DimensionalData
import Extents
import GEMB_ClimateForcing

const _CF = GEMB_ClimateForcing

# One GLO-30 tile at its published geometry (3600 rows, the latitude band's column count), with
# value `base + column` so a read in the wrong place is visible. Written as `<dir>/<id>.tif`.
function _write_dem_tile(dir, lat, lon, base)
    id = _CF._copernicus_dem_tile_id(lat, lon)
    nc = _CF._copernicus_dem_ncols(lat)
    x = X(Projected(range(lon; step = 1 / nc, length = nc); order = ForwardOrdered(),
                    span = Regular(1 / nc), sampling = Intervals(Start()), crs = EPSG(4326)))
    y = Y(Projected(range(lat + 1; step = -1 / 3600, length = 3600); order = ReverseOrdered(),
                    span = Regular(-1 / 3600), sampling = Intervals(Start()), crs = EPSG(4326)))
    data = Float32[base + i for i in 1:nc, _ in 1:3600]
    path = joinpath(mkpath(dir), id * ".tif")
    write(path, Raster(data, (x, y); missingval = nothing); force = true)
    return id
end

@testset "Copernicus DEM from a local archive" begin
    # Under the package rather than tempdir(): the network loader's tile cache refuses tempdir().
    work = mktempdir(@__DIR__)
    try
        archive = joinpath(work, "archive")
        cache = joinpath(work, "cache")
        ids = [_write_dem_tile(joinpath(archive, _CF._copernicus_dem_tile_id(89, lon)), 89, lon, base)
               for (lon, base) in ((10, 100), (11, 200))]
        listed_missing = _CF._copernicus_dem_tile_id(89, 12)
        write(joinpath(archive, "tileList.txt"), join([ids; listed_missing], "\n") * "\n")
        # The network loader's own cache layout, holding the same tiles.
        mkpath(joinpath(cache, "tiles"))
        for id in ids
            cp(joinpath(archive, id, id * ".tif"), joinpath(cache, "tiles", id * ".tif"))
        end
        write(joinpath(cache, "tileList.txt"), join(ids, "\n") * "\n")

        for extent in (Extents.Extent(X = (10.9, 11.1), Y = (89.4, 89.5)),   # two tiles, a VRT
                       Extents.Extent(X = (10.2, 10.3), Y = (89.4, 89.5)))   # one tile, read directly
            local_dem = copernicus_dem_archive(archive, extent)
            network = _CF.climate_model_invariant(; model = :copernicus_dem_30m, extent, cache_path = cache,
                                                  cache_tiles = true, verbose = false)
            a, b = read(local_dem), read(network)
            @test dims(a) == dims(b)
            @test parent(a) == parent(b)
            @test DimensionalData.metadata(local_dem)["archive"] == archive
        end
        dem = read(copernicus_dem_archive(archive, Extents.Extent(X = (10.9, 11.1), Y = (89.4, 89.5))))
        # Column 343 of the E010 tile (360 columns per degree at 89°N) and column 19 of E011.
        @test dem[X = Near(10.951), Y = Near(89.45)] == 100 + 343
        @test dem[X = Near(11.051), Y = Near(89.45)] == 200 + 19

        @test_throws "lacks 1 tile(s) its tileList.txt publishes" copernicus_dem_archive(
            archive, Extents.Extent(X = (11.5, 12.5), Y = (89.4, 89.5)))
        @test_throws "No Copernicus DEM tiles exist" copernicus_dem_archive(
            archive, Extents.Extent(X = (20.1, 20.2), Y = (89.4, 89.5)))
        @test_throws "no tileList.txt" copernicus_dem_archive(
            cache * "/tiles", Extents.Extent(X = (10.2, 10.3), Y = (89.4, 89.5)))
    finally
        rm(work; recursive = true, force = true)
    end

    cells = GEMB_GlacierSims.DataFrame(longitude = [10.25], latitude = [89.45])
    @test_throws "pass either a dem or a dem_archive" derive_bare_ice_albedo(
        cells; dem = zeros(1), dem_archive = "/nonexistent", verbose = false)

    @testset "the derivation scripts' archive setting" begin
        derive = read(joinpath(@__DIR__, "..", "scripts", "derive_downscaling_parameter_tiles.jl"), String)
        @test occursin("get(ENV, \"COPERNICUS_DEM_ARCHIVE\", nothing)", derive)
        @test occursin("dem_archive = DEM_ARCHIVE", derive)
        tile = read(joinpath(@__DIR__, "..", "scripts", "derive_bare_ice_albedo_tile.jl"), String)
        @test occursin("copernicus_dem_archive(DEM_ARCHIVE, tile_extent)", tile)
        @test occursin("dem_archive = DEM_ARCHIVE", tile)
    end
end
