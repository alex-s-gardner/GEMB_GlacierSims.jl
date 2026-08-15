module GEMB_GlacierSims

using GEMB
# CF metadata helpers GEMB uses internally but does not export; the NetCDF writer and the
# restart reader need them so profiles read back with the same layer metadata GEMB attaches.
using GEMB: cf_layermetadata, cf_layer_index_attributes, cf_time_attributes
using GEMB_ClimateForcing
using DimensionalData
using Rasters
using DataFrames
using GeoDataFrames
using SortTileRecursiveTree
using ProgressMeter
using Statistics
using Dates
import NCDatasets

include("util.jl")
export era5_land_invariant, wrap_lon, forcing_at_elevation

include("glacier_elevation_class.jl")
export gemb_glacier_elevation_class_runfile, hypsometry_bin_edges, glacier_hypsometry

include("glacier_run.jl")
export glacier_hypsometry_coverage, glacier_area_total, gemb_glacier_cell, GlacierCellRun
export forcing_is_complete, ForcingUpToDate, ForcingUnavailable, RestartParameterMismatch
export run_parameters
export CELL_MASS_VARIABLES, CELL_TOTAL_VARIABLES, PROFILE_VARIABLES

include("netcdf_output.jl")
export write_glacier_cell_netcdf, append_glacier_cell_netcdf, read_glacier_cell_restart
export read_glacier_cell_parameters


end # module
