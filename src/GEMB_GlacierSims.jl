module GEMB_GlacierSims

using GEMB
# CF metadata helpers GEMB uses internally but does not export; the NetCDF writer and the
# restart reader need them so profiles read back with the same layer metadata GEMB attaches.
using GEMB: cf_layermetadata, cf_layer_index_attributes, cf_time_attributes
using GEMB_ClimateForcing
# The per-region downscaling-parameter fits and the constants they are defined against. They live
# upstream because each one inverts a `climate_adjust_for_*` and is pure arithmetic on this
# package's forcing conventions; what stays here is the half that knows about glacier
# elevation-class tables. Imported explicitly, and re-exported below, so a caller does not have to
# know which package a given piece of the derivation lives in — and so the private names this file
# shares with upstream (`_FORCING_VARIABLES`, the domain limits) are visibly borrowed rather than
# silently reached for.
using GEMB_ClimateForcing: derive_decoupling_factor, derive_lapse_rate,
                           decoupling_factor_at_elevation,
                           _make_applicable, _decouple_per_timestep,
                           _cell_forcing_at_interval, _FORCING_VARIABLES,
                           _MIN_CELLS_DEFAULT, _DECOUPLING_FACTOR_LIMITS,
                           _LAPSE_RATE_LIMITS, _DECOUPLING_REFERENCE_TEMPERATURE,
                           _MIN_AMBIENT_EXCESS
using DimensionalData
using Rasters
using DataFrames
using GeoDataFrames
using SortTileRecursiveTree
using ProgressMeter
using Statistics
using Dates
using Extents
import GeoInterface
import GeometryOps as GO
import Tables
import NCDatasets

include("util.jl")
export era5_land_invariant, wrap_lon, forcing_at_elevation

include("glacier_elevation_class.jl")
export gemb_glacier_elevation_class_runfile, hypsometry_bin_edges, glacier_hypsometry

include("glacier_run.jl")
export glacier_hypsometry_coverage, glacier_area_total, glacier_area_column,
       gemb_glacier_cell, GlacierCellRun
export forcing_is_complete, ForcingUpToDate, ForcingUnavailable, RestartParameterMismatch,
       SpinupWindowUnavailable
export run_parameters, run_parameter_differences
export cell_decoupling_factor, resolve_decoupling_factor, decoupling_factor_label
export CELL_MASS_VARIABLES, CELL_TOTAL_VARIABLES, PROFILE_VARIABLES

include("netcdf_output.jl")
export write_glacier_cell_netcdf, append_glacier_cell_netcdf, read_glacier_cell_restart
export read_glacier_cell_parameters, read_glacier_cell_status

include("downscaling_parameters.jl")
export derive_downscaling_parameters, grid_cells_in_region
export derive_decoupling_factor, derive_lapse_rate, decoupling_factor_at_elevation


end # module
