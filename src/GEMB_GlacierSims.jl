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
                           _decouple_per_timestep, _ambient_forcing, _corrects_to_ambient,
                           _downscaling_sums, _accumulate_downscaling_cell!,
                           _cell_forcing_at_interval, _FORCING_VARIABLES,
                           _MIN_CELLS_DEFAULT, _DECOUPLING_FACTOR_LIMITS,
                           _DONOR_DECOUPLING_FACTOR_LIMITS,
                           _LAPSE_RATE_LIMITS, _DECOUPLING_REFERENCE_TEMPERATURE,
                           cloud_transmissivity, cloud_optical_thickness,
                           _MIN_AMBIENT_EXCESS, _haversine_km,
                           _glacier_term_at_elevation, GLACIER_CORRECTIONS,
                           DEFAULT_GLACIER_CORRECTION, AdditiveGlacierCorrection
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
# `setproperties` is how a band's `albedo_ice` is set: `GEMB.ModelParameters` is an immutable
# `@kwdef` struct with a derived field, so it cannot be rebuilt field-by-field through
# `initialize_parameters`.
import ConstructionBase
import Tables
import NCDatasets
# The sweep's summary and point->tile index are plain tables with no geometry column, so they are
# written with Parquet2 (which GeoParquet already carries) rather than through GeoDataFrames.
import GeoParquet
import GeoParquet.Parquet2
import TOML
import SHA
import CodecZlib
# Read only to learn the span of the cached ERA5-Land record, from the Zarr groups the forcing
# loader caches on disk.
import Zarr

include("util.jl")
export claim_tile!, claim_order, weight_balanced_blocks
export ClaimHeartbeat, start_claim_heartbeat, stop_claim_heartbeat!, with_claim_heartbeat
export era5_land_invariant, wrap_lon, forcing_at_elevation, software_provenance
export era5_land_record_span, run_time_range, spinup_window_range
export cell_output_name, parse_cell_lonlat, tile_output_name, parse_tile_index

include("volume_change.jl")
export surface_height_change, height_change_components, column_reaches_ice_density
export tile_volume_change, tile_mass_total, mie2cubickm, convergence_density_from_fac
export reference_discharge_rate, discharge_corrected_volume_change

include("glacier_elevation_class.jl")
export gemb_glacier_elevation_class_runfile, hypsometry_bin_edges, glacier_hypsometry

# Observed bare-ice albedo binned by elevation. GEMB_ClimateForcing's job ends at the per-cell pooled
# albedo `bare_ice_albedo` returns; both halves of the elevation reduction live here, because the
# binning is this package's — it is the glacier hypsometry.
#
# `_hyps` bins a set of (albedo, elevation) pairs into a callable best estimate; `_cells` knows what
# an ERA5-Land cell is, pools a tile's cells into one profile, and resolves that onto an elevation
# class. Must follow glacier_elevation_class.jl, whose hypsometry decoder `_cells` reads, and precede
# applied_downscaling.jl, which resolves a band's `albedo_ice` through it. `bare_ice_albedo` and
# `bare_ice_albedo_points` are re-exported so a caller need not know which package holds which piece.
include("bare_ice_albedo_hyps.jl")
include("bare_ice_albedo_cells.jl")
export era5_land_cell_polygon, era5_land_cell_key, bare_ice_albedo_tile
export derive_bare_ice_albedo, resolve_albedo_ice, copernicus_dem_archive
export BARE_ICE_ALBEDO_SOURCES, GEMB_ALBEDO_ICE_RANGE
export bare_ice_albedo_source_code, bare_ice_albedo_source_name
export bare_ice_albedo, bare_ice_albedo_points, bare_ice_albedo_hyps, BareIceAlbedoHyps
export AlbedoElevationFit, bare_ice_albedo_source, bare_ice_albedo_table

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

# Before `downscaling_parameters.jl`, which dispatches on `AppliedDownscaling` and interpolates this
# file's thresholds into its docstrings — both of which are resolved when the method is defined.
include("wind_correction.jl")
export slope_wind_multiplier, slope_wind_floor, slope_wind_correction, wind_correction_rule
export tile_interval_slopes, SLOPE_WIND_COEFFICIENTS, BAND_SLOPE_COLUMN_PREFIX, WIND_CORRECTIONS
include("applied_downscaling.jl")
export hypsometry_intervals, resolve_downscaling, AppliedDownscaling
export DOWNSCALING_SOURCES, downscaling_source_counts, decoupling_factor_prior
export APPLIED_LAPSE_RATE_STDERR_MAXIMUM, APPLIED_LAPSE_RATE_WINDOW
export BAND_DECOUPLING_RULES, band_decoupling_rule

include("forcing_cache.jl")
export CachedForcingLoader, forcing_cache_capacity, forcing_cache_report

include("downscaling_parameters.jl")
export derive_downscaling_parameters, grid_cells_in_region, RegionForcingUnavailable
export elevation_interval_forcing, derive_lapse_rate_uncertainty
export derive_decoupling_factor, derive_lapse_rate, decoupling_factor_at_elevation
export GLACIER_CORRECTIONS, DEFAULT_GLACIER_CORRECTION, AdditiveGlacierCorrection
export GLACIER_FRACTION_FORMS, GLACIER_INDICATOR_THRESHOLD, decoupling_glacier_fraction

include("netcdf_downscaling_parameters.jl")
export write_downscaling_tile_netcdf, write_sparse_downscaling_tile_netcdf
export read_downscaling_tile, read_downscaling_tile_status

include("downscaling_tiles.jl")
export downscaling_tiles, tile_index, tile_bounds, derive_downscaling_parameter_tiles

# Per-region background snow impurity loading. Before the tile run, which applies it.
include("background_black_carbon.jl")
export background_black_carbon, background_black_carbon_table, dust_bc_equivalent
export dang2015_albedo_reduction, snicar_adv3_albedo_reduction, gs2010_equivalent
export rgi_o1_region, tile_rgi_o1_region, tile_rgi_o1_region_shares
export BACKGROUND_BLACK_CARBON_TABLE, BACKGROUND_BLACK_CARBON_BASES, BACKGROUND_BLACK_CARBON_OPTICS
export SNICAR_ADV3_ALBEDO_TABLE, GS2010_EQUIVALENT_REFERENCE, RGI7_O1_REGIONS_FILE

# Shaw et al. (2025) glacier areas per band: the area-weighted prior and the `:shaw` band decoupling.
include("shaw_glacier_bands.jl")
export shaw_glacier_bands, shaw_decoupling_prior, shaw_band_decoupling, tile_interval_areas
export SHAW_GLACIER_BANDS_TABLE, SHAW_FILL_DECOUPLING_FACTOR

# The physics members a sweep can be run under. After the background loading, whose table reader
# the effective loading shares; before the tile run, which records the member.
include("physics_members.jl")
export physics_member, physics_model_parameters, physics_member_parameters, effective_black_carbon
export PHYSICS_GRAIN_VARIANTS, PHYSICS_BAND_DECOUPLING, PHYSICS_IMPURITY

include("glacier_tile_run.jl")
export gemb_glacier_tile, GlacierTileRun, tile_run_parameters, geotile_id
export TILE_MASS_VARIABLES, TILE_HEIGHT_VARIABLES, TILE_DIAGNOSTIC_VARIABLES
export LOST_AREA_FRACTION_MAXIMUM, DROPPED_BAND_CAUSES, dropped_band_cause_code

include("netcdf_tile_run.jl")
# Column profiles at requested dates for selected bands, kept apart from the tile files.
include("column_profiles.jl")
export ColumnProfileRecorder, column_profile_top, write_column_profiles_netcdf, read_column_profiles
export COLUMN_PROFILE_MASS_MAXIMUM
include("netcdf_tile_merge.jl")
export merge_tile_perturbations
export write_glacier_tile_netcdf, append_glacier_tile_netcdf
export read_glacier_tile_status, read_glacier_tile_restart
export tile_run_disposition, output_period_bound

# After the tile run, which a native cell run is: one cell's forcing, the tile's downscaling, and the
# calibrated correction as a one-point perturbation grid.
include("forcing_corrections.jl")
include("glacier_native_run.jl")
export read_forcing_corrections, forcing_correction, native_cells, gemb_native_cell,
       native_cell_spinup_seed, native_tile_lost_area
export reference_forcing_corrections, gga_reference_ensemble_file,
       forcing_corrections_provenance_path, assert_corrections_fitted_on

end # module
