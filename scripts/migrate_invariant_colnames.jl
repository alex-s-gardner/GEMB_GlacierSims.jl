# One-off migration: rename the ERA5-Land invariant columns in the cached glacier elevation-class
# parquet from the old `<field>_frac` spelling to the bare field name (`glm_frac` -> `glm`,
# `lsm_frac` -> `lsm`).
#
# Motivation: `gemb_glacier_elevation_class_runfile` names invariant columns from the layer keys of
# the stack it is handed (`glacier_elevation_class.jl`), and `era5_example.jl` hands it
# `era5_land_invariant(parameter = (:glm, :lsm, :cl))` — so a table built today carries `:glm`.
# `cell_decoupling_factor` reads `row.glm` to weight the decoupling factor by the non-glacier
# fraction. The shipped parquet predates that naming, so `:glm` was absent and the weighting
# silently collapsed to the unweighted published factor on every cell. Renaming the columns is what
# makes the data match the code; the multi-hour builder does NOT need to rerun.
#
# Run once:  julia --project=. scripts/migrate_invariant_colnames.jl

using GeoDataFrames
using GeoParquet          # GeoParquet backend for GeoDataFrames read/write
using DataFrames

const PARQUET = joinpath(@__DIR__, "..", "data", "era5land_glacier_elevation_classes.parquet")

# Old name => new name. Only `glm` is read by the package today, but `lsm` is renamed alongside it
# so the whole table speaks one convention rather than a mix of two.
const RENAMES = [:glm_frac => :glm, :lsm_frac => :lsm]

function main()
    isfile(PARQUET) || error("parquet not found: $PARQUET")

    println("Reading: $PARQUET")
    df = GeoDataFrames.read(PARQUET)
    println("  nrow=$(nrow(df)) ncol=$(ncol(df))")

    present = [p for p in RENAMES if string(first(p)) in names(df)]
    if isempty(present)
        # Idempotent: a second run is a no-op, not an error.
        all(p -> string(last(p)) in names(df), RENAMES) ||
            error("neither the old nor the new invariant column names are present; " *
                  "expected one of $(first.(RENAMES)) or $(last.(RENAMES))")
        println("Already migrated (columns $(last.(RENAMES)) present); nothing to do.")
        return
    end

    # Back up before overwriting.
    bak = PARQUET * ".bak_colnames"
    if !isfile(bak)
        cp(PARQUET, bak)
        println("Backed up cache -> $bak")
    else
        println("Backup already exists, leaving it: $bak")
    end

    # Keep the pre-rename values so the write can be verified against them rather than against a
    # re-read of the file we just wrote.
    before = Dict(last(p) => copy(df[!, first(p)]) for p in present)
    nrow_before = nrow(df)
    ncol_before = ncol(df)
    hyps_before = filter(c -> startswith(c, "hyps_"), names(df))

    for p in present
        println("Renaming $(first(p)) -> $(last(p))")
    end
    rename!(df, present)

    println("Writing: $PARQUET")
    GeoDataFrames.write(PARQUET, df)

    # Verify the round-trip: schema, row count, values, and that the hypsometry columns are intact.
    println("Re-reading migrated file:")
    df2 = GeoDataFrames.read(PARQUET)
    @assert nrow(df2) == nrow_before "row count changed: $(nrow(df2)) vs $nrow_before"
    @assert ncol(df2) == ncol_before "column count changed: $(ncol(df2)) vs $ncol_before"
    for p in present
        @assert string(last(p)) in names(df2) "missing renamed column $(last(p)) after write"
        @assert !(string(first(p)) in names(df2)) "old column $(first(p)) still present"
        # `isequal` rather than `==` so `missing` compares equal to `missing`.
        @assert all(isequal.(df2[!, last(p)], before[last(p)])) "values changed for $(last(p))"
    end
    @assert filter(c -> startswith(c, "hyps_"), names(df2)) == hyps_before "hyps_* columns changed"
    println("  nrow=$(nrow(df2)) ncol=$(ncol(df2))")

    println("Migration complete.")
end

main()
