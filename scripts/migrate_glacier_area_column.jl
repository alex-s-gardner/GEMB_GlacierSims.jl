# One-off migration: add the precomputed `:glacier_area_km2` column to the cached glacier
# elevation-class parquet.
#
# Motivation: `glacier_area_total` is the area screen every sweep runs over the whole table, and
# summing 100 `hyps_*` columns per row costs ~1.8 s per pass on the ~47,000-row global table. The
# total is already determined by the per-bin columns, so caching it as a column makes the screen a
# scalar read (~2 ms). `gemb_glacier_elevation_class_runfile` writes the column for tables built
# from now on; this brings the shipped table up to the same shape.
#
# The column is computed *from the existing per-bin columns*, so the multi-hour hypsometry builder
# does NOT rerun — nothing here recomputes any glacier area, it only sums what is already stored.
#
# Run once:  julia --project=. scripts/migrate_glacier_area_column.jl

using GEMB_GlacierSims
using GEMB_GlacierSims: GLACIER_AREA_COLUMN
using GeoDataFrames
using GeoParquet          # GeoParquet backend for GeoDataFrames read/write
using DataFrames

const PARQUET = joinpath(@__DIR__, "..", "data", "era5land_glacier_elevation_classes.parquet")

function main()
    isfile(PARQUET) || error("parquet not found: $PARQUET")

    println("Reading: $PARQUET")
    df = GeoDataFrames.read(PARQUET)
    println("  nrow=$(nrow(df)) ncol=$(ncol(df))")

    if string(GLACIER_AREA_COLUMN) in names(df)
        # Idempotent: a second run is a no-op, not an error.
        println("Already migrated (column $GLACIER_AREA_COLUMN present); nothing to do.")
        return
    end

    # Computed by the same function the package reads the column with, so the stored values and a
    # fallback sum cannot disagree: this is `glacier_area_column`'s no-column branch, which sums the
    # per-bin columns in a fixed column order.
    total = glacier_area_column(df)
    length(total) == nrow(df) || error("area column length $(length(total)) != nrow $(nrow(df))")

    # Checked against the row-wise decoder on a sample rather than on all ~47,000 rows: the point is
    # that the cached column agrees with the bin decoding it stands in for, and a sample establishes
    # that without a second slow pass. `glacier_hypsometry_coverage` is the full decoder — bin
    # sorting and nearest-bin reassignment — so agreeing with its `total_area` is the real check.
    sample = [i for i in 1:max(1, nrow(df) ÷ 500):nrow(df)]
    rows = eachrow(df)
    for i in sample
        total[i] == 0 && continue   # a cell with no glacier area has no bins to decode
        decoded = glacier_hypsometry_coverage(rows[i]; coverage = 0.95).total_area
        isapprox(total[i], decoded; rtol = 1e-12) ||
            error("row $i: cached area $(total[i]) != decoded $(decoded)")
    end
    println("Verified $(length(sample)) sampled rows against the full bin decoding.")

    # Back up before overwriting.
    bak = PARQUET * ".bak_area_column"
    if !isfile(bak)
        cp(PARQUET, bak)
        println("Backed up cache -> $bak")
    else
        println("Backup already exists, leaving it: $bak")
    end

    # Keep the pre-write state so the round-trip is verified against it rather than against a
    # re-read of the file just written.
    nrow_before = nrow(df)
    ncol_before = ncol(df)
    hyps_before = filter(c -> startswith(c, "hyps_"), names(df))
    other_before = Dict(c => copy(df[!, c])
                        for c in names(df) if !startswith(c, "hyps_") && c != "geometry")

    df[!, GLACIER_AREA_COLUMN] = total
    println("Added $GLACIER_AREA_COLUMN: min=$(minimum(total)) max=$(round(maximum(total), digits=3)) " *
            "sum=$(round(sum(total), digits=3)) km2")

    println("Writing: $PARQUET")
    GeoDataFrames.write(PARQUET, df)

    println("Re-reading migrated file:")
    df2 = GeoDataFrames.read(PARQUET)
    @assert nrow(df2) == nrow_before "row count changed: $(nrow(df2)) vs $nrow_before"
    @assert ncol(df2) == ncol_before + 1 "column count changed: $(ncol(df2)) vs $(ncol_before + 1)"
    @assert string(GLACIER_AREA_COLUMN) in names(df2) "migrated file lacks $GLACIER_AREA_COLUMN"
    # Exact, not approximate: a Float64 column round-trips through Parquet bit for bit, and the
    # column is a cache — a value that shifted in the write no longer matches the bins it sums.
    @assert df2[!, GLACIER_AREA_COLUMN] == total "$GLACIER_AREA_COLUMN changed in the round trip"
    @assert filter(c -> startswith(c, "hyps_"), names(df2)) == hyps_before "hyps_* columns changed"
    # `isequal` rather than `==` so `missing` compares equal to `missing`.
    for (c, v) in other_before
        @assert all(isequal.(df2[!, c], v)) "values changed for $c"
    end

    # And the package now takes the fast path on the migrated file.
    @assert glacier_area_column(df2) == total "glacier_area_column disagrees with the stored column"
    @assert glacier_area_total(first(eachrow(df2))) == total[1] "glacier_area_total disagrees"

    println("  nrow=$(nrow(df2)) ncol=$(ncol(df2))")
    println("Migration complete.")
end

main()
