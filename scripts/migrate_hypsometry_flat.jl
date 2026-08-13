# One-off migration: convert the cached glacier elevation-class parquet from the old nested
# `:glacier_hypsometry` column (a per-row Vector, stored on disk as a Parquet LIST) to the flat
# layout — one scalar Float64 column per elevation bin (`hyps_<lo>_<hi>`).
#
# Motivation: the nested column deserializes into a boxed, type-unstable
# `Vector{Union{Missing, AbstractDict, Vector{Any}}}` and dominates read time (~11 s of ~11.5 s).
# The flat scalar columns read in milliseconds. This reuses the already-computed values so the
# multi-hour builder does NOT need to rerun.
#
# Run once:  julia --project=. scripts/migrate_hypsometry_flat.jl

using GeoDataFrames
using GeoParquet          # GeoParquet backend for GeoDataFrames read/write
using DataFrames
using GEMB_GlacierSims: _hyps_colnames

const PARQUET = joinpath(@__DIR__, "..", "data", "era5land_glacier_elevation_classes.parquet")
const BIN_EDGES = 0:100:10000          # must match the build (0:100:10000 → 100 bins)
const N_BINS = length(BIN_EDGES) - 1

# Normalize one cell to a concrete Vector{Float64}. On read the nested column comes back boxed as
# Vector{Any} (and, for some rows, an ordered Dict of index=>value); handle both and force Float64.
function tofloatvec(v)
    if v isa AbstractDict
        return Float64[v[k] for k in sort(collect(keys(v)))]
    else
        return Float64.(collect(v))
    end
end

function main()
    isfile(PARQUET) || error("parquet not found: $PARQUET")

    # Back up before overwriting.
    bak = PARQUET * ".bak"
    if !isfile(bak)
        cp(PARQUET, bak)
        println("Backed up cache -> $bak")
    else
        println("Backup already exists, leaving it: $bak")
    end

    println("Reading (slow, one last time): $PARQUET")
    @time df = GeoDataFrames.read(PARQUET)
    println("  nrow=$(nrow(df)) ncol=$(ncol(df))")

    "glacier_hypsometry" in names(df) ||
        error("no :glacier_hypsometry column — already migrated?")

    # Coerce every cell and validate the bin count before touching the file.
    vecs = tofloatvec.(df.glacier_hypsometry)
    lens = length.(vecs)
    all(==(N_BINS), lens) ||
        error("expected $N_BINS bins per cell; found lengths $(sort(unique(lens)))")

    # Row-major stack into an nrow × N_BINS matrix.
    H = reduce(vcat, permutedims.(vecs))
    @assert size(H) == (nrow(df), N_BINS)

    # Preserve the total area for a post-write sanity check.
    total_before = sum(sum, vecs)

    colnames = _hyps_colnames(BIN_EDGES)
    @assert length(colnames) == N_BINS
    for (b, name) in enumerate(colnames)
        df[!, name] = H[:, b]
    end
    select!(df, Not(:glacier_hypsometry))
    metadata!(df, "hypsometry_bin_edges", collect(BIN_EDGES); style=:note)

    println("Writing flat layout: $PARQUET")
    GeoDataFrames.write(PARQUET, df)

    # Verify the round-trip: fast read, schema, and totals.
    println("Re-reading migrated file:")
    @time df2 = GeoDataFrames.read(PARQUET)
    hcols = _hyps_colnames(BIN_EDGES)
    @assert all(in(names(df2)), string.(hcols)) "missing hyps_* columns after write"
    @assert !("glacier_hypsometry" in names(df2)) "old column still present"
    @assert all(c -> eltype(df2[!, c]) == Float64, hcols) "hyps_* columns are not Float64"
    total_after = sum(Matrix(df2[!, hcols]))
    println("  nrow=$(nrow(df2)) ncol=$(ncol(df2))")
    println("  total glacier area  before=$(total_before)  after=$(total_after)")
    @assert isapprox(total_before, total_after; rtol=1e-9) "area mismatch after migration"

    println("Migration complete.")
end

main()
