"""
    COLUMN_PROFILE_MASS_MAXIMUM

How deep [kg m⁻²] a recorded column profile reaches below the surface: 6000 kg m⁻², several times any
seasonal snowpack the WGMS density records describe.
"""
const COLUMN_PROFILE_MASS_MAXIMUM = 6000.0

# The time origin of the profile files' date variables.
const _NC_EPOCH = DateTime(1900, 1, 1)

"""
    column_profile_top(output, date; mass_max = COLUMN_PROFILE_MASS_MAXIMUM) -> NamedTuple or nothing

The top of a GEMB column at the output time nearest `date`: layer thickness `dz` [m], `density`
[kg m⁻³] and mass-weighted `age` [d], surface first, down to the first layer at which the cumulative
mass reaches `mass_max`. Returns `(; time, dz, density, age)` with `time` the output time used, or
`nothing` when `date` lies outside the output. The output times are the run's output frequency, so a
weekly run samples within 3.5 days of `date`.
"""
function column_profile_top(output, date; mass_max::Real = COLUMN_PROFILE_MASS_MAXIMUM)
    times = collect(dims(output, Ti))
    (isempty(times) || date < first(times) - Day(7) || date > last(times) + Day(7)) && return nothing
    _, t = findmin(x -> abs(Dates.value(x - DateTime(date))), times)
    dz = collect(Float64, output[:dz][:, t])
    rho = collect(Float64, output[:density][:, t])
    age = collect(Float64, output[:age][:, t])
    keep = .!isnan.(dz) .& .!isnan.(rho) .& .!isnan.(age) .& (dz .> 0)
    dz, rho, age = dz[keep], rho[keep], age[keep]
    n = something(findfirst(>=(mass_max), cumsum(dz .* rho)), length(dz))
    return (; time = times[t], dz = dz[1:n], density = rho[1:n], age = age[1:n])
end

"""
    ColumnProfileRecorder(band_centers, dates; mass_max = COLUMN_PROFILE_MASS_MAXIMUM)

An `on_output` callback for [`gemb_glacier_tile`](@ref) that keeps the top of the column
([`column_profile_top`](@ref)) at each of `dates` for the bands centred at `band_centers`, at every
(ΔT, pscale) node. Safe to call from the run's threads. [`write_column_profiles_netcdf`](@ref) writes
what it holds.
"""
struct ColumnProfileRecorder
    band_centers::Vector{Float64}
    dates::Vector{DateTime}
    mass_max::Float64
    profiles::Dict{Tuple{Float64,Float64,Float64},Vector{Any}}
    lock::ReentrantLock
end

function ColumnProfileRecorder(band_centers, dates; mass_max::Real = COLUMN_PROFILE_MASS_MAXIMUM)
    mass_max > 0 || throw(ArgumentError("mass_max must be positive, got $mass_max"))
    return ColumnProfileRecorder(sort!(unique!(collect(Float64, band_centers))),
                                 sort!(unique!(collect(DateTime, dates))), Float64(mass_max),
                                 Dict{Tuple{Float64,Float64,Float64},Vector{Any}}(), ReentrantLock())
end

function (r::ColumnProfileRecorder)(output; band, delta, pscale)
    Float64(band.center) in r.band_centers || return nothing
    columns = [column_profile_top(output, d; r.mass_max) for d in r.dates]
    lock(r.lock) do
        r.profiles[(Float64(band.center), Float64(delta), Float64(pscale))] = columns
    end
    return nothing
end

"""
    write_column_profiles_netcdf(path, recorder::ColumnProfileRecorder; tile_name, delta_temperatures,
                                 precipitation_scalings) -> String

Write the column profiles `recorder` holds to `path`, over (layer, date, band, delta_temperature,
precipitation_scaling): `dz` [m], `density` [kg m⁻³], `age` [d] (NaN below the profile's last layer),
the output `sample_time` each date was taken at, and `n_layers`. Written whole to a temporary file and
renamed. Throws when a band, node or date the recorder was given has no profile, so a missing column
cannot pass as a recorded one.
"""
function write_column_profiles_netcdf(path::AbstractString, r::ColumnProfileRecorder; tile_name,
                                      delta_temperatures, precipitation_scalings)
    dts = collect(Float64, delta_temperatures); pss = collect(Float64, precipitation_scalings)
    nb, nd, nt, np = length(r.band_centers), length(r.dates), length(dts), length(pss)
    for b in r.band_centers, t in dts, p in pss
        haskey(r.profiles, (b, t, p)) || throw(ArgumentError(
            "no column profile for band $b m at ΔT $t, pscale $p; the run did not reach the recorder"))
    end
    nl = maximum((c === nothing ? 0 : length(c.dz)) for v in values(r.profiles) for c in v; init = 1)
    dz = fill(NaN, nl, nd, nb, nt, np); rho = similar(dz); age = similar(dz)
    sample = fill(NaN, nd, nb, nt, np); nlay = zeros(Int32, nd, nb, nt, np)
    for (ib, b) in pairs(r.band_centers), (it, t) in pairs(dts), (ip, p) in pairs(pss)
        for (id, c) in pairs(r.profiles[(b, t, p)])
            c === nothing && continue
            n = length(c.dz)
            dz[1:n, id, ib, it, ip] .= c.dz; rho[1:n, id, ib, it, ip] .= c.density
            age[1:n, id, ib, it, ip] .= c.age
            sample[id, ib, it, ip] = Dates.value(c.time - _NC_EPOCH) / 86_400_000
            nlay[id, ib, it, ip] = n
        end
    end
    tmp = path * ".part"
    NCDatasets.NCDataset(tmp, "c") do ds
        ds.attrib["title"] = "GEMB column profiles at requested dates, for selected bands of one tile"
        ds.attrib["tile_name"] = tile_name
        ds.attrib["profile_mass_maximum_kg_m2"] = r.mass_max
        ds.attrib["comment"] = "Top of the column at the output time nearest each date, surface first, " *
                               "down to the first layer at which the cumulative mass reaches " *
                               "profile_mass_maximum_kg_m2. age is GEMB's mass-weighted layer age."
        NCDatasets.defDim(ds, "layer", nl); NCDatasets.defDim(ds, "date", nd)
        NCDatasets.defDim(ds, "band", nb); NCDatasets.defDim(ds, "delta_temperature", nt)
        NCDatasets.defDim(ds, "precipitation_scaling", np)
        # Plain day counts rather than CF time units, so a reader gets the numbers it wrote.
        NCDatasets.defVar(ds, "date", [Dates.value(d - _NC_EPOCH) / 86_400_000 for d in r.dates],
                          ("date",); attrib = ["units" => "d", "comment" => "days since 1900-01-01T00:00"])
        NCDatasets.defVar(ds, "band_center", r.band_centers, ("band",); attrib = ["units" => "m"])
        NCDatasets.defVar(ds, "delta_temperature", dts, ("delta_temperature",); attrib = ["units" => "K"])
        NCDatasets.defVar(ds, "precipitation_scaling", pss, ("precipitation_scaling",);
                          attrib = ["units" => "1"])
        dims5 = ("layer", "date", "band", "delta_temperature", "precipitation_scaling")
        NCDatasets.defVar(ds, "dz", dz, dims5; attrib = ["units" => "m"])
        NCDatasets.defVar(ds, "density", rho, dims5; attrib = ["units" => "kg m-3"])
        NCDatasets.defVar(ds, "age", age, dims5; attrib = ["units" => "d"])
        NCDatasets.defVar(ds, "sample_time", sample, dims5[2:end];
                          attrib = ["units" => "d", "comment" => "days since 1900-01-01T00:00"])
        NCDatasets.defVar(ds, "n_layers", nlay, dims5[2:end])
    end
    mv(tmp, path; force = true)
    return path
end

"""
    read_column_profiles(path) -> NamedTuple

Read a [`write_column_profiles_netcdf`](@ref) file: `(; tile_name, dates, band_centers,
delta_temperatures, precipitation_scalings, profile)`, with `profile(date, band_center, ΔT, pscale)`
returning that column's `(; time, dz, density, age)`, or `nothing` where none was taken.
"""
function read_column_profiles(path::AbstractString)
    NCDatasets.NCDataset(path, "r") do ds
        days(x) = _NC_EPOCH + Millisecond(round(Int64, x * 86_400_000))
        dates = days.(collect(Float64, ds["date"][:]))
        bands = collect(Float64, ds["band_center"][:])
        dts = collect(Float64, ds["delta_temperature"][:])
        pss = collect(Float64, ds["precipitation_scaling"][:])
        dz = coalesce.(ds["dz"][:, :, :, :, :], NaN); rho = coalesce.(ds["density"][:, :, :, :, :], NaN)
        age = coalesce.(ds["age"][:, :, :, :, :], NaN)
        sample = coalesce.(ds["sample_time"][:, :, :, :], NaN); nlay = ds["n_layers"][:, :, :, :]
        tile_name = ds.attrib["tile_name"]
        function profile(date, band, t, p)
            i = (findfirst(==(DateTime(date)), dates), findfirst(==(Float64(band)), bands),
                 findfirst(==(Float64(t)), dts), findfirst(==(Float64(p)), pss))
            any(isnothing, i) && throw(ArgumentError(
                "no profile slot for $date, band $band m, ΔT $t, pscale $p in $path"))
            n = Int(nlay[i...])
            n == 0 && return nothing
            return (; time = days(sample[i...]), dz = dz[1:n, i...], density = rho[1:n, i...],
                    age = age[1:n, i...])
        end
        (; tile_name, dates, band_centers = bands, delta_temperatures = dts,
         precipitation_scalings = pss, profile)
    end
end
