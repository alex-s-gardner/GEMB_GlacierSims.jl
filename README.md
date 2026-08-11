# GEMB_GlacierSims

A Julia package for running glacier simulations using [GEMB.jl](https://github.com/alex-s-gardner/GEMB.jl) and [GEMB_ClimateForcing.jl](https://github.com/alex-s-gardner/GEMB_ClimateForcing.jl).

## Installation

Since GEMB.jl and GEMB_ClimateForcing.jl are not yet in the General registry, you'll need to add them from GitHub:

```julia
using Pkg

# Add GEMB and GEMB_ClimateForcing from GitHub
Pkg.add(url="https://github.com/alex-s-gardner/GEMB.jl")
Pkg.add(url="https://github.com/alex-s-gardner/GEMB_ClimateForcing.jl")

# Add GEMB_GlacierSims (adjust path as needed)
Pkg.add(url="https://github.com/alex-s-gardner/GEMB_GlacierSims.jl")
```

## Setup

Before running simulations with ERA5 data, you need a CDS API key:

1. Get a CDS API key from: https://cds.climate.copernicus.eu/api-how-to
2. Set the environment variable:
   ```julia
   ENV["CDS_API_KEY"] = "your-key-here"
   ```

## Quick Start

```julia
using GEMB_GlacierSims
using Dates

# Set your CDS API key
ENV["CDS_API_KEY"] = "your-key-here"

# Run simulation for Summit Station, Greenland
result = run_era5_simulation(72.58, -38.48;
                              time_range=(DateTime(2020,1,1), DateTime(2020,12,31)),
                              spinup_years=100)

# Access results
output = result.output
forcing = result.forcing
profile = result.profile_spunup

# Example post-processing
using Statistics
mean_albedo = mean(parent(output[:albedo_surface]))
println("Mean surface albedo: $mean_albedo")
```

## Features

- Automated download and formatting of ERA5-Land climate data
- Integrated spin-up procedure for firn equilibration
- Simple API for running glacier energy and mass balance simulations
- Support for any location covered by ERA5-Land reanalysis

## Examples

The `src/era5_example.jl` file contains a detailed example script that can be run standalone or used as a reference.

## License

See LICENSE file.
