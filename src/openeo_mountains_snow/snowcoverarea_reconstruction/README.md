# Historical Snow Cover Reconstruction Pipeline

A comprehensive Python pipeline for reconstructing historical snow cover area (SCA) from satellite data using openEO and AWS S3. This system combines high-resolution Sentinel-2 imagery, MODIS conditional probabilities, and climate data downscaling to produce detailed snow cover reconstructions over mountainous regions.

## Overview

This pipeline performs the following key tasks:

1. **Snow Cover Fraction (SCF) Computation** - Calculates conditional probability maps from MODIS data
2. **Historical Reconstruction** - Reconstructs snow cover using high-resolution satellite data and historical patterns
3. **Climate Downscaling** - Downscales coarse-resolution AGERA5 climate data to high-resolution DEM
4. **Topographic Corrections** - Applies solar position and incidence angle corrections to radiation
5. **Batch Execution** - Orchestrates distributed processing via openEO backend

## Project Structure

```
snowcoverarea_reconstruction/
├── README.md                              # This file
├── config.py                              #  Centralized configuration hub
├── main.py                                # Main execution orchestrator
├── scf_processing.py                      # Snow Cover Fraction functions
├── s3_utils.py                            # S3 credential management for debugging
├── downscale_variables.py                 # Climate data downscaling
├── downscaling_distribution.py            # Distribution downscaling utilities
├── highresolution_gapfilling.py           # Gap-filling for high-resolution data
├── utils_gapfilling.py                    # General gap-filling utilities
├── utils_scf.py                           # SCF utility functions
├── run_agera_downscaler.py                # AGERA5 downscaling runner
├── incidence_angle_udf.py                 #                                  #  User-Defined Functions folder
    ├── __init__.py
    ├── historical_reconstruction_udf.py   # Main reconstruction UDF
    ├── solar_position_udf.py              # Solar position calculation
    └── incidence_angle_udf.py             # Solar incidence angle calculation
```

## Quick Start

### Prerequisites

- Python 3.11+
- Access to openEO Copernicus DataSpace backend
- OIDC credentials for authentication

### Running the Pipeline

```python
from main import main

# Execute the full reconstruction pipeline
main()
```

All configuration is managed in `config.py` - modify parameters there before running.

## Configuration (`config.py`)

The `config.py` file centralizes all parameters. Key sections:

### Temporal Configuration
```python
START_DATE = '2023-02-01'              # Main reconstruction period start
END_DATE = '2025-06-30'                # Main reconstruction period end

MODIS_TEMPORAL_EXTENT = ["2023-01-20", "2023-01-21"]    # MODIS data range
AGERA_TEMPORAL_EXTENT = ['2024-07-01', '2024-07-05']    # Climate data range
DEM_TEMPORAL_EXTENT = None             # DEM is static (no temporal)
DEM_GEOPOTENTIAL_LABEL = '2025-09-29'  # Static geopotential label
```

### Spatial Configuration
```python
CRS = 32632                # Coordinate Reference System (UTM Zone 32N)
RESOLUTION = 20.          # Output resolution in meters
SPATIAL_EXTENT = {        # Bounding box in CRS units
    "west": 631800.,
    "south": 5170700.,
    "east": 635800.,
    "north": 5174200.,
    "crs": "EPSG:32632"
}
```

### Processing Parameters
```python
CLOUD_PROB = 80              # Maximum cloud probability (%)
PIXEL_RATIO = 25             # Ratio between low-res and high-res pixels
N_DAYS_TO_RECONSTRUCT = 10   # Number of days to reconstruct
NEIGHBORHOOD_SIZE = {        # Spatial window for UDF processing
    'x': {'value': 128, 'unit': 'px'},
    'y': {'value': 128, 'unit': 'px'},
}
```

### SCF (Snow Cover Fraction) Parameters
```python
DELTA = 10      # Step size for SCF range bins (%)
EPSILON = 10    # Security buffer for SCF ranges (%)
```

### Backend Configuration
```python
BACKEND = "https://openeo.dataspace.copernicus.eu/"
JOB_OPTIONS = {
    "executor-memory": "8G",
    "executor-memoryOverhead": "500m",
    "python-memory": "15G",
    "load_stac_apply_lcfm_improvements": True,
    "split_strategy": "crossbackend"
}
```

## Module Descriptions

### `main.py`
Main orchestration script that coordinates the entire pipeline:
1. Authenticates with openEO backend
2. Computes SCF masks and conditional probabilities
3. Loads high-resolution Sentinel-2 data
4. Runs historical reconstruction UDF
5. Loads and downscales climate data
6. Executes batch job on backend

**Key Function**: `main()` - Execute full pipeline

### `config.py`
Centralized configuration hub containing all constants and parameters used across the pipeline. Imported by all other modules.

**Key Exports**:
- Temporal extents (START_DATE, END_DATE, AGERA_TEMPORAL_EXTENT, etc.)
- Spatial configuration (CRS, RESOLUTION, SPATIAL_EXTENT)
- Processing parameters (CLOUD_PROB, PIXEL_RATIO, etc.)
- UDF paths (HISTORICAL_RECONSTRUCTION_UDF, SOLAR_POSITION_UDF, etc.)

### `scf_processing.py`
Snow Cover Fraction computation functions using MODIS data.

**Key Functions**:
- `compute_scf_masks(connection)` - Compute SCF masks and conditional probabilities
- `low_resolution_snow_cover_fraction_mask(connection, total_mask)` - MODIS-based SCF
- `create_modis_scf_cube(connection, temporal_extent, spatial_extent)` - Load and clean MODIS data

### `s3_utils.py`
AWS S3 credential management for checkpoint storage during processing.

**Key Class**: `S3Manager`
- `authenticate()` - OIDC → STS → S3FS setup
- `get_checkpoint_config()` - Return config dict for UDF context
- `print_credentials_info()` - Debug output

### `downscale_variables.py`
Climate data downscaling using lapse rates and topographic corrections.

**Key Functions**:
- `downscale_temperature_humidity(agera, dem, geopotential)` - Downscale T/Td
- `downscale_shortwave_radiation(agera, slope_aspect)` - Topographic correction with solar position/incidence angles
- Temperature/humidity lapse rate functions

### `utils_gapfilling.py`
Gap-filling utilities for missing or invalid data.

**Key Functions**:
- `calculate_snow()` - Calculate snow from temperature/precipitation
- `get_scf_ranges()` - Generate SCF range bins
- `salomonson()` - NDSI-based snow detection
- `binarize()` - Convert values to binary snow/no-snow

### `highresolution_gapfilling.py`
High-resolution specific gap-filling algorithms for Sentinel-2 data.

## UDFs (User-Defined Functions)

UDFs are stored in the `udfs/` folder and handle distributed processing on the openEO backend.

### `historical_reconstruction_udf.py`
Main reconstruction UDF that:
- Loads historical snow cover patterns
- Uses HR reconstruction and SCF-based methods
- Fills cloud pixels iteratively
- Saves checkpoints to S3

**Key Function**: `apply_datacube(cube, context)` - Main UDF entry point

### `solar_position_udf.py`
Computes solar position (zenith and azimuth angles) for each pixel and timestamp.

**Key Function**: `apply_datacube(cube, context)` - Append solar angles to data cube

### `incidence_angle_udf.py`
Computes solar incidence angle on sloped terrain from slope, aspect, and solar angles.

**Key Function**: `apply_datacube(cube, context)` - Return incidence angle

## Data Flows

### Input Data Sources
- **Sentinel-2**: High-resolution (10-20m) multispectral imagery
- **MODIS**: Low-resolution (500m) snow cover and conditional probabilities
- **AGERA5**: Coarse climate data (temperature, humidity, radiation)
- **COPERNICUS DEM**: Elevation and slope/aspect data
- **Geopotential**: Static atmospheric geopotential data

### Processing Steps

```
1. Load MODIS data → Compute SCF masks & conditional probabilities
                  ↓
2. Load Sentinel-2 data → Extract snow cover band
                       ↓
3. Merge with conditional probabilities & historical patterns
                       ↓
4. Apply Historical Reconstruction UDF (distributed on openEO backend)
                       ↓
5. Load AGERA5 climate data
                       ↓
6. Downscale to DEM resolution (with topographic corrections)
                       ↓
7. Execute batch job and save results
```

## Key Concepts

### Snow Cover Fraction (SCF)
Represents the fraction of a pixel covered by snow (0-100%). Computed from MODIS data at low resolution and used to constrain high-resolution reconstructions.

### Conditional Probability
P(Snow | MODIS_condition) - Historical likelihood of snow given current MODIS observations. Pre-computed from training period and applied during reconstruction.

### Historical Reconstruction
Two-stage iterative approach:
1. **HR Reconstruction**: Uses similar historical scenes to fill clouds
2. **SCF Reconstruction**: Uses conditional probabilities and MODIS SCF to fill remaining gaps

### Topographic Correction
Adjusts shortwave radiation based on:
- Solar position (zenith/azimuth angles)
- Terrain slope and aspect
- Solar incidence angle

## Configuration Examples

### Adjusting Temporal Periods
To reconstruct a different time period with different climate data:

```python
# config.py
START_DATE = '2022-10-01'
END_DATE = '2023-09-30'
AGERA_TEMPORAL_EXTENT = ['2022-07-01', '2023-07-01']  # Climate period
```

### Changing Spatial Extent
Update coordinates in SPATIAL_EXTENT for different mountain regions:

```python
SPATIAL_EXTENT = {
    "west": 631800.,      # Western boundary
    "south": 5170700.,    # Southern boundary
    "east": 635800.,      # Eastern boundary
    "north": 5174200.,    # Northern boundary
    "crs": "EPSG:32632"   # UTM Zone 32N
}
```

### Adjusting Processing Parameters
Fine-tune for different conditions:

```python
CLOUD_PROB = 50              # More lenient cloud filtering
PIXEL_RATIO = 20             # Finer high-res resolution
N_DAYS_TO_RECONSTRUCT = 20   # Longer reconstruction window
```

## Dependencies

Key Python packages:
- **openeo** - Earth observation processing
- **xarray** - Multidimensional array operations
- **boto3** - AWS S3 access
- **s3fs** - S3 filesystem interface
- **netCDF4** - NetCDF file I/O
- **numpy, pandas** - Data manipulation



## References

- [openEO Documentation](https://openeo.org/)


