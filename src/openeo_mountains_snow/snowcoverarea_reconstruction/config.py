"""
Configuration parameters for historical snow cover reconstruction.

Contains all configuration parameters and constants used across the pipeline.
"""

import os

# ==============================
# Output Configuration
# ==============================

OUTPUT_DIR = "../results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# Temporal Configuration
# ==============================

START_DATE = '2023-02-01'
END_DATE = '2025-06-30'

# Separate temporal extent for MODIS data
MODIS_TEMPORAL_EXTENT = ["2023-01-20", "2023-01-21"]

# Separate temporal extent for climate data (AGERA5)
AGERA_TEMPORAL_EXTENT = ['2024-07-01', '2024-07-05']

# DEM is static with a fixed geopotential label
DEM_GEOPOTENTIAL_LABEL = '2025-09-29'

# ==============================
# Spatial Configuration
# ==============================

CRS = 32632
RESOLUTION = 20.  # Output resolution in meters

SPATIAL_EXTENT = {
    "west": 631800.,
    "south": 5170700.,
    "east": 635800.,
    "north": 5174200.,
    "crs": f"EPSG:{CRS}"
}

# ==============================
# Processing Parameters
# ==============================

CLOUD_PROB = 80  # Maximum cloud probability (%)
PIXEL_RATIO = 25  # Ratio between LR and HR pixel sizes

# Non-valid values configuration
INVALID_CODES = [205, 210, 254, 255]
INVALID_VALUE = 255
INVALID_THRESHOLD = 10  # % invalid pixels allowed in LR pixel

# SCF range parameters
DELTA = 10  # Step size for SCF ranges (%)
EPSILON = 10  # Security buffer for SCF ranges (%)

# ==============================
# openEO Backend Configuration
# ==============================

BACKEND = "https://openeofed.dataspace.copernicus.eu"

# ==============================
# Reconstruction Parameters
# ==============================

N_DAYS_TO_RECONSTRUCT = 10
NEIGHBORHOOD_SIZE = {
    'x': {'value': 128, 'unit': 'px'},
    'y': {'value': 128, 'unit': 'px'},
}

# ==============================
# Job Configuration
# ==============================

JOB_OPTIONS = {
    "executor-memory": "8G",
    "executor-memoryOverhead": "500m",
    "python-memory": "15G",
    "load_stac_apply_lcfm_improvements": True,
    "split_strategy": "crossbackend"
}

# ==============================
# MODIS Configuration
# ==============================

MODIS_RESOLUTION = 25  # meters
# ==============================
# UDF Configuration
# ==============================

from pathlib import Path

UDF_DIR = Path(__file__).parent / 'udfs'
HISTORICAL_RECONSTRUCTION_UDF = UDF_DIR / 'historical_reconstruction_udf.py'
SOLAR_POSITION_UDF = UDF_DIR / 'solar_position_udf.py'
INCIDENCE_ANGLE_UDF = UDF_DIR / 'incidence_angle_udf.py'