"""
Shared configuration for the snow cover area and snow water equivalent pipeline (main.py)
and for upscaling. Single source for processing constants.
"""

from pathlib import Path

# ==============================
# Temporal Configuration
# ==============================

TEMPORAL_EXTENT = ['2021-09-01', '2024-09-30']

# Separate temporal extent for MODIS data
MODIS_TEMPORAL_EXTENT = ["2023-01-20", "2023-01-21"]

# Separate temporal extent for climate data (AGERA5)
AGERA_TEMPORAL_EXTENT = ['2023-07-01', '2023-07-05']



# ==============================
# Spatial Configuration
# ==============================

CRS = 32632
RESOLUTION = 20.  # Output resolution in meters

SPATIAL_EXTENT = {
    "west": 631800.,
    "south": 5170700.,
    "east": 637800.,
    "north": 5176200.,
    "crs": f"EPSG:{CRS}"
}

# ==============================
# Processing Parameters
# ==============================

CLOUD_PROB = 80  # Maximum cloud probability (%)
PIXEL_RATIO = 25  # Ratio between LR and HR pixel sizes

# Non-valid values configuration
INVALID_CODES = [205, 210, 254, 255]
CLOUD_VALUE = 205  # cloud code in snow maps; the UDFs hardcode the same value
INVALID_VALUE = 255
INVALID_THRESHOLD = 10  # % invalid pixels allowed in LR pixel
SNOW_FRACTION_THRESHOLD = 50  # SCF (%) at or above which a pixel counts as snow

# SCF range parameters
DELTA = 10  # Step size for SCF ranges (%)
EPSILON = 10  # Security buffer for SCF ranges (%)

# ==============================
# openEO Backend Configuration
# ==============================

BACKEND = "https://openeo.dataspace.copernicus.eu/"

# ==============================
# Reconstruction Parameters
# ==============================

N_DAYS_TO_RECONSTRUCT = 5
NEIGHBORHOOD_SIZE = 64

# ==============================
# Job Configuration
# ==============================

JOB_OPTIONS = {
    "executor-memory": "7G",
    "executor-memoryOverhead": "3G",
    "driver-memory": "2G",
    "driver-memoryOverhead": "2G",
    "python-memory": None, 
}


# ==============================
# MODIS Configuration
# ==============================

MODIS_RESOLUTION = 25  # meters
MODIS_STAC_URL = "https://stac.eurac.edu/collections/MOD10A1v61"

# ==============================
# UDF Configuration
# ==============================

SCA_UDF_DIR = Path(__file__).parent / 'snow_cover_area' / 'udfs'
SWE_UDF_DIR = Path(__file__).parent / 'snow_water_equivalent' / 'udfs'
SCA_RECONSTRUCTION_UDF = SCA_UDF_DIR / 'snow_cover_area_reconstruction_udf.py'
SWE_RECONSTRUCTION_UDF = SWE_UDF_DIR / 'snow_water_equivalent_udf.py'
SOLAR_POSITION_UDF = SWE_UDF_DIR / 'solar_position_udf.py'
INCIDENCE_ANGLE_UDF = SWE_UDF_DIR / 'incidence_angle_udf.py'