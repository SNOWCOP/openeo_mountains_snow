"""
Shared configuration for the snow cover area and snow water equivalent pipeline (main.py)
and for upscaling. Single source for processing constants.
"""

from pathlib import Path
from types import SimpleNamespace

# ==============================
# Temporal Configuration
# ==============================

TEMPORAL_EXTENT = ['2021-09-01', '2024-09-30']

# Single temporal extent policy for all pipeline inputs.
# Backward-compatible aliases kept for older scripts.
MODIS_TEMPORAL_EXTENT = TEMPORAL_EXTENT.copy()

AGERA_TEMPORAL_EXTENT = TEMPORAL_EXTENT.copy()



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
MODIS_STAC_URL = "https://stac.dataspace.copernicus.eu/v1/collections/modis-terra-mod10a1"
MODIS_SNOW_BAND = "NDSI_Snow_Cover"  # 0-100 valid, >100 flag codes (cloud, water, fill...)

# ==============================
# UDF Configuration
# ==============================

SCA_UDF_DIR = Path(__file__).parent / 'snow_cover_area' / 'udfs'
SWE_UDF_DIR = Path(__file__).parent / 'snow_water_equivalent' / 'udfs'
SCA_RECONSTRUCTION_UDF = SCA_UDF_DIR / 'snow_cover_area_reconstruction_udf.py'
SWE_RECONSTRUCTION_UDF = SWE_UDF_DIR / 'snow_water_equivalent_udf.py'
SOLAR_POSITION_UDF = SWE_UDF_DIR / 'solar_position_udf.py'
INCIDENCE_ANGLE_UDF = SWE_UDF_DIR / 'incidence_angle_udf.py'


# ==============================
# SCF (Step-1) Configuration
# ==============================

# Keep the SnowFLAKES step-1 settings in Python so scripts can run without OmegaConf.
SCF_SETTINGS = {
    "sentinel2_l2a": {
        "collection": "SENTINEL2_L2A",
        "scl_band": "SCL",
        "cloud_values": [8, 9, 10],
    },
    "sentinel2_l1c": {
        "collection": "SENTINEL2_L1C",
        "bands": ["B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B08", "B11", "B12", "sunZenithAngles", "sunAzimuthAngles"],
    },
    "water_mask": {
        "collection": "ESA_WORLDCOVER_10M_2021_V2",
        "band": "MAP",
        "water_values": [80],
    },
    "omnicloudmask": {"model_url": "https://s3.waw3-1.cloudferro.com/project_dependencies/snowcop/omnicloudmask_v4.json"},
    "glacier_mask": {"stac_url": None, "band": "glacier"},
}


def _as_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _as_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_as_namespace(v) for v in value]
    return value


def build_scf_config() -> SimpleNamespace:
    """Return a config object with attribute access expected by snow_cover_fraction.py."""
    return _as_namespace(SCF_SETTINGS)