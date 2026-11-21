"""
Shared configuration for the snow cover area and snow water equivalent pipeline (main.py)
and for extras/upscaling. Single source for processing constants.
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


def utm_epsg(spatial_extent: dict) -> int:
    """EPSG code of the WGS84 UTM zone at the centre of the extent (32632 for the Senales valley)."""
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(spatial_extent.get("crs", "EPSG:4326"), "EPSG:4326", always_xy=True).transform(
        (spatial_extent["west"] + spatial_extent["east"]) / 2, (spatial_extent["south"] + spatial_extent["north"]) / 2)
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1

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
INVALID_PIXEL_THRESHOLD = 40  # LR pixel invalid from this many invalid HR pixels (original nv_thres)
SNOW_FRACTION_THRESHOLD = 15  # SCF (%) above which a pixel counts as snow (original dailyHR_SCA snow_thres)

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

N_DAYS_TO_RECONSTRUCT = None  # None = every day (SWE needs the full season)
NEIGHBORHOOD_SIZE = 64
# SCA tiles get one LR pixel of context on each side so LR pixels on tile edges stay complete.
SCA_NEIGHBORHOOD_OVERLAP = PIXEL_RATIO

# ==============================
# Job Configuration
# ==============================

# Tuned from batch-job usage metrics (peak executor memory, cpu_utilization_ratio).
# Heavy: Sentinel-2 SCF chain (peak ~14.9 GB, cpu ratio 0.82).
JOB_OPTIONS = {
    "executor-memory": "8G",
    "executor-memoryOverhead": "8G",
    'executor-cores': 1,
    "driver-memory": "2G",
    "driver-memoryOverhead": "2G",
    "python-memory": "disable",
    'max-executors': 6
}
# Medium: meteo, swe, sca, cp_occ (peak ~9 GB, cpu ratio 0.71-0.80).
JOB_OPTIONS_MEDIUM = {**JOB_OPTIONS, "executor-memoryOverhead": "4G", 'max-executors': 4}
# Small: modis_scf, snow, terrain (peak ~1 GB, cpu ratio 0.10-0.60).
JOB_OPTIONS_SMALL = {
    **JOB_OPTIONS,
    "executor-memory": "2G",
    "executor-memoryOverhead": "2G",
    'max-executors': 2,
}



# ==============================
# MODIS Configuration
# ==============================

MODIS_RESOLUTION = PIXEL_RATIO * RESOLUTION  # meters (500 m LR grid, aligned with the HR grid)
MODIS_STAC_URL = "https://stac.dataspace.copernicus.eu/v1/collections/modis-terra-mod10a1"
MODIS_SNOW_BAND = "NDSI_Snow_Cover"  # 0-100 valid, >100 flag codes (cloud, water, fill...)

# ERA5 surface geopotential z (m2/s2, 0.1 deg), reference height for the AGERA5 temperature downscaling.
GEOPOTENTIAL_STAC_URL = "https://s3.waw3-1.cloudferro.com/project_dependencies/snowcop/geopotential_cdse.json"

# ==============================
# UDF Configuration
# ==============================

SCA_UDF_DIR = Path(__file__).parent / 'snow_cover_area' / 'udfs'
SWE_UDF_DIR = Path(__file__).parent / 'snow_water_equivalent' / 'udfs'
SCF_UDF_DIR = Path(__file__).parent / 'snow_cover_fraction' / 'udfs'
COMMON_UDF_DIR = Path(__file__).parent / 'common' / 'udfs'
SCA_RECONSTRUCTION_UDF = SCA_UDF_DIR / 'snow_cover_area_reconstruction_udf.py'
SWE_RECONSTRUCTION_UDF = SWE_UDF_DIR / 'snow_water_equivalent_udf.py'
TEMPERATURE_DOWNSCALING_UDF = SWE_UDF_DIR / 'temperature_downscaling_udf.py'
REPRESENTATIVE_PIXELS_UDF = SCF_UDF_DIR / 'representative_pixels.py'
SOLAR_POSITION_UDF = COMMON_UDF_DIR / 'solar_position_udf.py'
INCIDENCE_ANGLE_UDF = COMMON_UDF_DIR / 'incidence_angle_udf.py'


# ==============================
# SCF (Step-1) Configuration
# ==============================

# Keep the SnowFLAKES step-1 settings in Python so scripts can run without OmegaConf.
SCF_SETTINGS = {
    "sentinel2_l2a": {
        "collection": "SENTINEL2_L2A",
        "scl_band": "SCL",
        "cloud_values": [8, 9, 3, 10],
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
    "glacier_mask": {"stac_url": "https://s3.waw3-1.cloudferro.com/project_dependencies/snowcop/glacier_mask.json", "band": "glacier"},
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