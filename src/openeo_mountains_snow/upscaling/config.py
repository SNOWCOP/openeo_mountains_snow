"""Configuration for conditional probability upscaling over the Sentinel era."""

from pathlib import Path
import json

from openeo_mountains_snow.snowcoverarea_reconstruction.config import (  # noqa: F401
    BACKEND,
    JOB_OPTIONS,
    CRS,
)

# Load full Senales valley boundary from GeoJSON and extract bounding box.
_GEOJSON_PATH = Path(__file__).parent.parent / "senales_wgs84.geojson"
with open(_GEOJSON_PATH) as f:
    _geojson = json.load(f)

# Extract bounding box from all coordinates (WGS84).
_all_coords = []
for feature in _geojson["features"]:
    if feature["geometry"]["type"] == "MultiPolygon":
        for polygon in feature["geometry"]["coordinates"]:
            for ring in polygon:
                _all_coords.extend(ring)

_lons = [c[0] for c in _all_coords]
_lats = [c[1] for c in _all_coords]
SPATIAL_EXTENT_WGS84 = {
    "west": min(_lons),
    "south": min(_lats),
    "east": max(_lons),
    "north": max(_lats),
    "crs": "EPSG:4326",
}

# SWE-relevant region centered on Senales.
# Fixed to ~30 x 30 km so we get a deterministic 3x3 grid of 10 x 10 km jobs.
SPATIAL_EXTENT = {
    "west": 10.6879,
    "south": 46.5868,
    "east": 11.0799,
    "north": 46.8568,
    "crs": "EPSG:4326",
}

# Workspace for persisting results as a STAC collection.
WORKSPACE = "snowcop-workspace"

# Full Sentinel-2 era used for every upscaling job.
FULL_SENTINEL_TEMPORAL_EXTENT = ["2015-01-01", "2025-12-31"]

# Hydra config defaults for snow_cover_fraction_cube
DEFAULT_SCF_CONFIG = {
    "connection": {
        "endpoint": "https://openeo.dataspace.copernicus.eu/"
    },
    "sentinel2_l2a": {
        "collection": "SENTINEL2_L2A",
        "scl_band": "SCL",
        "cloud_values": [8, 9, 3, 10]
    },
    "sentinel2_l1c": {
        "collection": "SENTINEL2_L1C",
        "bands": ["B02", "B03", "B04", "B08", "B11", "sunZenithAngles", "sunAzimuthAngles"]
    },
    "water_mask": {
        "collection": "ESA_WORLDCOVER_10M_2021_V2",
        "band": "MAP",
        "water_values": [80]
    }
}

# Standard spatial tile size (meters): 10 km.
DEFAULT_TILE_SIZE_M = 10000

# How many jobs to run in parallel on the backend.
MAX_PARALLEL_JOBS = 2

# Local paths for job tracking (Parquet-based, standard openEO format).
UPSCALE_ROOT = Path("data") / "upscaling"
JOB_DATABASE_PATH = UPSCALE_ROOT / "job_database.parquet"
