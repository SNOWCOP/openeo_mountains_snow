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

# This extent will be re-projected by split_area() to EPSG:32632 for tiling.
SPATIAL_EXTENT = SPATIAL_EXTENT_WGS84

# Workspace for persisting results as a STAC collection.
WORKSPACE = "snowcop-workspace"

# Full Sentinel-2 era used for every upscaling job.
FULL_SENTINEL_TEMPORAL_EXTENT = ["2015-01-01", "2025-12-31"]

# Last date available in the MODIS STAC collection.
MODIS_END_DATE = "2023-12-31"

# Standard spatial tile size (meters): 20 km.
DEFAULT_TILE_SIZE_M = 20000

# How many jobs to run in parallel on the backend.
MAX_PARALLEL_JOBS = 5

# Local paths for job tracking (Parquet-based, standard openEO format).
UPSCALE_ROOT = Path("data") / "upscaling"
JOB_DATABASE_PATH = UPSCALE_ROOT / "job_database.parquet"
