"""Configuration for conditional probability upscaling over the Sentinel era."""

from pathlib import Path

from openeo_mountains_snow.snowcoverarea_reconstruction.config import (  # noqa: F401
    BACKEND,
    JOB_OPTIONS,
    SPATIAL_EXTENT,
)

# Workspace for persisting results as a STAC collection.
WORKSPACE = "snowcop-waw4-1-stac-workspace"

# Sentinel-2 operational era. Adapt the end date to the latest validated period.
FULL_SENTINEL_TEMPORAL_EXTENT = ["2024-01-01", "2025-12-31"]

# Temporal chunking: each job covers this many months.
WINDOW_MONTHS = 12

# How many jobs to run in parallel on the backend.
MAX_PARALLEL_JOBS = 2

# Local paths for job tracking (Parquet-based, standard openEO format).
UPSCALE_ROOT = Path("data") / "upscaling"
JOB_DATABASE_PATH = UPSCALE_ROOT / "job_database.parquet"
