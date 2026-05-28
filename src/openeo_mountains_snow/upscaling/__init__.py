"""Upscaling orchestration for Sentinel-era conditional probability production.

Uses openeo.extra.job_management (MultiBackendJobManager + ParquetJobDatabase)
for robust distributed job tracking and retry logic.
"""

from .config import FULL_SENTINEL_TEMPORAL_EXTENT
from .job_manager import create_job_database, run, start_job
from .pipeline import build_conditional_probability_cube

__all__ = [
    "FULL_SENTINEL_TEMPORAL_EXTENT",
    "build_conditional_probability_cube",
    "create_job_database",
    "run",
    "start_job",
]
