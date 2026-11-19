"""Upscaling orchestration for Sentinel-era conditional probability production.

Uses openeo.extra.job_management (MultiBackendJobManager + ParquetJobDatabase)
for robust distributed job tracking and retry logic.
"""

from .config import FULL_SENTINEL_TEMPORAL_EXTENT
from .job_manager import (
    create_job_database,
    preview_jobs,
    run,
    start_job,
)

__all__ = [
    "FULL_SENTINEL_TEMPORAL_EXTENT",
    "create_job_database",
    "preview_jobs",
    "run",
    "start_job",
]
