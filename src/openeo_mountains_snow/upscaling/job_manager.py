"""Job management for large-scale SWE upscaling using the openEO standard API."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import openeo
from openeo.extra.job_management import MultiBackendJobManager, ParquetJobDatabase

from openeo_mountains_snow.upscaling.config import (
    BACKEND,
    FULL_SENTINEL_TEMPORAL_EXTENT,
    JOB_DATABASE_PATH,
    JOB_OPTIONS,
    MAX_PARALLEL_JOBS,
    SPATIAL_EXTENT,
    WINDOW_MONTHS,
    WORKSPACE,
)
from openeo_mountains_snow.upscaling.pipeline import build_conditional_probability_cube

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Temporal windowing helpers
# ---------------------------------------------------------------------------

def _sentinel_windows(extent: list[str]) -> list[list[str]]:
    """Split the full temporal extent into WINDOW_MONTHS-sized chunks."""
    global_start = date.fromisoformat(extent[0])
    global_end = date.fromisoformat(extent[1])

    windows: list[list[str]] = []
    current = global_start

    while current < global_end:
        end_year = current.year + (current.month + WINDOW_MONTHS - 1 - 1) // 12
        end_month = (current.month + WINDOW_MONTHS - 1 - 1) % 12 + 1
        last_day = (date(end_year + end_month // 12, end_month % 12 + 1, 1) - date(end_year, end_month, 1)).days
        window_end = min(date(end_year, end_month, last_day), global_end)
        windows.append([current.isoformat(), window_end.isoformat()])
        # Advance to next window
        next_month = current.month + WINDOW_MONTHS
        next_year = current.year + (next_month - 1) // 12
        next_month = (next_month - 1) % 12 + 1
        current = date(next_year, next_month, 1)

    return windows


# ---------------------------------------------------------------------------
# Job database creation
# ---------------------------------------------------------------------------

def create_job_dataframe() -> pd.DataFrame:
    """Build a DataFrame with one row per temporal window, ready for ParquetJobDatabase."""
    windows = _sentinel_windows(FULL_SENTINEL_TEMPORAL_EXTENT)

    rows = []
    for window in windows:
        rows.append({
            "temporal_extent": window,
            "spatial_extent": SPATIAL_EXTENT,
            "title": f"swe_upscale_{window[0]}_{window[1]}",
        })

    return pd.DataFrame(rows)


def create_job_database(db_path: Path = JOB_DATABASE_PATH) -> ParquetJobDatabase:
    """Initialize or load an existing ParquetJobDatabase for the upscaling run."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    job_db = ParquetJobDatabase(db_path)
    df = create_job_dataframe()
    return job_db.initialize_from_df(df, on_exists="skip")


# ---------------------------------------------------------------------------
# start_job callback (called by MultiBackendJobManager per row)
# ---------------------------------------------------------------------------

def start_job(row: pd.Series, connection: openeo.Connection, **kwargs) -> openeo.BatchJob:
    """Build the conditional probability cube for one temporal window and create a batch job."""
    temporal_extent = row["temporal_extent"]
    spatial_extent = row["spatial_extent"]
    title = row.get("title", "swe_upscale")

    logger.info("Building conditional probability cube for %s", title)

    cube = build_conditional_probability_cube(
        connection=connection,
        temporal_extent=temporal_extent,
        spatial_extent=spatial_extent,
    )

    result = cube.save_result(format="netCDF")
    result = result.export_workspace(
        workspace=WORKSPACE,
        merge=title,
    )

    return result.create_job(title=title, job_options=JOB_OPTIONS)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(db_path: Path = JOB_DATABASE_PATH) -> None:
    """Connect, initialize the database, and run the job manager."""
    logger.info("Connecting to %s", BACKEND)
    connection = openeo.connect(BACKEND, auto_validate=False)
    connection.authenticate_oidc()

    logger.info("Initializing job database at %s", db_path)
    job_db = create_job_database(db_path)

    job_manager = MultiBackendJobManager(root_dir=str(db_path.parent / "results"))
    job_manager.add_backend("cdse", connection=connection, parallel_jobs=MAX_PARALLEL_JOBS)

    logger.info("Starting job manager...")
    job_manager.run_jobs(start_job=start_job, job_db=job_db)
    logger.info("Job manager finished.")
