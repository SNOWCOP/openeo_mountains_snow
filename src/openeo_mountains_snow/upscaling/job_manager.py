"""Job management for large-scale SWE upscaling using the openEO standard API."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import openeo
from pyproj import Transformer
from openeo.extra.job_management import MultiBackendJobManager, ParquetJobDatabase

from openeo_mountains_snow.upscaling.config import (
    BACKEND,
    DEFAULT_TILE_SIZE_M,
    FULL_SENTINEL_TEMPORAL_EXTENT,
    JOB_DATABASE_PATH,
    JOB_OPTIONS,
    MAX_PARALLEL_JOBS,
    SPATIAL_EXTENT,
    WORKSPACE,
)
from openeo_mountains_snow.snowcoverarea_reconstruction.config import CRS
from openeo_mountains_snow.upscaling.pipeline import (
    build_conditional_probability_with_inputs_cube,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def _square_extent_from_center(extent: dict, tile_size_m: int) -> dict:
    """Create a square extent in target CRS units (meters), centered on input AOI."""
    west = float(extent["west"])
    east = float(extent["east"])
    south = float(extent["south"])
    north = float(extent["north"])
    source_crs = extent["crs"]
    target_crs = f"EPSG:{CRS}"

    center_x_src = (west + east) / 2.0
    center_y_src = (south + north) / 2.0

    if source_crs == target_crs:
        center_x = center_x_src
        center_y = center_y_src
    else:
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        center_x, center_y = transformer.transform(center_x_src, center_y_src)

    half = tile_size_m / 2.0

    return {
        "west": center_x - half,
        "south": center_y - half,
        "east": center_x + half,
        "north": center_y + half,
        "crs": target_crs,
    }


# ---------------------------------------------------------------------------
# Job database creation
# ---------------------------------------------------------------------------

def create_job_dataframe(
    max_spatial_tiles: int | None = None,
) -> pd.DataFrame:
    """Build a DataFrame with one row for the standard 20 km tile-size setup.

    All jobs use the full configured timeline. Temporal windowing is disabled.
    No split_area call is used.
    """
    selected_tile_sizes = [DEFAULT_TILE_SIZE_M]
    if max_spatial_tiles is not None and max_spatial_tiles < 1:
        logger.info("max_spatial_tiles < 1 requested; forcing to one standard tile-size scenario")

    rows = []
    for tile_idx_local, tile_size_m in enumerate(selected_tile_sizes):
        tile = _square_extent_from_center(SPATIAL_EXTENT, tile_size_m)
        rows.append({
            "temporal_extent": FULL_SENTINEL_TEMPORAL_EXTENT,
            "spatial_extent": tile,
            "tile_size_m": tile_size_m,
            "title": (
                f"swe_upscale_{FULL_SENTINEL_TEMPORAL_EXTENT[0]}_"
                f"{FULL_SENTINEL_TEMPORAL_EXTENT[1]}_"
                f"{tile_size_m // 1000}km_tile{tile_idx_local:02d}"
            ),
        })

    df = pd.DataFrame(rows)
    logger.info("Job dataframe contains %d row(s) for the fixed 20 km tile-size", len(df))
    return df


def create_job_database(
    db_path: Path = JOB_DATABASE_PATH,
    max_spatial_tiles: int | None = None,
) -> ParquetJobDatabase:
    """Initialize or load an existing ParquetJobDatabase for the upscaling run."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    job_db = ParquetJobDatabase(db_path)
    df = create_job_dataframe(
        max_spatial_tiles=max_spatial_tiles,
    )
    return job_db.initialize_from_df(df, on_exists="skip")


# ---------------------------------------------------------------------------
# start_job callback (called by MultiBackendJobManager per row)
# ---------------------------------------------------------------------------

def start_job(row: pd.Series, connection: openeo.Connection, **kwargs) -> openeo.BatchJob:
    """Build the output cube (CP + pre-division inputs) for one tile and submit."""
    temporal_extent = row["temporal_extent"]
    spatial_extent = row["spatial_extent"]
    title = row.get("title", "swe_upscale")

    logger.info("Building CP and pre-division inputs cube for %s", title)

    cube = build_conditional_probability_with_inputs_cube(
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

def run(
    db_path: Path = JOB_DATABASE_PATH,
    max_spatial_tiles: int | None = None,
) -> None:
    """Connect, initialize the database, and run the job manager."""
    logger.info("Connecting to %s", BACKEND)
    connection = openeo.connect(BACKEND, auto_validate=False)
    connection.authenticate_oidc()

    logger.info("Initializing job database at %s", db_path)
    job_db = create_job_database(
        db_path=db_path,
        max_spatial_tiles=max_spatial_tiles,
    )

    job_manager = MultiBackendJobManager(root_dir=str(db_path.parent / "results"))
    job_manager.add_backend("cdse", connection=connection, parallel_jobs=MAX_PARALLEL_JOBS)

    logger.info("Starting job manager...")
    job_manager.run_jobs(start_job=start_job, job_db=job_db)
    logger.info("Job manager finished.")


