"""Job management for large-scale snowflakes upscaling using the openEO standard API."""

from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime, timezone
import uuid
import os

import pandas as pd
import openeo
from openeo.extra.job_management import (
    MultiBackendJobManager,
    CsvJobDatabase,
    split_area,
)

from openeo_mountains_snow.upscaling.config import (
    BACKEND,
    DEFAULT_SCF_CONFIG,
    DEFAULT_TILE_SIZE_M,
    FULL_SENTINEL_TEMPORAL_EXTENT,
    JOB_DATABASE_PATH,
    JOB_OPTIONS,
    MAX_PARALLEL_JOBS,
    SPATIAL_EXTENT,
    WORKSPACE,
)
from openeo_mountains_snow.snowcoverarea_reconstruction.config import CRS
from openeo_mountains_snow.snow_cover_fraction import snow_cover_fraction_cube

try:
    from openeo_mountains_snow.upscaling.local_auth import (
        OPENEO_AUTH_CLIENT_ID as LOCAL_OPENEO_AUTH_CLIENT_ID,
        OPENEO_AUTH_CLIENT_SECRET as LOCAL_OPENEO_AUTH_CLIENT_SECRET,
        OPENEO_AUTH_PROVIDER_ID as LOCAL_OPENEO_AUTH_PROVIDER_ID,
    )
except ImportError:
    LOCAL_OPENEO_AUTH_CLIENT_ID = None
    LOCAL_OPENEO_AUTH_CLIENT_SECRET = None
    LOCAL_OPENEO_AUTH_PROVIDER_ID = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Job database creation
# ---------------------------------------------------------------------------

def create_job_dataframe(
    max_spatial_tiles: int | None = None,
) -> pd.DataFrame:
    """Build a DataFrame with one row per split_area tile over Senales AOI.

    Uses the official openEO split_area helper to tile the configured AOI.
    All jobs use the full configured timeline.
    """
    from pyproj import Transformer
    from shapely.geometry import box
    from shapely.ops import transform as shp_transform

    # Build AOI box in WGS84 and reproject to the tiling CRS (EPSG:32632).
    aoi_wgs84 = box(
        SPATIAL_EXTENT["west"], SPATIAL_EXTENT["south"],
        SPATIAL_EXTENT["east"], SPATIAL_EXTENT["north"],
    )
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{CRS}", always_xy=True)
    aoi_projected = shp_transform(transformer.transform, aoi_wgs84)

    gdf = split_area(
        aoi=aoi_projected,
        tile_size=DEFAULT_TILE_SIZE_M,
        projection=f"EPSG:{CRS}",
    )

    rows = []
    for tile_idx, geom in enumerate(gdf.geometry):
        bounds = geom.bounds  # (minx, miny, maxx, maxy) in projection CRS
        tile_extent = {
            "west": bounds[0],
            "south": bounds[1],
            "east": bounds[2],
            "north": bounds[3],
            "crs": f"EPSG:{CRS}",
        }
        rows.append({
            "temporal_extent": FULL_SENTINEL_TEMPORAL_EXTENT,
            "spatial_extent": tile_extent,
            "tile_size_m": DEFAULT_TILE_SIZE_M,
            "title": (
                f"snowflakes_upscale_{FULL_SENTINEL_TEMPORAL_EXTENT[0]}_"
                f"{FULL_SENTINEL_TEMPORAL_EXTENT[1]}_"
                f"{DEFAULT_TILE_SIZE_M // 1000}km_tile{tile_idx:02d}"
            ),
        })

    if max_spatial_tiles is not None:
        rows = rows[:max_spatial_tiles]

    df = pd.DataFrame(rows)
    logger.info(
        "Job dataframe contains %d row(s) from split_area tiles of %d km",
        len(df),
        DEFAULT_TILE_SIZE_M // 1000,
    )
    return df


def create_job_database(
    db_path: Path = JOB_DATABASE_PATH,
) -> CsvJobDatabase:
    """Initialize or load an existing CsvJobDatabase for the upscaling run."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    job_db = CSVJobDatabase(db_path)
    df = create_job_dataframe()
    return job_db.initialize_from_df(df, on_exists="skip")


# ---------------------------------------------------------------------------
# start_job callback (called by MultiBackendJobManager per row)
# ---------------------------------------------------------------------------

def _to_wgs84(spatial_extent: dict) -> dict:
    """Convert a projected spatial extent to WGS84 for STAC endpoints."""
    from pyproj import Transformer

    source_crs = str(spatial_extent.get("crs", "EPSG:4326"))
    if source_crs == "EPSG:4326":
        return spatial_extent
    tr = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    west, south = tr.transform(spatial_extent["west"], spatial_extent["south"])
    east, north = tr.transform(spatial_extent["east"], spatial_extent["north"])
    return {"west": west, "south": south, "east": east, "north": north, "crs": "EPSG:4326"}


def start_job(
    row: pd.Series,
    connection: openeo.Connection,
    **kwargs,
) -> openeo.BatchJob:
    """Start a single upscaling job for one tile using snow_cover_fraction_cube.

    The row must contain 'spatial_extent' (in EPSG:32632) and 'temporal_extent'.
    This function converts the spatial extent to WGS84 before passing to the pipeline.
    """
    logger.info("Starting job '%s' (tile_size=%d m)", row["title"], row["tile_size_m"])

    # Compute neighborhood_size from the projected tile bounds (EPSG:32632, in metres).
    # This ensures apply_neighborhood receives exactly the tile area with no NaN padding.
    import math
    SENTINEL2_RESOLUTION_M = 10.0
    projected = row["spatial_extent"]
    width_px = math.ceil((projected["east"] - projected["west"]) / SENTINEL2_RESOLUTION_M)
    height_px = math.ceil((projected["north"] - projected["south"]) / SENTINEL2_RESOLUTION_M)
    neighborhood_size = max(width_px, height_px)
    logger.info("Computed neighborhood_size=%d px (%dx%d)", neighborhood_size, width_px, height_px)

    # Convert tile extent from EPSG:32632 to WGS84 for OpenEO.
    spatial_extent_wgs84 = _to_wgs84(row["spatial_extent"])

    # snow_cover_fraction_cube expects a shapely geometry, not a bbox dict.
    from shapely.geometry import box as shapely_box
    aoi = shapely_box(
        spatial_extent_wgs84["west"],
        spatial_extent_wgs84["south"],
        spatial_extent_wgs84["east"],
        spatial_extent_wgs84["north"],
    )

    # Build Hydra-style config from defaults.
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(DEFAULT_SCF_CONFIG)

    # Convert temporal extent to list (row returns numpy array)
    temporal_extent = list(row["temporal_extent"])

    # Call the snow_cover_fraction_cube pipeline.
    result_cube = snow_cover_fraction_cube(
        aoi=aoi,
        time_period=temporal_extent,
        c=connection,
        cfg=cfg,
        neighborhood_size=neighborhood_size,
    )

    # Persist each tile result in user workspace with a unique merge path.
    unique_suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    merge_path = f"snowflakes_upscaling/{row['title']}_{unique_suffix}"
    result = result_cube.save_result(format="NetCDF").export_workspace(
        workspace=WORKSPACE,
        merge=merge_path,
    )

    batch_job = result.create_job(
        title=row["title"],
        job_options=JOB_OPTIONS,
    )

    return batch_job



# ---------------------------------------------------------------------------
# Job preview/validation
# ---------------------------------------------------------------------------

def preview_jobs() -> None:
    """Preview and validate the job dataframe before running."""
    df = create_job_dataframe()
    
    print(f"\n{'='*70}")
    print(f"UPSCALING JOB PREVIEW")
    print(f"{'='*70}")
    print(f"Total jobs: {len(df)}")
    print(f"Temporal extent: {df.iloc[0]['temporal_extent']}")
    print(f"Tile size: {df.iloc[0]['tile_size_m'] / 1000:.0f} km")
    print(f"\nJob titles:")
    for idx, row in df.iterrows():
        extent = row["spatial_extent"]
        print(f"  {idx:2d}: {row['title']}")
        print(f"       extent: {extent['west']:.0f}, {extent['south']:.0f} → {extent['east']:.0f}, {extent['north']:.0f} ({extent['crs']})")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(
    db_path: Path = JOB_DATABASE_PATH,
) -> None:
    """Connect, initialize the database, and run the job manager."""
    logger.info("Connecting to %s", BACKEND)
    connection = openeo.connect(BACKEND, auto_validate=False)

    client_id = os.environ.get("OPENEO_AUTH_CLIENT_ID") or LOCAL_OPENEO_AUTH_CLIENT_ID
    client_secret = os.environ.get("OPENEO_AUTH_CLIENT_SECRET") or LOCAL_OPENEO_AUTH_CLIENT_SECRET
    provider_id = os.environ.get("OPENEO_AUTH_PROVIDER_ID") or LOCAL_OPENEO_AUTH_PROVIDER_ID
    if not client_id or not client_secret:
        raise RuntimeError(
            "Missing OPENEO_AUTH_CLIENT_ID and/or OPENEO_AUTH_CLIENT_SECRET for client credentials authentication."
        )
    connection.authenticate_oidc_client_credentials(
        client_id=client_id,
        client_secret=client_secret,
        provider_id=provider_id,
    )

    logger.info("Initializing job database at %s", db_path)
    job_db = create_job_database(db_path=db_path)

    job_manager = MultiBackendJobManager(root_dir=str(db_path.parent / "results"))
    job_manager.add_backend("cdse", connection=connection, parallel_jobs=MAX_PARALLEL_JOBS)

    logger.info("Starting job manager...")
    #job_manager.run_jobs(start_job=start_job, job_db=job_db)
    #logger.info("Job manager finished.")


