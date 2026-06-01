"""CLI entrypoint for Sentinel-era SWE upscaling."""

from pathlib import Path

from openeo_mountains_snow.upscaling.config import JOB_DATABASE_PATH
from openeo_mountains_snow.upscaling.job_manager import run


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SWE upscaling job manager")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=JOB_DATABASE_PATH,
        help="Path to Parquet job database.",
    )
    parser.add_argument(
        "--max-spatial-tiles",
        type=int,
        default=None,
        help="Compatibility option; tile size is fixed to the 20 km standard.",
    )
    args = parser.parse_args()
    run(
        db_path=args.db_path,
        max_spatial_tiles=args.max_spatial_tiles,
    )
