"""CLI entrypoint for snow cover fraction upscaling."""

from pathlib import Path

from extras.upscaling.config import JOB_DATABASE_PATH
from extras.upscaling.job_manager import run


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run snow cover fraction upscaling job manager")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=JOB_DATABASE_PATH,
        help="Path to Parquet job database.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview jobs without submitting.",
    )
    args = parser.parse_args()
    
    if args.preview:
        from extras.upscaling import preview_jobs
        preview_jobs()
    else:
        run(db_path=args.db_path)
