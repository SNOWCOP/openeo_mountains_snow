"""
Unified entry point for all openEO snow pipelines.

Run experiments via Hydra:
    python -m openeo_mountains_snow.main +experiment=andes_scf
    python -m openeo_mountains_snow.main +experiment=reconstruction
"""

import hydra
import openeo
from omegaconf import DictConfig, OmegaConf

from openeo_mountains_snow.spatial_extent_utils import resolve_aoi


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Connect to openEO backend
    conn = openeo.connect(cfg.connection.endpoint).authenticate_oidc()

    # Resolve area of interest into a standard bbox dict
    spatial_extent = resolve_aoi(cfg.experiment)

    # Dispatch to the requested pipeline
    pipeline = cfg.experiment.pipeline

    if pipeline == "scf":
        _run_scf(cfg, conn, spatial_extent)
    elif pipeline == "reconstruction":
        _run_reconstruction(cfg, conn, spatial_extent)
    else:
        raise ValueError(f"Unknown pipeline: {pipeline!r}. Use 'scf' or 'reconstruction'.")


def _run_scf(cfg: DictConfig, conn: openeo.Connection, spatial_extent: dict):
    """Run the snow cover fraction pipeline."""
    from openeo_mountains_snow.snow_cover_fraction import snow_cover_fraction_cube

    time_period = list(cfg.experiment.temporal_extent)
    cube = snow_cover_fraction_cube(spatial_extent, time_period, conn, cfg)

    job_options = OmegaConf.to_container(cfg.experiment.job_options, resolve=True)
    cube.execute_batch(
        "representative_pixels.nc",
        title=cfg.experiment.title_prefix or "SCF",
        filename_prefix=cfg.experiment.title_prefix,
        job_options=job_options,
    )


def _run_reconstruction(cfg: DictConfig, conn: openeo.Connection, spatial_extent: dict):
    """Run the full historical reconstruction pipeline."""
    from openeo_mountains_snow.snowcoverarea_reconstruction.pipeline import run_reconstruction

    run_reconstruction(cfg, conn, spatial_extent)


if __name__ == "__main__":
    main()
