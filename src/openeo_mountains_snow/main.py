"""
Unified entry point for all openEO snow pipelines.

Run experiments via Hydra:
    python -m openeo_mountains_snow.main +experiment=andes_scf
    python -m openeo_mountains_snow.main +experiment=reconstruction
"""

import json
from pathlib import Path

import hydra
import openeo
import shapely
from omegaconf import DictConfig, OmegaConf
from pyproj import Transformer


def resolve_aoi(cfg: DictConfig):
    """
    Resolve the experiment AOI into a format suitable for openEO.

    Returns a GeoJSON-like geometry or a spatial extent dict depending on format.
    """
    exp = cfg.experiment

    if exp.aoi is None:
        # Default: load the bundled Senales GeoJSON
        return json.load(open(Path(__file__).parent / "senales_wgs84.geojson"))

    aoi = exp.aoi

    # Dict-style AOI {west, south, east, north, crs} — return as-is
    if isinstance(aoi, DictConfig):
        return OmegaConf.to_container(aoi, resolve=True)

    # List-style AOI [west, south, east, north] — convert to shapely box → GeoJSON
    box = shapely.box(*aoi)
    if exp.aoi_crs is not None:
        transformer = Transformer.from_crs(exp.aoi_crs, "EPSG:4326", always_xy=True)
        from shapely.ops import transform
        box = transform(transformer.transform, box)
    return shapely.geometry.mapping(box)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Connect to openEO backend
    conn = openeo.connect(cfg.connection.endpoint).authenticate_oidc()

    # Resolve area of interest
    aoi = resolve_aoi(cfg)

    # Dispatch to the requested pipeline
    pipeline = cfg.experiment.pipeline

    if pipeline == "scf":
        _run_scf(cfg, conn, aoi)
    elif pipeline == "reconstruction":
        _run_reconstruction(cfg, conn, aoi)
    else:
        raise ValueError(f"Unknown pipeline: {pipeline!r}. Use 'scf' or 'reconstruction'.")


def _run_scf(cfg: DictConfig, conn: openeo.Connection, aoi):
    """Run the snow cover fraction pipeline."""
    from openeo_mountains_snow.snow_cover_fraction import snow_cover_fraction_cube

    time_period = list(cfg.experiment.temporal_extent)
    cube = snow_cover_fraction_cube(aoi, time_period, conn, cfg)

    job_options = OmegaConf.to_container(cfg.experiment.job_options, resolve=True)
    cube.execute_batch(
        "representative_pixels.nc",
        title=cfg.experiment.title_prefix or "SCF",
        filename_prefix=cfg.experiment.title_prefix,
        job_options=job_options,
    )


def _run_reconstruction(cfg: DictConfig, conn: openeo.Connection, aoi):
    """Run the full historical reconstruction pipeline."""
    from openeo_mountains_snow.snowcoverarea_reconstruction.pipeline import run_reconstruction

    run_reconstruction(cfg, conn, aoi)


if __name__ == "__main__":
    main()
