import json
import math
from pathlib import Path

import hydra
import openeo
import shapely
from omegaconf import DictConfig, OmegaConf
from openeo import UDF
from openeo.api.process import Parameter
from openeo.processes import array_append, array_create
from pyproj import Geod, Transformer

from openeo_mountains_snow.common.masks import cloud_water_mask
from openeo_mountains_snow.common.sun import aoi_shape
from openeo_mountains_snow.common.terrain import local_incidence_angle, shadow_mask
from openeo_mountains_snow.config import REPRESENTATIVE_PIXELS_UDF
from openeo_mountains_snow.snow_cover_fraction.udfs.representative_pixels import REPRESENTATIVE_PIXEL_BAND_NAME


def collect_training(inputs_cube):
    """
    Given a cube with preprocessed inputs, collect training data.

    Inputs needed:
    - Sentinel-2 bands (e.g., B02, B03, B04, B08)
    - local solar incidence angle
    - shadow index
    - distance index
    - NDVI, NDSI, diff_B_NIR


    """
    #TODO: pre-mask on NDSI of 0.7??
    collect_training_udf = None
    inputs_cube.apply_polygon(collect_training_udf)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run_openeo(cfg : DictConfig) -> None:
    """
    Run a specific experiment on the command line via:
    python -m openeo_mountains_snow.snow_cover_fraction.snow_cover_fraction +experiment=andes_snow_cover_fraction
    """
    print(OmegaConf.to_yaml(cfg))
    c = openeo.connect(cfg.connection.endpoint).authenticate_oidc()

    if cfg.experiment.aoi is None:
        aoi = json.load(open(Path(__file__).parent.parent / "senales_wgs84.geojson"))
    else:
        aoi = shapely.box(*cfg.experiment.aoi)
        if cfg.experiment.aoi_crs is not None:
            transformer = Transformer.from_crs(cfg.experiment.aoi_crs, "EPSG:4326", always_xy=True)
            from shapely.ops import transform
            aoi = transform(transformer.transform, aoi)


    # define time period
    time_period = list(cfg.experiment.temporal_extent) or ['2022-09-01', '2023-05-01']

    representative_pixels = snow_cover_fraction_cube(aoi,time_period, c, cfg )

    job_options = dict(cfg.experiment.job_options)
    representative_pixels.execute_batch( "representative_pixels_senales_multirange_classified.nc", title=(cfg.experiment.title_prefix or "") , filename_prefix=cfg.experiment.title_prefix , job_options=job_options)


def apply_representative_pixels(bands_indices: openeo.DataCube, neighborhood_size: int = 500) -> openeo.DataCube:
    """Apply the SnowFLAKES representative pixels UDF to a pre-built inputs cube.

    Sensor-agnostic: any cube whose bands match the names expected by
    representative_pixels.py (B02, B03, B04, B08, B11, NDVI, NDSI,
    diff_B_NIR, SI, local_solar_incidence_angle) can be passed in, regardless
    of the original sensor (Sentinel-2, Landsat, …).

    Args:
        bands_indices: Pre-built inputs cube with the standard SnowFLAKES bands.
        neighborhood_size: apply_neighborhood window size in pixels (x and y).
            Use ≈ AOI_size_m / sensor_resolution_m so the UDF gets the full
            tile in one block without NaN padding.
            Defaults to 500 px (≈5 km at 10 m, suitable for small test areas).
    """
    result = bands_indices.apply_neighborhood(
        process=UDF.from_file(str(REPRESENTATIVE_PIXELS_UDF), runtime="Python", version="3.11", context={"classify": True}),
        size=[
            {"dimension": "x", "value": neighborhood_size, "unit": "px"},
            {"dimension": "y", "value": neighborhood_size, "unit": "px"},
            {"dimension": "t", "value": "P1D"},
        ],
    )
    return result.rename_labels(dimension="bands", target=[REPRESENTATIVE_PIXEL_BAND_NAME])


def snow_cover_fraction_cube(aoi, time_period, c, cfg, neighborhood_size: int | None = None):
    """Build the SCF cube for the given AOI.

    Args:
        neighborhood_size: apply_neighborhood window size in pixels (x and y).
            Default: derived from the AOI (see `aoi_neighborhood_size`) so the UDF trains
            once on the full area; 3038 for parameterised (UDP) AOIs.
    """
    if neighborhood_size is None:
        neighborhood_size = 1024 if isinstance(aoi, Parameter) else aoi_neighborhood_size(aoi)
    bands_indices = snowflake_inputs_cube(aoi, time_period, c, cfg)
    return apply_representative_pixels(bands_indices, neighborhood_size)


def aoi_neighborhood_size(aoi, resolution: float = 10, tile: int = 128) -> int:
    """Window (px) covering the whole AOI in one apply_neighborhood chunk.

    The backend's chunk grid is not aligned to the AOI corner (seen: 128 px offset), so a
    window equal to the AOI size splits it into small edge chunks that lack training
    samples. Pad by two tiles and round up to a tile multiple.
    """
    x0, y0, x1, y1 = aoi_shape(aoi).bounds
    geod = Geod(ellps="WGS84")
    width = max(geod.inv(x0, y0, x1, y0)[2], geod.inv(x0, y1, x1, y1)[2])
    height = geod.inv(x0, y0, x0, y1)[2]
    return (math.ceil(max(width, height) / resolution / tile) + 2) * tile


def snowflake_inputs_cube(aoi, time_period, connection, cfg):
    mask = cloud_water_mask(aoi, time_period, connection, cfg)
    sentinel2_bands = connection.load_collection(
        cfg.sentinel2_l1c.collection,
        spatial_extent=aoi,
        temporal_extent=time_period,
        bands=cfg.sentinel2_l1c.bands)
    # DN -> reflectance: the training thresholds are in reflectance; sun angle bands stay in degrees.
    names = sentinel2_bands.metadata.band_names
    sentinel2_bands = sentinel2_bands.apply_dimension(
        dimension="bands",
        process=lambda x: array_create([x[i] / 10000 if n.startswith("B") else x[i] for i, n in enumerate(names)]),
    ).rename_labels(dimension="bands", target=names)
    masked_s2 = sentinel2_bands.mask(mask)

    s2_with_local_angle = local_incidence_angle(masked_s2, aoi, time_period, connection, cfg,
                                                bands_to_retain=[n for n in names if n.startswith("B")])
    return add_indices(shadow_mask(s2_with_local_angle, aoi))


def add_indices(cube):
    """Append NDSI, NDWI and diff_B_NIR (blue - nir), from reflectance bands."""
    from openeo.processes import normalized_difference
    def compute_indices(bands):
        green = bands["B03"]
        nir = bands["B08"]
        updated = array_append(bands, normalized_difference(green, bands["B11"]), "NDSI")
        updated = array_append(updated, normalized_difference(green, nir), "NDWI")
        return array_append(updated, bands["B02"] - nir, "diff_B_NIR")

    return cube.apply_dimension(dimension="bands", process=compute_indices).rename_labels(
        dimension="bands", target=cube.metadata.band_names + ["NDSI", "NDWI", "diff_B_NIR"])


if "__main__" == __name__:
    run_openeo()