import importlib
import json
import math
from importlib.resources import Package, Resource, files
from pathlib import Path
from typing import TextIO

import hydra
import openeo
from omegaconf import DictConfig, OmegaConf
from openeo import UDF
from openeo.extra.spectral_indices import compute_indices
from openeo.processes import any, array_append, process, cos, sin, arccos, array_create, quantiles

from openeo_mountains_snow.representative_pixels import REPRESENTATIVE_PIXEL_BAND_NAME


def elevation_mask(region, conn: openeo.Connection, cfg:DictConfig):
    elevation = conn.load_collection("COPERNICUS_30", spatial_extent=region).max_time()
    percentile10 = elevation.aggregate_spatial(region, reducer=lambda x: x.quantiles(probabilities = [0.1])).vector_to_raster(target=elevation).rename_labels(dimension="bands", target=["percentile10"])


    return elevation.merge_cubes(percentile10).reduce_dimension(dimension="bands", reducer=lambda x: x[0] < x[1] - 200)

    #altitude_min_threshold = 1600 - 200
    #return conn.load_collection("COPERNICUS_30", spatial_extent=region).max_time().apply(lambda x: x < altitude_min_threshold)





def cloud_water_mask(region, time_period, conn: openeo.Connection, cfg:DictConfig):
    scl = conn.load_collection(
        cfg.sentinel2_l2a.collection,
        spatial_extent=region,
        temporal_extent=time_period,

        bands=[cfg.sentinel2_l2a.scl_band])

    cloud_mask = scl.reduce_dimension(dimension="bands", reducer=lambda x: any([ x == cloud_value for cloud_value in cfg.sentinel2_l2a.cloud_values]))

    water = conn.load_collection(
        cfg.water_mask.collection,
        spatial_extent=region,
        bands=[cfg.water_mask.band]).max_time()

    water_mask = water.reduce_dimension(dimension="bands", reducer=lambda x: any(
        [x == cloud_value for cloud_value in cfg.water_mask.water_values]))

    return cloud_mask | water_mask


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

@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_openeo(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    c = openeo.connect(cfg.connection.endpoint).authenticate_oidc()
    aoi = json.load(open(Path(__file__).parent / ".." / ".."/ "auxiliary"/"senales_wgs84.geojson"))

    # define time period
    time_period = ['2023-02-01', '2023-02-28']

    bands_indices = snowflake_inputs_cube(aoi, time_period, c, cfg)

    representative_pixels = bands_indices.apply_neighborhood(process=get_udf("representative_pixels.py"), size=[{"dimension": "x", "value": 2460, "unit": "px"},{"dimension": "y", "value": 1800, "unit": "px"},{"dimension": "t", "value": "P1D"}])
    representative_pixels = representative_pixels.rename_labels(dimension="bands", target=[REPRESENTATIVE_PIXEL_BAND_NAME])

    job_options = {
        "executor-memory": "3G",
        "python-memory": "disable",
        "executor-memoryOverhead": "4G"
        #"image-name": "python311-dev"
    }
    representative_pixels.execute_batch("representative_pixels_senales_multirange.nc", job_options=job_options)



def get_udf(name):
    with (files('snowflakes_openeo') / name).open('r') as fp:
        udf_code = fp.read()
        return UDF( code= udf_code, runtime="Python", version="3.8")


def snowflake_inputs_cube(aoi, time_period, connection, cfg):
    mask = cloud_water_mask(aoi, time_period, connection, cfg)
    dem_mask = elevation_mask(aoi, connection, cfg)
    sentinel2_bands = connection.load_collection(
        cfg.sentinel2_l1c.collection,
        spatial_extent=aoi,
        temporal_extent=time_period,

        bands=cfg.sentinel2_l1c.bands)
    masked_s2 = sentinel2_bands.mask(mask | dem_mask)

    s2_with_local_angle = local_incidence_angle(masked_s2, aoi, connection, cfg)




    from openeo.processes import normalized_difference
    def compute_indices(bands):
        nir = bands["B08"]
        ndvi = normalized_difference(nir, bands["B04"])
        updated = array_append(bands, ndvi, "NDVI")
        green = bands["B03"]
        swir = bands["B11"]
        ndsi = (green - swir) / (green + swir)
        updated = array_append(updated, ndsi, "NDSI")

        blue = bands["B02"]
        diff_B_NIR = (blue - nir) / (blue + nir)
        updated = array_append(updated, diff_B_NIR, "diff_B_NIR")

        SI = ((green - swir) / (green + swir) / green)
        updated = array_append(updated, SI, "SI")
        return updated

    bands_indices = s2_with_local_angle.apply_dimension(dimension="bands", process=compute_indices).rename_labels(
        dimension="bands", target=s2_with_local_angle.metadata.band_names + ["NDVI", "NDSI", "diff_B_NIR", "SI"])



    return bands_indices


def slope_aspect(aoi, connection, cfg):
    aspect = connection.load_stac(
        "https://stac.openeo.vito.be/collections/DEM_aspec_30m",
        spatial_extent=aoi
    ).reduce_dimension(dimension='t', reducer='mean')

    slope = connection.load_stac(
        "https://stac.openeo.vito.be/collections/DEM_slope_30m",
        spatial_extent=aoi
    ).reduce_dimension(dimension='t', reducer='mean')

    return aspect.merge_cubes(slope).rename_labels(dimension="bands", target=["aspect", "slope"])



def local_incidence_angle(s2_cube, aoi, connection, cfg, bands_to_retain = ["B02", "B03", "B04", "B08", "B11"]):
    slope_aspect_cube = slope_aspect(aoi, connection, cfg)
    combined = s2_cube.merge_cubes(slope_aspect_cube)

    # degree - radians conversions
    deg2rad = math.pi / 180.0
    rad2deg = 180.0 / math.pi

    # define the solar incidence angle function per-pixel
    def compute_sia(data):
        zenith_rad = data["sunZenithAngles"] * deg2rad
        azimuth_rad = data["sunAzimuthAngles"] * deg2rad
        slope_rad = data["slope"] * deg2rad
        aspect_rad = data["aspect"] * deg2rad

        cos_theta = (
            cos(zenith_rad) * cos(slope_rad) +
                sin(zenith_rad) * sin(slope_rad) *
                cos(aspect_rad - azimuth_rad)
        )

        cos_theta_clipped = cos_theta.max(-1).min(1)
        local_angle = arccos(cos_theta_clipped) * rad2deg

        bands = [data[b] for b in bands_to_retain]
        bands.append(local_angle)

        return array_create( bands )

    # apply the function
    extended_cube = combined.apply_dimension(
        dimension="bands",
        process=lambda data: compute_sia(data)
    )
    extended_cube = extended_cube.rename_labels(dimension="bands", target=bands_to_retain + ["local_solar_incidence_angle"])

    return extended_cube



if "__main__" == __name__:
    run_openeo()