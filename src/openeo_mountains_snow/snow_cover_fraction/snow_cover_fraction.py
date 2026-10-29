import importlib
import json
import math
from pathlib import Path
from typing import TextIO

import hydra
import openeo
import shapely
from omegaconf import DictConfig, OmegaConf
from openeo import UDF
from openeo.extra.spectral_indices import compute_indices
from openeo.processes import any, array_append, process, cos, sin, arccos, array_create, quantiles, if_
from pyproj import Transformer

from openeo_mountains_snow.snow_cover_fraction.udfs.representative_pixels import REPRESENTATIVE_PIXEL_BAND_NAME


def elevation_mask(region, conn: openeo.Connection, cfg:DictConfig):
    elevation = conn.load_collection("COPERNICUS_30", spatial_extent=region).max_time()
    percentile10 = elevation.aggregate_spatial(region, reducer=lambda x: x.quantiles(probabilities = [0.1])).vector_to_raster(target=elevation).rename_labels(dimension="bands", target=["percentile10"])


    return elevation.merge_cubes(percentile10).reduce_dimension(dimension="bands", reducer=lambda x: x[0] < x[1] - 200)

    #altitude_min_threshold = 1600 - 200
    #return conn.load_collection("COPERNICUS_30", spatial_extent=region).max_time().apply(lambda x: x < altitude_min_threshold)



def omnicloudmask_classes(region, time_period, conn: openeo.Connection, cfg: DictConfig, dilation_iterations: int = 3):
    """OmniCloudMask on L1C via predict_onnx: 0 clear, 1 thick cloud, 2 thin cloud, 3 shadow, 255 nodata.

    Thick cloud is dilated like SnowFLAKES `create_omnicloudmask` (scipy binary_dilation, cross
    structure, `dilation_iterations` times == diamond of that radius).
    """
    rgn = conn.load_collection(
        cfg.sentinel2_l1c.collection,
        spatial_extent=region,
        temporal_extent=time_period,
        bands=["B04", "B03", "B8A"])
    rgn = rgn.process("convert_data_type", data=rgn, data_type="float32")
    predicted = rgn.process("predict_onnx", data=rgn, model=cfg.omnicloudmask.model_url)
    classes = predicted.reduce_dimension(dimension="bands", reducer=lambda x: x.array_element(0))

    r = dilation_iterations
    kernel = [[1 if abs(i) + abs(j) <= r else 0 for j in range(-r, r + 1)] for i in range(-r, r + 1)]
    thick_dilated = (classes.apply(lambda x: if_(x == 1, 1, 0))
                     .apply_kernel(kernel)
                     .apply(lambda x: x > 0))
    return classes.mask(thick_dilated, replacement=1)


def glacier_mask(region, conn: openeo.Connection, cfg: DictConfig):
    """True on glacier pixels from the hosted glacier STAC collection, None if not configured."""
    if cfg.glacier_mask.stac_url is None:
        return None
    glacier = conn.load_stac(cfg.glacier_mask.stac_url, spatial_extent=region, bands=[cfg.glacier_mask.band])
    return glacier.max_time().reduce_dimension(dimension="bands", reducer=lambda x: x[0] == 1)


def cloud_water_mask(region, time_period, conn: openeo.Connection, cfg:DictConfig):
    """Return (cloud | water mask, cloud shadow mask)."""
    if cfg.omnicloudmask.model_url is not None:
        classes = omnicloudmask_classes(region, time_period, conn, cfg)
        cloud_mask = classes.apply(lambda x: (x == 1).or_(x == 2))
        cloud_shadow = classes.apply(lambda x: x == 3)
    else:
        scl = conn.load_collection(
            cfg.sentinel2_l2a.collection,
            spatial_extent=region,
            temporal_extent=time_period,

            bands=[cfg.sentinel2_l2a.scl_band])

        cloud_mask = scl.reduce_dimension(dimension="bands", reducer=lambda x: any([x[0] == cloud_value for cloud_value in cfg.sentinel2_l2a.cloud_values]))
        cloud_shadow = scl.reduce_dimension(dimension="bands", reducer=lambda x: x[0] == 3)

    water = conn.load_collection(
        cfg.water_mask.collection,
        spatial_extent=region,
        bands=[cfg.water_mask.band]).max_time()

    water_mask = water.reduce_dimension(dimension="bands", reducer=lambda x: any(
        [x[0] == water_value for water_value in cfg.water_mask.water_values]))

    return cloud_mask | water_mask, cloud_shadow


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
    representative_pixels.py (B02, B03, B04, B08, B11 as reflectance in [0, 1],
    cloud_shadow, local_solar_incidence_angle, NDSI, NDWI, diff_B_NIR = B02 - B08)
    can be passed in, regardless of the original sensor (Sentinel-2, Landsat, …).

    Args:
        bands_indices: Pre-built inputs cube with the standard SnowFLAKES bands.
        neighborhood_size: apply_neighborhood window size in pixels (x and y).
            Use ≈ AOI_size_m / sensor_resolution_m so the UDF gets the full
            tile in one block without NaN padding.
            Defaults to 500 px (≈5 km at 10 m, suitable for small test areas).
    """
    result = bands_indices.apply_neighborhood(
        process=get_udf("representative_pixels.py"),
        size=[
            {"dimension": "x", "value": neighborhood_size, "unit": "px"},
            {"dimension": "y", "value": neighborhood_size, "unit": "px"},
            {"dimension": "t", "value": "P1D"},
        ],
    )
    return result.rename_labels(dimension="bands", target=[REPRESENTATIVE_PIXEL_BAND_NAME])


def snow_cover_fraction_cube(aoi, time_period, c, cfg, neighborhood_size: int = 3038):
    """Build the SCF cube for the given AOI.

    Args:
        neighborhood_size: apply_neighborhood window size in pixels (x and y).
            Set this to the pixel dimensions of the input AOI at 10 m resolution
            so the UDF receives the full area in one block without NaN padding.
            Defaults to 3038 (≈30×30 km at 10 m, the original full-area size).
    """
    bands_indices = snowflake_inputs_cube(aoi, time_period, c, cfg)
    return apply_representative_pixels(bands_indices, neighborhood_size)


def get_udf(name):
    with (Path(__file__).parent / "udfs" / name).open('r') as fp:
        udf_code = fp.read()
        return UDF( code= udf_code, runtime="Python", version="3.11", context={"classify":True})


def snowflake_inputs_cube(aoi, time_period, connection, cfg):
    mask, cloud_shadow = cloud_water_mask(aoi, time_period, connection, cfg)
    dem_mask = elevation_mask(aoi, connection, cfg)
    sentinel2_bands = connection.load_collection(
        cfg.sentinel2_l1c.collection,
        spatial_extent=aoi,
        temporal_extent=time_period,
        bands=cfg.sentinel2_l1c.bands)
    sentinel2_bands = sentinel2_bands.merge_cubes(
        cloud_shadow.add_dimension(name="bands", label="cloud_shadow", type="bands"))
    masked_s2 = sentinel2_bands.mask(mask | dem_mask)

    spectral = ["B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B08", "B11", "B12"]
    s2_with_local_angle = local_incidence_angle(masked_s2, aoi, connection, cfg,
                                                bands_to_retain=spectral + ["cloud_shadow"])

    # SnowFLAKES reference: reflectance in [0, 1], diff_B_NIR = B02 - B08.
    def compute_indices(bands):
        reflectances = [bands[b] / 10000 for b in spectral]
        blue, green, nir, swir = (reflectances[spectral.index(b)] for b in ["B02", "B03", "B08", "B11"])
        return array_create(reflectances + [
            bands["cloud_shadow"], bands["local_solar_incidence_angle"],
            (green - swir) / (green + swir),
            (green - nir) / (green + nir),
            blue - nir,
        ])

    inputs = s2_with_local_angle.apply_dimension(dimension="bands", process=compute_indices).rename_labels(
        dimension="bands",
        target=spectral + ["cloud_shadow", "local_solar_incidence_angle", "NDSI", "NDWI", "diff_B_NIR"])

    # Cast-shadow CLOSDI threshold: 10th percentile over self-shadow pixels of the whole AOI, per date.
    def self_shadow_closdi(bands):
        red, nir = bands["B04"], bands["B08"]
        closdi = (1 - 1.5 * nir - 0.1 * red) * 100 / (1 + 3.5 * nir + 4.9 * red)
        return array_create([if_(bands["local_solar_incidence_angle"] > 90, closdi)])

    closdi_threshold = (inputs.apply_dimension(dimension="bands", process=self_shadow_closdi)
                        .aggregate_spatial(aoi, reducer=lambda x: x.quantiles(probabilities=[0.1]))
                        .vector_to_raster(target=inputs)
                        .rename_labels(dimension="bands", target=["closdi_threshold"]))
    return inputs.merge_cubes(closdi_threshold)


def slope_aspect(aoi, connection, cfg):
    dem_spacetime = connection.load_collection("COPERNICUS_30", spatial_extent=aoi)
    dem = dem_spacetime.reduce_dimension(dimension='t', reducer='mean')

    aspect = dem.aspect()
    slope = dem.slope()

    aspect_slope = aspect.merge_cubes(slope).rename_labels(
        dimension="bands", target=["aspect", "slope"]
    )
    return aspect_slope




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