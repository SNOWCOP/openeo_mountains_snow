import importlib
import json
import math
from pathlib import Path
from typing import TextIO

import hydra
import openeo
import requests
import shapely
from omegaconf import DictConfig, OmegaConf
from openeo import UDF
from openeo.api.process import Parameter
from openeo.extra.spectral_indices import compute_indices
from openeo.processes import any, array_append, process, cos, sin, arccos, array_create, quantiles
from pyproj import Transformer

from openeo_mountains_snow.config import SOLAR_POSITION_UDF
from openeo_mountains_snow.snow_cover_fraction.udfs.representative_pixels import REPRESENTATIVE_PIXEL_BAND_NAME


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
    mask = cloud_water_mask(aoi, time_period, connection, cfg)
    dem_mask = elevation_mask(aoi, connection, cfg)
    sentinel2_bands = connection.load_collection(
        cfg.sentinel2_l1c.collection,
        spatial_extent=aoi,
        temporal_extent=time_period,
        bands=cfg.sentinel2_l1c.bands)
    masked_s2 = sentinel2_bands.mask(mask | dem_mask)

    s2_with_local_angle = local_incidence_angle(masked_s2, aoi, time_period, connection, cfg)


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
    dem_spacetime = connection.load_collection("COPERNICUS_30", spatial_extent=aoi)
    dem = dem_spacetime.reduce_dimension(dimension='t', reducer='mean')

    aspect = dem.aspect()
    slope = dem.slope()

    aspect_slope = aspect.merge_cubes(slope).rename_labels(
        dimension="bands", target=["aspect", "slope"]
    )
    return aspect_slope




def s2_tile_sun_angles(aoi, time_period, collection="sentinel-2-l1c"):
    """Per date, ``[lon, lat, sun_zenith, sun_azimuth]`` at each S2 tile centre, from the CDSE STAC.

    S2 sun angles are tile means, valid at the tile centre: the solar position UDF
    recovers the overpass time there instead of at the (offset) chunk centre.
    """
    if isinstance(aoi, dict):
        features = aoi.get("features", [aoi])
        aoi = shapely.unary_union([shapely.geometry.shape(f.get("geometry", f)) for f in features])
    url = "https://stac.dataspace.copernicus.eu/v1/search"
    body = {"collections": [collection], "bbox": list(aoi.bounds), "limit": 1000,
            "datetime": f"{time_period[0]}T00:00:00Z/{time_period[1]}T23:59:59Z"}
    tiles = {}
    while body:
        response = requests.post(url, json=body, timeout=60)
        response.raise_for_status()
        page = response.json()
        for item in page["features"]:
            props = item["properties"]
            asset = next(a for a in item["assets"].values() if "proj:bbox" in a)
            x0, y0, x1, y1 = asset["proj:bbox"]
            lon, lat = Transformer.from_crs(asset["proj:code"], "EPSG:4326", always_xy=True).transform(
                (x0 + x1) / 2, (y0 + y1) / 2)
            tiles.setdefault(props["datetime"][:10], []).append(
                [lon, lat, 90 - props["view:sun_elevation"], props["view:sun_azimuth"]])
        body = next((link.get("body") for link in page.get("links", []) if link["rel"] == "next"), None)
    return tiles


def local_incidence_angle(s2_cube, aoi, time_period, connection, cfg, bands_to_retain = ["B02", "B03", "B04", "B08", "B11"]):
    # Per-pixel sun angles (radians) at the overpass, replacing the scene-constant S2 angles.
    # UDP builds have parameterised AOI/time: no STAC lookup, the UDF uses the S2 angle bands.
    context = {}
    if not isinstance(aoi, Parameter) and not isinstance(time_period, Parameter):
        context = {"s2_tiles": s2_tile_sun_angles(aoi, time_period)}
    s2_cube = s2_cube.apply_dimension(
        dimension="bands", process=UDF.from_file(str(SOLAR_POSITION_UDF), context=context)
    ).rename_labels(dimension="bands", target=s2_cube.metadata.band_names + ["zenith", "azimuth"])

    slope_aspect_cube = slope_aspect(aoi, connection, cfg)
    combined = s2_cube.merge_cubes(slope_aspect_cube)

    # degree - radians conversions
    deg2rad = math.pi / 180.0
    rad2deg = 180.0 / math.pi

    # define the solar incidence angle function per-pixel
    def compute_sia(data):
        zenith_rad = data["zenith"]
        azimuth_rad = data["azimuth"]
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