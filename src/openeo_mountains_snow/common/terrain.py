"""DEM slope/aspect, local solar incidence angle and terrain shadow (openEO process graphs)."""

import math

from openeo import UDF
from openeo.api.process import Parameter
from openeo.processes import arccos, array_create, cos, quantiles, sin

from openeo_mountains_snow.common.sun import s2_tile_sun_angles
from openeo_mountains_snow.config import SOLAR_POSITION_UDF


# openEO slope()/aspect() output unit on CDSE: radians (checked: slope max 1.19, aspect max 6.28).
SLOPE_ASPECT_IN_RADIANS = True


def shadow_mask(s2_cube, region, nir_band="B8A", cloud_shadow=None):
    """Add the scene-wide composite shadow mask (self-shadow | spectral shadow | cloud shadow) as a new band.

    `cloud_shadow`: optional boolean cube (t, y, x), e.g. OmniCloudMask class 3, as in SnowFLAKES.
    """
    def select_band(cube, name):
        # ``DataCube.band`` enables a client-side band-math mode that cannot be
        # combined with the aggregated statistic cubes below.
        return cube.filter_bands([name]).reduce_dimension(dimension="bands", reducer="first")

    nir = select_band(s2_cube, nir_band)
    red = select_band(s2_cube, "B04")
    sia = select_band(s2_cube, "local_solar_incidence_angle")

    curr_angle_valid = (sia >= 70) & (sia <= 90)
    self_shadow = sia > 90

    # spectral shadows
    CLOSDI = (1 - 1.5 * nir - 0.1 * red) * 100 / (1 + 3.5 * nir + 4.9 * red)

    # threshold = 10th percentile of CLOSDI over self-shadow pixels
    threshold = CLOSDI.mask(~self_shadow).aggregate_spatial(
        geometries=region,
        reducer=lambda data: quantiles(data, probabilities=[0.1], ignore_nodata=True)
    ).vector_to_raster(target=CLOSDI)

    spectral_shadow = (CLOSDI - threshold) > 0

    shadow = (spectral_shadow & curr_angle_valid) | self_shadow
    if cloud_shadow is not None:
        shadow = shadow | cloud_shadow

    shadow = shadow.process_with_node(shadow.result_node(), metadata=sia.metadata)
    shadow_band = shadow.add_dimension(name="bands", label="shadow_mask", type="bands")

    return s2_cube.merge_cubes(shadow_band)


def slope_aspect(aoi, connection, cfg):
    dem_spacetime = connection.load_collection("COPERNICUS_30", spatial_extent=aoi)
    dem = dem_spacetime.reduce_dimension(dimension='t', reducer='mean')

    aspect = dem.aspect()
    slope = dem.slope()

    aspect_slope = aspect.merge_cubes(slope).rename_labels(
        dimension="bands", target=["aspect", "slope"]
    )
    return aspect_slope


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
        slope_rad = data["slope"] if SLOPE_ASPECT_IN_RADIANS else data["slope"] * deg2rad
        aspect_rad = data["aspect"] if SLOPE_ASPECT_IN_RADIANS else data["aspect"] * deg2rad

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
