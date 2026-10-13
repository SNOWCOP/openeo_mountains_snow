"""High-resolution snow maps for the snow cover area step: band `snow` = 0 / 100 / CLOUD_VALUE.

Two sources:
- `snow_map_from_snow_cover_fraction`: step-1 representative-pixels SCF, binarized (main pipeline).
- `calculate_snow`: Sentinel-2 L2A SCL classes (used by upscaling).
"""

import openeo
from openeo.processes import if_, is_nan, array_create

from openeo_mountains_snow.config import CLOUD_VALUE, CRS, RESOLUTION, SNOW_FRACTION_THRESHOLD


def snow_map_from_snow_cover_fraction(snow_cover_fraction: openeo.DataCube) -> openeo.DataCube:
    """Binarize step-1 SCF (0-100, 255 = masked/no data) to snow (100) / no snow (0) / cloud.

    `snow_cover_fraction` is the cube from `snow_cover_fraction.snow_cover_fraction_cube`
    (band `representative`), run with classify=True.
    """
    def binarize(bands):
        scf = bands[0]
        snow = if_(scf >= SNOW_FRACTION_THRESHOLD, 100.0, 0.0)
        return if_(is_nan(scf).or_(scf > 100), float(CLOUD_VALUE), snow)

    return (snow_cover_fraction
            .resample_spatial(resolution=RESOLUTION, projection=CRS, method="near")  # "average" would mix 255 into valid SCF
            .apply_dimension(dimension="bands", process=binarize)
            .rename_labels(dimension="bands", target=["snow"]))


def calculate_snow(connection, temporal_extent, spatial_extent, cloud_prob=80.0) -> openeo.DataCube:
    """Classify Sentinel-2 L2A SCL into 0 (no snow), 100 (snow), CLOUD_VALUE (cloud)."""
    scl = connection.load_collection(
        "SENTINEL2_L2A",
        temporal_extent=temporal_extent,
        spatial_extent=spatial_extent,
        bands=["SCL"],
        max_cloud_cover=cloud_prob,
    )
    scl = scl.resample_spatial(resolution=RESOLUTION, projection=CRS, method="near")

    def snow_callback(scl_data):
        classification = scl_data["SCL"]
        snow_pixel = (classification == 11) * 100.0
        is_cloud = (classification == 7).or_(classification == 8).or_(classification == 9)
        return if_(is_cloud, float(CLOUD_VALUE), snow_pixel)

    snow = scl.apply_dimension(dimension="bands", process=snow_callback)
    return snow.rename_labels(dimension="bands", target=["snow"])


def create_mask(snow: openeo.DataCube) -> openeo.DataCube:
    """Return two bands: `valid` (not cloud) and `snow` (snow pixel)."""
    def valid_snow(bands):
        snow_band = bands["snow"]
        return array_create([(snow_band <= 100) * 1.0, (snow_band == 100) * 1.0])

    return (snow.apply_dimension(dimension="bands", process=valid_snow)
            .rename_labels(dimension="bands", target=["valid", "snow"]))
