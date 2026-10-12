"""Conditional probability of high-resolution snow given low-resolution MODIS snow cover fraction.

Single implementation used by both `main.py` and `upscaling/pipeline.py`.
"""

import openeo
from openeo import DataCube
from openeo.processes import eq, gt, or_, if_, is_nan, array_create

from openeo_mountains_snow.config import (
    CLOUD_PROB, CLOUD_VALUE, CRS, DELTA, EPSILON, INVALID_THRESHOLD, INVALID_VALUE,
    MODIS_RESOLUTION, MODIS_STAC_URL, RESOLUTION,
)
from openeo_mountains_snow.snow_cover_area.sentinel2_snow_mask import calculate_snow, create_mask


def get_snow_cover_fraction_ranges(delta, epsilon):
    """Map buffered range keys like "0_20" to base (lower, upper) SCF intervals."""
    ranges = {}
    for lower in range(0, 100, delta):
        upper = lower + delta
        ranges[f"{max(0, lower - epsilon)}_{min(100, upper + epsilon)}"] = (lower, upper)
    return ranges


def low_resolution_snow_cover_fraction(
    connection: openeo.Connection,
    valid_and_snow_mask: DataCube,
    temporal_extent: list[str],
    spatial_extent: dict,
) -> DataCube:
    """Aggregate the high-resolution snow mask to the MODIS grid as SCF (0-100)."""
    modis = connection.load_stac(
        MODIS_STAC_URL,
        temporal_extent=temporal_extent,
        spatial_extent=spatial_extent,
    )
    average = valid_and_snow_mask.resample_cube_spatial(modis, method="average")

    def to_scf(bands):
        valid = bands["valid"]
        scf = 100.0 * bands["snow"] / valid
        scf = if_(is_nan(scf), CLOUD_VALUE, scf)
        return if_(valid <= 1 - INVALID_THRESHOLD / 100, INVALID_VALUE, scf)

    return (average.apply_dimension(dimension="bands", process=to_scf)
            .rename_labels(dimension="bands", target=["scf"]))


def compute_snow_cover_fraction_masks(
    connection: openeo.Connection,
    temporal_extent: list[str],
    spatial_extent: dict,
    modis_temporal_extent: list[str] | None = None,
) -> tuple[DataCube, list[str]]:
    """Per-SCF-range binary masks at high resolution, merged with `valid`/`snow` bands."""
    snow = calculate_snow(connection, temporal_extent, spatial_extent, CLOUD_PROB)
    valid_and_snow_mask = create_mask(snow)
    scf_low_res = low_resolution_snow_cover_fraction(
        connection, valid_and_snow_mask, modis_temporal_extent or temporal_extent, spatial_extent
    )

    ranges = get_snow_cover_fraction_ranges(DELTA, EPSILON)

    def to_range_bands(scf):
        masks = []
        for key in ranges:
            lower, upper = (int(v) for v in key.split("_"))
            above = scf >= lower if lower == 0 else scf > lower
            masks.append(above.and_(scf <= upper) * 1.0)
        return array_create(masks)

    labels = [f"scf_{lower}_{upper}" for lower, upper in ranges.values()]
    range_masks = (scf_low_res.apply_dimension(to_range_bands, dimension="bands")
                   .rename_labels(dimension="bands", target=labels)
                   .resample_spatial(resolution=RESOLUTION, projection=CRS, method="near")
                   .resample_cube_spatial(snow))
    return range_masks.merge_cubes(valid_and_snow_mask), labels


def compute_conditional_probability(
    all_masks: DataCube,
    labels: list[str],
) -> tuple[DataCube, DataCube, DataCube]:
    """Return (conditional_probability, snow_sum, occurrences), all with the SCF range labels."""
    def snow_in_range(mask_cube):
        return mask_cube.and_(mask_cube.array_element(label="snow")) * 1.0

    snow_sum = (all_masks.apply(process=snow_in_range).filter_bands(bands=labels)
                .reduce_dimension(reducer="sum", dimension="t"))
    occurrences = all_masks.reduce_dimension(reducer="sum", dimension="t").filter_bands(bands=labels)
    return snow_sum / occurrences, snow_sum, occurrences


def create_modis_snow_cover_fraction_cube(
    connection: openeo.Connection,
    temporal_extent: list[str],
    spatial_extent: dict,
) -> DataCube:
    """MODIS SCF band `scf`: 0-100 valid, CLOUD_VALUE for cloud/water/no-data."""
    modis = connection.load_stac(
        url=MODIS_STAC_URL,
        temporal_extent=temporal_extent,
        spatial_extent=spatial_extent,
        bands=["SCF"],
    )

    def clean(pixel):
        value = pixel.array_element(index=0)
        # MODIS codes: 254 water, 255 no data, >100 other invalid.
        invalid = or_(eq(value, 254), or_(eq(value, 255), gt(value, 100)))
        return if_(invalid, CLOUD_VALUE, value)

    return (modis.apply(clean)
            .resample_spatial(resolution=MODIS_RESOLUTION, projection=CRS, method="near")
            .rename_labels(dimension="bands", target=["scf"]))
