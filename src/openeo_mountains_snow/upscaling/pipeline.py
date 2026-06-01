"""Pipeline assembly for conditional probability upscaling runs.

Mirrors the workflow in downscaling_distribution.py, parametrized for
submission via the openEO MultiBackendJobManager.
"""

from __future__ import annotations

import openeo
from openeo.processes import is_nan, if_, array_create, ProcessBuilder

from openeo_mountains_snow.upscaling.config import MODIS_END_DATE
from openeo_mountains_snow.snowcoverarea_reconstruction.utils_gapfilling import (
    calculate_snow,
    get_scf_ranges,
    create_mask,
)

# Processing parameters (same as downscaling_distribution.py)
CLOUD_PROB = 80
RESOLUTION = 20.0
INVALID_VALUE = 205
INVALID_THRESHOLD = 10
DELTA = 10
EPSILON = 10


def _low_resolution_scf(
    connection: openeo.Connection,
    total_mask: openeo.DataCube,
    temporal_extent: list[str],
    spatial_extent: dict,
) -> openeo.DataCube:
    """Compute low-resolution SCF from MODIS, masked by valid pixel fraction."""
    modis_temporal_extent = [
        temporal_extent[0],
        min(temporal_extent[1], MODIS_END_DATE),
    ]
    modis = connection.load_stac(
        "https://stac.eurac.edu/collections/MOD10A1v61",
        temporal_extent=modis_temporal_extent,
        spatial_extent=spatial_extent,
    )

    average = total_mask.resample_cube_spatial(modis, method="average")

    def create_scf_lr_masked(average_bands: ProcessBuilder):
        snow_band = average_bands["snow"]
        valid_band = average_bands["valid"]

        scf_lr = 100.0 * snow_band / valid_band
        scf_lr = if_(is_nan(scf_lr), INVALID_VALUE, scf_lr)

        valid_threshold = 1 - INVALID_THRESHOLD / 100
        return if_(valid_band <= valid_threshold, INVALID_VALUE, scf_lr)

    scf_lr_masked = average.apply_dimension(
        dimension="bands", process=create_scf_lr_masked
    )
    return scf_lr_masked.rename_labels(dimension="bands", target=["scf"])


def _compute_scf_masks(
    connection: openeo.Connection,
    temporal_extent: list[str],
    spatial_extent: dict,
) -> tuple[openeo.DataCube, list[str]]:
    """Compute SCF binary masks at HR, merged with valid/snow masks."""
    snow = calculate_snow(connection, temporal_extent, spatial_extent, CLOUD_PROB)
    total_mask = create_mask(snow)

    scf_lr_masked = _low_resolution_scf(
        connection, total_mask, temporal_extent, spatial_extent
    )

    scf_dic = get_scf_ranges(DELTA, EPSILON)

    def scf_to_bands(scf_lr_masked):
        result = []
        for key in scf_dic:
            scf_1 = int(key.split("_")[0])
            scf_2 = int(key.split("_")[1])
            if scf_1 == 0:
                mask_scf = (
                    (scf_lr_masked >= scf_1).and_(scf_lr_masked <= scf_2)
                ) * 1.0
            else:
                mask_scf = (
                    (scf_lr_masked > scf_1).and_(scf_lr_masked <= scf_2)
                ) * 1.0
            result.append(mask_scf)
        return array_create(result)

    labels_scf = [f"scf_{v[0]}_{v[1]}" for v in scf_dic.values()]

    all_scf_masks = scf_lr_masked.apply_dimension(scf_to_bands, dimension="bands")
    all_scf_masks = all_scf_masks.rename_labels(dimension="bands", target=labels_scf)

    mask_scf_hr = (
        all_scf_masks
        .resample_spatial(resolution=RESOLUTION, projection=32632, method="near")
        .resample_cube_spatial(snow)
    )

    return mask_scf_hr.merge_cubes(total_mask), labels_scf


def build_conditional_probability_cube(
    connection: openeo.Connection,
    temporal_extent: list[str],
    spatial_extent: dict,
) -> openeo.DataCube:
    """Build the conditional probability datacube for one temporal window.

    This is the upscalable version of downscaling_distribution.py.
    """
    all_masks, labels_scf = _compute_scf_masks(
        connection, temporal_extent, spatial_extent
    )

    def merge_masks(mask_cube):
        return mask_cube.and_(mask_cube.array_element(label="snow")) * 1.0

    mask_cp_snow = all_masks.apply(process=merge_masks).filter_bands(bands=labels_scf)
    sum_cp_snow = mask_cp_snow.reduce_dimension(reducer="sum", dimension="t")

    occurrences = all_masks.reduce_dimension(reducer="sum", dimension="t")
    occurrences = occurrences.filter_bands(bands=labels_scf)

    cp = sum_cp_snow / occurrences

    return cp


def build_conditional_probability_with_inputs_cube(
    connection: openeo.Connection,
    temporal_extent: list[str],
    spatial_extent: dict,
) -> openeo.DataCube:
    """Build CP and export the two pre-division inputs in one datacube.

    Output bands are prefixed to keep labels unique in a single merged product:
    - cp_*: conditional probabilities
    - numerator_*: sum of snow-conditioned mask values over time
    - denominator_*: total SCF-class occurrences over time
    """
    all_masks, labels_scf = _compute_scf_masks(
        connection, temporal_extent, spatial_extent
    )

    def merge_masks(mask_cube):
        return mask_cube.and_(mask_cube.array_element(label="snow")) * 1.0

    mask_cp_snow = all_masks.apply(process=merge_masks).filter_bands(bands=labels_scf)
    sum_cp_snow = mask_cp_snow.reduce_dimension(reducer="sum", dimension="t")

    occurrences = all_masks.reduce_dimension(reducer="sum", dimension="t")
    occurrences = occurrences.filter_bands(bands=labels_scf)

    cp = sum_cp_snow / occurrences

    cp_labels = [f"cp_{label}" for label in labels_scf]
    numerator_labels = [f"snow_sum_{label}" for label in labels_scf]
    denominator_labels = [f"snow_occur_{label}" for label in labels_scf]

    cp = cp.rename_labels(dimension="bands", target=cp_labels)
    sum_cp_snow = sum_cp_snow.rename_labels(dimension="bands", target=numerator_labels)
    occurrences = occurrences.rename_labels(dimension="bands", target=denominator_labels)

    return cp.merge_cubes(sum_cp_snow).merge_cubes(occurrences)
