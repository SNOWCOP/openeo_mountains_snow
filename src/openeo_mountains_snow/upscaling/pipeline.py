"""Per-tile conditional probability process graphs for upscaling runs.

Thin wrappers around snow_cover_area/conditional_probability.py; only the MODIS date cap
and the output band naming are specific to upscaling.
"""

from __future__ import annotations

import openeo

from openeo_mountains_snow.config import CLOUD_PROB
from openeo_mountains_snow.snow_cover_area.conditional_probability import (
    compute_conditional_probability,
    compute_snow_cover_fraction_range_masks,
)
from openeo_mountains_snow.snow_cover_area.snow_map import calculate_snow
from openeo_mountains_snow.upscaling.config import MODIS_END_DATE


def _conditional_probability(connection, temporal_extent, spatial_extent):
    modis_temporal_extent = [temporal_extent[0], min(temporal_extent[1], MODIS_END_DATE)]
    snow = calculate_snow(connection, temporal_extent, spatial_extent, CLOUD_PROB)
    all_masks, labels = compute_snow_cover_fraction_range_masks(
        connection, snow, modis_temporal_extent, spatial_extent
    )
    return compute_conditional_probability(all_masks, labels), labels


def build_conditional_probability_cube(
    connection: openeo.Connection,
    temporal_extent: list[str],
    spatial_extent: dict,
) -> openeo.DataCube:
    """Conditional probability datacube for one temporal window."""
    (conditional_probability, _, _), _ = _conditional_probability(connection, temporal_extent, spatial_extent)
    return conditional_probability


def build_conditional_probability_with_inputs_cube(
    connection: openeo.Connection,
    temporal_extent: list[str],
    spatial_extent: dict,
) -> openeo.DataCube:
    """Conditional probability plus its numerator (snow_sum_*) and denominator (snow_occur_*)."""
    (conditional_probability, snow_sum, occurrences), labels = _conditional_probability(
        connection, temporal_extent, spatial_extent
    )
    return (
        conditional_probability.rename_labels(dimension="bands", target=[f"cp_{b}" for b in labels])
        .merge_cubes(snow_sum.rename_labels(dimension="bands", target=[f"snow_sum_{b}" for b in labels]))
        .merge_cubes(occurrences.rename_labels(dimension="bands", target=[f"snow_occur_{b}" for b in labels]))
    )
