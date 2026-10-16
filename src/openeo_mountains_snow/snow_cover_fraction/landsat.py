"""
Landsat process-graph pipeline for SnowFLAKES representative pixels.

This module builds everything as an openEO process graph and avoids local
NetCDF extraction for intermediate steps.

Expected source bands (example LANDSAT_L1C stack):
- B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, SZA, VZA

The representative-pixels UDF expects these logical inputs:
- B02, B03, B04, B08, B11, local_solar_incidence_angle,
  NDVI, NDSI, diff_B_NIR, SI

Angles are scaled upfront and SZA is renamed to local_solar_incidence_angle
before entering the representative-pixels UDF.
"""

from __future__ import annotations

from typing import Iterable
from pathlib import Path
import sys

import openeo
from openeo.processes import array_append, array_create


# Map Landsat L1C names to the Sentinel-2-like names used downstream.
LANDSAT_TO_SNOWFLAKES = {
    "B1": "B01",
    "B2": "B02",
    "B3": "B03",
    "B4": "B04",
    "B5": "B08",
    "B6": "B11",
    "B7": "B12",
    "B8": "B8A",
    "B9": "B09",
    "B10": "B10",
    # Keep SWIR mapping on B6->B11; rename thermal B11 to avoid duplicate labels.
    "B11": "B11_THERMAL",
    "SZA": "SZA",
    "VZA": "VZA",
    "SAA": "SAA",
    "VAA": "VAA",

}

# Raw collection bands we try to load from LANDSAT_L1C.
LANDSAT_INPUT_BANDS = list(LANDSAT_TO_SNOWFLAKES.keys())

# Minimal spectral contract required by representative_pixels.py
REQUIRED_SPECTRAL_BANDS = ["B02", "B03", "B04", "B08", "B11"]
REQUIRED_ANGLE_BAND = "SZA"
ANGLE_BANDS = {"SZA", "VZA", "SAA", "VAA"}
DERIVED_INDEX_BANDS = ["NDVI", "NDSI", "diff_B_NIR", "SI"]

# LANDSAT_L1C scaling assumptions for this backend.
REFLECTANCE_SCALE = 1e-4
ANGLE_SCALE = 1e-2


def _make_unique_band_names(names: list[str]) -> list[str]:
    """Return unique band labels while preserving order.

    If duplicates exist, suffix later occurrences as <name>_<n>.
    """
    counts: dict[str, int] = {}
    unique: list[str] = []

    for name in names:
        count = counts.get(name, 0)
        if count == 0:
            unique.append(name)
        else:
            unique.append(f"{name}_{count}")
        counts[name] = count + 1

    return unique


def _require_bands(
    cube: openeo.DataCube,
    *,
    required_bands: Iterable[str],
    context: str,
) -> None:
    """Fail fast when mandatory inputs are absent."""
    available = set(cube.metadata.band_names)
    missing = [band for band in required_bands if band not in available]
    if missing:
        raise ValueError(
            f"Missing required {context} bands: {missing}. Available bands: {sorted(available)}"
        )


def _scale_angles_and_set_local_incidence(cube: openeo.DataCube) -> openeo.DataCube:
    """Scale Landsat angle bands and rename SZA to local_solar_incidence_angle."""
    band_names = list(cube.metadata.band_names)

    def scale_angles(bands):
        scaled = []
        for name in band_names:
            value = bands[name]
            if name in ANGLE_BANDS:
                value = value * ANGLE_SCALE
            scaled.append(value)
        return array_create(scaled)

    scaled_cube = cube.apply_dimension(dimension="bands", process=scale_angles).rename_labels(
        dimension="bands", target=band_names
    )

    renamed = [
        "local_solar_incidence_angle" if b == "SZA" else b
        for b in band_names
    ]
    return scaled_cube.rename_labels(dimension="bands", target=renamed)


def _compute_indices(cube: openeo.DataCube) -> openeo.DataCube:
    """Compute only missing NDVI/NDSI/diff_B_NIR/SI bands."""

    existing = set(cube.metadata.band_names)
    missing_indices = [band for band in DERIVED_INDEX_BANDS if band not in existing]
    if not missing_indices:
        return cube

    need_ndvi = "NDVI" in missing_indices
    need_ndsi = "NDSI" in missing_indices
    need_diff_b_nir = "diff_B_NIR" in missing_indices
    need_si = "SI" in missing_indices

    def add_indices(bands):
        eps = 1e-6
        nir = bands["B08"] * REFLECTANCE_SCALE
        red = bands["B04"] * REFLECTANCE_SCALE
        blue = bands["B02"] * REFLECTANCE_SCALE
        green = bands["B03"] * REFLECTANCE_SCALE
        swir = bands["B11"] * REFLECTANCE_SCALE

        updated = bands
        if need_ndvi:
            ndvi = (nir - red) / (nir + red + eps)
            updated = array_append(updated, ndvi, "NDVI")

        if need_ndsi:
            ndsi = (green - swir) / (green + swir + eps)
            updated = array_append(updated, ndsi, "NDSI")

        if need_diff_b_nir:
            diff_b_nir = (blue - nir) / (blue + nir + eps)
            updated = array_append(updated, diff_b_nir, "diff_B_NIR")

        if need_si:
            si = ((green - swir) / (green + swir + eps)) / (green + eps)
            updated = array_append(updated, si, "SI")

        return updated

    base_names = list(cube.metadata.band_names)
    target_names = base_names + missing_indices
    return cube.apply_dimension(dimension="bands", process=add_indices).rename_labels(
        dimension="bands", target=target_names
    )


def landsat_inputs_cube(
    aoi: dict,
    time_period: list[str],
    connection: openeo.Connection,
    *,
    max_cloud_cover: int = 70,
) -> openeo.DataCube:
    """Build the representative-pixels input cube from LANDSAT_L1C.

    All operations stay in the process graph. No local NetCDF is required.
    """
    landsat = connection.load_collection(
        "LANDSAT_L1C",
        spatial_extent=aoi,
        temporal_extent=time_period,
        bands=LANDSAT_INPUT_BANDS,
        max_cloud_cover=max_cloud_cover,
    )

    original_names = list(landsat.metadata.band_names)
    mapped_names = [LANDSAT_TO_SNOWFLAKES.get(name, name) for name in original_names]
    mapped_names = _make_unique_band_names(mapped_names)
    mapped = landsat.rename_labels(dimension="bands", target=mapped_names)

    _require_bands(mapped, required_bands=REQUIRED_SPECTRAL_BANDS, context="spectral")
    _require_bands(mapped, required_bands=[REQUIRED_ANGLE_BAND], context="angle")

    with_scaled_angles = _scale_angles_and_set_local_incidence(mapped)
    return _compute_indices(with_scaled_angles)


def landsat_representative_pixels_cube(
    aoi: dict,
    time_period: list[str],
    connection: openeo.Connection,
    *,
    max_cloud_cover: int = 70,
    neighborhood_size: int = 100,
) -> openeo.DataCube:
    """Run representative-pixels directly on the Landsat process graph."""
    inputs = landsat_inputs_cube(
        aoi=aoi,
        time_period=time_period,
        connection=connection,
        max_cloud_cover=max_cloud_cover,
    )
    return apply_representative_pixels(inputs, neighborhood_size=neighborhood_size)
