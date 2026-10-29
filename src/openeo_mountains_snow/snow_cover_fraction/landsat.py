"""
Landsat process-graph pipeline for SnowFLAKES representative pixels.

This module builds everything as an openEO process graph and avoids local
NetCDF extraction for intermediate steps.

Expected source bands (example LANDSAT_L1C stack):
- B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, SZA, VZA

The representative-pixels UDF expects these logical inputs:
- B02, B03, B04, B08, B11, B12 (reflectance), cloud_shadow,
  local_solar_incidence_angle, NDSI, NDWI, diff_B_NIR

Angles are scaled upfront and SZA is renamed to local_solar_incidence_angle
before entering the representative-pixels UDF.
"""

from __future__ import annotations

from typing import Iterable

import openeo
from openeo.processes import array_create

from openeo_mountains_snow.snow_cover_fraction.snow_cover_fraction import apply_representative_pixels


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
REQUIRED_SPECTRAL_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
REQUIRED_ANGLE_BAND = "SZA"
ANGLE_BANDS = {"SZA", "VZA", "SAA", "VAA"}

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
    """Build the UDF inputs: reflectance bands, cloud_shadow, angle, NDSI, NDWI, diff_B_NIR = B02 - B08."""
    spectral = REQUIRED_SPECTRAL_BANDS

    def add_indices(bands):
        reflectances = [bands[b] * REFLECTANCE_SCALE for b in spectral]
        blue, green, nir, swir = (reflectances[spectral.index(b)] for b in ["B02", "B03", "B08", "B11"])
        return array_create(reflectances + [
            blue * 0,  # no cloud-shadow source for Landsat yet
            bands["local_solar_incidence_angle"],
            (green - swir) / (green + swir),
            (green - nir) / (green + nir),
            blue - nir,
        ])

    return cube.apply_dimension(dimension="bands", process=add_indices).rename_labels(
        dimension="bands",
        target=spectral + ["cloud_shadow", "local_solar_incidence_angle", "NDSI", "NDWI", "diff_B_NIR"],
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
