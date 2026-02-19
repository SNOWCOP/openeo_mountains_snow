"""
Snow cover fraction downscaling and distribution analysis.

Computes conditional probabilities of high-resolution snow cover based on
low-resolution snow cover fractions.

Created on Tue Apr 22 20:16:37 2025
@author: vpremier
"""

from typing import Tuple
import os

import openeo
from openeo.processes import (
    and_, is_nan, if_, array_create, ProcessBuilder
)

from utils_gapfilling import *


# ==============================
# Configuration Parameters
# ==============================

BACKEND = "https://openeo.dataspace.copernicus.eu/"
os.makedirs("../results", exist_ok=True)

# Temporal and spatial extent
TEMPORAL_EXTENT = ("2021-02-01", "2025-06-30")

SPATIAL_EXTENT = dict(
    west=631800.,
    south=5167700.,
    east=655800.,
    north=5184200.,
    crs=32632
)

# Processing parameters
CLOUD_PROB = 80  # Maximum cloud probability (%)
RESOLUTION = 20.  # Output resolution (m)
PIXEL_RATIO = 25  # Ratio between LR and HR pixel sizes
INVALID_CODES = [205, 210, 254, 255]  # No-data value codes
INVALID_VALUE = 205  # No-data fill value
INVALID_THRESHOLD = 10  # Max % invalid pixels allowed in LR pixel

# SCF range parameters
DELTA = 10  # Step size for SCF ranges (%)
EPSILON = 10  # Security buffer for SCF ranges (%)

# ==============================
# Main Processing Functions
# ==============================


def compute_scf_masks(
    connection: openeo.Connection,
    temporal_extent: Tuple[str, str],
    spatial_extent: dict
) -> Tuple[openeo.DataCube, list]:
    """
    Compute snow cover fraction masks at multiple resolution levels.
    
    Args:
        connection: openEO Connection object
        temporal_extent: Temporal extent as (start_date, end_date)
        spatial_extent: Spatial extent dictionary with bounds and CRS
        
    Returns:
        Tuple of (merged SCF masks, list of SCF range labels)
    """
    snow = calculate_snow(
        connection,
        temporal_extent,
        spatial_extent,
        CLOUD_PROB
    )

    total_mask = create_mask(snow)

    scf_lr_masked = low_resolution_snow_cover_fraction_mask(
        connection,
        total_mask,
        temporal_extent,
        spatial_extent
    )

    scf_dic = get_scf_ranges(DELTA, EPSILON)

    def scf_to_bands(scf_lr_masked):
        """Convert SCF map to binary bands for each SCF range."""
        result = []

        for key in scf_dic:
            scf_1 = int(key.split("_")[0])
            scf_2 = int(key.split("_")[1])

            print(f"Computing CP for {scf_1} < SCF ≤ {scf_2}")

            if scf_1 == 0:
                mask_scf = (
                    (scf_lr_masked >= scf_1)
                    .and_(scf_lr_masked <= scf_2)
                ) * 1.0
            else:
                mask_scf = (
                    (scf_lr_masked > scf_1)
                    .and_(scf_lr_masked <= scf_2)
                ) * 1.0

            result.append(mask_scf)

        return array_create(result)

    labels_scf = [f"scf_{v[0]}_{v[1]}" for v in scf_dic.values()]

    all_scf_masks = scf_lr_masked.apply_dimension(
        scf_to_bands,
        dimension="bands"
    )

    all_scf_masks = all_scf_masks.rename_labels(
        dimension="bands",
        target=labels_scf
    )

    mask_scf_hr = (
        all_scf_masks
        .resample_spatial(resolution=RESOLUTION, projection=32632, method="near")
        .resample_cube_spatial(snow)
    )

    return mask_scf_hr.merge_cubes(total_mask), labels_scf


def low_resolution_snow_cover_fraction_mask(
    connection,
    total_mask,
    temporal_extent,
    spatial_extent
):
    """
    Calculate low-resolution snow cover fraction (SCF) from MODIS data.
    
    Args:
        connection: openEO Connection object
        total_mask: Valid and snow pixel masks
        temporal_extent: Temporal extent
        spatial_extent: Spatial extent
        
    Returns:
        Low-resolution SCF data cube
    """
    modis = connection.load_stac(
        "https://stac.eurac.edu/collections/MOD10A1v61",
        temporal_extent=temporal_extent,
        spatial_extent=spatial_extent
    )

    average = total_mask.resample_cube_spatial(modis, method="average")

    def create_scf_lr_masked(average_bands: ProcessBuilder):
        """Compute SCF from valid and snow pixel fractions."""
        snow_band = average_bands["snow"]
        valid_band = average_bands["valid"]

        scf_lr = 100.0 * snow_band / valid_band
        scf_lr = if_(is_nan(scf_lr), INVALID_VALUE, scf_lr)

        valid_threshold = 1 - INVALID_THRESHOLD / 100

        scf_lr_masked = if_(
            valid_band <= valid_threshold,
            INVALID_VALUE,
            scf_lr
        )

        return scf_lr_masked

    scf_lr_masked = average.apply_dimension(
        dimension="bands",
        process=create_scf_lr_masked
    )

    scf_lr_masked = scf_lr_masked.rename_labels(
        dimension="bands",
        target=["scf"]
    )

    return scf_lr_masked

# ==============================
# Execution
# ==============================


if __name__ == "__main__":

    eoconn = openeo.connect(BACKEND, auto_validate=False)
    eoconn.authenticate_oidc()

    all_masks, labels_scf = compute_scf_masks(
        eoconn,
        TEMPORAL_EXTENT,
        SPATIAL_EXTENT
    )

    def merge_masks(all_masks):
        """Merge masks with snow pixels."""
        return all_masks.and_(
            all_masks.array_element(label="snow")
        ) * 1.0

    mask_cp_snow = all_masks.apply(process=merge_masks)

    mask_cp_snow = mask_cp_snow.filter_bands(
        bands=labels_scf
    )

    sum_cp_snow = mask_cp_snow.reduce_dimension(
        reducer="sum",
        dimension="t"
    )

    occurences = all_masks.reduce_dimension(
        reducer="sum",
        dimension="t"
    )

    occurences = occurences.filter_bands(
        bands=labels_scf
    )

    cp = sum_cp_snow / occurences

    cp.execute_batch(
        "../results/scf_lr_masked_new.nc",
        title="scf_lr_masked_new",
        job_options={
            "executor-memory": "8G",
            "executor-memoryOverhead": "4G",
            "python-memory": "disable"
        }
    )