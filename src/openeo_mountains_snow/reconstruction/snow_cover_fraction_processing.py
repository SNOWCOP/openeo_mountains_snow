"""
Snow cover fraction (SCF) processing utilities.

Contains functions for computing SCF masks, conditional probabilities,
and loading MODIS data.
"""

from typing import Tuple, List, Dict
import openeo
from openeo.processes import eq, is_nan, gt, or_, if_, array_create, ProcessBuilder
from openeo import DataCube

from openeo_mountains_snow.reconstruction.utils_gapfilling import calculate_snow, create_mask, get_scf_ranges
from openeo_mountains_snow.reconstruction.config import (
    TEMPORAL_EXTENT, SPATIAL_EXTENT, CLOUD_PROB, DELTA, EPSILON,
    RESOLUTION, CRS, INVALID_THRESHOLD, INVALID_VALUE, MODIS_TEMPORAL_EXTENT,
    MODIS_RESOLUTION
)


def compute_snow_cover_fraction_masks(
    connection: openeo.Connection,
) -> Tuple[openeo.DataCube, list]:
    """
    Compute snow cover fraction masks and conditional probabilities.
    
    Args:
        connection: openEO Connection object
        
    Returns:
        Tuple of (merged SCF masks, list of SCF range labels)
    """
    snow_cover_mask = calculate_snow(
        connection,
        TEMPORAL_EXTENT,
        SPATIAL_EXTENT,
        CLOUD_PROB
    )

    # Create masks for valid and snow pixels
    valid_and_snow_mask = create_mask(snow_cover_mask)
    low_resolution_snow_cover_fraction_mask_cube = low_resolution_snow_cover_fraction_mask(
        connection,
        valid_and_snow_mask,
    )

    # ==============================
    # Conditional Probabilities
    # ==============================
    snow_cover_fraction_ranges = get_scf_ranges(DELTA, EPSILON)

    def snow_cover_fraction_to_bands(low_resolution_snow_cover_fraction):
        """Convert SCF map to binary bands for each SCF range."""
        result = []
        for key in snow_cover_fraction_ranges:
            lower_bound = int(key.split('_')[0])
            upper_bound = int(key.split('_')[1])

            # Define mask for lower_bound < scf <= upper_bound
            if lower_bound == 0:
                snow_cover_fraction_mask = (
                    low_resolution_snow_cover_fraction >= lower_bound
                ).and_(low_resolution_snow_cover_fraction <= upper_bound) * 1.0
            else:
                snow_cover_fraction_mask = (
                    low_resolution_snow_cover_fraction > lower_bound
                ).and_(low_resolution_snow_cover_fraction <= upper_bound) * 1.0

            result.append(snow_cover_fraction_mask)

        return array_create(result)

    # Generate labels for SCF masks
    snow_cover_fraction_band_labels = [
        f'scf_{v[0]}_{v[1]}' for v in snow_cover_fraction_ranges.values()
    ]

    # Apply dimension over bands
    snow_cover_fraction_band_masks = low_resolution_snow_cover_fraction_mask_cube.apply_dimension(
        snow_cover_fraction_to_bands,
        dimension='bands',
    )

    # Rename labels
    snow_cover_fraction_band_masks = snow_cover_fraction_band_masks.rename_labels(
        dimension="bands",
        target=snow_cover_fraction_band_labels,
    )

    # Upsample back to HR resolution
    high_resolution_snow_cover_fraction_band_masks = (
        snow_cover_fraction_band_masks
        .resample_spatial(resolution=RESOLUTION, projection=CRS, method="near")
        .resample_cube_spatial(snow_cover_mask)
    )

    return (
        high_resolution_snow_cover_fraction_band_masks.merge_cubes(valid_and_snow_mask),
        snow_cover_fraction_band_labels,
    )


def compute_scf_masks(connection: openeo.Connection) -> Tuple[openeo.DataCube, list]:
    """Backward-compatible alias for compute_snow_cover_fraction_masks."""
    return compute_snow_cover_fraction_masks(connection)


def low_resolution_snow_cover_fraction_mask(connection, total_mask):
    """
    Calculate low-resolution snow cover fraction (SCF) from MODIS data.
    
    Args:
        connection: openEO connection
        total_mask: Valid and snow pixel masks
        
    Returns:
        Low-resolution SCF data cube
    """
    modis = connection.load_stac(
        "https://stac.eurac.edu/collections/MOD10A1v61",
        temporal_extent=MODIS_TEMPORAL_EXTENT
    )
    
    # Resample mask to MODIS resolution and compute statistics
    average_mask_by_modis_pixel = total_mask.resample_cube_spatial(modis, method="average")

    def create_low_resolution_snow_cover_fraction_mask(average_bands: ProcessBuilder):
        """Compute SCF from valid and snow pixel fractions."""
        snow_band = average_bands["snow"]
        valid_band = average_bands["valid"]

        # Compute SCF as percentage
        low_resolution_snow_cover_fraction = 100.0 * snow_band / valid_band
        low_resolution_snow_cover_fraction = if_(
            is_nan(low_resolution_snow_cover_fraction),
            205,
            low_resolution_snow_cover_fraction,
        )

        # Replace pixels with too many invalid values
        valid_threshold = 1 - INVALID_THRESHOLD / 100
        masked_low_resolution_snow_cover_fraction = if_(
            valid_band <= valid_threshold,
            INVALID_VALUE,
            low_resolution_snow_cover_fraction,
        )

        return masked_low_resolution_snow_cover_fraction

    masked_low_resolution_snow_cover_fraction = average_mask_by_modis_pixel.apply_dimension(
        dimension="bands",
        process=create_low_resolution_snow_cover_fraction_mask,
    )
    masked_low_resolution_snow_cover_fraction = masked_low_resolution_snow_cover_fraction.rename_labels(
        dimension="bands",
        target=['scf'],
    )
    return masked_low_resolution_snow_cover_fraction


def create_modis_snow_cover_fraction_cube(
    connection: openeo.Connection,
    temporal_extent: List[str],
    spatial_extent: Dict,
) -> DataCube:
    """
    Load and prepare MODIS snow cover fraction (SCF) data.
    
    Args:
        connection: openEO connection
        temporal_extent: [start_date, end_date]
        spatial_extent: Dictionary with bbox and crs
        
    Returns:
        DataCube with single 'scf' band containing:
        - 0-100: Valid snow cover fraction (%)
        - 205: Cloud/invalid pixels
    """
    # Load MODIS from the STAC endpoint
    modis_cube = connection.load_stac(
        url="https://stac.eurac.edu/collections/MOD10A1v61",
        temporal_extent=temporal_extent,
        spatial_extent=spatial_extent,
        bands=["SCF"]
    )
    
    def clean_modis_snow_cover_fraction(pixel: ProcessBuilder) -> ProcessBuilder:
        """
        Clean MODIS SCF values by handling invalid data codes.
        
        MODIS SCF values:
        - 0-100: Valid snow cover fraction (%)
        - 205: Cloud (preserved)
        - 254: Water
        - 255: No data
        """
        scf_value = pixel.array_element(index=0)
        
        # Identify invalid values
        is_water = eq(scf_value, 254)
        is_no_data = eq(scf_value, 255)
        is_other_invalid = gt(scf_value, 100)
        
        should_replace = or_(is_water, or_(is_no_data, is_other_invalid))
        
        # Replace invalid with cloud code, keep valid values
        return if_(should_replace, 205, scf_value)
    
    # Apply cleaning
    cleaned_cube = modis_cube.apply(clean_modis_snow_cover_fraction)
    
    # Resample to consistent resolution
    cleaned_cube = cleaned_cube.resample_spatial(
        resolution=MODIS_RESOLUTION,
        projection=CRS,
        method="near"
    )
    
    # Rename output band
    return cleaned_cube.rename_labels(dimension="bands", target=["scf"])


def create_modis_scf_cube(
    connection: openeo.Connection,
    temporal_extent: List[str],
    spatial_extent: Dict,
) -> DataCube:
    """Backward-compatible alias for create_modis_snow_cover_fraction_cube."""
    return create_modis_snow_cover_fraction_cube(connection, temporal_extent, spatial_extent)
