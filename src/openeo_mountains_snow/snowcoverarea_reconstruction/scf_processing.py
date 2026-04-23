"""
Snow cover fraction (SCF) processing utilities.

Contains functions for computing SCF masks, conditional probabilities,
and loading MODIS data.
"""

from typing import Tuple, List, Dict
import openeo
from omegaconf import DictConfig
from openeo.processes import eq, is_nan, gt, or_, if_, array_create, ProcessBuilder
from openeo import DataCube

from openeo_mountains_snow.snowcoverarea_reconstruction.utils_gapfilling import (
    create_mask, get_scf_ranges,
)
from openeo_mountains_snow.snow_cover_fraction import snow_cover_fraction_cube


def compute_scf_masks(
    connection: openeo.Connection,
    cfg: DictConfig,
    spatial_extent: dict,
    temporal_extent: list,
) -> Tuple[openeo.DataCube, list]:
    """
    Compute snow cover fraction masks and conditional probabilities.

    Args:
        connection: openEO Connection object
        cfg: Hydra configuration
        spatial_extent: Spatial extent dict
        temporal_extent: [start_date, end_date]

    Returns:
        Tuple of (merged SCF masks, list of SCF range labels)
    """
    proc = cfg.processing

    # Use the improved spectral-index-based snow cover fraction
    snow = snow_cover_fraction_cube(
        spatial_extent=spatial_extent,
        time_period=temporal_extent,
        c=connection,
        cfg=cfg,
    ).rename_labels(dimension="bands", target=["snow"])

    # Create masks for valid and snow pixels
    total_mask = create_mask(snow)
    scf_lr_masked = low_resolution_snow_cover_fraction_mask(
        connection, cfg, total_mask, temporal_extent, spatial_extent
    )

    # ==============================
    # Conditional Probabilities
    # ==============================
    scf_dic = get_scf_ranges(proc.delta, proc.epsilon)

    def scf_to_bands(scf_lr_masked):
        """Convert SCF map to binary bands for each SCF range."""
        result = []
        for key in scf_dic:
            scf_1 = int(key.split('_')[0])
            scf_2 = int(key.split('_')[1])

            # Define mask for scf_1 < scf <= scf_2
            if scf_1 == 0:
                mask_scf = (scf_lr_masked >= scf_1).and_(scf_lr_masked <= scf_2) * 1.0
            else:
                mask_scf = (scf_lr_masked > scf_1).and_(scf_lr_masked <= scf_2) * 1.0

            result.append(mask_scf)

        return array_create(result)

    # Generate labels for SCF masks
    labels_scf = [f'scf_{v[0]}_{v[1]}' for v in scf_dic.values()]

    # Apply dimension over bands
    all_scf_masks = scf_lr_masked.apply_dimension(scf_to_bands, dimension='bands')

    # Rename labels
    all_scf_masks = all_scf_masks.rename_labels(dimension="bands", target=labels_scf)

    # Upsample back to HR resolution
    mask_scf_hr = (all_scf_masks
                   .resample_spatial(resolution=proc.resolution, projection=proc.crs, method="near")
                   .resample_cube_spatial(snow))

    return mask_scf_hr.merge_cubes(total_mask), labels_scf


def low_resolution_snow_cover_fraction_mask(connection, cfg, total_mask, temporal_extent, spatial_extent=None):
    """
    Calculate low-resolution snow cover fraction (SCF) from MODIS data.
    
    Args:
        connection: openEO connection
        total_mask: Valid and snow pixel masks
        temporal_extent: [start_date, end_date]
        spatial_extent: bbox dict (optional, passed to load_stac)
        
    Returns:
        Low-resolution SCF data cube
    """
    proc = cfg.processing
    load_kwargs = {"temporal_extent": temporal_extent}
    if spatial_extent is not None:
        load_kwargs["spatial_extent"] = spatial_extent
    modis = connection.load_stac(
        cfg.modis.stac_url,
        **load_kwargs,
    )
    
    # Resample mask to MODIS resolution and compute statistics
    average = total_mask.resample_cube_spatial(modis, method="average")

    def create_scf_lr_masked(average_bands: ProcessBuilder):
        """Compute SCF from valid and snow pixel fractions."""
        snow_band = average_bands["snow"]
        valid_band = average_bands["valid"]

        # Compute SCF as percentage
        scf_lr = 100.0 * snow_band / valid_band
        scf_lr = if_(is_nan(scf_lr), 205, scf_lr)

        # Replace pixels with too many invalid values
        valid_threshold = 1 - proc.invalid_threshold / 100
        scf_lr_masked = if_(valid_band <= valid_threshold, proc.invalid_value, scf_lr)

        return scf_lr_masked

    scf_lr_masked = average.apply_dimension(dimension="bands", process=create_scf_lr_masked)
    scf_lr_masked = scf_lr_masked.rename_labels(dimension="bands", target=['scf'])
    return scf_lr_masked


def create_modis_scf_cube(connection: openeo.Connection,
                          cfg: DictConfig,
                          temporal_extent: List[str],
                          spatial_extent: Dict) -> DataCube:
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
    proc = cfg.processing
    modis_cube = connection.load_stac(
        url=cfg.modis.stac_url,
        temporal_extent=temporal_extent,
        spatial_extent=spatial_extent,
        bands=["SCF"]
    )
    
    def clean_modis_scf(pixel: ProcessBuilder) -> ProcessBuilder:
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
    cleaned_cube = modis_cube.apply(clean_modis_scf)
    
    # Resample to consistent resolution
    cleaned_cube = cleaned_cube.resample_spatial(
        resolution=proc.modis_resolution,
        projection=proc.crs,
        method="near"
    )
    
    # Rename output band
    return cleaned_cube.rename_labels(dimension="bands", target=["scf"])
