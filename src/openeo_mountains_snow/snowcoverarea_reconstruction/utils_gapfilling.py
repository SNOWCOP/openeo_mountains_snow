#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gap-filling and snow cover fraction utilities.

Created on Fri Jun  7 10:08:55 2024
@author: vpremier
"""

import openeo
import openeo.processes
from openeo.processes import if_, is_nan, array_create

from typing import List, Optional

import logging
_log = logging.getLogger(__name__)


def get_scf_ranges(delta, epsilon):
    """
    Generate SCF (Snow Cover Fraction) ranges for conditional probability computation.
    
    Creates ranges with a buffer (epsilon) around base delta-sized intervals for
    computing conditional probabilities.
    
    Args:
        delta: Step size for SCF ranges (e.g., 10 for 10% steps)
        epsilon: Security buffer around each range
        
    Returns:
        Dictionary mapping range keys (e.g., "0_20") to tuples of (scf1, scf2)
    """
    SCF_1 = list(range(0, 100, delta))
    SCF_2 = list(range(delta, 100 + delta, delta))

    scf_range_dic = {}

    for scf1, scf2 in zip(SCF_1, SCF_2):
        scf_l = max(0, scf1 - epsilon)
        scf_u = min(100, scf2 + epsilon)
        key = f"{scf_l}_{scf_u}"
        scf_range_dic[key] = (scf1, scf2)

    return scf_range_dic



def calculate_cloud_mask(scl: openeo.DataCube) -> openeo.DataCube:
    """
    Calculate cloud mask from SCL (Scene Classification Layer) data.
    
    Identifies clouds, cloud shadows, and cirrus clouds (SCL classes 7, 8, 9).
    
    Args:
        scl: SCL data cube
        
    Returns:
        Binary cloud mask data cube with dimension "bands"
    """
    _log.info('Calculating cloud mask')

    classification = scl.band("SCL")
    binary = (classification == 7) | (classification == 8) | (classification == 9)
    binary = binary.add_dimension(name="bands", label="clouds", type="bands")
    return binary



def calculate_snow(connection, temporal_extent, spatial_extent, cloud_prob=80.0) -> openeo.DataCube:
    """
    Calculate snow cover from Sentinel-2 L2A scene classification.
    
    Args:
        connection: openEO Connection object
        temporal_extent: Temporal extent as [start_date, end_date]
        spatial_extent: Spatial extent dictionary with crs and bounds
        cloud_prob: Maximum cloud probability threshold (default: 80%)
        
    Returns:
        DataCube with snow classification (0: no snow, 100: snow, 205: clouds)
    """
    scl = connection.load_collection(
        "SENTINEL2_L2A",
        temporal_extent=temporal_extent,
        spatial_extent=spatial_extent,
        bands=['SCL'],
        max_cloud_cover=cloud_prob,
    )

    scl = scl.resample_spatial(resolution=20, projection=32632, method="near")

    def snow_callback(scl_data: openeo.processes.ProcessBuilder):
        classification = scl_data["SCL"]
        snow_pixel = (classification == 11) * 100.0
        return if_((classification == 7).or_(classification == 8).or_(classification == 9), 205.0, snow_pixel)

    snow = scl.apply_dimension(dimension="bands", process=snow_callback)
    snow = snow.rename_labels(dimension="bands", target=["snow"])

    return snow

def _calculate_snow(scl: openeo.DataCube) -> openeo.DataCube:
    """
    Calculate snow mask from SCL data (legacy implementation).
    
    Args:
        scl: SCL data cube
        
    Returns:
        Snow classification data cube
    """
    _log.info('Calculating snow mask')

    classification = scl.band("SCL")
    clouds = ((classification == 7) | (classification == 8) | (classification == 9)) * 1.0
    snow = (classification == 11) * 100
    snow = snow.mask(clouds, replacement=205)
    snow = snow.add_dimension(name="bands", label="snow", type="bands")
    return snow


def calculate_ndsi(s2: openeo.DataCube) -> openeo.DataCube:
    """
    Calculate Normalized Difference Snow Index (NDSI).
    
    NDSI = (B03 - B11) / (B03 + B11)
    
    Args:
        s2: Sentinel-2 data cube with B03 (Red) and B11 (SWIR) bands
        
    Returns:
        NDSI data cube
    """
    _log.info('Calculating NDSI')

    B03 = s2.band("B03")
    B11 = s2.band("B11")

    ndsi = (B03 - B11) / (B03 + B11)
    ndsi = ndsi.add_dimension(name="bands", label="ndsi", type="bands")
    return ndsi


def salomonson(ndsi: openeo.DataCube) -> openeo.DataCube:
    """
    Apply Salomonson and Appel (2006) method to compute fractional snow cover.
    
    SCF = (-0.01 + 1.45 * NDSI) * 100
    
    Args:
        ndsi: NDSI data cube
        
    Returns:
        Snow Cover Fraction (SCF) data cube clipped to [0, 100]
    """
    _log.info('Applying Salomonson snow cover fraction retrieval')
    
    NDSI = ndsi.band("ndsi")
    scf = (-0.01 + 1.45 * NDSI) * 100

    scf = scf.mask(scf > 100, replacement=100)
    scf = scf.mask(scf < 0, replacement=0)
    scf = scf.apply(lambda x: if_(is_nan(x), 205, x))

    scf = scf.add_dimension(name="bands", label="scf", type="bands")

    return scf


def binarize(scf: openeo.DataCube, snowT: Optional[int] = 20) -> openeo.DataCube:
    """
    Binarize SCF to binary classification (0: no snow, 100: snow, 205: clouds).
    
    Args:
        scf: Snow Cover Fraction data cube
        snowT: Snow threshold (default: 20%)
        
    Returns:
        Binary snow classification data cube
    """
    scf = scf.band("scf")
    mask_100 = (scf >= snowT) & (scf <= 100) & (scf != 205)
    scf = scf.mask(mask_100, replacement=100)

    mask_0 = (scf < snowT) & (scf <= 100) & (scf != 205)
    scf = scf.mask(mask_0, replacement=0)

    scf = scf.add_dimension(name="bands", label="scf", type="bands")

    return scf


def create_mask(snow: openeo.DataCube) -> openeo.DataCube:
    """
    Create valid and snow masks from classified snow data.
    
    Creates two masks:
    - valid: pixels with valid snow classification (not clouds)
    - snow: pixels classified as snow
    
    Args:
        snow: Snow data cube (0: no snow, 100: snow, 205: clouds)
        
    Returns:
        Data cube with two bands: [valid_mask, snow_mask]
    """
    def valid_snow(bands):
        snow_band = bands["snow"]
        mask_valid = (snow_band <= 100) * 1.0
        mask_snow = (snow_band == 100) * 1.0
        return array_create([mask_valid, mask_snow])

    return (snow.apply_dimension(dimension="bands", process=valid_snow)
            .rename_labels(dimension="bands", target=["valid", "snow"]))

