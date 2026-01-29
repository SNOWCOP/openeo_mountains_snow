#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 10:08:55 2024

@author: vpremier
"""

import openeo
import openeo.processes
from openeo.processes import if_, is_nan, array_create

from typing import List, Optional

import logging
_log = logging.getLogger(__name__)

# def cloud_mask_compute(data):
    
#     band = data.band("SCL")
#     # include also 10 cirrus, 2 and 3 cloud shadows
#     combined_mask = (band==7) | (band==8) | (band==9)
    
#     return combined_mask*1.0


def get_scf_ranges(delta, epsilon):
    # get a dictionary with the SCF ranges for which you want to compute
    # the conditional probability
    SCF_1 = list(range(0, 100, delta))
    SCF_2 = list(range(delta, 100 + delta, delta))

    # INFO about SCF variability --- load information
    scf_range_dic = {}

    for scf1, scf2 in zip(SCF_1, SCF_2):
        scf_l = scf1 - epsilon
        scf_u = scf2 + epsilon
        if scf_l < 0:
            scf_l = 0
        if scf_u > 100:
            scf_u = 100
        key = str(scf_l) + '_' + str(scf_u)

        scf_range_dic[key] = (scf1, scf2)

    return scf_range_dic



def calculate_cloud_mask(scl: openeo.DataCube) -> openeo.DataCube:
    """
    Calculate cloud mask from SCL data.
    Args:
        scl (openeo.datacube.DataCube): SCL data cube.
    Returns:
        openeo.datacube.DataCube: Cloud mask data cube.
    """
    _log.info(f'calculating cloud mask')

    classification = scl.band("SCL")
    binary = (classification == 7) | (classification == 8) | (classification == 9) 
    binary = binary.add_dimension(name="bands", label="clouds", type="bands")
    return binary



def calculate_snow(connection,temporal_extent,spatial_extent, cloud_prob = 80.0) -> openeo.DataCube:
    # ==============================
    # Load collections
    # ==============================

    scl = connection.load_collection(
        "SENTINEL2_L2A",
        temporal_extent=temporal_extent,
        spatial_extent=spatial_extent,
        bands=['SCL'],
        max_cloud_cover=cloud_prob,
    )

    scl = scl.resample_spatial(resolution=20,
                          projection=32632,
                          method="near")

    # ==============================
    # Get the snow cover information
    # ==============================

    def snow_callback(scl_data: openeo.processes.ProcessBuilder):
        classification: openeo.processes.ProcessBuilder = scl_data["SCL"]
        snow_pixel = (classification == 11) * 100.0
        return if_((classification == 7).or_(classification == 8).or_(classification == 9), 205.0, snow_pixel)

    snow = scl.apply_dimension(dimension="bands", process=snow_callback)
    snow = snow.rename_labels(dimension="bands", target=["snow"])


    return snow

def _calculate_snow(scl: openeo.DataCube) -> openeo.DataCube:
    """
    Calculate snow mask from SCL data.
    TODO: needs latest code

    Args:
        scl (openeo.datacube.DataCube): SCL data cube.
    Returns:
        openeo.datacube.DataCube: Cloud mask data cube.
    """
    _log.info(f'calculating snow mask')

    classification = scl.band("SCL")
    clouds = ((classification == 7) | (classification == 8) | (classification == 9) ) * 1.0
    snow = (classification == 11) * 100
    snow = snow.mask(clouds, replacement = 205)
    snow = snow.add_dimension(name="bands", label="snow", type="bands")
    return snow


def calculate_ndsi(s2: openeo.DataCube) -> openeo.DataCube:
    """
    Calculate NDSI.
    Args:
        s2 (openeo.datacube.DataCube): S2 data cube.
    Returns:
        openeo.datacube.DataCube: NDSI data cube.
    """
    _log.info(f'calculating ndsi')

    B03 = s2.band("B03")
    B11 = s2.band("B11")

    ndsi = (B03 - B11)/(B03+B11)
    ndsi = ndsi.add_dimension(name="bands", label="ndsi", type="bands")
    return ndsi


def salomonson(ndsi: openeo.DataCube) -> openeo.DataCube:
    """
    Apply the approach of Salomonson and Appel (2006) to get fractional snow cover
    Args:
        ndsi (openeo.datacube.DataCube): ndsi data cube.
    Returns:
        openeo.datacube.DataCube: SCF data cube.
    """
    
    NDSI = ndsi.band("ndsi")
    scf = (- 0.01  + 1.45 * NDSI) * 100

    scf = scf.mask(scf>100,replacement=100)  
    scf = scf.mask(scf<0,replacement=0)  

    scf = scf.apply(lambda x: if_(is_nan(x), 205, x))
    
    scf = scf.add_dimension(name="bands", label="scf", type="bands")

    return scf


def binarize(scf: openeo.DataCube,
             snowT: Optional[int] = 20 ) -> openeo.DataCube:
    """
    Binarize the scf to 0 and 100 (205 clouds)
    Args:
        scf (openeo.datacube.DataCube): ndsi data cube.
    Returns:
        openeo.datacube.DataCube: binary SCF
    """
    
    scf = scf.band("scf")
    mask_100 = (scf >= snowT) & (scf <= 100) & (scf!=205)
    scf = scf.mask(mask_100,replacement=100)  

    mask_0 = (scf < snowT) & (scf <= 100) & (scf!=205)
    scf = scf.mask(mask_0,replacement=0)  
    
    scf = scf.add_dimension(name="bands", label="scf", type="bands")

    return scf


def create_mask(snow: openeo.DataCube) -> openeo.DataCube:
    """
    Create two masks: mask with valid pixels and mask with 
    pixels classified as snow.
    Args:
        snow (openeo.datacube.DataCube): snow data cube (0:snow free, 100: snow, 205: clouds).
    Returns:
        openeo.datacube.DataCube: mask
    """

    def valid_snow(bands):
        snow = bands["snow"]
        mask_valid = (snow <= 100 )*1.0
        mask_snow = (snow == 100 )*1.0
        return array_create([mask_valid,mask_snow])

    return snow.apply_dimension(dimension="bands",process=valid_snow).rename_labels(dimension="bands", target=["valid","snow"])

