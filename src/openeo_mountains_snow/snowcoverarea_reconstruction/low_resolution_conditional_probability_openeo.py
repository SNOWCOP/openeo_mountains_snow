#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 20:16:37 2025

@author: vpremier
"""
from typing import Tuple

import openeo

import os
from dotenv import load_dotenv
from shapely import box


load_dotenv(dotenv_path="../.env")

import xarray as xr

import geopandas as gpd
from openeo.processes import (and_, is_nan, is_valid, not_, is_nodata, or_, 
                              if_, array_create, ProcessBuilder)

from snowcoverarea_openeo.utils_gapfilling import *
from snowcoverarea_openeo.utils import *


# ==============================
# User Configuration Section
# ==============================


# openEO backend
backend = 'https://openeo.dataspace.copernicus.eu/'

# out directory
os.makedirs("../results", exist_ok=True)

# period to be downloaded
startdate = '2021-02-01'
enddate = '2025-06-30'

# cloud probability 
cloud_prob = 80

# extent
west=631800.
south=5167700.
east=655800.
north=5184200.

# resolution
res = 20.

# Ratio betweeen the size of a LR and a HR pixel, e.g., 500 m and 20 m.
pixel_ratio = 25 
# non-valid values
codes = [205, 210, 254, 255] 
nv_value = 205
# Threshold of non valid HR pixels allowed within a LR pixel [%]
nv_thres = 10 

# delta and epsilon: are used to define the SCF ranges. 
# The delta defines the steps, while epsilon represents a security buffer
delta = 10
epsilon = 10


def compute_scf_masks(connection: openeo.Connection) -> Tuple[openeo.DataCube, list]:

    snow = calculate_snow(connection,[startdate, enddate],dict( west=west,
                                 south=south,
                                 east=east,
                                 north=north,
                                 crs=32632), cloud_prob)

    # resample to 20 m spatial resolution
    # snow_rsmpl = snow.resample_spatial(resolution=res,
    #                                    projection=32632,
    #                                    method="near")
    # snow_rsmpl.download('../results/snow_rsmpl.nc')


    # mask with valid and snow pixels
    total_mask = create_mask(snow)

    scf_lr_masked = low_resolution_snow_cover_fraction_mask(connection, total_mask)

    # ==============================
    # Conditional probabilities
    # ==============================
    scf_dic = get_scf_ranges(delta, epsilon)

    # we need to add the dimension bands before applying the function
    # scf_lr_masked = scf_lr_masked.add_dimension(type="bands",name="bands",label='scf')

    def scf_to_bands(scf_lr_masked):
        result = []
        for i, key in enumerate(scf_dic):
            # range with a buffer - to be considered for the CP computation
            scf_1 = int(key.split('_')[0])
            scf_2 = int(key.split('_')[1])
            print(f'Computing CP by considering {scf_1}<SCF<={scf_2}')

            # define the mask scf_1 < scf <= scf_2
            if scf_1 == 0:
                mask_scf = (scf_lr_masked >= scf_1).and_(scf_lr_masked <= scf_2) * 1.0
            else:
                mask_scf = (scf_lr_masked > scf_1).and_(scf_lr_masked <= scf_2) * 1.0

            result.append(mask_scf)

        return array_create(result)


    # new labels for SCF masks
    labels_scf = [f'scf_{v[0]}_{v[1]}' for v in scf_dic.values()]


    # apply dimension should be applied over bands
    all_scf_masks = scf_lr_masked.apply_dimension(scf_to_bands, dimension='bands')

    # rename labels
    all_scf_masks = all_scf_masks.rename_labels(dimension = "bands", target =labels_scf)

    # upsample back to HR
    mask_scf_hr = all_scf_masks.resample_spatial(resolution=20,
                          projection=32632,
                          method="near").resample_cube_spatial(snow)

    return mask_scf_hr.merge_cubes(total_mask), labels_scf


def low_resolution_snow_cover_fraction_mask(connection, total_mask):
    modis = connection.load_stac("https://stac.eurac.edu/collections/MOD10A1v61",
                                 temporal_extent=["2023-01-20", "2023-01-21"])
    # modis.download('../results/modis.nc')
    # get SCF
    average = total_mask.resample_cube_spatial(modis, method="average")



    def create_scf_lr_masked(average_bands: ProcessBuilder):
        # SCF [0-1]
        snow_band = average_bands["snow"]
        valid_band = average_bands["valid"]

        scf_lr = 100.0 * snow_band / valid_band
        scf_lr = if_(is_nan(scf_lr), 205, scf_lr)

        # Compute the minimum SCF (SCF that you would obtain if the non valid pixels are replaced with 0-snow free)
        # SCF min and max are not used here (but will be used in another step of the workflow..)
        # scf_min = snow_band
        # scf_max = 1 - valid_band + snow_band
        # Replace pixels with non valid data < threshold with 205
        valid_threshold = 1 - nv_thres / 100
        scf_lr_masked = if_(valid_band <= valid_threshold, nv_value, scf_lr)

        return scf_lr_masked

    scf_lr_masked = average.apply_dimension(dimension="bands", process=create_scf_lr_masked)
    scf_lr_masked = scf_lr_masked.rename_labels(dimension="bands", target=['scf'])
    return scf_lr_masked


if __name__ == "__main__":
    # ==============================
    # Authentication
    # ==============================

    eoconn = openeo.connect(backend, auto_validate=False)
    eoconn.authenticate_oidc()

    all_masks, labels_scf = compute_scf_masks(eoconn)

    def merge_masks(all_masks):
        # multiply x snow
        return all_masks.and_(all_masks.array_element(label="snow")) * 1.0

    # all_masks.download("../results/all_masks.nc")

    mask_cp_snow = all_masks.apply(process=merge_masks)
    mask_cp_snow = mask_cp_snow.filter_bands(bands = labels_scf)

    # sum of all the snow pixels over time
    sum_cp_snow = mask_cp_snow.reduce_dimension(reducer="sum",
                                                dimension="t")


    # mask of all the scf occurences over time
    occurences = all_masks.reduce_dimension(reducer="sum",
                                                dimension="t")
    occurences = occurences.filter_bands(bands = labels_scf)

    # conditional probabilities
    cp = sum_cp_snow/occurences

    #job = cp.create_job(title="cp")
    #job.start()


    job = cp.execute_batch("../results/scf_lr_masked_new.nc",title="scf_lr_masked_new", job_options={"executor-memory":"8G", "executor-memoryOverhead": "4G", "python-memory": "disable"})

    ds = xr.open_dataset(r'../results/cp_test3.nc')
    ds.scf_60_70.plot()

    ds = xr.open_dataset(r'../results/cp_test1.nc')
    ds.scf_60_70.plot()

    ds = xr.open_dataset(r'../results/scf_lr_masked_old.nc')
    ds.scf_60_70.plot()

    ds.snow.sel(t='2025-04-02').plot()



    """
    # To dos
    - fare il downscaling di modis una volta che sono pronte le CP
    - provare a caricare MOD10A1
    - landsat?
    
    """