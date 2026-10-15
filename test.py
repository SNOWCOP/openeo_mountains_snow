#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 14:37:36 2026

@author: vpremier
"""
import numpy as np
import xarray
inputs = xarray.load_dataset("/mnt/CEPH_PROJECTS/SNOWCOP/SWE_processing/Aguirre/OpenEO_input/Aguirre snow cover fraction.nc", engine="netcdf4")
inputs = inputs.drop_vars(["crs"])
array = inputs.to_array(dim="bands").astype("float")





solar_incidence_angle = np.squeeze(inputs['local_solar_incidence_angle'].values)





# SIA between 70 and 180
curr_angle_valid = np.logical_and.reduce((solar_incidence_angle >= 70,
                                          solar_incidence_angle <= 90))

mask = np.squeeze((solar_incidence_angle > 90))


CLOSDI = np.squeeze((1 - 1.5*inputs["B08"].values -0.1*inputs["B04"].values)*100/(1+3.5*inputs["B08"].values+4.9*inputs["B04"].values))

threshold = np.nanpercentile(CLOSDI, 10)

# DIFFERENT SHADOWS: i) cloud shadows, ii) self-shadows, and iii) cast-shadows 
# cloud_shadow = cloud_mask == 3
self_shadow = (solar_incidence_angle > 90)

spectral_shadow = CLOSDI > threshold
# spectral_shadow = shadow_score > threshold
casted_shadow = np.logical_and(spectral_shadow, curr_angle_valid)

shadow_mask = np.logical_or.reduce((casted_shadow, self_shadow))


# Save shadow mask to GeoTIFF
shadow_mask_path = os.path.join(curr_aux_folder, f'{scene_id}_shadow_mask.tif')

save_tif(shadow_mask, path, shadow_mask_path, dtype=rasterio.uint8)





