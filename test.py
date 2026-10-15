#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 14:37:36 2026

@author: vpremier
"""

import xarray
inputs = xarray.load_dataset("/mnt/CEPH_PROJECTS/SNOWCOP/SWE_processing/Aguirre/OpenEO_input/Aguirre snow cover fraction.nc", engine="netcdf4")
inputs = inputs.drop_vars(["crs"])
array = inputs.to_array(dim="bands").astype("float")


