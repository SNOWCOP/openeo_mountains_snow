#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 14:37:36 2026

@author: vpremier
"""
import numpy as np
import xarray
inputs = xarray.load_dataset("/home/vpremier/Documents/git/OPENEO/openeo_mountains_snow/representative_pixels_senales_multirange_classified.nc", engine="netcdf4")
inputs = inputs.drop_vars(["crs"])
cube = inputs.to_array(dim="bands").astype("float")



