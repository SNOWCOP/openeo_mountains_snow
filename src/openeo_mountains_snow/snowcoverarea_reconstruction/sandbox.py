#%%

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import sys

# Load dataset
path = r"C:\\Users\\VROMPAYH\\Downloads\\swe_input.nc"
inputs = xr.open_dataset(path)

inputs = inputs.drop_vars(["crs"])
array = inputs.to_array(dim="bands").astype("float")
array = array.transpose('t', 'bands', 'y', 'x')

# Replace bands 1–3 with t=0 values
array_modified = array.copy()
array_modified[:, 1:, :, :] = array.isel(t=0, bands=slice(1, None))

# Filter based on band 0 NaNs
band0 = array_modified.isel(bands=0)
nan_fraction = band0.isnull().mean(dim=("y", "x"))
valid_timesteps = np.where(nan_fraction <= 0.2)[0]

array_filtered = array_modified.isel(t=valid_timesteps)



sys.path.append(r"C:\Git_projects\openeo_mountains_snow\src")
from openeo_mountains_snow.snowcoverarea_reconstruction.udfs.swe_udf import apply_datacube

result = apply_datacube(array_filtered, {})

# %%

import xarray


inputs = xarray.load_dataset("sca_input.nc")
inputs = inputs.drop_vars(["crs"])
array = inputs.to_array(dim="bands").astype("float")

import sys
sys.path.append(r"C:\Git_projects\openeo_mountains_snow\src")
from openeo_mountains_snow.snowcoverarea_reconstruction.udfs.historical_reconstruction_udf import apply_datacube

result = apply_datacube(array, {})

# %%

