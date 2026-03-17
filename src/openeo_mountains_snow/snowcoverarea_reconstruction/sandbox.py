
import openeo
from config import BACKEND

connection = openeo.connect(BACKEND, auto_validate=False).authenticate_oidc()

job = connection.job('j-2603060857054ee194786f92560e32df')
result = job.get_result()

result.download_files('./sca_input')




#%%

import xarray as xr
import matplotlib.pyplot as plt
import math

# Load dataset
path = r"C:\Users\VROMPAYH\Downloads\openEO (11).nc"
inputs = xr.open_dataset(path)

inputs = inputs.drop_vars(["crs"])
array = inputs.to_array(dim="bands").astype("float")
array = array.transpose('t', 'bands', 'y', 'x')
from udfs.swe_udf import apply_datacube
result = apply_datacube(array, None)

for i in range(result.sizes['t']):
    plt.figure()
    result.isel(t=i, bands=0).plot()
    plt.title(f"time index {i}")
    plt.show()


#%%
n_vars = len(variables)

cols = 3
rows = math.ceil(n_vars / cols)

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
axes = axes.flatten()

for i, var in enumerate(variables):
    
    # First variable -> timestep 5
    if i == 0:
        data = ds[var].isel(t=5)
        timestep_used = 5
    else:
        data = ds[var].isel(t=0)
        timestep_used = 0

    # Compute min/max ignoring NaNs
    vmin = float(data.min(skipna=True))
    vmax = float(data.max(skipna=True))

    im = axes[i].imshow(data, vmin=vmin, vmax=vmax)
    axes[i].set_title(f"{var} (t={timestep_used})")
    axes[i].axis("off")

    plt.colorbar(im, ax=axes[i])

# Hide unused axes
for j in range(n_vars, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()
# %%
















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


#%%
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

inputs = xr.load_dataset("sca_input.nc")
inputs = inputs.drop_vars(["crs"])
snow = inputs['snow']

# Assuming your DataArray is called 'snow'
# Select the last 10 timesteps
snow_last10 = snow.isel(t=slice(-10, None))

# Determine min and max for consistent color scale, ignoring NaNs
vmin = np.nanmin(snow_last10.values)
vmax = np.nanmax(snow_last10.values)

# Create figure with GridSpec: 2 rows x 5 columns + extra space for colorbar
fig = plt.figure(figsize=(22, 8))
gs = GridSpec(2, 6, width_ratios=[1, 1, 1, 1, 1, 0.05], figure=fig)  # last column for colorbar

axes = []
n_timesteps = snow_last10.sizes['t']

for i in range(n_timesteps):
    row = i // 5
    col = i % 5
    ax = fig.add_subplot(gs[row, col])
    
    im = ax.imshow(snow_last10.isel(t=i), cmap='viridis', vmin=vmin, vmax=vmax, origin='upper')
    
    # Set date as title
    ax.set_title(str(snow_last10['t'].values[i])[:10], fontsize=10)
    ax.axis('off')
    axes.append(ax)

# Add a single colorbar in the last column
cbar_ax = fig.add_subplot(gs[:, 5])  # span all rows
fig.colorbar(im, cax=cbar_ax, orientation='vertical', label='Snow [units]')

plt.tight_layout()
plt.show()

#%%


import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

path = r"C:\\Users\\VROMPAYH\\Downloads\\openEO (11).nc"
ds = xr.open_dataset(path)


data = ds['sca']

# Select last 10 timesteps
n_plot = 10
data_to_plot = data.isel(t=slice(-n_plot, None))

# Determine min and max for consistent color scale
vmin = data_to_plot.min().values
vmax = data_to_plot.max().values

# Create figure with GridSpec: 2 rows x 5 columns + space for colorbar
fig = plt.figure(figsize=(22, 8))
gs = GridSpec(2, 6, width_ratios=[1,1,1,1,1,0.05], figure=fig)  # last column for colorbar

axes = []
for i in range(n_plot):
    row = i // 5
    col = i % 5
    ax = fig.add_subplot(gs[row, col])
    im = ax.imshow(data_to_plot.isel(t=i), cmap='viridis', vmin=vmin, vmax=vmax, origin='upper')
    ax.set_title(str(data_to_plot['t'].values[i])[:10], fontsize=10)
    ax.axis('off')
    axes.append(ax)

# Single colorbar spanning all rows
cbar_ax = fig.add_subplot(gs[:, 5])
fig.colorbar(im, cax=cbar_ax, orientation='vertical', label='SCA')

plt.tight_layout()
plt.show()
