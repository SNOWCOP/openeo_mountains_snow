
import openeo
from config import BACKEND

connection = openeo.connect(BACKEND, auto_validate=False).authenticate_oidc()

job = connection.job('j-2603060857054ee194786f92560e32df')
result = job.get_result()

result.download_files('./sca_input')

#%%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

SCF_RANGES = [
    (0, 20), (0, 30), (10, 40), (20, 50), (30, 60),
    (40, 70), (50, 80), (60, 90), (70, 100), (80, 100)
]

START_X = 0
START_Y = 0
CHUNK_SIZE = 64
SNOW = 100
CLOUD = 205
NO_DATA = 255


n_days_actual = 10
n_ranges = len(SCF_RANGES)

path = "C:\\Git_projects\\openeo_mountains_snow\\src\\openeo_mountains_snow\\snowcoverarea_reconstruction\\sca_reconstruction.nc"
ds = xr.open_dataset(path)
ds = ds.set_coords(['x', 'y'])

# Now slice lazily - no data is loaded yet
ds_sliced = ds.isel(
    x=slice(START_X, START_X + CHUNK_SIZE),
    y=slice(START_Y, START_Y + CHUNK_SIZE)
)

# Access the data variable
data_var = ds_sliced['__xarray_dataarray_variable__']

# If you need to reorganize dimensions
cube = data_var.rename({'variable': 'bands'})

total_days = cube.shape[0]


hist_end = total_days - n_days_actual

historical_cp_maps = cube.isel(time=0, bands=slice(2, 2 + n_ranges)).values.astype(np.uint8)
historical_occ_maps = cube.isel(time=0, bands=slice(2 + n_ranges, 2 + 2 * n_ranges)).values.astype(np.uint8)
historical_snow = cube.isel(bands=0).values.astype(np.uint8)

np.nan_to_num(historical_cp_maps, nan=NO_DATA, copy=False)
np.nan_to_num(historical_occ_maps, nan=NO_DATA, copy=False)
np.nan_to_num(historical_snow, nan=NO_DATA, copy=False)


coords_t = cube.coords["time"].values[hist_end:hist_end + n_days_actual]
coords_y = cube.coords["y"].values
coords_x = cube.coords["x"].values


reconstructed_days  = []
day_idx = 7

snow_map  = cube.isel(time=hist_end + day_idx, bands=0).values.astype(np.uint8)
scf_map  = cube.isel(time=hist_end + day_idx, bands=1).values.astype(np.uint8)
cloud_mask = (snow_map == CLOUD)

plt.figure(figsize=(6, 6))
plt.imshow(snow_map, cmap='Blues')  # adjust vmin/vmax as needed
plt.title(f"snow_map pre_hr")
plt.colorbar(label='Snow Cover (%)')
plt.show()

reconstructed_hr = hr_reconstruction_single(
        snow_map,
        historical_snow
    )

update_mask_hr = cloud_mask & (reconstructed_hr != NO_DATA)
snow_map[update_mask_hr] = reconstructed_hr[update_mask_hr]

plt.figure(figsize=(6, 6))
plt.imshow(snow_map, cmap='Blues')  # adjust vmin/vmax as needed
plt.title(f"snow_map post_hr")
plt.colorbar(label='Snow Cover (%)')
plt.show()

reconstructed_scf = scf_reconstruction_single(
    snow_map,
    scf_map,
    historical_cp_maps,
    historical_occ_maps,
    SCF_RANGES
)

plt.figure(figsize=(6, 6))
plt.imshow(reconstructed_scf, cmap='Blues')  # adjust vmin/vmax as needed
plt.title(f"reconstructed_scf")
plt.colorbar(label='Snow Cover (%)')
plt.show()



# Update snow map with SCF reconstruction
update_mask_scf = cloud_mask & (reconstructed_scf != NO_DATA) 
snow_map[update_mask_scf] = reconstructed_scf[update_mask_scf]



plt.figure(figsize=(6, 6))
plt.imshow(reconstructed_scf, cmap='Blues')  # adjust vmin/vmax as needed
plt.title(f"reconstructed_scf")
plt.colorbar(label='Snow Cover (%)')
plt.show()






"""
    
    # Update snow map
    update_mask_hr = cloud_mask & (reconstructed_hr != NO_DATA)
    snow_map[update_mask_hr] = reconstructed_hr[update_mask_hr]
    logger.info(f"HR update non NAN {np.sum((reconstructed_hr != NO_DATA))} pixels")

    del reconstructed_hr
    del update_mask_hr
    gc.collect()
            
    # Update cloud mask after HR reconstruction
    cloud_mask = (snow_map == CLOUD)
    
    # ----- Step 2: SCF-based reconstruction -----
    if not cloud_mask.any():
        logger.info("No clouds remaining - stopping iterations")
        break
        
    # Call your scf_reconstruction function
    reconstructed_scf = scf_reconstruction_single(
        snow_map,
        scf_map,
        hist_cp_maps,
        hist_occ_maps,
        scf_ranges
    )
    
    # Update snow map with SCF reconstruction
    update_mask_scf = cloud_mask & (reconstructed_scf != NO_DATA) 
    snow_map[update_mask_scf] = reconstructed_scf[update_mask_scf]
    
    del reconstructed_scf
    del update_mask_scf
    gc.collect()
    
logger.info(f" Completed after {iteration} iterations")
return snow_map

#check gap filling; this is also done it in a loop.
# this is also an itteration in a loop based on this daily date thing

"""

#%%
scf_map

plt.figure(figsize=(6, 6))
plt.imshow(snow_map, cmap='Blues')  # adjust vmin/vmax as needed
plt.title(f"scf_map")
plt.colorbar(label='Snow Cover (%)')
plt.show()


#%%

# Define which timesteps to plot: every 30th timestep
timesteps_to_plot = np.arange(0, total_days, 30)

# Example: Plotting historical snow
for t in timesteps_to_plot:
    plt.figure(figsize=(6, 6))
    plt.imshow(historical_snow[t], cmap='Blues')  # adjust vmin/vmax as needed
    plt.title(f"Snow Cover Day {t}")
    plt.colorbar(label='Snow Cover (%)')
    plt.show()



#%%

import xarray as xr
import matplotlib.pyplot as plt
import math

# Load dataset
path = r"C:\Users\VROMPAYH\Downloads\openEO (11).nc"
ds = xr.open_dataset(path)

# Keep only spatial variables (exclude CRS)
variables = [
    v for v in ds.data_vars
    if ds[v].dims == ("t", "y", "x")
]

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
