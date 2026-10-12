#%%
"""Interactive Landsat runs, moved out of snow_cover_fraction/landsat.py so importing it has no side effects."""

import os

import matplotlib.pyplot as plt
import openeo
import xarray as xr

from openeo_mountains_snow.snow_cover_fraction.landsat import (
    landsat_inputs_cube,
    landsat_representative_pixels_cube,
)

url = "https://openeo.prod.eu-west-1.openeo-int.v1.dataspace.copernicus.eu/openeo"
conn = openeo.connect(url).authenticate_oidc()

#%% Inputs only

aoi = {
        "west": 10.851177,
        "south": 46.699372,
        "east": 10.916695,
        "north": 46.744288,
        "crs": "EPSG:4326",
    }
time_period = ["2025-04-01", "2025-09-01"]

inputs = landsat_inputs_cube(
        aoi=aoi,
        time_period=time_period,
        connection=conn,
        max_cloud_cover=70,
    )

#job = inputs.save_result(format="NetCDF").create_job(title="Landsat representative pixels")
#job.start_and_wait()

#%% Representative pixels

aoi = {
    "west": 10.818380,
    "south": 46.676915,
    "east": 10.949492,
    "north": 46.766745,
    "crs": "EPSG:4326",
}
time_period = ["2015-09-01", "2025-09-01"]

result_cube = landsat_representative_pixels_cube(
    aoi=aoi,
    time_period=time_period,
    connection=conn,
    neighborhood_size=100,
    max_cloud_cover=70,
)

job_options ={
  "driver-memory": "2G",
  "driver-memoryOverhead": "2G",
  "executor-memory": "7G",
  "executor-memoryOverhead": "3G",
}

job = result_cube.save_result(format="NetCDF").create_job(title="Landsat representative pixels", job_options=job_options)
job.start_and_wait()

#%% Download and inspect

results = job.get_results()
download_dir = os.path.join("src", "openeo_mountains_snow", "tmp", "landsat_snowflakes_AWS_output2")
os.makedirs(download_dir, exist_ok=True)
results.download_files(download_dir)

ds = xr.open_dataset(os.path.join(download_dir, "openEO.nc"))
print("Downloaded NetCDF bands:", list(ds.data_vars))

#%% Plot

da = ds['representative'].isel(t=0)

plt.figure(figsize=(8,6))
da.plot(cmap='viridis')
plt.title("Representative pixels")
plt.show()
