
#%%
import sys
sys.path.insert(0, r'c:\Git_projects\openeo_mountains_snow\src')

import xarray

inputs = xarray.open_dataset(r"C:\Git_projects\openeo_mountains_snow\src\openeo_mountains_snow\tmp\landsat_input_cube\openEO.nc")
print("Dataset:", inputs)

array = inputs.to_array(dim="bands").astype("float")
print(f"\nArray shape: {array.shape}, bands: {array.bands.values}")

from openeo_mountains_snow.snow_cover_fraction.udfs.representative_pixels import apply_datacube
context = {"total_samples": 500, "classify": True}

result = apply_datacube(array, context)
print(f"✓ Success! Result shape: {result.shape}")

