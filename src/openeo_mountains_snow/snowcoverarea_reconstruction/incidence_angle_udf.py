# /// script
# dependencies = [
#   "pvlib"
# ]
import xarray
import pandas as pd
import pvlib
import numpy as np

def apply_datacube(cube: xarray.DataArray, context) -> xarray.DataArray:
    incidence = pvlib.irradiance.aoi(
        surface_tilt=np.degrees(slope_rad),
        surface_azimuth=np.degrees(aspect_rad),
        solar_zenith=solpos['zenith'].values[0],
        solar_azimuth=solpos['azimuth'].values[0]
    )
    return incidence
