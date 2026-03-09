# /// script
# dependencies = [
#   "pvlib"
# ]
# ///
"""
Solar incidence angle calculation UDF.

Computes the angle of incidence of solar radiation on a sloped surface
using slope, aspect, and solar position angles.

#TODO: pvlib is used  for very minimal math,  consider inlining the code
"""

import xarray
import pandas as pd
import pvlib
import numpy as np
from openeo.udf import inspect

def apply_datacube(cube: xarray.DataArray, context) -> xarray.DataArray:
    """
    Compute solar incidence angle on sloped terrain.
    
    Args:
        cube: Input xarray DataArray with slope, aspect, zenith, azimuth bands
        context: Execution context
        
    Returns:
        Solar incidence angle (angle of incidence in degrees)
    """
    slope = cube. sel(bands="slope").values
    aspect_deg = cube.sel(bands="aspect").values
    zenith = cube.sel(bands="zenith").values
    azimuth = cube.sel(bands="azimuth").values
    agera_ssrd_resampled = cube.sel(bands="solar-radiation-flux").values

    # Extract solar angles (placeholder implementation)
    incidence = pvlib.irradiance.aoi(
        surface_tilt=slope,
        surface_azimuth=aspect_deg,
        solar_zenith=np.degrees(zenith),
        solar_azimuth=np.degrees(azimuth)
    )

    inspect(data=incidence.shape, message=f"Incidence shape: {incidence.shape}")
    inspect(data=incidence, message=f"Incidence: {incidence}")

    #downscale shortwave radiation using incidence angle
    cosZ = np.cos(zenith)
    cos_i = np.clip(np.cos(np.radians(incidence)), 0, 1)
    Qsi = agera_ssrd_resampled * (cos_i / (cosZ + 1e-6))

    slope_band = cube.sel(bands="slope")
    new_data = xarray.DataArray(
        Qsi,
        dims=slope_band.dims,
        coords={dim: slope_band.coords[dim] for dim in slope_band.dims if dim in slope_band.coords}
    ).expand_dims(bands=["shortwave-radiation-flux-downscaled"])

    return new_data

    # alternative to return also inputs
    #return xarray.concat([cube, new_data], dim="bands")
