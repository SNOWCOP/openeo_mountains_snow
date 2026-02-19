# /// script
# dependencies = [
#   "pvlib"
# ]
# ///
"""
Solar incidence angle calculation UDF.

Computes the angle of incidence of solar radiation on a sloped surface
using slope, aspect, and solar position angles.
"""

import xarray
import pandas as pd
import pvlib
import numpy as np


def apply_datacube(cube: xarray.DataArray, context) -> xarray.DataArray:
    """
    Compute solar incidence angle on sloped terrain.
    
    Args:
        cube: Input xarray DataArray with slope, aspect, zenith, azimuth bands
        context: Execution context
        
    Returns:
        Solar incidence angle (angle of incidence in degrees)
    """
    # TODO: Extract slope, aspect, zenith, azimuth from cube
    # Extract solar angles (placeholder implementation)
    incidence = pvlib.irradiance.aoi(
        surface_tilt=np.degrees(slope_rad),
        surface_azimuth=np.degrees(aspect_rad),
        solar_zenith=solpos['zenith'].values[0],
        solar_azimuth=solpos['azimuth'].values[0]
    )
    return incidence
