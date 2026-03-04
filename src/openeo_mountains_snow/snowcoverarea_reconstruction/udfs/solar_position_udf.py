# /// script
# dependencies = [
#   "pvlib"
# ]
# ///
"""
Solar position calculation UDF.

Computes solar position (zenith and azimuth angles) for each pixel
using pvlib based on pixel coordinates and timestamp.
"""

import xarray
import pandas as pd
import datetime
import numpy as np
from openeo.udf import inspect


def apply_datacube(cube: xarray.DataArray, context) -> xarray.DataArray:
    """
    Compute solar position angles and append to data cube.
    
    Args:
        cube: Input xarray DataArray with shape (time, x, y, bands)
        context: Execution context containing metadata
        
    Returns:
        DataArray with two new bands appended: [zenith, azimuth] angles
    """
    assert len(cube.shape) == 3, "Expected 3D data array (time, x, y)"
    assert "t" in cube.attrs, "UDF requires 't' attribute with timestamp"
    assert isinstance(cube.attrs["t"], datetime.datetime), \
        f"Time must be datetime, got: {cube.attrs['t']}"
    
    inspect(data=cube.dims, message=f"Input dimensions: {cube.dims}")
    
    # Prepare output shape with 2 additional bands for zenith and azimuth
    new_shape = list(cube.shape)
    new_shape[cube.dims.index("bands")] = 2
    new_shape = tuple(new_shape)
    inspect(data=new_shape, message=f"Output shape: {new_shape}")

    longitude = cube.coords["x"].mean().item()
    latitude = cube.coords["y"].mean().item()
    inspect(data=longitude, message=f"Longitude: {longitude}") #check for correctness
    inspect(data=latitude, message=f"Latitude: {latitude}")

    chunk_timestamp: datetime.datetime = cube.attrs["t"]
    inspect(data=chunk_timestamp, message=f"Timestamp: {chunk_timestamp}")

    # Calculate midnight UTC and solar noon
    midnight = datetime.datetime.fromordinal(chunk_timestamp.date().toordinal()).replace(
        tzinfo=datetime.timezone.utc
    )
    solar_noon_utc = midnight + pd.Timedelta(hours=12)

    assert solar_noon_utc.tzinfo is not None, "Datetime has no timezone information"
    assert solar_noon_utc.tzinfo == datetime.timezone.utc, "Datetime is not in UTC timezone"

    # TODO: Use pvlib to compute actual solar position
    # For now using placeholder values
    zenith = 0.4  # radians
    azimuth = 0.5  # radians

    # Create output array
    full_array = np.full(new_shape, zenith)
    full_array[1, :, :] = azimuth

    new_data = xarray.DataArray(
        full_array,
        dims=["bands", "x", "y"],
        coords={"bands": ["zenith", "azimuth"]}
    )

    # Concatenate with input data
    return xarray.concat([cube, new_data], dim="bands")
