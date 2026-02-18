# /// script
# dependencies = [
#   "pvlib"
# ]
# ///
import xarray
import pandas as pd
import datetime

import numpy as np
from openeo.udf import inspect


def apply_datacube(cube: xarray.DataArray, context) -> xarray.DataArray:
    assert len(cube.shape) == 3
    assert "t" in cube.attrs, "UDF requires 't' attribute with timestamp, as provided by Geotrellis backend."
    assert isinstance(cube.attrs["t"], datetime.datetime ), f"time has to be provided as datetime, but got: {cube.attrs['t']}"
    inspect(data=cube.dims, message = f"dims {cube.dims}")
    new_shape = list(cube.shape)
    new_shape[cube.dims.index("bands")] = 2
    new_shape = tuple(new_shape)
    inspect(data=new_shape, message=f"shape {new_shape}")


    longitude = cube.coords["x"].mean().item()
    latitude = cube.coords["y"].mean().item()
    inspect(data=longitude, message=f"longitude {longitude}")

    chunk_timestamp: datetime.datetime = cube.attrs["t"]

    inspect(data=chunk_timestamp, message=f"date {chunk_timestamp}")

    print(f"SOME LOGGING FROM PRINT!!! {cube}")
    #import pvlib
    #location = pvlib.location.Location(latitude, longitude)
    #inspect(data=location, message=f"location {location}")
    midnight = datetime.datetime.fromordinal(chunk_timestamp.date().toordinal()).replace(tzinfo=datetime.timezone.utc)

    solar_noon_utc = midnight + pd.Timedelta(hours=12)# - location.longitude / 15)

    assert solar_noon_utc.tzinfo is not None, "The datetime object has no timezone information."
    assert solar_noon_utc.tzinfo == datetime.timezone.utc, "The datetime object is not in UTC timezone."


    # Get solar position at estimated solar noon
    solpos = dict(zenith=45, azimuth=60)#location.get_solarposition(solar_noon_utc)

    # Extract solar angles
    zenith = 0.4#np.radians(solpos['zenith'].values[0])  # radians
    azimuth = 0.5 #np.radians(solpos['azimuth'].values[0])  # radians

    full_array = np.full(new_shape, zenith)
    full_array[1,:,:] = azimuth

    # New data to add
    new_data = xarray.DataArray(
        full_array,  # Shape: (1, x, y)
        dims=["bands", "x", "y"],
        coords={"bands": ["zenith", "azimuth"]}
    )

    # Concatenate along the band dimension
    return xarray.concat([cube, new_data], dim="bands")

