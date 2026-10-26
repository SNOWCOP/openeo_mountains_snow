# /// script
# dependencies = [
#   "pvlib"
# ]
# ///
"""
Solar position calculation UDF.

Computes per-pixel solar zenith and azimuth angles (radians, azimuth clockwise
from north) at local solar noon of the chunk centre, for the date of the chunk.
All pixels share the same instant, so the angles vary spatially as they would
in a single acquisition.
"""

import xarray
import numpy as np
import pandas as pd
from pvlib import solarposition
from pyproj import Transformer
from openeo.udf import XarrayDataCube
from openeo.udf.udf_data import UdfData


def apply_udf_data(data: UdfData):
    """
    Append [zenith, azimuth] bands to the input cube (dims: bands, y, x).

    Uses ``UdfData.proj`` for the cube CRS, falling back to ``context["epsg"]``.
    """
    cube = data.get_datacube_list()[0].get_array()
    assert cube.dims == ("bands", "y", "x"), f"Expected dims (bands, y, x), got {cube.dims}"
    assert "t" in cube.attrs, "UDF requires 't' attribute with timestamp"

    epsg = (data.proj or {}).get("EPSG") or (data.user_context or {}).get("epsg")
    assert epsg, "Cube CRS unknown: no UdfData.proj and no 'epsg' in context"

    xx, yy = np.meshgrid(cube.coords["x"].values, cube.coords["y"].values)
    lon, lat = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform(xx, yy)

    # At solar noon of the chunk centre, the hour angle of each pixel is its
    # longitude offset from the centre (no equation-of-time term needed).
    hour_angle = np.radians(lon - lon.mean())
    declination = solarposition.declination_spencer71(pd.Timestamp(cube.attrs["t"]).dayofyear)
    lat_rad = np.radians(lat)

    zenith = solarposition.solar_zenith_analytical(lat_rad, hour_angle, declination)
    azimuth = solarposition.solar_azimuth_analytical(lat_rad, hour_angle, declination, zenith)

    new_data = xarray.DataArray(
        np.stack([zenith, azimuth]),
        dims=("bands", "y", "x"),
        coords={"bands": ["zenith", "azimuth"], "y": cube.coords["y"], "x": cube.coords["x"]},
    )
    data.set_datacube_list([XarrayDataCube(xarray.concat([cube, new_data], dim="bands"))])
