# /// script
# dependencies = [
#   "pvlib"
# ]
# ///
"""
Solar position calculation UDF.

Computes per-pixel solar zenith and azimuth angles (radians, azimuth clockwise
from north) for the date of the chunk. All pixels share the same instant, so the
angles vary spatially as they would in a single acquisition.

The instant is the S2 overpass: the time at which NREL SPA reproduces the S2
tile-mean sun angles at the tile centre. These come from ``context["s2_tiles"][date]``
(``[lon, lat, zenith_deg, azimuth_deg]`` per tile, from STAC; nearest tile used),
else from the ``sunZenithAngles``/``sunAzimuthAngles`` bands at the chunk centre.
Without either: local solar noon of the chunk centre.
"""

import xarray
import numpy as np
import pandas as pd
from pvlib import spa
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

    date = pd.Timestamp(cube.attrs["t"]).normalize()

    # Reference point (tile or chunk centre) and its sun angles in degrees.
    ref_lon, ref_lat, sun = lon.mean(), lat.mean(), None
    tiles = (data.user_context or {}).get("s2_tiles", {}).get(f"{date:%Y-%m-%d}")
    bands = list(cube.coords["bands"].values)
    if tiles:
        ref_lon, ref_lat, *sun = min(tiles, key=lambda t: (t[0] - ref_lon) ** 2 + (t[1] - ref_lat) ** 2)
    elif "sunZenithAngles" in bands and "sunAzimuthAngles" in bands:
        sun = [float(cube.sel(bands="sunZenithAngles").mean()), float(cube.sel(bands="sunAzimuthAngles").mean())]

    # Overpass instant: the time at which the sun at the reference point matches the
    # S2 angles (smallest angular separation on the sky), or is highest (solar noon)
    # without them. Searched over the local day (approx. noon +/- 12 h, 240 s per degree
    # of longitude). Minute grid, then seconds.
    def best(seconds):
        zen, azi = _spa(seconds, ref_lat, ref_lon, date)
        if sun is None:
            miss = zen
        else:
            z, a, (z0, a0) = np.radians(zen), np.radians(azi), np.radians(sun)
            miss = -(np.cos(z) * np.cos(z0) + np.sin(z) * np.sin(z0) * np.cos(a - a0))
        return seconds[np.argmin(miss)]

    local_noon = date.timestamp() + 43200 - ref_lon * 240
    t = best(local_noon + np.arange(-43200.0, 43200, 60))
    t = best(t + np.arange(-60.0, 61))
    zenith, azimuth = _spa(np.array([t]), lat, lon, date)

    new_data = xarray.DataArray(
        np.radians(np.stack([zenith, azimuth])),
        dims=("bands", "y", "x"),
        coords={"bands": ["zenith", "azimuth"], "y": cube.coords["y"], "x": cube.coords["x"]},
    )
    data.set_datacube_list([XarrayDataCube(xarray.concat([cube, new_data], dim="bands"))])


def _spa(unixtime, lat, lon, date):
    """Geometric (unrefracted) solar zenith and azimuth in degrees, NREL SPA."""
    delta_t = spa.calculate_deltat(date.year, date.month)
    _, zenith, _, _, azimuth, _ = spa.solar_position(unixtime, lat, lon, 0, 1013.25, 12, delta_t, 0.5667)
    return zenith, azimuth
