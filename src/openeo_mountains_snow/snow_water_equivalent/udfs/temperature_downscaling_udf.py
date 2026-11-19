"""Downscale daily air temperature to the DEM with a monthly lapse rate (MicroMet, Liston & Elder 2006).

Input bands (per t): 0 = coarse air temperature (K), 1 = its reference height (m), 2 = DEM (m).
Context: ``southern_hemisphere`` (bool, default False) selects the lapse-rate table.
Output band ``temperature`` in °C.
"""
import numpy as np
import xarray as xr

# K/m, January..December
LAPSE_RATE_NORTH = np.array([4.4, 5.9, 7.1, 7.8, 8.1, 8.2, 8.1, 8.1, 7.7, 6.8, 5.5, 4.7]) / 1000.0
LAPSE_RATE_SOUTH = np.array([8.1, 8.1, 7.7, 6.8, 5.5, 4.7, 4.4, 5.9, 7.1, 7.8, 8.1, 8.2]) / 1000.0


def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:
    table = LAPSE_RATE_SOUTH if (context or {}).get("southern_hemisphere") else LAPSE_RATE_NORTH
    lapse_rate = xr.DataArray(table[cube.t.dt.month.values - 1], dims="t", coords={"t": cube.t})
    temperature, height, dem = (cube.isel(bands=i, drop=True) for i in range(3))
    out = temperature - 273.15 + lapse_rate * (height - dem)
    return out.expand_dims(bands=["temperature"]).transpose(*cube.dims)
