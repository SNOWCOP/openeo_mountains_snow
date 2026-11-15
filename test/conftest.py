from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from openeo.udf import UdfData, XarrayDataCube
from openeo.udf.run_code import run_udf_code


@pytest.fixture
def make_cube():
    """Mock openEO cube from {band: array}; (t, y, x) arrays give dims (t, bands, y, x), (y, x) give (bands, y, x)."""
    def make(bands, x0=650_000.0, y0=5_150_000.0, res=10.0, start="2023-01-01"):
        data = np.stack([np.asarray(v, dtype=np.float32) for v in bands.values()])
        ny, nx = data.shape[-2:]
        coords = {"bands": list(bands), "y": y0 - res * np.arange(ny), "x": x0 + res * np.arange(nx)}
        if data.ndim == 4:
            coords["t"] = pd.date_range(start, periods=data.shape[1], freq="D")
            return xr.DataArray(data, dims=("bands", "t", "y", "x"), coords=coords).transpose("t", "bands", "y", "x")
        return xr.DataArray(data, dims=("bands", "y", "x"), coords=coords)
    return make


@pytest.fixture
def run_udf():
    """Run a UDF file the way the backend does: source text through openeo's run_udf_code."""
    def run(path, cube, context=None, epsg=None):
        data = UdfData(proj={"EPSG": epsg} if epsg else None, datacube_list=[XarrayDataCube(cube)],
                       user_context=context or {})
        return run_udf_code(code=Path(path).read_text(), data=data).get_datacube_list()[0].get_array()
    return run
