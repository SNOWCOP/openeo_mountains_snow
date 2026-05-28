"""
Integration tests for meteorological downscaling.

These tests require a live openEO connection and are meant to be run
manually or in CI with credentials configured:

    pytest test/snowflakes_openeo_tests/ -m integration
"""

from pathlib import Path

import numpy as np
import openeo
import pytest
import xarray

from openeo_mountains_snow.swe.downscale_variables import downscale_shortwave_radiation

pytestmark = pytest.mark.integration

# ---- Shared test extent (small, fast) ----
SPATIAL_EXTENT = {
    "south": 5816500,
    "north": 5816500 + 128 * 30,
    "west": 271000,
    "east": 271000 + 128 * 30,
    "crs": "EPSG:32719",
}
TEMPORAL_EXTENT = "2025-07"
JOB_OPTIONS = {
    "executor-memory": "5G",
    "executor-memoryOverhead": "5G",
}


@pytest.fixture(scope="module")
def openeoplatform_connection() -> openeo.Connection:
    return openeo.connect("openeo-dev.vito.be").authenticate_oidc()


def test_shortwave_radiation(openeoplatform_connection, tmp_path):
    """End-to-end: downscale shortwave radiation and verify output."""
    agera = openeoplatform_connection.load_collection(
        "AGERA5",
        spatial_extent=SPATIAL_EXTENT,
        temporal_extent=TEMPORAL_EXTENT,
        bands=["solar-radiation-flux"],
    )
    agera = agera.rename_labels(dimension="bands", target=["solar-radiation-flux"])

    dem_spacetime = openeoplatform_connection.load_collection(
        "COPERNICUS_30", spatial_extent=SPATIAL_EXTENT
    )
    dem = dem_spacetime.reduce_dimension(dimension="t", reducer="mean")

    aspect = dem.aspect()
    slope = dem.slope()
    slope_aspect = aspect.merge_cubes(slope).rename_labels(
        dimension="bands", target=["aspect", "slope"]
    )

    agera = agera.resample_cube_spatial(dem_spacetime)
    shortwave_rad_cube = downscale_shortwave_radiation(agera, slope_aspect)

    out_file = tmp_path / "shortwave_rad_downscaled.nc"
    shortwave_rad_cube.execute_batch(
        str(out_file),
        title="shortwave radiation test",
        job_options=JOB_OPTIONS,
    )

    ds = xarray.open_dataset(out_file)
    assert len(ds.data_vars) > 0, "Output should contain at least one variable"
    for var in ds.data_vars:
        vals = ds[var].values
        assert (vals[~np.isnan(vals)] >= 0).all(), (
            f"Radiation variable {var} should be non-negative"
        )


