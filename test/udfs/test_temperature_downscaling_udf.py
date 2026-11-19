import numpy as np

from openeo_mountains_snow.config import TEMPERATURE_DOWNSCALING_UDF


def temperature_cube(make_cube, start, temperature_k=273.15, height=1000.0, dem=2000.0):
    shape = (2, 2, 3)
    return make_cube({"temperature-mean": np.full(shape, temperature_k), "geopotential": np.full(shape, height),
                      "DEM": np.full(shape, dem)}, start=start)


def test_january_north_uses_winter_lapse_rate(make_cube, run_udf):
    result = run_udf(TEMPERATURE_DOWNSCALING_UDF, temperature_cube(make_cube, "2023-01-01"))
    assert list(result.bands.values) == ["temperature"]
    assert result.dims == ("t", "bands", "y", "x")
    # 1000 m above the reference height at 4.4 K/km
    np.testing.assert_allclose(result.values, -4.4, atol=1e-4)


def test_hemisphere_and_month_select_lapse_rate(make_cube, run_udf):
    cube = temperature_cube(make_cube, "2023-07-01")
    north = run_udf(TEMPERATURE_DOWNSCALING_UDF, cube)
    south = run_udf(TEMPERATURE_DOWNSCALING_UDF, cube, context={"southern_hemisphere": True})
    np.testing.assert_allclose(north.values, -8.1, atol=1e-4)
    np.testing.assert_allclose(south.values, -4.4, atol=1e-4)
