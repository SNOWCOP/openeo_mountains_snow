import numpy as np

from openeo_mountains_snow.config import SWE_RECONSTRUCTION_UDF


def swe_cube(make_cube, sca, ta=10.0, precip=0.0, sw=200.0):
    """Bands as merged by snow_water_equivalent_cube: SCA, temperature, precipitation, shortwave."""
    return make_cube({"sca": sca, "ta": np.full_like(sca, ta), "precip": np.full_like(sca, precip),
                      "sw": np.full_like(sca, sw)})


def test_snow_free_gives_zero_swe(make_cube, run_udf):
    cube = swe_cube(make_cube, np.zeros((6, 3, 4)))
    result = run_udf(SWE_RECONSTRUCTION_UDF, cube)
    assert result.dims == ("t", "bands", "y", "x")
    assert list(result.bands.values) == ["swe"]
    np.testing.assert_array_equal(result.t.values, cube.t.values)
    np.testing.assert_array_equal(result.values, 0)


def test_cloud_gives_nodata(make_cube, run_udf):
    sca = np.zeros((6, 3, 4))
    sca[3] = 205
    result = run_udf(SWE_RECONSTRUCTION_UDF, swe_cube(make_cube, sca))
    assert np.isnan(result.values[3]).all()
    assert (result.values[[0, 1, 2, 4, 5]] == 0).all()


def test_daily_meteo_is_used(make_cube, run_udf):
    # snow onset, two cold snowy days, three melt days, melt-out
    sca = np.repeat(np.array([0, 100, 100, 100, 100, 100, 0, 0], float)[:, None, None], 3, axis=1).repeat(4, axis=2)
    ta = np.repeat(np.array([5, -5, -5, 5, 5, 5, 5, 5], float)[:, None, None], 3, axis=1).repeat(4, axis=2)
    precip = np.where(ta < 0, 10.0, 0.0)  # above the 5 mm/day accumulation threshold
    hotter = ta.copy()
    hotter[4] = 15.0  # differs only after day 0
    mild = run_udf(SWE_RECONSTRUCTION_UDF, swe_cube(make_cube, sca, ta=ta, precip=precip))
    hot = run_udf(SWE_RECONSTRUCTION_UDF, swe_cube(make_cube, sca, ta=hotter, precip=precip))
    assert np.nanmax(mild.values) > 0
    assert not np.array_equal(mild.values, hot.values, equal_nan=True)


def test_basin_sca_context_replaces_neighborhood_sca(make_cube, run_udf):
    # warm, wet snowfall day: only accumulation if the SCA increases by more than 10 points
    sca = np.repeat(np.array([0, 100, 100, 100, 0, 0], float)[:, None, None], 3, axis=1).repeat(4, axis=2)
    precip = np.zeros_like(sca)
    precip[1] = 10.0
    cube = swe_cube(make_cube, sca, ta=5.0, precip=precip)
    local = run_udf(SWE_RECONSTRUCTION_UDF, cube)
    flat_basin = {str(t)[:10]: 20.0 for t in cube.t.values}
    basin = run_udf(SWE_RECONSTRUCTION_UDF, cube, context={"basin_sca": flat_basin})
    assert np.nanmax(local.values) > 0
    np.testing.assert_array_equal(basin.values, 0)
