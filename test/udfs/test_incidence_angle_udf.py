import numpy as np
import pytest

pytest.importorskip("pvlib")

from openeo_mountains_snow.config import INCIDENCE_ANGLE_UDF


def test_flat_terrain_keeps_radiation(make_cube, run_udf):
    shape = (1, 3, 4)
    cube = make_cube({
        "slope": np.zeros(shape),
        "aspect": np.full(shape, 90.0),
        "zenith": np.full(shape, 0.5),
        "azimuth": np.full(shape, 3.0),
        "solar-radiation-flux": np.full(shape, 500.0),
    })
    result = run_udf(INCIDENCE_ANGLE_UDF, cube)
    assert list(result.bands.values) == ["shortwave-radiation-flux-downscaled"]
    assert result.sizes == {"bands": 1, "t": 1, "y": 3, "x": 4}
    np.testing.assert_allclose(result.values, 500.0, rtol=1e-5)
