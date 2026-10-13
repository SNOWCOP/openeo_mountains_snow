import numpy as np
import pytest
import xarray

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from openeo_mountains_snow.snow_cover_fraction.udfs.representative_pixels import apply_datacube

BANDS = ["B02", "B03", "B04", "B08", "B11", "local_solar_incidence_angle", "NDVI", "NDSI", "diff_B_NIR", "SI"]


def test_apply_datacube_returns_single_band_with_input_grid():
    rng = np.random.default_rng(0)
    size = 20
    data = rng.uniform(0, 1, size=(len(BANDS), size, size)).astype(float)
    data[BANDS.index("local_solar_incidence_angle")] = rng.uniform(0, 89, size=(size, size))
    cube = xarray.DataArray(
        data,
        dims=("bands", "y", "x"),
        coords={"bands": BANDS, "y": np.arange(size), "x": np.arange(size)},
    )

    result = apply_datacube(cube, {"total_samples": 50, "classify": False})

    assert list(result.bands.values) == ["B03"]
    assert result.shape == (1, size, size)
    assert np.isfinite(result.values).all()
