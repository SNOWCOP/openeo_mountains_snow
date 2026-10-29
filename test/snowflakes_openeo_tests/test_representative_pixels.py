import numpy as np
import pytest
import xarray

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from openeo_mountains_snow.snow_cover_fraction.udfs.representative_pixels import apply_datacube

BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B08", "B11", "B12", "cloud_shadow", "local_solar_incidence_angle", "NDSI", "NDWI", "diff_B_NIR"]


def training_cube(size=60):
    rng = np.random.default_rng(0)
    shape = (size, size)
    snow = np.indices(shape)[1] % 2 == 0
    shadow = np.zeros(shape)
    shadow[20:40, 20:40] = 1
    in_shadow = shadow == 1
    green = np.where(snow, rng.uniform(0.65, 0.95, shape), rng.uniform(0.1, 0.4, shape))
    green = np.where(in_shadow, np.where(snow, rng.uniform(0.15, 0.25, shape), rng.uniform(0.01, 0.05, shape)), green)
    swir = np.where(snow, rng.uniform(0.03, 0.1, shape), rng.uniform(0.25, 0.45, shape))
    bands = {
        "B02": green,
        "B03": green,
        "B04": green,
        "B05": green,
        "B06": green,
        "B07": green,
        "B8A": np.full(shape, 0.3),
        "B08": np.full(shape, 0.3),
        "B11": swir,
        "B12": swir,
        "cloud_shadow": shadow,
        "local_solar_incidence_angle": np.repeat([10.0, 30.0, 55.0], size // 3)[:, None] * np.ones(shape),
        "NDSI": np.where(snow, rng.uniform(0.75, 0.95, shape), rng.uniform(-0.15, 0.15, shape)),
        "NDWI": np.zeros(shape),
        "diff_B_NIR": np.where(snow, 0.2, 0.01),
    }
    return xarray.DataArray(
        np.stack([bands[b] for b in BANDS]),
        dims=("bands", "y", "x"),
        coords={"bands": BANDS, "y": np.arange(size), "x": np.arange(size)},
    ), snow, in_shadow


def test_apply_datacube_selects_snow_and_snowfree_in_sun_and_shadow():
    cube, snow, in_shadow = training_cube()

    result = apply_datacube(cube, {"total_samples": 200, "classify": False})

    labels = result.sel(bands="B03").values
    assert list(result.bands.values) == ["B03"]
    assert set(np.unique(labels)) <= {0, 1, 2}
    for region in (in_shadow, ~in_shadow):
        assert np.any(labels[region] == 1) and np.any(labels[region] == 2)
    assert np.all(snow[labels == 1]) and not np.any(snow[labels == 2])
