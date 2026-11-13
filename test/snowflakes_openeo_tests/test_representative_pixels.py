from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from openeo_mountains_snow.snow_cover_fraction.udfs import representative_pixels as rp

SPECTRAL = ["B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B08", "B11", "B12"]


def training_cube():
    rng = np.random.default_rng(42)
    shape = (2, 48, 64)
    snow = np.indices(shape)[2] % 2 == 0
    green = np.where(snow, rng.uniform(0.65, 0.95, shape), rng.uniform(0.1, 0.4, shape))
    swir = np.where(snow, rng.uniform(0.03, 0.1, shape), rng.uniform(0.25, 0.45, shape))
    angles = np.broadcast_to(np.repeat([10., 30., 55., 90.], 16), shape)
    shadow = np.zeros(shape)
    shadow[:, 12:36, 25:39] = 1
    green = np.where((shadow == 1) & ~snow, rng.uniform(0.01, 0.05, shape), green)
    bands = {
        "B03": green,
        "B11": swir,
        "diff_B_NIR": np.where(snow, 0.2, 0.01),
        "NDSI": np.where(snow, rng.uniform(0.75, 0.95, shape), rng.uniform(-0.15, 0.15, shape)),
        "NDWI": np.zeros(shape),
        "shadow_mask": shadow,
        "local_solar_incidence_angle": angles,
    }
    return xr.DataArray(
        np.stack(list(bands.values())), dims=("bands", "t", "y", "x"),
        coords={"bands": list(bands)},
    )


def test_classification_trains_on_snow_and_snowfree_labels():
    cube = training_cube()
    for band in ["B02", "B04", "B05", "B06", "B07", "B8A", "B08", "B12"]:
        cube = xr.concat([cube, cube.sel(bands=["B03"]).assign_coords(bands=[band])], dim="bands")
    cube.loc[{"bands": "B02", "t": 0, "y": 0, "x": 0}] = np.nan
    context = {"total_samples": 240, "training_seed": 7}
    with patch.object(rp, "inspect"):
        labels = rp.apply_datacube(cube, context).sel(bands="B03").values
    spectral = cube.sel(bands=SPECTRAL).values
    predicted = np.full(labels.shape, 50.0)
    with patch.object(rp, "inspect"), patch.object(
        rp, "model_training_svm", return_value=(None, "normalizer", "svm", None)
    ) as train, patch.object(rp, "classify", return_value=predicted) as classify:
        result = rp.apply_datacube(cube, {**context, "classify": True})
    train.assert_called_once()
    np.testing.assert_array_equal(train.call_args.args[0], spectral[:, labels == 1].T)
    np.testing.assert_array_equal(train.call_args.args[1], spectral[:, labels == 2].T)
    classify.assert_called_once()
    np.testing.assert_array_equal(classify.call_args.args[1], np.all(np.isfinite(spectral), axis=0))
    np.testing.assert_array_equal(result.sel(bands="B03").values, predicted)


def test_classification_uses_available_bands_only():
    """Landsat-like input without red-edge bands still trains on the bands present."""
    cube = training_cube()
    for band in ["B02", "B04", "B08"]:
        cube = xr.concat([cube, cube.sel(bands=["B03"]).assign_coords(bands=[band])], dim="bands")
    with patch.object(rp, "inspect"), patch.object(
        rp, "model_training_svm", return_value=(None, "normalizer", "svm", None)
    ) as train, patch.object(rp, "classify", return_value=np.zeros((2, 48, 64))):
        rp.apply_datacube(cube, {"total_samples": 240, "training_seed": 7, "classify": True})
    assert train.call_args.args[0].shape[1] == 5


def test_sun_rules_match_original_threshold_conditions():
    cube = training_cube()
    eligible = cube.sel(bands="shadow_mask").values == 0
    eligible[:, :5] = False
    medians = [0.75, 0.25, 0.07, 0.35]
    with patch.object(rp, "define_threshold", return_value=np.array([0.2, 0.7])), patch.object(
        rp, "fit_distribution_and_median",
        side_effect=[{"fitted_median": median} for median in medians],
    ):
        snow, snowfree = rp.get_pixels_sun(
            *(cube.sel(bands=b).values for b in ("NDSI", "NDWI", "B03", "B11")), eligible
        )
    green = cube.sel(bands="B03").values
    swir = cube.sel(bands="B11").values
    ndsi = cube.sel(bands="NDSI").values
    ndwi = cube.sel(bands="NDWI").values
    np.testing.assert_array_equal(
        snow, eligible & (ndwi < 0.1) & (ndsi > 0.7) & (green > 0.75) & (swir < 0.07)
    )
    np.testing.assert_array_equal(
        snowfree, eligible & (ndsi < 0.2) & (green < 0.25) & (swir > 0.35)
    )


def test_sampling_covers_sun_angles_and_preserves_shadow_labels():
    cube = training_cube()
    before = cube.copy(deep=True)
    context = {"total_samples": 240, "training_seed": 7}
    with patch.object(rp, "inspect"):
        result = rp.apply_datacube(cube, context)
        repeated = rp.apply_datacube(cube, context)
    xr.testing.assert_identical(cube, before)
    xr.testing.assert_identical(result, repeated)
    labels = result.sel(bands="B03").values
    shadow = cube.sel(bands="shadow_mask").values == 1
    angles = cube.sel(bands="local_solar_incidence_angle").values
    assert set(np.unique(labels)) == {0, 1, 2}
    assert np.any(labels[shadow] == 1)
    assert np.any(labels[shadow] == 2)
    for angle in [10, 30, 55, 90]:
        selected = labels[(angles == angle) & ~shadow]
        assert np.any(selected == 1), f"No sunlit snow samples at {angle}"
        assert np.any(selected == 2), f"No sunlit snow-free samples at {angle}"
    # Boundary erosion excludes the outer three rows/columns.
    assert not np.any(labels[:, :3])
    assert not np.any(labels[:, :, :3])


def test_empty_and_missing_scenes():
    cube = training_cube()
    empty = np.zeros(cube.sel(bands="B03").shape, dtype=bool)
    with patch.object(rp, "define_threshold") as threshold:
        snow, snowfree = rp.get_pixels_sun(
            *(cube.sel(bands=b).values for b in ("NDSI", "NDWI", "B03", "B11")), empty
        )
    threshold.assert_not_called()
    assert not np.any(snow | snowfree)
    cube[:] = np.nan
    with patch.object(rp, "inspect"), np.errstate(invalid="raise"):
        result = rp.apply_datacube(cube, {})
    assert not np.any(result.values)


def test_failed_distribution_fits_use_default_median():
    values = np.zeros((5, 5))
    result = rp.fit_distribution_and_median(values, np.ones_like(values, dtype=bool), default_median=0.6)
    assert not result["fit_success"]
    assert result["fitted_median"] == 0.6


def test_distribution_fit_recovers_median():
    values = np.random.default_rng(0).normal(0.6, 0.05, (40, 40))
    result = rp.fit_distribution_and_median(values, np.ones_like(values, dtype=bool))
    assert result["fit_success"]
    assert result["fitted_median"] == pytest.approx(0.6, abs=0.01)
