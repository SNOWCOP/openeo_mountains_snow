import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr

import representative_pixels as rp
from fit_distribution import fit_distribution_and_median


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


class SunTrainingTests(unittest.TestCase):
    def test_classification_trains_on_snow_and_snowfree_labels(self):
        cube = training_cube()
        for band in ["B02", "B04", "B05", "B06", "B07", "B8A", "B08", "B12"]:
            cube = xr.concat([cube, cube.sel(bands=["B03"]).assign_coords(bands=[band])], dim="bands")
        cube.loc[{"bands": "B02", "t": 0, "y": 0, "x": 0}] = np.nan
        context = {"reflectance_scale": 1, "total_samples": 240, "training_seed": 7}
        with patch.object(rp, "inspect"):
            labels = rp.apply_datacube(cube, context).sel(bands="B03").values
        spectral = cube.sel(bands=["B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B08", "B11", "B12"]).values
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

    def test_sun_rules_match_original_threshold_conditions(self):
        cube = training_cube()
        eligible = cube.sel(bands="shadow_mask").values == 0
        eligible[:, :5] = False
        medians = [0.75, 0.25, 0.07, 0.35]
        with patch.object(rp, "define_threshold", return_value=np.array([0.2, 0.7])), patch.object(
            rp, "fit_distribution_and_median",
            side_effect=[{"fitted_median": median} for median in medians],
        ):
            snow, snowfree = rp.get_pixels_sun(cube, eligible)
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

    def test_sampling_covers_sun_angles_and_preserves_shadow_labels(self):
        cube = training_cube()
        before = cube.copy(deep=True)
        context = {"reflectance_scale": 1, "total_samples": 240, "training_seed": 7}
        with patch.object(rp, "inspect"):
            result = rp.apply_datacube(cube, context)
            repeated = rp.apply_datacube(cube, context)
        xr.testing.assert_identical(cube, before)
        xr.testing.assert_identical(result, repeated)
        labels = result.sel(bands="B03").values
        shadow = cube.sel(bands="shadow_mask").values == 1
        angles = cube.sel(bands="local_solar_incidence_angle").values
        self.assertEqual(set(np.unique(labels)), {0, 1, 2})
        self.assertTrue(np.any(labels[shadow] == 1))
        self.assertTrue(np.any(labels[shadow] == 2))
        for angle in [10, 30, 55, 90]:
            selected = labels[(angles == angle) & ~shadow]
            self.assertTrue(np.any(selected == 1), f"No sunlit snow samples at {angle}")
            self.assertTrue(np.any(selected == 2), f"No sunlit snow-free samples at {angle}")
        # Boundary erosion excludes the outer three rows/columns.
        self.assertFalse(np.any(labels[:, :3]))
        self.assertFalse(np.any(labels[:, :, :3]))

        scaled = cube.copy(deep=True)
        spectral = ["B03", "B11", "diff_B_NIR"]
        scaled.loc[{"bands": spectral}] *= 10000
        with patch.object(rp, "inspect"):
            scaled_result = rp.apply_datacube(
                scaled, {"total_samples": 240, "training_seed": 7}
            )
        xr.testing.assert_identical(result, scaled_result)

    def test_empty_and_missing_scenes(self):
        cube = training_cube()
        empty = np.zeros(cube.sel(bands="B03").shape, dtype=bool)
        with patch.object(rp, "define_threshold") as threshold:
            snow, snowfree = rp.get_pixels_sun(cube, empty)
        threshold.assert_not_called()
        self.assertFalse(np.any(snow | snowfree))
        cube[:] = np.nan
        with patch.object(rp, "inspect"), np.errstate(invalid="raise"):
            result = rp.apply_datacube(cube, {})
        self.assertFalse(np.any(result.values))

    def test_failed_distribution_fits_use_default_median(self):
        values = np.zeros((5, 5))
        result = fit_distribution_and_median(values, np.ones_like(values, dtype=bool), default_median=0.6)
        self.assertFalse(result["fit_success"])
        self.assertEqual(result["fitted_median"], 0.6)


if __name__ == "__main__":
    unittest.main()
