"""
Unit tests for UDFs.

These tests run the UDF apply_datacube functions locally with synthetic data,
without requiring an openEO connection.
"""

import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr


# ---------------------------------------------------------------------------
# Solar position UDF
# ---------------------------------------------------------------------------

class TestSolarPositionUDF:
    """Tests for swe/udfs/solar_position_udf.py"""

    @staticmethod
    def _make_cube(nx=4, ny=4):
        """Create a minimal input cube for the solar position UDF."""
        data = np.random.rand(1, nx, ny).astype(np.float32)
        cube = xr.DataArray(
            data,
            dims=["bands", "x", "y"],
            coords={
                "bands": ["solar-radiation-flux"],
                "x": np.linspace(11.0, 11.1, nx),
                "y": np.linspace(46.5, 46.6, ny),
            },
        )
        cube.attrs["t"] = datetime.datetime(2023, 1, 15, 12, 0, tzinfo=datetime.timezone.utc)
        return cube

    def test_output_has_three_bands(self):
        from openeo_mountains_snow.swe.udfs.solar_position_udf import apply_datacube

        cube = self._make_cube()
        result = apply_datacube(cube, {})

        assert "bands" in result.dims
        assert list(result.coords["bands"].values) == [
            "solar-radiation-flux",
            "zenith",
            "azimuth",
        ]

    def test_output_shape(self):
        from openeo_mountains_snow.swe.udfs.solar_position_udf import apply_datacube

        cube = self._make_cube(nx=8, ny=6)
        result = apply_datacube(cube, {})

        assert result.shape == (3, 8, 6)

    def test_input_band_preserved(self):
        from openeo_mountains_snow.swe.udfs.solar_position_udf import apply_datacube

        cube = self._make_cube()
        result = apply_datacube(cube, {})

        np.testing.assert_array_equal(
            result.sel(bands="solar-radiation-flux").values,
            cube.sel(bands="solar-radiation-flux").values,
        )

    def test_requires_time_attribute(self):
        from openeo_mountains_snow.swe.udfs.solar_position_udf import apply_datacube

        cube = self._make_cube()
        del cube.attrs["t"]

        with pytest.raises(AssertionError, match="'t' attribute"):
            apply_datacube(cube, {})


# ---------------------------------------------------------------------------
# Incidence angle UDF
# ---------------------------------------------------------------------------

class TestIncidenceAngleUDF:
    """Tests for swe/udfs/incidence_angle_udf.py"""

    @staticmethod
    def _make_cube(nx=4, ny=4):
        """Create a 5-band input cube matching the expected merge order."""
        bands = ["solar-radiation-flux", "zenith", "azimuth", "aspect", "slope"]
        data = np.zeros((len(bands), nx, ny), dtype=np.float32)
        # solar flux (MJ/m²)
        data[0] = 15.0
        # zenith ~30° in radians
        data[1] = np.radians(30.0)
        # azimuth ~180° in radians
        data[2] = np.radians(180.0)
        # aspect (degrees) — south-facing
        data[3] = 180.0
        # slope (degrees) — moderate
        data[4] = 20.0

        return xr.DataArray(
            data,
            dims=["bands", "x", "y"],
            coords={
                "bands": bands,
                "x": np.linspace(11.0, 11.1, nx),
                "y": np.linspace(46.5, 46.6, ny),
            },
        )

    def test_output_single_band(self):
        pytest.importorskip("pvlib")
        from openeo_mountains_snow.swe.udfs.incidence_angle_udf import apply_datacube

        cube = self._make_cube()
        result = apply_datacube(cube, None)

        assert "bands" in result.dims
        assert list(result.coords["bands"].values) == [
            "shortwave-radiation-flux-downscaled"
        ]

    def test_output_shape_matches_spatial(self):
        pytest.importorskip("pvlib")
        from openeo_mountains_snow.swe.udfs.incidence_angle_udf import apply_datacube

        cube = self._make_cube(nx=8, ny=6)
        result = apply_datacube(cube, None)

        assert result.shape == (1, 8, 6)

    def test_output_non_negative(self):
        pytest.importorskip("pvlib")
        from openeo_mountains_snow.swe.udfs.incidence_angle_udf import apply_datacube

        cube = self._make_cube()
        result = apply_datacube(cube, None)

        assert np.all(result.values >= 0), "Downscaled radiation should be non-negative"

    def test_flat_terrain_preserves_flux(self):
        """On flat terrain (slope=0) the correction factor should be ~1."""
        pytest.importorskip("pvlib")
        from openeo_mountains_snow.swe.udfs.incidence_angle_udf import apply_datacube

        cube = self._make_cube()
        cube.loc["slope"] = 0.0  # flat

        result = apply_datacube(cube, None)

        np.testing.assert_allclose(
            result.sel(bands="shortwave-radiation-flux-downscaled").values,
            cube.sel(bands="solar-radiation-flux").values,
            rtol=0.05,
        )


# ---------------------------------------------------------------------------
# Historical reconstruction UDF
# ---------------------------------------------------------------------------

class TestHistoricalReconstructionUDF:
    """Tests for sca/udfs/historical_reconstruction_udf.py"""

    @staticmethod
    def _make_cube(n_time=15, ny=10, nx=10, n_ranges=10):
        """Build a synthetic input cube matching the UDF's expected layout.

        Bands per timestep:
          0: snow (0/100/205)
          1: scf (0-100/205)
          2..2+n_ranges-1: cp maps
          2+n_ranges..2+2*n_ranges-1: occurrence maps
        """
        n_bands = 2 + 2 * n_ranges
        data = np.zeros((n_time, n_bands, ny, nx), dtype=np.float32)

        # Snow band: mix of snow, no-snow, cloud
        rng = np.random.RandomState(42)
        snow_vals = rng.choice([0, 100, 205], size=(n_time, ny, nx), p=[0.3, 0.4, 0.3])
        data[:, 0, :, :] = snow_vals

        # SCF band
        data[:, 1, :, :] = rng.choice([0, 50, 100, 205], size=(n_time, ny, nx))

        # CP maps (static, same for all timesteps)
        for i in range(n_ranges):
            data[:, 2 + i, :, :] = rng.choice([0, 100], size=(ny, nx))
            data[:, 2 + n_ranges + i, :, :] = rng.randint(10, 50, size=(ny, nx))

        cube = xr.DataArray(
            data,
            dims=["t", "bands", "y", "x"],
            coords={
                "t": [f"2023-01-{d+1:02d}" for d in range(n_time)],
                "bands": [f"band_{i}" for i in range(n_bands)],
                "y": np.arange(ny),
                "x": np.arange(nx),
            },
        )
        return cube

    def test_output_dimensions(self):
        from openeo_mountains_snow.sca.udfs.historical_reconstruction_udf import apply_datacube

        cube = self._make_cube(n_time=15)
        result = apply_datacube(cube, {"n_days_to_reconstruct": 5})

        assert result.dims == ("t", "bands", "y", "x")
        assert result.shape[1] == 1  # single output band
        assert result.shape[0] == 5  # n_days_to_reconstruct

    def test_output_values_valid(self):
        from openeo_mountains_snow.sca.udfs.historical_reconstruction_udf import apply_datacube

        cube = self._make_cube(n_time=15)
        result = apply_datacube(cube, {"n_days_to_reconstruct": 5})

        values = result.values
        # All values should be valid snow codes
        valid = np.isin(values, [0, 100, 205, 255]) | np.isnan(values)
        assert np.all(valid), f"Unexpected values: {np.unique(values[~valid])}"

    def test_single_timestep_returns_empty(self):
        """With only 1 timestep there's nothing to reconstruct."""
        from openeo_mountains_snow.sca.udfs.historical_reconstruction_udf import apply_datacube

        cube = self._make_cube(n_time=1)
        result = apply_datacube(cube, {"n_days_to_reconstruct": 5})

        assert result.shape[0] == 0

    def test_cloud_pixels_reduced(self):
        """After reconstruction, there should be fewer cloud pixels than in the input."""
        from openeo_mountains_snow.sca.udfs.historical_reconstruction_udf import apply_datacube

        cube = self._make_cube(n_time=20)
        n_days = 5
        result = apply_datacube(cube, {"n_days_to_reconstruct": n_days})

        # Count clouds in the reconstructed days of the input vs output
        input_clouds = np.sum(cube.values[-n_days:, 0, :, :] == 205)
        output_clouds = np.sum(result.values[:, 0, :, :] == 205)

        assert output_clouds <= input_clouds


# ---------------------------------------------------------------------------
# SWE UDF
# ---------------------------------------------------------------------------

class TestSweUDF:
    """Tests for swe/udfs/swe_udf.py"""

    @staticmethod
    def _make_cube(n_time=30, ny=4, nx=4):
        """Build a synthetic 4-band input cube for the SWE UDF.

        Bands: sca, temperature, humidity, shortwave radiation.
        """
        data = np.zeros((n_time, 4, ny, nx), dtype=np.float32)

        # SCA: snow in first half, no snow in second half
        data[:n_time // 2, 0, :, :] = 100  # snow
        data[n_time // 2:, 0, :, :] = 0    # no snow

        # Temperature: cold then warm
        data[:n_time // 2, 1, :, :] = -5.0
        data[n_time // 2:, 1, :, :] = 5.0

        # Humidity (relative, %)
        data[:, 2, :, :] = 2.0  # mm/day precipitation proxy

        # Shortwave radiation (MJ/m²/day)
        data[:, 3, :, :] = 10.0

        dates = pd.date_range("2023-01-01", periods=n_time, freq="D")

        cube = xr.DataArray(
            data,
            dims=["t", "bands", "y", "x"],
            coords={
                "t": dates,
                "bands": ["sca", "temperature", "humidity", "shortwave"],
                "y": np.arange(ny),
                "x": np.arange(nx),
            },
        )
        return cube

    def test_output_dimensions(self):
        from openeo_mountains_snow.swe.udfs.swe_udf import apply_datacube

        cube = self._make_cube()
        result = apply_datacube(cube, {})

        assert result.dims == ("t", "bands", "y", "x")
        assert list(result.coords["bands"].values) == ["swe"]

    def test_output_time_matches_input(self):
        from openeo_mountains_snow.swe.udfs.swe_udf import apply_datacube

        cube = self._make_cube()
        result = apply_datacube(cube, {})

        assert result.shape[0] == cube.shape[0]

    def test_swe_non_negative(self):
        from openeo_mountains_snow.swe.udfs.swe_udf import apply_datacube

        cube = self._make_cube()
        result = apply_datacube(cube, {})

        swe_vals = result.values
        finite = swe_vals[np.isfinite(swe_vals)]
        assert np.all(finite >= 0), "SWE should be non-negative"

    def test_no_snow_means_zero_swe(self):
        """Pixels that are always snow-free should have SWE = 0."""
        from openeo_mountains_snow.swe.udfs.swe_udf import apply_datacube

        cube = self._make_cube()
        cube[:, 0, :, :] = 0  # no snow ever

        result = apply_datacube(cube, {})

        np.testing.assert_array_equal(
            result.values[:, 0, :, :],
            0,
        )


# ---------------------------------------------------------------------------
# Utility functions from legacy (still used by active pipeline)
# ---------------------------------------------------------------------------

class TestScaUtils:
    """Tests for sca/utils.py — get_scf_ranges and create_mask."""

    def test_get_scf_ranges_default(self):
        from openeo_mountains_snow.sca.utils import get_scf_ranges

        ranges = get_scf_ranges(delta=10, epsilon=10)

        assert isinstance(ranges, dict)
        assert len(ranges) == 10  # 0-100 in steps of 10
        # All keys should be "lower_upper"
        for key in ranges:
            parts = key.split("_")
            assert len(parts) == 2
            lower, upper = int(parts[0]), int(parts[1])
            assert 0 <= lower < upper <= 100

    def test_get_scf_ranges_no_overlap(self):
        from openeo_mountains_snow.sca.utils import get_scf_ranges

        ranges = get_scf_ranges(delta=10, epsilon=0)

        values = list(ranges.values())
        for i in range(len(values) - 1):
            assert values[i][1] == values[i + 1][0], \
                f"Ranges should be contiguous: {values[i]} vs {values[i+1]}"
