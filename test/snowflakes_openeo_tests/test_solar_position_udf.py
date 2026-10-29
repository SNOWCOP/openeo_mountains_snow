import pandas as pd
import numpy as np

from openeo_mountains_snow.snow_water_equivalent.udfs.solar_position_udf import compute_solar_angles


def test_compute_solar_angles_depends_on_acquisition_time():
    lat = np.array([[0.0]])
    lon = np.array([[0.0]])

    noon_zenith, noon_azimuth = compute_solar_angles(pd.Timestamp("2024-06-21T12:00:00Z"), lat, lon)
    morning_zenith, morning_azimuth = compute_solar_angles(pd.Timestamp("2024-06-21T08:00:00Z"), lat, lon)

    assert noon_zenith.shape == lat.shape
    assert morning_zenith.shape == lat.shape
    assert not np.allclose(noon_zenith, morning_zenith)
    assert not np.allclose(noon_azimuth, morning_azimuth)
