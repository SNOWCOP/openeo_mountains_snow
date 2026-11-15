import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pvlib")

from pyproj import Transformer

from openeo_mountains_snow.common.udfs.solar_position_udf import _spa
from openeo_mountains_snow.config import SOLAR_POSITION_UDF

LON, LAT, EPSG = 11.1, 46.5, 32632
OVERPASS = pd.Timestamp("2023-07-05 10:15:00", tz="UTC")


def chunk(make_cube, extra_bands=None):
    """5x5 px, 2 km spacing, centre pixel exactly at (LON, LAT); t attr date-only as on CDSE."""
    x, y = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True).transform(LON, LAT)
    bands = {"B02": np.full((5, 5), 0.1), **(extra_bands or {})}
    cube = make_cube(bands, x0=x - 4000, y0=y + 4000, res=2000)
    cube.attrs["t"] = OVERPASS.normalize().tz_localize(None).to_pydatetime()
    return cube


def reference_angles():
    zen, azi = _spa(np.array([OVERPASS.timestamp()]), LAT, LON, OVERPASS.normalize())
    return float(zen[0]), float(azi[0])


def centre_degrees(result):
    return [float(np.degrees(result.sel(bands=b).values[2, 2])) for b in ("zenith", "azimuth")]


def test_recovers_overpass_from_stac_context(make_cube, run_udf):
    zen, azi = reference_angles()
    context = {"s2_tiles": {OVERPASS.strftime("%Y-%m-%d"): [[LON, LAT, zen, azi], [LON + 5, LAT, 0.0, 0.0]]}}
    result = run_udf(SOLAR_POSITION_UDF, chunk(make_cube), context, epsg=EPSG)
    assert result.dims == ("bands", "y", "x")
    assert list(result.bands.values) == ["B02", "zenith", "azimuth"]
    assert centre_degrees(result) == pytest.approx([zen, azi], abs=0.02)
    assert np.ptp(result.sel(bands="zenith").values) > 0, "angles must vary per pixel"


def test_falls_back_to_sun_angle_bands(make_cube, run_udf):
    zen, azi = reference_angles()
    cube = chunk(make_cube, {"sunZenithAngles": np.full((5, 5), zen), "sunAzimuthAngles": np.full((5, 5), azi)})
    result = run_udf(SOLAR_POSITION_UDF, cube, epsg=EPSG)
    assert centre_degrees(result) == pytest.approx([zen, azi], abs=0.02)


def test_without_angles_uses_local_solar_noon(make_cube, run_udf):
    zen, azi = centre_degrees(run_udf(SOLAR_POSITION_UDF, chunk(make_cube), epsg=EPSG))
    assert azi == pytest.approx(180, abs=0.5)
    assert zen == pytest.approx(LAT - 22.7, abs=1)  # early-July declination ~22.7 deg


def test_requires_crs(make_cube, run_udf):
    with pytest.raises(AssertionError, match="CRS"):
        run_udf(SOLAR_POSITION_UDF, chunk(make_cube))
