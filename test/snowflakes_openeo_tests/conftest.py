import json
from pathlib import Path

import openeo
import pytest
from hydra import initialize, compose

from snowflakes_openeo.collect_training import snowflake_inputs_cube
import snowflakes_openeo.collect_training

@pytest.fixture
def cdse_connection() -> openeo.Connection:
    return openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()

@pytest.fixture
def cdse_staging_connection() -> openeo.Connection:
    return openeo.connect("openeo-staging.dataspace.copernicus.eu").authenticate_oidc()

@pytest.fixture
def elevation_mask(cdse_connection):
    out = Path(__file__).parent / "testdata" / "elevation_mask2.nc"
    if not out.exists():
        aoi = json.load(open(Path(__file__).parent / ".." / ".." / "auxiliary" / "senales_wgs84.geojson"))

        with initialize(version_base=None, config_path="../../src/snowflakes_openeo/conf"):
            # config is relative to a module
            cfg = compose(config_name="config", overrides=[])

            mask = snowflakes_openeo.collect_training.elevation_mask(aoi, cdse_connection, cfg)
            mask.execute_batch(str(out))
    return out


@pytest.fixture
def local_cube(cdse_connection):
    out = Path(__file__).parent / "testdata" / "snowflake_area1.nc"
    if not out.exists():
        aoi = json.load(open(Path(__file__).parent / "andes_area1.geojson"))

        # define time period
        time_period = ['2023-02-01', '2023-02-28']

        with initialize(version_base=None, config_path="../../src/snowflakes_openeo/conf"):
            # config is relative to a module
            cfg = compose(config_name="config", overrides=[])

            bands_indices = snowflake_inputs_cube(aoi, time_period, cdse_connection, cfg)
            bands_indices.execute_batch(str(out))
    return out


