"""Run snow cover fraction -> snow cover area -> snow water equivalent as one openEO batch job.

    python -m openeo_mountains_snow.main
"""

from pathlib import Path

import openeo
from omegaconf import OmegaConf
from shapely.geometry import box
from shapely.ops import transform
from pyproj import Transformer

from openeo_mountains_snow.config import BACKEND, JOB_OPTIONS, SPATIAL_EXTENT, TEMPORAL_EXTENT
from openeo_mountains_snow.snow_cover_area.snow_cover_area import snow_cover_area_cube
from openeo_mountains_snow.snow_cover_area.snow_map import snow_map_from_snow_cover_fraction
from openeo_mountains_snow.snow_cover_fraction.snow_cover_fraction import snow_cover_fraction_cube
from openeo_mountains_snow.snow_water_equivalent.snow_water_equivalent import snow_water_equivalent_cube


def main():
    connection = openeo.connect(BACKEND, auto_validate=False)
    connection.authenticate_oidc()

    # Step 1 expects a WGS84 geometry and the Hydra collection config.
    to_wgs84 = Transformer.from_crs(SPATIAL_EXTENT["crs"], "EPSG:4326", always_xy=True).transform
    aoi = transform(to_wgs84, box(SPATIAL_EXTENT["west"], SPATIAL_EXTENT["south"],
                                   SPATIAL_EXTENT["east"], SPATIAL_EXTENT["north"]))
    scf_cfg = OmegaConf.load(Path(__file__).parent / "conf" / "config.yaml")
    snow_cover_fraction = snow_cover_fraction_cube(aoi, TEMPORAL_EXTENT, connection, scf_cfg)

    high_resolution_snow = snow_map_from_snow_cover_fraction(snow_cover_fraction)
    sca = snow_cover_area_cube(connection, high_resolution_snow, SPATIAL_EXTENT)
    swe = snow_water_equivalent_cube(connection, sca, SPATIAL_EXTENT)
    swe.execute_batch(title="swe", job_options=JOB_OPTIONS)


if __name__ == "__main__":
    main()
