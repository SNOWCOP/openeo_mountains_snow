"""Run the snow cover area -> snow water equivalent pipeline as one openEO batch job.

    python -m openeo_mountains_snow.main
"""

import openeo

from openeo_mountains_snow.config import BACKEND, JOB_OPTIONS, SPATIAL_EXTENT, TEMPORAL_EXTENT
from openeo_mountains_snow.snow_cover_area.snow_cover_area import snow_cover_area_cube
from openeo_mountains_snow.snow_water_equivalent.snow_water_equivalent import snow_water_equivalent_cube


def main():
    connection = openeo.connect(BACKEND, auto_validate=False)
    connection.authenticate_oidc()

    sca = snow_cover_area_cube(connection, TEMPORAL_EXTENT, SPATIAL_EXTENT)
    swe = snow_water_equivalent_cube(connection, sca, SPATIAL_EXTENT)
    swe.execute_batch(title="swe", job_options=JOB_OPTIONS)


if __name__ == "__main__":
    main()
