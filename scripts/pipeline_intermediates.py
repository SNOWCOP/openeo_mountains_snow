"""Run every intermediate cube of the SCF -> SCA -> SWE pipeline as a separate openEO batch job.

Run cell by cell (VS Code "# %%" cells / Jupyter interactive window). Each `submit(...)` starts a
batch job without blocking; the last cell waits for all started jobs and downloads the results.
Note: every job recomputes its upstream graph, so later steps are more expensive.
"""

# %% Setup
from pathlib import Path

import openeo
from omegaconf import OmegaConf
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform

from openeo_mountains_snow.config import (
    BACKEND, JOB_OPTIONS, MODIS_TEMPORAL_EXTENT, SPATIAL_EXTENT, TEMPORAL_EXTENT,
)
from openeo_mountains_snow.snow_cover_area.conditional_probability import (
    compute_conditional_probability,
    compute_snow_cover_fraction_range_masks,
    create_modis_snow_cover_fraction_cube,
)
from openeo_mountains_snow.snow_cover_area.snow_cover_area import snow_cover_area_cube
from openeo_mountains_snow.snow_cover_area.snow_map import snow_map_from_snow_cover_fraction
from openeo_mountains_snow.snow_cover_fraction.snow_cover_fraction import (
    apply_representative_pixels,
    snowflake_inputs_cube,
)
from openeo_mountains_snow.snow_water_equivalent.snow_water_equivalent import snow_water_equivalent_cube

REPO = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO / "tmp" / "intermediates"
cfg = OmegaConf.load(REPO / "src" / "openeo_mountains_snow" / "conf" / "config.yaml")
print(OmegaConf.to_yaml(cfg))

connection = openeo.connect(BACKEND, auto_validate=False)
connection.authenticate_oidc()

to_wgs84 = Transformer.from_crs(SPATIAL_EXTENT["crs"], "EPSG:4326", always_xy=True).transform
aoi = transform(to_wgs84, box(SPATIAL_EXTENT["west"], SPATIAL_EXTENT["south"],
                              SPATIAL_EXTENT["east"], SPATIAL_EXTENT["north"]))

jobs: dict[str, openeo.BatchJob] = {}


def submit(cube: openeo.DataCube, name: str) -> openeo.BatchJob:
    """Start a batch job saving `cube` as netCDF; keeps a handle in `jobs`."""
    job = cube.save_result(format="netCDF").create_job(title=f"intermediate_{name}", job_options=JOB_OPTIONS)
    job.start()
    jobs[name] = job
    print(f"{name}: {job.job_id}")
    return job


# %% Build all cubes (graph construction only, nothing runs yet)
inputs = snowflake_inputs_cube(aoi, TEMPORAL_EXTENT, connection, cfg)
snow_cover_fraction = apply_representative_pixels(inputs, neighborhood_size=3038)
high_resolution_snow = snow_map_from_snow_cover_fraction(snow_cover_fraction)

modis_scf = create_modis_snow_cover_fraction_cube(connection, MODIS_TEMPORAL_EXTENT, SPATIAL_EXTENT)

all_masks, labels = compute_snow_cover_fraction_range_masks(
    connection, high_resolution_snow, MODIS_TEMPORAL_EXTENT, SPATIAL_EXTENT
)
conditional_probability, snow_sum, occurrences = compute_conditional_probability(all_masks, labels)

sca = snow_cover_area_cube(connection, high_resolution_snow, SPATIAL_EXTENT)
swe = snow_water_equivalent_cube(connection, sca, SPATIAL_EXTENT)

# %% Step 1: snow cover fraction
submit(inputs, "01_scf_inputs")
submit(snow_cover_fraction, "02_snow_cover_fraction")

# %% Step 2a: high-resolution snow map and MODIS SCF
submit(high_resolution_snow, "03_high_resolution_snow")
submit(modis_scf, "04_modis_scf")

# %% Step 2b: conditional probability
submit(all_masks, "05_scf_range_masks")
submit(conditional_probability, "06_conditional_probability")
submit(occurrences, "07_occurrences")

# %% Step 2c: snow cover area
submit(sca, "08_snow_cover_area")

# %% Step 3: snow water equivalent
submit(swe, "09_snow_water_equivalent")

# %% Wait for all started jobs and download results
for name, job in jobs.items():
    try:
        if job.status() != "finished":
            job.start_and_wait()  # start is ignored for queued/running jobs; then polls
        job.get_results().download_files(OUTPUT_DIR / name)
        print(f"{name}: downloaded")
    except openeo.rest.JobFailedException as e:
        print(f"{name}: FAILED ({e}); logs: {job.logs(level='error')}")
