"""Tiny CDSE test: raw inputs, pipeline intermediates and backend UDF runs as small netCDFs.

    .venv\\Scripts\\python scripts\\validation\\cdse_tiny_inputs.py            # start missing jobs, wait, download
    .venv\\Scripts\\python scripts\\validation\\cdse_tiny_inputs.py raw_dem    # only the named cubes

Job ids are kept in tmp/cdse_tiny/jobs.json, so a rerun resumes instead of resubmitting.
Downstream cubes load the finished upstream job results (load_stac_from_job, GTiff) instead of
rebuilding the upstream graph; a cube is only submitted once its DEPENDS have finished.
Validate with scripts/validate_cdse_tiny.ipynb.
"""
import json
import sys
import time
from pathlib import Path

import openeo
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform

from openeo_mountains_snow.config import (
    BACKEND, GEOPOTENTIAL_STAC_URL, JOB_OPTIONS, MODIS_RESOLUTION, MODIS_SNOW_BAND, MODIS_STAC_URL,
    N_DAYS_TO_RECONSTRUCT, NEIGHBORHOOD_SIZE, SCA_NEIGHBORHOOD_OVERLAP, SCA_RECONSTRUCTION_UDF, SWE_RECONSTRUCTION_UDF, build_scf_config,
)
from openeo_mountains_snow.snow_cover_area.conditional_probability import create_modis_snow_cover_fraction_cube
from openeo_mountains_snow.snow_cover_area.snow_cover_area import conditional_probability_inputs
from openeo_mountains_snow.snow_cover_area.snow_map import snow_map_from_snow_cover_fraction
from openeo_mountains_snow.snow_cover_fraction.snow_cover_fraction import snow_cover_fraction_cube
from openeo_mountains_snow.snow_water_equivalent.snow_water_equivalent import swe_meteo_cube

# 1.5 x 1.5 km = 3 x 3 MODIS (500 m) pixels, aligned to the 500 m grid, inside the Senales extent.
EXTENT = {"west": 634000.0, "south": 5172500.0, "east": 635500.0, "north": 5174000.0, "crs": "EPSG:32632"}
PERIOD = ["2022-10-01", "2023-09-01"]  # end exclusive
OUT = Path(__file__).resolve().parents[2] / "tmp" / "cdse_tiny"
# Executors running the S2 snow-chain UDFs were OOM-killed with 3G overhead (Python UDFs live there).
TINY_JOB_OPTIONS = {**JOB_OPTIONS, "executor-memoryOverhead": "6G"}
# snow still lost executors (exit 137) at 7G/6G, worse with the GTiff writer.
JOB_OPTIONS_BY_CUBE = {"snow": {**TINY_JOB_OPTIONS, "executor-memory": "10G", "executor-memoryOverhead": "8G"}}

connection = openeo.connect(BACKEND, auto_validate=False).authenticate_oidc()
aoi = transform(Transformer.from_crs(EXTENT["crs"], "EPSG:4326", always_xy=True).transform,
                box(EXTENT["west"], EXTENT["south"], EXTENT["east"], EXTENT["north"]))


def snow():
    return snow_map_from_snow_cover_fraction(snow_cover_fraction_cube(aoi, PERIOD, connection, build_scf_config()))


def result(name):
    """Finished job result of an upstream cube (must be GTiff: CDSE load_stac cannot read netCDF results)."""
    return connection.load_stac_from_job(load_ids()[name])


def sca_inputs():
    """Same bands as snow_cover_area_inputs, built from the snow, modis_scf and cp_occ results."""
    snow_cube = result("snow")
    cp_occ = result("cp_occ")
    # Client cannot parse the dateless result item (metadata None); backend load_stac still adds a t dimension.
    if cp_occ.metadata is None or cp_occ.metadata.has_temporal_dimension():
        cp_occ = cp_occ.reduce_dimension(dimension="t", reducer="max")
    first_date = f"{snow_cube.metadata.temporal_dimension.extent[0][:10]}T00:00:00Z"  # backend rejects bare dates
    return (snow_cube.merge_cubes(result("modis_scf"))
            .merge_cubes(cp_occ.add_dimension(name="time", label=first_date, type="temporal")))


def neighborhood(cube, udf_path, context=None, overlap=0):
    """apply_neighborhood as in snow_cover_area_cube / snow_water_equivalent_cube."""
    return cube.apply_neighborhood(
        process=openeo.UDF.from_file(str(udf_path), context=context),
        size=[{"dimension": "x", "value": NEIGHBORHOOD_SIZE, "unit": "px"},
              {"dimension": "y", "value": NEIGHBORHOOD_SIZE, "unit": "px"}],
        overlap=[{"dimension": "x", "value": overlap, "unit": "px"},
                 {"dimension": "y", "value": overlap, "unit": "px"}] if overlap else None,
    )


# Upstream cubes whose results are loaded by downstream cubes; saved as GTiff.
DEPENDS = {
    "cp_occ": ["snow"],
    "sca_inputs": ["snow", "modis_scf", "cp_occ"],
    "sca": ["sca_inputs"],
    "swe": ["sca", "swe_meteo"],
}
GTIFF = {dep for deps in DEPENDS.values() for dep in deps}


CUBES = {
    # Raw inputs, no scaling applied: for checking units and scale factors.
    "raw_agera5": lambda: connection.load_stac(
        "https://stac.openeo.vito.be/collections/agera5_daily", spatial_extent=EXTENT, temporal_extent=PERIOD),
    "raw_geopotential": lambda: connection.load_stac(GEOPOTENTIAL_STAC_URL, spatial_extent=EXTENT, bands=["geopotential"]),
    "raw_dem": lambda: connection.load_collection("COPERNICUS_30", spatial_extent=EXTENT),
    "raw_slope_aspect_vito": lambda: connection.load_stac(
        "https://stac.openeo.vito.be/collections/DEM_slope_30m", spatial_extent=EXTENT).merge_cubes(
        connection.load_stac("https://stac.openeo.vito.be/collections/DEM_aspec_30m", spatial_extent=EXTENT)),
    "raw_modis": lambda: connection.load_stac(
        MODIS_STAC_URL, spatial_extent=EXTENT, temporal_extent=PERIOD, bands=[MODIS_SNOW_BAND]),
    # Pipeline intermediates = the UDF inputs.
    "modis_scf": lambda: create_modis_snow_cover_fraction_cube(connection, PERIOD, EXTENT),
    "swe_meteo": lambda: swe_meteo_cube(connection, EXTENT, PERIOD[0], PERIOD),
    # SCA UDF inputs as separate, lighter jobs, each loading the finished upstream results.
    "snow": snow,
    "cp_occ": lambda: conditional_probability_inputs(connection, result("snow"), EXTENT, PERIOD),
    "sca_inputs": sca_inputs,
    # UDFs on the backend.
    "sca": lambda: neighborhood(result("sca_inputs"), SCA_RECONSTRUCTION_UDF,
                                {"n_days_to_reconstruct": N_DAYS_TO_RECONSTRUCT, "lr_pixel_size": MODIS_RESOLUTION},
                                SCA_NEIGHBORHOOD_OVERLAP),
    "swe": lambda: neighborhood(result("sca").merge_cubes(result("swe_meteo")), SWE_RECONSTRUCTION_UDF)
    .rename_labels(dimension="bands", target=["swe"]),
}


def retry(fn, attempts=10):
    """Call fn, backing off when CDSE answers 429 Too Many Requests."""
    for attempt in range(attempts):
        try:
            return fn()
        except openeo.rest.OpenEoApiError as e:
            if e.http_status_code != 429 or attempt == attempts - 1:
                raise
        except openeo.rest.OpenEoApiPlainError as e:
            if e.http_status_code != 429 or attempt == attempts - 1:
                raise
        time.sleep(10 * (attempt + 1))


def load_ids():
    ids_file = OUT / "jobs.json"
    return json.loads(ids_file.read_text()) if ids_file.exists() else {}


def submit(names):
    """Create and start the jobs that are not in jobs.json yet, once their DEPENDS have finished."""
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        ids = load_ids()
        try:
            if name not in ids:
                deps = DEPENDS.get(name, [])
                download(deps)  # waits for the upstream jobs
                if not all((OUT / dep / "done").exists() for dep in deps):
                    print(name, "SKIPPED: upstream not finished:", deps, flush=True)
                    continue
                out_format = "GTiff" if name in GTIFF else "netCDF"
                cube = retry(CUBES[name])
                job = retry(lambda: cube.create_job(out_format=out_format, title=f"tiny_{name}",
                                                    job_options=JOB_OPTIONS_BY_CUBE.get(name, TINY_JOB_OPTIONS)))
                ids[name] = job.job_id
                (OUT / "jobs.json").write_text(json.dumps(ids, indent=1))
                print(name, job.job_id, flush=True)
            job = connection.job(ids[name])
            if retry(job.status) == "created":
                retry(job.start)
        except Exception as e:  # one broken cube must not stop the others
            print(name, "SUBMIT FAILED:", type(e).__name__, str(e)[:300], flush=True)


def download(names, wait=True):
    """Download finished jobs; with wait=False, skip jobs that are still queued/running."""
    ids = load_ids()
    for name in names:
        target = OUT / name
        if name not in ids or (target / "done").exists():
            continue
        try:
            job = connection.job(ids[name])
            while (status := retry(job.status)) not in ("finished", "error", "canceled"):
                if not wait:
                    break
                time.sleep(30)
            if status == "finished":
                retry(lambda: job.get_results().download_files(target))
                (target / "done").touch()
                print(name, "downloaded", flush=True)
            elif status in ("error", "canceled"):
                (OUT / f"{name}_error.txt").write_text(
                    "\n".join(str(l.get("message")) for l in job.logs(level="error")), encoding="utf-8")
                print(name, status, "see", f"{name}_error.txt", flush=True)
            else:
                print(name, status, flush=True)
        except Exception as e:
            print(name, "DOWNLOAD FAILED:", type(e).__name__, str(e)[:300], flush=True)


if __name__ == "__main__":
    names = sys.argv[1:] or list(CUBES)
    submit(names)
    download(names)
