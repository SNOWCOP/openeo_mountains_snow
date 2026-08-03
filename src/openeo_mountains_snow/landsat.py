#%%

"""
Landsat process-graph pipeline for SnowFLAKES representative pixels.

This module builds everything as an openEO process graph and avoids local
NetCDF extraction for intermediate steps.

Expected source bands (example LANDSAT_L1C stack):
- B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, SZA, VZA

The representative-pixels UDF expects these logical inputs:
- B02, B03, B04, B08, B11, local_solar_incidence_angle,
  NDVI, NDSI, diff_B_NIR, SI

Missing bands are currently filled with constants so graph execution can
continue while sensor-specific refinements are still in progress.
"""

from __future__ import annotations

from typing import Iterable
from pathlib import Path
import sys

import openeo
from openeo.processes import array_append


from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent
_src_pkg_dir = _pkg_dir.parent / "src" / "openeo_mountains_snow"

if _src_pkg_dir.is_dir():
    __path__ = [str(_src_pkg_dir)]
else:
    __path__ = [str(_pkg_dir)]


src_dir = Path(__file__).resolve().parents[1]
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
from openeo_mountains_snow.snow_cover_fraction import apply_representative_pixels


# Map Landsat L1C names to the Sentinel-2-like names used downstream.
LANDSAT_TO_SNOWFLAKES = {
    "B1": "B01",
    "B2": "B02",
    "B3": "B03",
    "B4": "B04",
    "B5": "B08",
    "B6": "B11",
    "B7": "B12",
    "B8": "B8A",
    "B9": "B09",
    "B10": "B10",
    # Keep SWIR mapping on B6->B11; rename thermal B11 to avoid duplicate labels.
    "B11": "B11_THERMAL",
    "SZA": "SZA",
    "VZA": "VZA",
}

# Raw collection bands we try to load from LANDSAT_L1C.
LANDSAT_INPUT_BANDS = list(LANDSAT_TO_SNOWFLAKES.keys())

# Minimal spectral contract required by representative_pixels.py
REQUIRED_SPECTRAL_BANDS = ["B02", "B03", "B04", "B08", "B11"]

# Temporary placeholders when required inputs are missing.
PLACEHOLDER_REFLECTANCE = 0.0
PLACEHOLDER_LOCAL_ANGLE_DEGREES = 45.0


def _make_unique_band_names(names: list[str]) -> list[str]:
    """Return unique band labels while preserving order.

    If duplicates exist, suffix later occurrences as <name>_<n>.
    """
    counts: dict[str, int] = {}
    unique: list[str] = []

    for name in names:
        count = counts.get(name, 0)
        if count == 0:
            unique.append(name)
        else:
            unique.append(f"{name}_{count}")
        counts[name] = count + 1

    return unique


def _append_constant_band(
    cube: openeo.DataCube,
    *,
    band_name: str,
    value: float,
) -> openeo.DataCube:
    """Append a constant-valued band through apply_dimension on 'bands'."""

    def add_constant(bands):
        return array_append(bands, value, band_name)

    old_names = list(cube.metadata.band_names)
    return cube.apply_dimension(dimension="bands", process=add_constant).rename_labels(
        dimension="bands", target=old_names + [band_name]
    )


def _ensure_required_bands(
    cube: openeo.DataCube,
    *,
    required_bands: Iterable[str],
    fill_value: float,
) -> openeo.DataCube:
    """Ensure required bands exist by appending placeholders when absent."""
    current = list(cube.metadata.band_names)
    fixed = cube

    for band in required_bands:
        if band not in current:
            fixed = _append_constant_band(fixed, band_name=band, value=fill_value)
            current.append(band)

    return fixed


def _add_local_incidence_angle(cube: openeo.DataCube) -> openeo.DataCube:
    """Provide local_solar_incidence_angle from SZA or a fallback constant."""
    band_names = list(cube.metadata.band_names)
    if "local_solar_incidence_angle" in band_names:
        return cube

    if "SZA" in band_names:
        renamed = [
            "local_solar_incidence_angle" if b == "SZA" else b
            for b in band_names
        ]
        return cube.rename_labels(dimension="bands", target=renamed)

    return _append_constant_band(
        cube,
        band_name="local_solar_incidence_angle",
        value=PLACEHOLDER_LOCAL_ANGLE_DEGREES,
    )


def _compute_indices(cube: openeo.DataCube) -> openeo.DataCube:
    """Compute NDVI, NDSI, diff_B_NIR and SI as extra bands."""

    def add_indices(bands):
        eps = 1e-6
        nir = bands["B08"]
        red = bands["B04"]
        blue = bands["B02"]
        green = bands["B03"]
        swir = bands["B11"]

        ndvi = (nir - red) / (nir + red + eps)
        updated = array_append(bands, ndvi, "NDVI")

        ndsi = (green - swir) / (green + swir + eps)
        updated = array_append(updated, ndsi, "NDSI")

        diff_b_nir = (blue - nir) / (blue + nir + eps)
        updated = array_append(updated, diff_b_nir, "diff_B_NIR")

        si = ndsi / (green + eps)
        updated = array_append(updated, si, "SI")
        return updated

    base_names = list(cube.metadata.band_names)
    target_names = base_names + ["NDVI", "NDSI", "diff_B_NIR", "SI"]
    return cube.apply_dimension(dimension="bands", process=add_indices).rename_labels(
        dimension="bands", target=target_names
    )


def landsat_inputs_cube(
    aoi: dict,
    time_period: list[str],
    connection: openeo.Connection,
    *,
    max_cloud_cover: int = 70,
) -> openeo.DataCube:
    """Build the representative-pixels input cube from LANDSAT_L1C.

    All operations stay in the process graph. No local NetCDF is required.
    """
    landsat = connection.load_collection(
        "LANDSAT_L1C",
        spatial_extent=aoi,
        temporal_extent=time_period,
        bands=LANDSAT_INPUT_BANDS,
        max_cloud_cover=max_cloud_cover,
    )

    original_names = list(landsat.metadata.band_names)
    mapped_names = [LANDSAT_TO_SNOWFLAKES.get(name, name) for name in original_names]
    mapped_names = _make_unique_band_names(mapped_names)
    mapped = landsat.rename_labels(dimension="bands", target=mapped_names)

    with_required_spectral = _ensure_required_bands(
        mapped,
        required_bands=REQUIRED_SPECTRAL_BANDS,
        fill_value=PLACEHOLDER_REFLECTANCE,
    )

    with_angle = _add_local_incidence_angle(with_required_spectral)

    return _compute_indices(with_angle)

#%%

url = "https://openeo.prod.eu-west-1.openeo-int.v1.dataspace.copernicus.eu/openeo"
conn = openeo.connect(url).authenticate_oidc()


aoi = {
        "west": 10.851177,
        "south": 46.699372,
        "east": 10.916695,
        "north": 46.744288,
        "crs": "EPSG:4326",
    }
time_period = ["2025-04-01", "2025-09-01"]

inputs = landsat_inputs_cube(
        aoi=aoi,
        time_period=time_period,
        connection=conn,
        max_cloud_cover=70,
    )

#job = inputs.save_result(format="NetCDF").create_job(title="Landsat representative pixels")
#job.start_and_wait()

#%%
import os
import xarray as xr

results = job.get_results()
download_dir = r"C:\Git_projects\openeo_mountains_snow\src\openeo_mountains_snow\tmp\landsat_input_cube2"
os.makedirs(download_dir, exist_ok=True)
results.download_files(download_dir)

# open the downloaded openEO.nc if present and show basic info
nc_path = os.path.join(download_dir, "openEO.nc")
if os.path.exists(nc_path):
    ds = xr.open_dataset(nc_path)
    print("Downloaded NetCDF bands:", list(ds.coords.get("bands", [])))
    ds.close()
else:
    print("No openEO.nc found in", download_dir)
#%%


def landsat_representative_pixels_cube(
    aoi: dict,
    time_period: list[str],
    connection: openeo.Connection,
    *,
    max_cloud_cover: int = 70,
    neighborhood_size: int = 100,
) -> openeo.DataCube:
    """Run representative-pixels directly on the Landsat process graph."""
    inputs = landsat_inputs_cube(
        aoi=aoi,
        time_period=time_period,
        connection=connection,
        max_cloud_cover=max_cloud_cover,
    )
    return apply_representative_pixels(inputs, neighborhood_size=neighborhood_size)



# %%

url = "https://openeo.prod.eu-west-1.openeo-int.v1.dataspace.copernicus.eu/openeo"
conn = openeo.connect(url).authenticate_oidc()

aoi = {
        "west": 10.851177,
        "south": 46.699372,
        "east": 10.916695,
        "north": 46.744288,
        "crs": "EPSG:4326",
    }
time_period = ["2024-09-01", "2025-09-01"]

result_cube = landsat_representative_pixels_cube(
    aoi=aoi,
    time_period=time_period,
    connection=conn,
    neighborhood_size=100,
    max_cloud_cover=70,
)

job_options ={
    "driver-memory": "3g",
    "driver-memoryOverhead": "3g",
    "executor-memory": "5g",
    "executor-memoryOverhead": "3g",
    "max-executors": 20,
    "python-memory": "5g"}

job = result_cube.save_result(format="NetCDF").create_job(title="Landsat representative pixels", job_options=job_options)
job.start_and_wait()



#%%

import os
import xarray as xr

results = job.get_results()
download_dir = r"C:\Git_projects\openeo_mountains_snow\src\openeo_mountains_snow\tmp\landsat_snowflakes_AWS_output"
os.makedirs(download_dir, exist_ok=True)
results.download_files(download_dir)

nc_path = os.path.join(download_dir, "openEO.nc")
ds = xr.open_dataset(nc_path)
ds

# %%

import matplotlib.pyplot as plt


da = ds['representative'].isel(t=6)

plt.figure(figsize=(8,6))
da.plot(cmap='viridis')  # or 'gray', 'Blues', etc.
plt.title("My Data")
plt.show()
