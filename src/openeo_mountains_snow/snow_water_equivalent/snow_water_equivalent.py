"""Step 3: snow water equivalent (SWE) from snow cover area plus downscaled meteo."""

import math

import numpy as np
import openeo
import xarray as xr
from pyproj import Transformer

from openeo_mountains_snow.common import terrain
from openeo_mountains_snow.config import (
    GEOPOTENTIAL_STAC_URL, NEIGHBORHOOD_SIZE, RESOLUTION, SWE_RECONSTRUCTION_UDF, TEMPORAL_EXTENT, utm_epsg,
)

# AGERA5 and ERA5 geopotential are 0.1 deg; load them this much wider so bilinear resampling has neighbours.
COARSE_BUFFER_DEG = 0.2
from openeo_mountains_snow.snow_water_equivalent.downscale_variables import (
    downscale_shortwave_radiation,
    downscale_temperature,
)


def extent_hemisphere(spatial_extent: dict) -> str:
    """"north" or "south", from the latitude of the extent centre."""
    _, (south, north) = Transformer.from_crs(spatial_extent.get("crs", "EPSG:4326"), "EPSG:4326", always_xy=True).transform(
        [spatial_extent["west"]] * 2, [spatial_extent["south"], spatial_extent["north"]])
    return "south" if south + north < 0 else "north"


def swe_meteo_cube(
    connection: openeo.Connection,
    spatial_extent: dict,
    first_date: str,
    temporal_extent: list[str] = TEMPORAL_EXTENT,
) -> openeo.DataCube:
    """Downscaled AGERA5 meteo on the SCA grid (UTM, `RESOLUTION` m): temperature (°C), precipitation (mm/day), shortwave (W/m²)."""
    first_date = f"{first_date[:10]}T00:00:00Z"  # backend add_dimension rejects bare dates
    dem = connection.load_collection("COPERNICUS_30", spatial_extent=spatial_extent)
    if dem.metadata.has_temporal_dimension():
        dem = dem.reduce_dimension(dimension="t", reducer="max")
    dem = dem.resample_spatial(resolution=RESOLUTION, projection=utm_epsg(spatial_extent), method="bilinear")
    # Slope/aspect from the Copernicus DEM (global); the VITO DEM_slope/aspect STAC collections only cover Europe.
    slope_aspect = dem.aspect().merge_cubes(dem.slope()).rename_labels(dimension="bands", target=["aspect", "slope"])
    dem = dem.add_dimension(name="t", label=first_date, type="temporal")

    lons, lats = Transformer.from_crs(spatial_extent.get("crs", "EPSG:4326"), "EPSG:4326", always_xy=True).transform(
        [spatial_extent["west"], spatial_extent["east"]] * 2,
        [spatial_extent["south"]] * 2 + [spatial_extent["north"]] * 2)
    coarse_extent = {"west": min(lons) - COARSE_BUFFER_DEG, "east": max(lons) + COARSE_BUFFER_DEG,
                     "south": min(lats) - COARSE_BUFFER_DEG, "north": max(lats) + COARSE_BUFFER_DEG, "crs": "EPSG:4326"}

    def to_dem_grid(cube):
        # resample the buffered coarse cube, then clip to the AOI so the UDFs only see the AOI
        return cube.resample_cube_spatial(dem, method="bilinear").filter_bbox(spatial_extent)

    agera = to_dem_grid(connection.load_stac(
        "https://stac.openeo.vito.be/collections/agera5_daily",
        spatial_extent=coarse_extent,
        temporal_extent=temporal_extent,
    ))
    # total_precipitation is stored with scale 0.01 -> mm/day.
    precipitation = (agera.filter_bands(bands=["total_precipitation"]) * 0.01).rename_labels(
        dimension="bands", target=["precipitation"])
    agera = agera.filter_bands(bands=["2m_temperature_mean", "solar_radiation_flux"])
    agera = agera.rename_labels(dimension="bands", target=["temperature-mean", "solar-radiation-flux"])

    geopotential = connection.load_stac(
        GEOPOTENTIAL_STAC_URL,
        spatial_extent=coarse_extent,
        bands=["geopotential"],
    )
    if not geopotential.metadata.has_temporal_dimension():
        geopotential.metadata = geopotential.metadata.add_dimension("t", label=first_date, type="temporal")
    geopotential = to_dem_grid(geopotential.max_time())
    temperature = downscale_temperature(agera, dem, geopotential, hemisphere=extent_hemisphere(spatial_extent))

    if terrain.SLOPE_ASPECT_IN_RADIANS:
        slope_aspect = slope_aspect * (180.0 / math.pi)  # the shortwave downscaling wants degrees
    shortwave_radiation = downscale_shortwave_radiation(agera, slope_aspect)
    return temperature.merge_cubes(precipitation).merge_cubes(shortwave_radiation)


def basin_snow_cover_area(snow_cover_area: xr.DataArray) -> dict[str, float]:
    """Basin-wide SCA (%) per date from a downloaded `sca` result (t, y, x): mean of the valid (<= 100) pixels."""
    percent = snow_cover_area.where(snow_cover_area <= 100).mean(dim=["y", "x"])
    return {str(t)[:10]: float(v) for t, v in zip(percent.t.values, percent.values) if np.isfinite(v)}


def snow_water_equivalent_cube(
    connection: openeo.Connection,
    snow_cover_area: openeo.DataCube,
    spatial_extent: dict,
    temporal_extent: list[str] = TEMPORAL_EXTENT,
    basin_sca: dict[str, float] | None = None,
) -> openeo.DataCube:
    """Downscale AGERA5 meteo to the DEM and run the SWE UDF. Output band: `swe`.

    `basin_sca` ({"YYYY-MM-DD": percent}, see `basin_snow_cover_area`) drives accumulation on
    warm days as in the original; without it each neighborhood uses its own SCA.
    """
    first_date = snow_cover_area.metadata.temporal_dimension.extent[0]
    meteo = swe_meteo_cube(connection, spatial_extent, first_date, temporal_extent)
    swe = (
        snow_cover_area
        .merge_cubes(meteo.resample_cube_spatial(snow_cover_area, method="near"))  # no-op when already aligned
        .apply_neighborhood(
            process=openeo.UDF.from_file(str(SWE_RECONSTRUCTION_UDF), context={
                "basin_sca": basin_sca or {},
                "hydrological_year_start": "04-01" if extent_hemisphere(spatial_extent) == "south" else "10-01",
            }),
            size=[
                {"dimension": "x", "value": NEIGHBORHOOD_SIZE, "unit": "px"},
                {"dimension": "y", "value": NEIGHBORHOOD_SIZE, "unit": "px"},
            ],
        )
    )
    return swe.rename_labels(dimension="bands", target=["swe"])
