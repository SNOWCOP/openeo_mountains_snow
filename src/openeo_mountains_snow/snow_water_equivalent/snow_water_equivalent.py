"""Step 3: snow water equivalent (SWE) from snow cover area plus downscaled meteo."""

import openeo

from openeo_mountains_snow.config import (
    NEIGHBORHOOD_SIZE, SWE_RECONSTRUCTION_UDF, TEMPORAL_EXTENT,
)
from openeo_mountains_snow.snow_water_equivalent.downscale_variables import (
    downscale_shortwave_radiation,
    downscale_temperature_humidity,
)


def swe_meteo_cube(
    connection: openeo.Connection,
    spatial_extent: dict,
    first_date: str,
    temporal_extent: list[str] = TEMPORAL_EXTENT,
) -> openeo.DataCube:
    """Downscaled AGERA5 meteo on the DEM grid: temperature (°C), precipitation (mm/day), shortwave (W/m²)."""
    dem = connection.load_collection("COPERNICUS_30", spatial_extent=spatial_extent)
    if dem.metadata.has_temporal_dimension():
        dem = dem.reduce_dimension(dimension="t", reducer="max")
    dem = dem.add_dimension(name="t", label=first_date, type="temporal")

    agera = connection.load_stac(
        "https://stac.openeo.vito.be/collections/agera5_daily",
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
    )
    # total_precipitation is stored with scale 0.01 -> mm/day.
    precipitation = (agera.filter_bands(bands=["total_precipitation"]) * 0.01).rename_labels(
        dimension="bands", target=["precipitation"])
    agera = agera.filter_bands(bands=["2m_temperature_mean", "dewpoint_temperature_mean", "solar_radiation_flux"])
    agera = agera.rename_labels(
        dimension="bands", target=["temperature-mean", "dewpoint-temperature", "solar-radiation-flux"]
    )

    # TODO(*): Also verify the `35885 / geopotential` term in
    # downscale_variables.preprocess_low_resolution_agera (ERA5: height = geopotential / 9.80665).
    geopotential = connection.load_stac(
        "https://artifactory.vgt.vito.be/artifactory/auxdata-public/geopotential.json",
        spatial_extent=spatial_extent,
        bands=["geopotential"],
    )
    if not geopotential.metadata.has_temporal_dimension():
        geopotential.metadata = geopotential.metadata.add_dimension("t", label=first_date, type="temporal")
    temperature = downscale_temperature_humidity(agera, dem, geopotential.max_time()).filter_bands(
        bands=["temperature_downscaled"])
    precipitation = precipitation.resample_cube_spatial(temperature, method="bilinear")

    aspect = connection.load_stac(
        "https://stac.openeo.vito.be/collections/DEM_aspec_30m", spatial_extent=spatial_extent
    ).reduce_dimension(dimension="t", reducer="mean")
    slope = connection.load_stac(
        "https://stac.openeo.vito.be/collections/DEM_slope_30m", spatial_extent=spatial_extent
    ).reduce_dimension(dimension="t", reducer="mean")
    slope_aspect = aspect.merge_cubes(slope).rename_labels(dimension="bands", target=["aspect", "slope"])
    shortwave_radiation = downscale_shortwave_radiation(agera, slope_aspect)
    return temperature.merge_cubes(precipitation).merge_cubes(shortwave_radiation)


def snow_water_equivalent_cube(
    connection: openeo.Connection,
    snow_cover_area: openeo.DataCube,
    spatial_extent: dict,
    temporal_extent: list[str] = TEMPORAL_EXTENT,
) -> openeo.DataCube:
    """Downscale AGERA5 meteo to the DEM and run the SWE UDF. Output band: `swe`."""
    first_date = snow_cover_area.metadata.temporal_dimension.extent[0]
    meteo = swe_meteo_cube(connection, spatial_extent, first_date, temporal_extent)
    swe = (
        snow_cover_area
        .merge_cubes(meteo)
        .apply_neighborhood(
            process=openeo.UDF.from_file(str(SWE_RECONSTRUCTION_UDF)),
            size=[
                {"dimension": "x", "value": NEIGHBORHOOD_SIZE, "unit": "px"},
                {"dimension": "y", "value": NEIGHBORHOOD_SIZE, "unit": "px"},
            ],
        )
    )
    return swe.rename_labels(dimension="bands", target=["swe"])
