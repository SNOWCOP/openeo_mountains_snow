"""Step 3: snow water equivalent (SWE) from snow cover area plus downscaled meteo."""

import openeo

from openeo_mountains_snow.config import (
    AGERA_TEMPORAL_EXTENT, NEIGHBORHOOD_SIZE, SWE_RECONSTRUCTION_UDF,
)
from openeo_mountains_snow.snow_water_equivalent.downscale_variables import (
    downscale_shortwave_radiation,
    downscale_temperature_humidity,
)


def snow_water_equivalent_cube(
    connection: openeo.Connection,
    snow_cover_area: openeo.DataCube,
    spatial_extent: dict,
) -> openeo.DataCube:
    """Downscale AGERA5 meteo to the DEM and run the SWE UDF. Output band: `swe`."""
    first_date = snow_cover_area.metadata.temporal_dimension.extent[0]
    dem = connection.load_collection("COPERNICUS_30", spatial_extent=spatial_extent)
    if dem.metadata.has_temporal_dimension():
        dem = dem.reduce_dimension(dimension="t", reducer="max")
    dem = dem.add_dimension(name="t", label=first_date, type="temporal")

    agera = connection.load_stac(
        "https://stac.openeo.vito.be/collections/agera5_daily",
        spatial_extent=spatial_extent,
        temporal_extent=AGERA_TEMPORAL_EXTENT,
    )
    agera = agera.filter_bands(bands=["2m_temperature_mean", "dewpoint_temperature_mean", "solar_radiation_flux"])
    agera = agera.rename_labels(
        dimension="bands", target=["temperature-mean", "dewpoint-temperature", "solar-radiation-flux"]
    )

    geopotential = connection.load_stac(
        "https://artifactory.vgt.vito.be/artifactory/auxdata-public/geopotential.json",
        spatial_extent=spatial_extent,
        bands=["geopotential"],
    )
    geopotential.metadata = geopotential.metadata.add_dimension("t", label=first_date, type="temporal")
    temperature_humidity = downscale_temperature_humidity(agera, dem, geopotential.max_time())

    aspect = connection.load_stac(
        "https://stac.openeo.vito.be/collections/DEM_aspec_30m", spatial_extent=spatial_extent
    ).reduce_dimension(dimension="t", reducer="mean")
    slope = connection.load_stac(
        "https://stac.openeo.vito.be/collections/DEM_slope_30m", spatial_extent=spatial_extent
    ).reduce_dimension(dimension="t", reducer="mean")
    slope_aspect = aspect.merge_cubes(slope).rename_labels(dimension="bands", target=["aspect", "slope"])
    shortwave_radiation = downscale_shortwave_radiation(agera, slope_aspect)

    swe = (
        snow_cover_area
        .merge_cubes(temperature_humidity)
        .merge_cubes(shortwave_radiation)
        .apply_neighborhood(
            process=openeo.UDF.from_file(str(SWE_RECONSTRUCTION_UDF)),
            size=[
                {"dimension": "x", "value": NEIGHBORHOOD_SIZE, "unit": "px"},
                {"dimension": "y", "value": NEIGHBORHOOD_SIZE, "unit": "px"},
            ],
        )
    )
    return swe.rename_labels(dimension="bands", target=["swe"])
