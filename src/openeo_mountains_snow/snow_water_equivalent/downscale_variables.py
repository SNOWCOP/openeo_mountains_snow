"""
Temperature and shortwave radiation downscaling module.

Downscales AGERA5 temperature and shortwave radiation from coarse resolution to the
high-resolution DEM using monthly lapse rates and topographic corrections.
"""

from openeo import DataCube, UDF

from openeo_mountains_snow.config import (
    INCIDENCE_ANGLE_UDF, NEIGHBORHOOD_SIZE, SOLAR_POSITION_UDF, TEMPERATURE_DOWNSCALING_UDF,
)


def downscale_temperature(agera_cube, elevation_cube, geopotential_cube, southern_hemisphere: bool):
    """
    Downscale AGERA5 mean temperature to the DEM with the monthly lapse rate of the hemisphere.

    Args:
        agera_cube: AGERA5 cube with band `temperature-mean` (K, scale 0.01)
        elevation_cube: DEM cube (band `DEM`, m)
        geopotential_cube: ERA5 surface geopotential (m2/s2), no time dimension
        southern_hemisphere: selects the southern-hemisphere lapse-rate table

    Returns:
        DataCube with band `temperature_downscaled` (°C)
    """
    temperature = agera_cube.filter_bands(["temperature-mean"]) * 0.01
    height = geopotential_cube / 9.80665  # geopotential (m2/s2) -> height (m)
    inputs = (temperature.merge_cubes(height)
              .resample_cube_spatial(elevation_cube, method="bilinear")
              .merge_cubes(elevation_cube.max_time()))
    udf = UDF.from_file(str(TEMPERATURE_DOWNSCALING_UDF), context={"southern_hemisphere": southern_hemisphere})
    return (inputs.apply_neighborhood(
                process=udf,
                size=[
                    {"dimension": "x", "value": NEIGHBORHOOD_SIZE, "unit": "px"},
                    {"dimension": "y", "value": NEIGHBORHOOD_SIZE, "unit": "px"},
                ])
            # backend keeps the 3 input labels for the 1-band UDF output; keep only the first
            .filter_bands(["temperature-mean"])
            .rename_labels(dimension="bands", target=["temperature_downscaled"]))


def downscale_shortwave_radiation(sw: DataCube,  slope_aspect: DataCube):
    """
    Downscale shortwave radiation using solar incidence angle correction.
    
    Requires slope and aspect in degrees (aspect clockwise from north) and computes
    topographic correction based on solar incidence angle.
    
    Args:
        sw: AGERA5 shortwave radiation DataCube (J/m^2/day) at DEM resolution
        slope_aspect: DataCube with slope and aspect bands in degrees
        
    Returns:
        DataCube with topographically corrected shortwave radiation

    """
    solar_flux = sw.filter_bands(["solar-radiation-flux"])
    solar_flux = solar_flux / 86400  # J/m^2/day -> W/m^2 (daily mean)
    # put the coarse AGERA5 flux on the slope/aspect grid first; else the merge happens on the ~10 km AGERA5 grid
    solar_flux = solar_flux.resample_cube_spatial(slope_aspect, method="bilinear")
    compute_solarposition = UDF.from_file(str(SOLAR_POSITION_UDF))
    
    solar_flux_with_sunpos = solar_flux.apply_dimension(dimension="bands", process=compute_solarposition)

    solar_flux_with_sunpos = solar_flux_with_sunpos.rename_labels(dimension="bands", target= ["solar-radiation-flux", "zenith", "azimuth"])

    compute_incidence = UDF.from_file(str(INCIDENCE_ANGLE_UDF))
    shortwave_radiation_downscaled = (solar_flux_with_sunpos
                                .merge_cubes(slope_aspect)
                                .apply_dimension(dimension="bands", process=compute_incidence)
                                # backend keeps the 5 input labels for the 1-band UDF output; keep only the first
                                .filter_bands(["solar-radiation-flux"])
                                .rename_labels(dimension="bands", target=["shortwave-radiation-flux-downscaled"]))

    return shortwave_radiation_downscaled