"""
Temperature and humidity downscaling module.

Provides functions for downscaling temperature, humidity, and shortwave radiation
from coarse resolution data to high resolution DEM data using lapse rates and
topographic corrections.
"""

from pathlib import Path
from openeo import DataCube, UDF
from openeo.processes import ProcessBuilder, array_create, exp, clip
import numpy as np

from config import SOLAR_POSITION_UDF, INCIDENCE_ANGLE_UDF

# Vapor pressure coefficients for Northern and Southern hemispheres
VP_COEFF_NOHEM = np.array([0.41, 0.42, 0.40, 0.39, 0.38, 0.36, 0.33, 0.33, 0.36, 0.37, 0.40, 0.40]) / 1000.0
VP_COEFF_SOHEM = np.array([0.38, 0.36, 0.33, 0.33, 0.36, 0.37, 0.40, 0.40, 0.41, 0.42, 0.40, 0.39]) / 1000.0

# Magnus formula constants
MAGNUS_A = 611.21
MAGNUS_B = 17.502
MAGNUS_C = 240.97

def preprocess_low_resolution_agera(cube: ProcessBuilder, lapse_rate, temp_index="temperature-mean",
                                      dewpoint_index=None, temp_scale=1.0) -> ProcessBuilder:
    """
    Preprocess low-resolution AGERA data for downscaling.
    
    Applies lapse rate correction using geopotential height reference.
    
    Args:
        cube: ProcessBuilder with temperature and geopotential bands
        lapse_rate: Temperature lapse rate (K/m)
        temp_index: Name of temperature band
        dewpoint_index: Optional index of dewpoint band (0-11 for month)
        temp_scale: Scale factor for temperature
        
    Returns:
        ProcessBuilder with preprocessed temperature and optionally dewpoint
    """
    t_raw = cube[temp_index] * temp_scale
    geopotential = 35885.0 / cube["geopotential"]
    t_0 = t_raw - lapse_rate * (0 - geopotential)

    if dewpoint_index is not None:
        d_0 = cube[dewpoint_index] * 0.01 - lapse_rate * (0 - geopotential)
        return array_create([t_0, d_0])
    else:
        return t_0


def downscale_t_dewpoint(cube: ProcessBuilder, lapse_rate, temp_index="temperature-mean",
                         dem_index="DEM") -> ProcessBuilder:
    """
    Downscale temperature and compute relative humidity.
    
    Args:
        cube: ProcessBuilder with preprocessed temperature and dewpoint
        lapse_rate: Temperature lapse rate (K/m)
        temp_index: Name of temperature band
        dem_index: Name of DEM elevation band
        
    Returns:
        ProcessBuilder with downscaled temperature and relative humidity
    """
    temperature_downscaled = cube[temp_index] - lapse_rate * (cube[dem_index] - 0)
    temperature_c = temperature_downscaled - 273.15
    dewpoint_c = cube["dewpoint-temperature"] - 273.15
    rh = relative_humidity_formula(temperature_c,
                                    dewpoint_c,
                                    cube[dem_index], 2)
    return array_create([temperature_c, rh])

def relative_humidity_formula(temperature_downscaled, dewpoint_temperature_coarse, elevation, month_index):
    """
    Calculate relative humidity using Magnus formula.
    
    Args:
        temperature_downscaled: Downscaled temperature in Celsius
        dewpoint_temperature_coarse: Coarse resolution dewpoint temperature in celcius
        elevation: Elevation in meters
        month_index: Month index (0-11) to select appropriate vapor pressure coefficient
        
    Returns:
        Relative humidity clipped to [0, 100]
    """
    vp_coeff_all = VP_COEFF_SOHEM  # TODO: select vp_coeff based on hemisphere
    vp_coeff = vp_coeff_all[month_index]
    d_t_lapse_rate = vp_coeff * MAGNUS_C / MAGNUS_B

    D_down = dewpoint_temperature_coarse - d_t_lapse_rate * (elevation - 0)
    es = MAGNUS_A * exp((MAGNUS_B * temperature_downscaled) / (temperature_downscaled + MAGNUS_C))
    e = MAGNUS_A * exp((MAGNUS_B * D_down) / (D_down + MAGNUS_C))
    return clip(100 * e / es, 0, 100)



def downscale_temperature_humidity(agera_cube, elevation_cube, geopotential_cube):
    """
    Downscale temperature and humidity data using elevation data.
    
    Args:
        agera_cube: AGERA5 data cube with temperature and dewpoint
        elevation_cube: DEM elevation data cube
        geopotential_cube: Geopotential height data cube
        
    Returns:
        DataCube with downscaled temperature and relative humidity
    """
    lapse_rate_nohem = np.array([4.4, 5.9, 7.1, 7.8, 8.1, 8.2, 8.1, 8.1, 7.7, 6.8, 5.5, 4.7]) / 1000.0
    lapse_rate_sohem = np.array([8.1, 8.1, 7.7, 6.8, 5.5, 4.7, 4.4, 5.9, 7.1, 7.8, 8.1, 8.2]) / 1000.0
    lapse_rate = lapse_rate_sohem[1]

    t0_cube = (agera_cube.merge_cubes(geopotential_cube)
               .apply_dimension(dimension="bands",
                                process=lambda x: preprocess_low_resolution_agera(
                                    x, lapse_rate, temp_index="temperature-mean",
                                    dewpoint_index=1, temp_scale=0.01))
               .rename_labels(target=["t0", "dewpoint-temperature"], dimension="bands"))
    
    downscale_inputs = (t0_cube.resample_cube_spatial(elevation_cube, method="bilinear")
                        .merge_cubes(elevation_cube.max_time()))

    downscaled = downscale_inputs.apply_dimension(
        dimension="bands",
        process=lambda x: downscale_t_dewpoint(x, lapse_rate, temp_index="t0", dem_index="DEM")
    )
    return downscaled.rename_labels(dimension="bands", target=["temperature_downscaled", "relative_humidity"])


def downscale_shortwave_radiation(sw: DataCube,  slope_aspect: DataCube):
    """
    Downscale shortwave radiation using solar incidence angle correction.
    
    Requires slope and aspect in radians and computes topographic correction
    based on solar incidence angle.
    
    Args:
        sw: AGERA5 shortwave radiation DataCube (J/m^2/day) at DEM resolution
        slope_aspect: DataCube with slope and aspect bands in radians
        
    Returns:
        DataCube with topographically corrected shortwave radiation

    """
    solar_flux = sw.filter_bands(["solar-radiation-flux"])
    
    # Scale to MJ/m^2 if needed
    solar_flux = solar_flux / 1000000  

    # NOW convert from MJ/m²/day to W/m²
    # 1 MJ/m²/day = 1e6 J/m² per day = 1e6 / 86400 W/m² ≈ 11.574 W/m²
    solar_flux = solar_flux* 11.574
    compute_solarposition = UDF.from_file(str(SOLAR_POSITION_UDF))
    
    solar_flux_with_sunpos = solar_flux.apply_dimension(dimension="bands", process=compute_solarposition)

    solar_flux_with_sunpos = solar_flux_with_sunpos.rename_labels(dimension="bands", target= ["solar-radiation-flux", "zenith", "azimuth"])

    compute_incidence = UDF.from_file(str(INCIDENCE_ANGLE_UDF))
    shortwave_radiation_downscaled = (solar_flux_with_sunpos
                                .merge_cubes(slope_aspect)
                                .apply_dimension(dimension="bands", process=compute_incidence))

    # Ensure output is properly labeled
    shortwave_radiation_downscaled = shortwave_radiation_downscaled.rename_labels(
        dimension="bands", target=["shortwave-radiation-flux-downscaled"]
    )
    
    return shortwave_radiation_downscaled


def downscale_precipitation(precip_cube, dem_cube, geopotential_cube, gamma):
    """
    Downscale precipitation using a constant gamma. #TODO expand for dyanamic gamma for multiple months
    
    Args:
        precip_cube: Precipitation cube in meters (after scaling)
        dem_cube: DEM cube (single band, e.g., "DEM")
        geopotential_cube: Geopotential cube with band "geopotential"
        gamma: Gamma value in m^-1 (e.g., june = 0.00025)
    
    Returns:
        DataCube with downscaled precipitation in mm/day
    """
    # Convert geopotential to elevation (m)
    elev_coarse = geopotential_cube.band("geopotential") / 9.81
    
    # Resample precipitation and coarse elevation to DEM grid
    precip_resampled = precip_cube.resample_cube_spatial(dem_cube, method="bilinear")
    elev_coarse_resampled = elev_coarse.resample_cube_spatial(dem_cube, method="bilinear")
    
    
    # Compute elevation difference
    dz = dem_cube - elev_coarse_resampled
    
    # Compute scaling factor: (1 + gamma * dz) / (1 + abs(gamma * dz))
    # Use process builder expressions
    gamma_dz = dz * gamma
    numerator = 1 + gamma_dz
    denominator = 1 + abs(gamma_dz)
    factor = numerator / denominator
    
    # Apply factor to precipitation
    precip_downscaled = precip_resampled * factor
    
    # Convert to mm/day
    precip_mm = precip_downscaled * 1000
    
    # Rename band
    precip_mm = precip_mm.rename_labels(dimension="bands", target=["precipitation_downscaled"])
    
    return precip_mm

    