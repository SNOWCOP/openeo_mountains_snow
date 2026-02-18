import importlib.resources

from openeo import DataCube, UDF, MultiResult
from openeo.processes import ProcessBuilder, array_create, exp, clip
import numpy as np

vp_coeff_nohem = np.array([0.41, 0.42, 0.40, 0.39, 0.38, 0.36, 0.33, 0.33, 0.36, 0.37, 0.40, 0.40]) / 1000.0
vp_coeff_sohem = np.array([0.38, 0.36, 0.33, 0.33, 0.36, 0.37, 0.40, 0.40, 0.41, 0.42, 0.40, 0.39]) / 1000.0
a, b, c = 611.21, 17.502, 240.97 # Magnus formula constants

def preprocess_low_resolution_agera(cube: ProcessBuilder, lapse_rate, temp_index="temperature-mean", dewpoint_index=None, temp_scale = 1.0) -> ProcessBuilder:
    """
    cube contains:
    "temperature"
    "geopotential"
    """

    t_raw = cube[temp_index] * temp_scale
    geopotential = 35885.0 /  cube["geopotential"]
    t_0 = t_raw - lapse_rate * (0 - geopotential)

    if (dewpoint_index is not None):
        d_0 = cube[dewpoint_index] *0.01 - lapse_rate * (0 - geopotential)
        return array_create([t_0, d_0])
    else:
        return t_0


def downscale_t_dewpoint(cube: ProcessBuilder, lapse_rate, temp_index="temperature-mean", dem_index="DEM") -> ProcessBuilder:
    """
    cube contains:
    "temperature"
    "elevation"
    """

    temperature_downscaled = cube[temp_index] - lapse_rate * (cube[dem_index] - 0)
    rh =  relative_humidity_formula(temperature_downscaled - 273.15, cube["dewpoint-temperature"], cube[dem_index], 2)
    return array_create([temperature_downscaled, rh])

def relative_humidity_formula( temperature_downscaled, dewpoint_temperature_coarse, elevation, month_index):
    # TODO select vp_coeff based on hemisphere
    vp_coeff_all = vp_coeff_sohem
    vp_coeff = vp_coeff_all[month_index]
    d_t_lapse_rate = vp_coeff * c / b


    D_down = dewpoint_temperature_coarse - d_t_lapse_rate * (elevation - 0) - 273.15
    es = a * exp((b * temperature_downscaled) / (temperature_downscaled + c))
    e = a * exp((b * D_down) / (D_down + c))
    return clip(100 * e / es, 0, 100)



def downscale_temperature_humidity(agera_cube, elevation_cube, geopotential_cube):
    lapse_rate_nohem = np.array([4.4, 5.9, 7.1, 7.8, 8.1, 8.2, 8.1, 8.1, 7.7, 6.8, 5.5, 4.7]) / 1000.0
    lapse_rate_sohem = np.array([8.1, 8.1, 7.7, 6.8, 5.5, 4.7, 4.4, 5.9, 7.1, 7.8, 8.1, 8.2]) / 1000.0
    lapse_rate = lapse_rate_sohem[1]

    t0_cube = agera_cube.merge_cubes(geopotential_cube).apply_dimension(dimension="bands", process=lambda x: preprocess_low_resolution_agera(x, lapse_rate, temp_index="temperature-mean",dewpoint_index=1, temp_scale= 0.01))\
        .rename_labels(target=["t0","dewpoint-temperature"], dimension="bands")
    downscale_inputs = t0_cube.resample_cube_spatial(elevation_cube,method="bilinear").merge_cubes(elevation_cube.max_time())

    downscaled =  downscale_inputs.reduce_dimension(dimension="bands", reducer=lambda x: downscale_t_dewpoint(x, lapse_rate, temp_index="t0", dem_index="DEM"))
    return MultiResult([
        downscaled.save_result("netCDF", dict(filename_prefix="downscaled_"))
        #t0_cube.save_result("GTIFF", dict(filename_prefix="t0_"))
        #downscale_inputs.save_result("netCDF", dict(filename_prefix="downscale_inputs_"))
    ])


def get_udf(name):
    return UDF( code= importlib.resources.read_text('meteo_downscaling_openeo', name), runtime="Python", version="3.11")


def downscale_shortwave_radiation(agera: DataCube, slope_aspect:DataCube):
    """

    requires slope and aspect in radians

    requires computation of solar incidence angle
    """



    compute_solarposition = get_udf('solar_position_udf.py')

    agera_with_sunpos = agera.apply_dimension(dimension="bands", process=compute_solarposition)

    return agera_with_sunpos

    compute_incidence = get_udf("incidence_angle_udf.py")

    radiation_with_incidence = (agera_with_sunpos.resample_cube_spatial(slope_aspect).merge_cubes(slope_aspect)
     .apply_dimension(dimension="bands", process=compute_incidence))

    def downscale_shortwave(radiation_incidence: ProcessBuilder) -> ProcessBuilder:
        ssrd = radiation_incidence["shortwave"]
        incidence = radiation_incidence["incidence_angle"]
        zenith = radiation_incidence["zenith"]


        cos_i = np.clip(np.cos(np.radians(incidence)), 0, 1)
        cosZ = np.cos(zenith)
        # Topographic correction
        Qsi_daily = ssrd * (cos_i / (cosZ + 1e-6))
        return Qsi_daily

    return radiation_with_incidence.apply_dimension(dimension="bands", process=downscale_shortwave)