from pathlib import Path

import openeo
import pytest
import xarray

from openeo_mountains_snow.snowcoverarea_reconstruction.downscale_variables import downscale_shortwave_radiation


@pytest.fixture
def openeoplatform_connection() -> openeo.Connection:
    return openeo.connect("openeo-dev.vito.be").authenticate_oidc()

spatial_extent = {
        "south": 5816500,
        "north": 5816500 + 128*30,
        "west": 271000,
        "east": 271000 + 128*30,
        "crs": "EPSG:32719"
}

temporal_extent = "2025-07"



def test_shortwave_radiation(openeoplatform_connection):

    agera = openeoplatform_connection.load_collection(
        "AGERA5",
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        bands=["solar-radiation-flux"]
    )

    #agera = agera.filter_bands(bands=["solar-radiation-flux"])
    agera = agera.rename_labels(dimension="bands", target=[ "solar-radiation-flux"])

    dem_spacetime = openeoplatform_connection.load_collection("COPERNICUS_30", spatial_extent=spatial_extent)
    dem = dem_spacetime.reduce_dimension(dimension='t', reducer='mean')

    aspect = dem.aspect()
    slope = dem.slope()

    slope_aspect = aspect.merge_cubes(slope).rename_labels(
        dimension="bands", target=["aspect", "slope"]
    )
    agera = agera.resample_cube_spatial(dem_spacetime)
    shortwave_rad_cube = downscale_shortwave_radiation(agera, slope_aspect)
    shortwave_rad_cube.execute_batch("shortwave_rad_downscaled.nc", title="shortwave radiation test", job_options={"executor-memory":"5G",  "executor-memoryOverhead": "5G"})

def test_shortwave_incidence_udf():
    import xarray
    inputs = xarray.load_dataset("shortwave_rad_input.nc")
    inputs = inputs.drop_vars(["crs","unkown_band_5"])
    array = inputs.to_array(dim="bands").astype("float")

    from openeo_mountains_snow.snowcoverarea_reconstruction.udfs.incidence_angle_udf import apply_datacube
    result = apply_datacube(array, None)
    angle = result.sel(bands="incidence_angle").isel(t=0)
    print(angle)
    #plot angle
    import matplotlib.pyplot as plt
    plt.imshow(angle, vmin=0, vmax=90)
    plt.colorbar(label="Incidence Angle (degrees)")
    plt.title("Solar Incidence Angle")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig("incidence_angle.png")
    


