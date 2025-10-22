import json
from pathlib import Path

import pytest
import xarray


def test_elevation_mask(elevation_mask):

    print(elevation_mask)

def test_udf_local(local_cube):
    from openeo_mountains_snow.representative_pixels import apply_datacube

    cube = xarray.open_dataset(local_cube)
    crs = cube.crs
    cube = cube.drop_vars("crs")


    input_da = cube.isel(t=1).to_dataarray(dim="bands")

    result = apply_datacube(input_da,{"classify":True})
    ds = result.to_dataset(dim="bands")
    ds = ds.assign(crs=crs)
    ds = ds.rename(dict(B03='snowmask'))
    print(ds.snowmask.values[ds.snowmask.values != 0].shape)
    #print(ds)
    ds.to_netcdf("snow_area1.nc")


def test_training(cdse_staging_connection):

    cdse_staging_connection.load_stac_from_job("j-251007173945406ba98784b7b3c5bf83").raster_to_vector().download("points.geojson")