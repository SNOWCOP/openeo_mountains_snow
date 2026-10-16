# swe — Step 3: snow water equivalent

Downscales AGERA5 meteo (temperature, humidity, shortwave radiation) to the DEM and runs the SWE model
on the reconstructed snow cover.

| File | Role |
|------|------|
| `downscale_variables.py` | Client code: lapse-rate and topographic downscaling. |
| `udfs/snow_water_equivalent_udf.py` | SWE model UDF. |
| `udfs/solar_position_udf.py` | Solar zenith/azimuth per pixel and timestamp. |
| `udfs/incidence_angle_udf.py` | Solar incidence angle on sloped terrain. |

- **Entry point:** `snow_water_equivalent.py` → `snow_water_equivalent_cube()`.
- **Config:** `../config.py` (UDF paths and extents live there).
- **Run:** as part of `python -m openeo_mountains_snow.main`.
