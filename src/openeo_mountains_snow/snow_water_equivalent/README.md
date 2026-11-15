# swe — Step 3: snow water equivalent

Downscales AGERA5 meteo (temperature, humidity, shortwave radiation) to the DEM and runs the SWE model
on the reconstructed snow cover.

| File | Role |
|------|------|
| `downscale_variables.py` | Client code: lapse-rate and topographic downscaling. |
| `udfs/snow_water_equivalent_udf.py` | SWE model UDF. |

Solar position and incidence angle UDFs are shared with SCF and live in `../common/udfs/`.

- **Entry point:** `snow_water_equivalent.py` → `snow_water_equivalent_cube()`.
- **Config:** `../config.py` (UDF paths and extents live there).
- **Run:** as part of `python -m openeo_mountains_snow.main`.
