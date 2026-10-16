# snow_cover_area — Step 2: daily snow cover area

Reconstructs a cloud-free daily snow cover area (SCA: 0 no snow, 100 snow, 205 cloud) from
Sentinel-2 snow maps, MODIS snow cover fraction and historical conditional probabilities.

| File | Role |
|------|------|
| `snow_cover_area.py` | Entry point: `snow_cover_area_cube()` builds the full SCA graph. |
| `conditional_probability.py` | SCF-range masks and P(snow \| MODIS SCF range). Also used by `upscaling/pipeline.py`. |
| `snow_map.py` | High-resolution 0/100/205 snow map: from the step-1 representative-pixels SCF (`snow_map_from_snow_cover_fraction`, main pipeline) or from Sentinel-2 SCL (`calculate_snow`, upscaling). |
| `udfs/snow_cover_area_reconstruction_udf.py` | Backend UDF: fills clouds via similar historical days, then via conditional probability. |

- **Config:** `../config.py`.
- **Run:** as part of `python -m openeo_mountains_snow.main`.
- **Method docs:** `docs/timeseries_reconstruction.qmd`.
