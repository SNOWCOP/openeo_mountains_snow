# scf — Step 1: snow cover fraction

Computes snow cover fraction (0–100 %) from Sentinel-2 using representative pixels.
The Landsat variant (work in progress) lives in `extras/landsat/`.

| File | Role |
|------|------|
| `snow_cover_fraction.py` | Client code. `snow_cover_fraction_cube()` builds the cube; `run_openeo()` is the Hydra entry point. |
| `udfs/representative_pixels.py` | UDF run on the backend. |
| `udp/snow_cover_fraction_udp.py` | Builds the UDP and writes `udp/sentinel2_snow_cover_fraction.json`. |

- **Config:** `../conf/config.yaml` + `../conf/experiment/*.yaml` (Hydra).
- **AOI default:** `../senales_wgs84.geojson`.
- **Run:** `python -m openeo_mountains_snow.snow_cover_fraction.snow_cover_fraction +experiment=andes_snow_cover_fraction`
