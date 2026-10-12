# experimental

Prototype scripts not part of the main production pipeline.

## Current file
- snow_corsa_ml_inference.py: experimental ML inference flow for snow analysis

## Notes
- Keep imports explicit and avoid relying on this folder from production modules.
- Move stable logic into snow_cover_fraction/, reconstruction/, snow_water_equivalent/ or upscaling/.
