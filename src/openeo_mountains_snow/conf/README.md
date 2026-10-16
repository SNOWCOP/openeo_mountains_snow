# conf

Hydra configuration for pipeline entry points.

## Contents
- config.yaml: shared defaults and backend settings
- experiment/: run presets (for example andes_area01, andes_scf)

## Used by
- snow_cover_fraction/snow_cover_fraction.py
- snow_cover_fraction/udp/snow_cover_fraction_udp.py
- experimental/snow_corsa_ml_inference.py

## Typical usage
Run an entry point with an experiment override:

```bash
python -m openeo_mountains_snow.snow_cover_fraction.snow_cover_fraction +experiment=andes_snow_cover_fraction
```
