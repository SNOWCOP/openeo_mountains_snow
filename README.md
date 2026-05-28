# openeo_mountains_snow

Snow cover fraction estimation and historical snow water equivalent (SWE) reconstruction over mountainous regions using [openEO](https://openeo.org/).

## Overview

This project provides two pipelines, both driven by [Hydra](https://hydra.cc/) configuration:

1. **Snow Cover Fraction (SCF)** — spectral-index-based sub-pixel snow classification from Sentinel-2, using representative-pixel SVM training.
2. **Historical Reconstruction** — combines Sentinel-2 SCF, MODIS conditional probabilities, and downscaled AGERA5 climate data to reconstruct daily snow cover area and snow water equivalent.

## Project Structure

```
src/openeo_mountains_snow/
├── main.py                        # Hydra entry point (dispatches to pipelines)
├── reconstruction_pipeline.py     # Full SCF → SCA → SWE orchestrator
├── spatial_extent_utils.py        # bbox / GeoJSON / CRS helpers
│
├── conf/                          # Hydra configuration
│   ├── config.yaml                #   base config (data sources, processing params)
│   └── experiment/                #   per-experiment overrides
│       ├── andes_scf.yaml
│       ├── andes_area01.yaml
│       └── reconstruction.yaml
│
├── scf/                           # Snow Cover Fraction (Sentinel-2)
│   ├── snow_cover_fraction.py     #   spectral indices + classification
│   ├── udfs/
│   │   └── representative_pixels.py  # SVM-based snow/no-snow classification
│   └── udp/                       #   openEO User Defined Process export
│       ├── scf_udp.py
│       └── sentinel2_snow_cover_fraction.json
│
├── sca/                           # Snow Cover Area reconstruction
│   ├── scf_processing.py          #   MODIS SCF masks, conditional probabilities
│   ├── utils.py                   #   create_mask, get_scf_ranges
│   └── udfs/
│       └── historical_reconstruction_udf.py
│
└── swe/                           # Snow Water Equivalent
    ├── downscale_variables.py     #   climate loading & downscaling (T, RH, SW)
    └── udfs/                      #   server-side UDFs
        ├── solar_position_udf.py
        ├── incidence_angle_udf.py
        └── swe_udf.py

# Repo root (outside the package)
data/                              # Static data files (default AOI GeoJSON)
legacy/                            # Deprecated modules kept for reference
scripts/                           # Standalone scripts (not part of library)
notebooks/                         # Interactive exploration notebooks
test/                              # Unit & integration tests
```

## Quick Start

### Prerequisites

- Python ≥ 3.11
- Access to [Copernicus Data Space](https://openeo.dataspace.copernicus.eu/) (OIDC credentials)

### Installation

```bash
pip install -e ".[dev]"
```

### Running a Pipeline

```bash
# Snow cover fraction (Andes example)
python -m openeo_mountains_snow.main +experiment=andes_scf

# Full historical reconstruction (Senales)
python -m openeo_mountains_snow.main +experiment=reconstruction
```

## Pipelines

### Snow Cover Fraction (`scf`)

Computes sub-pixel snow cover fraction from Sentinel-2 L1C using:
- Spectral indices (NDVI, NDSI, blue-NIR difference, shadow index)
- Local solar incidence angle from DEM slope/aspect
- Representative-pixel SVM classification per solar angle range

**Entry:** `snow_cover_fraction.py` → `snow_cover_fraction_cube()`

### Historical Reconstruction (`reconstruction`)

Reconstructs daily snow cover and SWE by:

1. **SCF masks & conditional probabilities** (`sca/scf_processing.py`) — MODIS-derived P(snow | SCF range)
2. **High-resolution data** — Sentinel-2 SCF + MODIS SCF
3. **Historical reconstruction UDF** (`sca/udfs/`) — iterative cloud-gap filling using similar historical scenes
4. **Climate downscaling** (`swe/downscale_variables.py`) — lapse-rate correction for temperature/humidity, topographic correction for shortwave radiation
5. **SWE computation UDF** (`swe/udfs/swe_udf.py`) — degree-day model driven by downscaled climate

**Entry:** `reconstruction_pipeline.py` → `run_reconstruction()`

## Configuration

All parameters are in `conf/config.yaml` with per-experiment overrides in `conf/experiment/`. Key sections:

| Section | Contents |
|---------|----------|
| `connection` | openEO backend endpoint |
| `sentinel2_l2a` / `sentinel2_l1c` | S2 collection names and bands |
| `modis` | MODIS STAC URL |
| `agera5` | AGERA5 STAC URL and band mappings |
| `geopotential` | Geopotential STAC URL |
| `dem` | Slope/aspect STAC URLs |
| `processing` | CRS, resolution, cloud thresholds, SCF range parameters |
| `reconstruction` | Neighbourhood size, number of days to reconstruct |

## Data Sources

| Dataset | Resolution | Source |
|---------|-----------|--------|
| Sentinel-2 L1C/L2A | 10–20 m | Copernicus Data Space |
| MODIS MOD10A1 v61 | 500 m | Eurac STAC |
| AGERA5 | ~10 km | VITO STAC |
| Copernicus DEM | 30 m | Copernicus Data Space |
| Geopotential | ~0.25° | VITO STAC |
| DEM slope/aspect | 30 m | VITO STAC |

## Dependencies

Core: `openeo`, `hydra-core`, `omegaconf`, `xarray`, `numpy`, `scipy`, `scikit-learn`, `shapely`, `pyproj`

Server-side UDFs additionally require: `pvlib` (declared via PEP 723 inline metadata)

## License

See [LICENSE](LICENSE).

## References

- [openEO Documentation](https://openeo.org/)
- [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/)