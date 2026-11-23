# openeo_mountains_snow
Snow processing in the mountains based on openEO.

**New here? Start with the [tutorial notebook](tutorial.ipynb).** It runs the whole chain on CDSE
(Sentinel-2 snow cover fraction → snow cover area → snow water equivalent) for a small Andean area,
shows every intermediate product and lists the caveats compared to the original method.

Full method docs: [docs/](docs/index.md) (Quarto).

Project history: [CHANGELOG.md](CHANGELOG.md). Technical before/after details:
[technical fixes report](docs/technical_fixes_report.md).

## Pipeline

```mermaid
flowchart LR
    S2[Sentinel-2] --> SCF["1. snow_cover_fraction/<br/>snow cover fraction"]
    SCF -- "binarized: snow_map_from_snow_cover_fraction" --> SCA["2. snow_cover_area/<br/>daily snow cover area"]
    MODIS[MODIS] --> SCA
    AGERA[AGERA5 + DEM] --> SWE["3. snow_water_equivalent/<br/>snow water equivalent"]
    SCA --> SWE
```

| Term | Meaning |
|------|---------|
| SCF  | Snow cover fraction, 0–100 % per pixel |
| SCA  | Snow cover area, binary snow / no snow |
| SWE  | Snow water equivalent |
| CP   | Conditional probability (MODIS SCF → high-res snow) |

## Install

```bash
pip install -e .[dev]
```

## Run

The [tutorial](tutorial.ipynb) runs each step as a separate job. The commands below run them as scripts.

| Step | Command | Config |
|------|---------|--------|
| SCF | `python -m openeo_mountains_snow.snow_cover_fraction.snow_cover_fraction +experiment=andes_snow_cover_fraction` | `conf/` (Hydra) |
| SCF UDP (writes JSON) | `python -m openeo_mountains_snow.snow_cover_fraction.udp.snow_cover_fraction_udp` | `conf/` (Hydra) |
| SCA reconstruction + SWE | `python -m openeo_mountains_snow.main` | `config.py` |
| Upscaling (extra, from repo root) | `python -m extras.upscaling.run_upscaling [--preview]` | `extras/upscaling/config.py` |
| Tests | `pytest` | |

## Layout

```
src/openeo_mountains_snow/
├── conf/            Hydra configs (config.yaml + experiment/*.yaml)
├── snow_cover_fraction/             Step 1: snow cover fraction
│   ├── udfs/        UDFs run on the backend
│   └── udp/         UDP builder + published JSON
├── config.py        Shared settings for main.py (and extras/upscaling)
├── main.py          Runs snow cover area -> snow water equivalent as one batch job
├── common/          Shared masks, sun and terrain helpers (+ udfs/)
├── snow_cover_area/ Step 2: daily snow cover area reconstruction (+ conditional probability)
│   └── udfs/
├── snow_water_equivalent/             Step 3: SWE + meteo downscaling
│   └── udfs/
└── senales_wgs84.geojson   Default AOI
extras/              Not installed; run from the repo root
├── upscaling/       Tile a large AOI into many batch jobs
└── landsat/         Landsat SCF variant (work in progress)
tutorial.ipynb       Entry point: step-by-step tutorial of the full chain
scripts/             export_omnicloudmask_onnx.py
docs/                Method documentation (Quarto)
test/                Tests
```

## Code rules

- **Client code** (`*.py` in a step folder) builds the openEO process graph locally.
- **UDFs** (`*/udfs/*.py`) are read as text and executed on the backend. Keep them
  self-contained: only import what the backend UDF runtime provides.
- **UDPs** (`*/udp/`) are generated JSON process definitions; regenerate, don't hand-edit.
