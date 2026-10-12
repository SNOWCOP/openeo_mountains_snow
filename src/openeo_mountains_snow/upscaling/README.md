# upscaling — run over large areas

Splits a large AOI into tiles and manages one openEO batch job per tile with
`openeo.extra.job_management.MultiBackendJobManager`.

| File | Role |
|------|------|
| `run_upscaling.py` | CLI entry point. |
| `job_manager.py` | Tile grid, job database, job submission. |
| `pipeline.py` | Per-tile process graph (mirrors `experimental/downscaling_distribution.py`). |
| `config.py` | Tiling, paths, job options (reuses `reconstruction/config.py`). |
| `local_auth.py` | Local authentication helper. |

**Run:** `python -m openeo_mountains_snow.upscaling.run_upscaling [--preview] [--db-path PATH]`
