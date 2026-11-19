# upscaling — run over large areas

Splits a large AOI into tiles and manages one openEO batch job per tile with
`openeo.extra.job_management.MultiBackendJobManager`.

| File | Role |
|------|------|
| `run_upscaling.py` | CLI entry point. |
| `job_manager.py` | Tile grid, job database, job submission. |
| `pipeline.py` | Per-tile conditional probability graph (wraps `openeo_mountains_snow.snow_cover_area.conditional_probability`). |
| `config.py` | Tiling, paths, job options (reuses `openeo_mountains_snow.config`). |
| `local_auth.py` | Local authentication helper (git-ignored). |

Not part of the installed package. **Run from the repo root:**
`python -m extras.upscaling.run_upscaling [--preview] [--db-path PATH]`
