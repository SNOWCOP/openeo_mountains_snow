# Changelog

## Unreleased

### Added

- End-to-end tutorial for the Sentinel-2 snow cover fraction, snow cover area, and snow water equivalent chain on CDSE.
- Shared OmniCloudMask, water, glacier, shadow, solar-position, and terrain-processing modules.
- Hemisphere-aware AGERA5 temperature downscaling using DEM and geopotential data.
- Daily SCA reconstruction using MODIS snow cover fraction and historical conditional probabilities.
- Focused tests for representative pixels, solar angles, incidence angles, temperature, SCA, and SWE reconstruction.

### Changed

- The representative-pixel workflow now uses reflectance inputs, scene-aware training, a derived AOI neighborhood, and a hosted glacier mask.
- SCA reconstruction now supports the complete selected time window and uses explicit low-resolution/edge overlap handling.
- SWE now consumes separate temperature, precipitation, and topographically corrected shortwave-radiation bands.
- Validation and upscaling material was reorganized for clearer repository entry points.
- Processing constants and UDF paths were centralized in `config.py`.

### Fixed

- Solar-position and local-incidence-angle calculations now use per-pixel terrain geometry and validated overpass angles.
- Thick-cloud masking is dilated consistently and cloud shadows are included in scene shadow detection.
- MODIS snow-fraction flags are cleaned separately from valid 0–100 snow fractions.
- Shortwave radiation is converted to W/m² and resampled onto the DEM grid before incidence correction.
- UDF state handling was improved for accumulation, melt, cloud, no-snow, and no-data pixels.

### Known limitations

- The reconstruction uses the selected window rather than a complete multi-year hydrological archive.
- Pre-window snow accumulation is attributed to accumulation days inside the selected window.
- The original GAM-based final SCA filling stage is not ported; unresolved pixels remain 205 and SWE remains NaN.
- The hosted glacier mask covers the supported Andes area; it can be disabled outside that area.

### Detailed report

See [Technical fixes report](docs/technical_fixes_report.md) for exact before-versus-now code comparisons.
