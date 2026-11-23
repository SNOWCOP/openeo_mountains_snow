# Technical Fixes Report

**Branch:** `tutorial-launch-comparison`  
**Comparison:** `main...HEAD`  
**Scope:** 52 changed files, +3074 / -1911 lines

## Executive summary

The branch evolved from several partly connected experiments into a reproducible three-step snow chain:

```text
main before:
separate prototype code and notebooks
→ representative-pixel experiment
→ limited SCA reconstruction
→ temperature/humidity/SWE prototype

now:
shared openEO masking and terrain modules
→ tested SnowFLAKES representative-pixel SCF
→ daily SCA reconstruction with MODIS and historical probabilities
→ DEM-downscaled AGERA5 meteorology and SWE reconstruction
→ launch-only tutorial with independent validation steps
```

## 1. Snow-cover fraction: cleaner inputs and scene-aware training

### Before

The SCF module contained masking, STAC solar-angle lookup, slope/aspect, incidence-angle, and S2 feature building in the same file. Sentinel-2 inputs were not normalized for the SnowFLAKES reflectance thresholds.

```python
# Before: raw Sentinel-2 data was masked, but not converted to reflectance
masked_s2 = sentinel2_bands.mask(mask | dem_mask)

s2_with_local_angle = local_incidence_angle(
    masked_s2, aoi, time_period, connection, cfg
)
```

The neighborhood was fixed:

```python
# Before
snow_cover_fraction_cube(..., neighborhood_size: int = 3038)

apply_representative_pixels(..., neighborhood_size=3038)
```

### Now

The SCF module reuses shared masking and terrain modules, normalizes L1C DN values to reflectance, adds the glacier band, and derives a chunk size from the AOI.

```python
# Now: reflectance normalization
names = sentinel2_bands.metadata.band_names
sentinel2_bands = sentinel2_bands.apply_dimension(
    dimension="bands",
    process=lambda x: array_create([
        x[i] / 10000 if n.startswith("B") else x[i]
        for i, n in enumerate(names)
    ]),
).rename_labels(dimension="bands", target=names)
```

```python
# Now: reuse OmniCloudMask, avoid duplicate inference, and add glacier/shadow bands
classes = None
if cfg.omnicloudmask.model_url is not None:
    classes = omnicloudmask_classes(aoi, time_period, connection, cfg)

mask = cloud_water_mask(
    aoi, time_period, connection, cfg, classes=classes
)

cloud_shadow = None if classes is None else classes.apply(lambda x: x == 3)
inputs = add_indices(
    shadow_mask(
        s2_with_local_angle,
        aoi,
        cloud_shadow=cloud_shadow,
    )
)

glacier = glacier_mask(aoi, connection, cfg)
if glacier is not None:
    inputs = inputs.merge_cubes(
        glacier.add_dimension(
            name="bands", label="glacier", type="bands"
        )
    )
```

```python
# Now: neighborhood covers the full AOI and is padded to the backend tile grid
def aoi_neighborhood_size(aoi, resolution: float = 10, tile: int = 128) -> int:
    x0, y0, x1, y1 = aoi_shape(aoi).bounds
    geod = Geod(ellps="WGS84")
    width = max(geod.inv(x0, y0, x1, y0)[2], geod.inv(x0, y1, x1, y1)[2])
    height = geod.inv(x0, y0, x0, y1)[2]
    return (math.ceil(max(width, height) / resolution / tile) + 2) * tile
```

**Impact:** training receives reflectance features and the whole AOI, while cloud, water, shadow, and glacier pixels are excluded or represented explicitly.

## 2. Masking and terrain: shared, reusable implementations

### Before

Cloud, water, glacier, slope/aspect, and incidence-angle logic lived in or around the SCF prototype. They were not consistently available to other pipeline stages.

Representative-pixel reduction also used a shape that could differ between fallback and OmniCloudMask paths:

```python
# Before
cloud_mask = scl.reduce_dimension(
    dimension="bands",
    reducer=lambda x: any([
        x == cloud_value
        for cloud_value in cfg.sentinel2_l2a.cloud_values
    ]),
)
```

### Now

A shared module provides:

```python
# src/openeo_mountains_snow/common/masks.py
predicted = rgn.process(
    "predict_onnx",
    data=rgn,
    model=cfg.omnicloudmask.model_url,
)
classes = predicted.reduce_dimension(
    dimension="bands",
    reducer=lambda x: x.array_element(0),
)

# 0 clear, 1 thick cloud, 2 thin cloud, 3 shadow, 255 nodata
cloud_mask = classes.apply(lambda x: (x == 1).or_(x == 2))
water_mask = water.reduce_dimension(
    dimension="bands",
    reducer=lambda x: any([
        x[0] == water_value
        for water_value in cfg.water_mask.water_values
    ]),
)
```

Glaciers are optional, configured, and reusable:

```python
def glacier_mask(region, conn, cfg):
    if cfg.glacier_mask.stac_url is None:
        return None

    glacier = conn.load_stac(
        cfg.glacier_mask.stac_url,
        spatial_extent=region,
        bands=[cfg.glacier_mask.band],
    )
    return glacier.max_time().reduce_dimension(
        dimension="bands",
        reducer=lambda x: x[0] == 1,
    )
```

The representative-pixel UDF now explicitly excludes glaciers from shadow training:

```python
# Now
glacier = (
    cube.sel(bands="glacier").values == 1
    if "glacier" in cube.bands.values
    else np.False_
)
eligible_shadow = (
    np.isfinite(shadow_mask)
    & (shadow_mask != 0)
    & np.isfinite(green)
    & np.isfinite(diff_B_NIR)
    & ~glacier
    & valid_sun
)
```

**Impact:** masks are configurable and shared, thick clouds are dilated, and glacier pixels cannot contaminate shadow training.

## 3. Solar position and incidence angle

### Before

Solar correction depended more heavily on scene-level metadata or synthetic/less validated calculations. The old local-incidence implementation converted DEM slope/aspect directly to radians.

### Now

The solar UDF uses a reference Sentinel-2 tile point, finds the overpass time matching Sentinel-2 zenith/azimuth angles, and uses a geometric NREL SPA calculation for every pixel:

```python
def best(seconds):
    zen, azi = _spa(seconds, ref_lat, ref_lon, date)
    if sun is None:
        miss = zen
    else:
        z, a, (z0, a0) = np.radians(zen), np.radians(azi), np.radians(sun)
        miss = -(
            np.cos(z) * np.cos(z0)
            + np.sin(z) * np.sin(z0) * np.cos(a - a0)
        )
    return seconds[np.argmin(miss)]

zenith, azimuth = _spa(np.array([t]), lat, lon, date)
```

The resulting angles are converted to radians and combined with slope/aspect:

```python
cos_theta = (
    cos(zenith_rad) * cos(slope_rad)
    + sin(zenith_rad)
    * sin(slope_rad)
    * cos(aspect_rad - azimuth_rad)
)
local_angle = arccos(cos_theta.max(-1).min(1)) * rad2deg
```

**Impact:** per-pixel terrain incidence is calculated consistently instead of using a scene-only approximation. Tests now compare the UDF against XML reference values.

## 4. SCA reconstruction: from iterative prototype to ported reconstruction stages

### Before

The old UDF reconstructed a limited number of days through a large iterative function, held historical maps separately, and returned reconstructed daily output only.

```python
n_days = context.get("n_days_to_reconstruct", 10)
n_days_actual = min(n_days, total_days - 1)

reconstructed = hist_rec_iterative(
    snow_map=snow_map,
    scf_map=scf_map,
    hist_snow=historical_snow,
    hist_cp_maps=historical_cp_maps,
    hist_occ_maps=historical_occ_maps,
    scf_ranges=SCF_RANGES,
    day_idx=day_idx,
)
```

### Now

The pipeline separates reusable input assembly from reconstruction:

```python
def conditional_probability_inputs(...):
    all_masks, labels = compute_snow_cover_fraction_range_masks(...)
    conditional_probability, _, occurrences = \
        compute_conditional_probability(all_masks, labels)
    return ...  # cp_* and occ_* bands


def snow_cover_area_inputs(...):
    cp_occ = conditional_probability_inputs(...)
    modis_snow_cover_fraction = \
        create_modis_snow_cover_fraction_cube(...)

    return (
        high_resolution_snow
        .merge_cubes(modis_snow_cover_fraction)
        .merge_cubes(cp_occ.add_dimension(...))
    )
```

The reconstruction UDF now maps 500 m low-resolution pixels to high-resolution labels and performs vectorized block calculations:

```python
def lr_labels(y, x, lr_size):
    iy = np.unique(np.floor(np.asarray(y) / lr_size), return_inverse=True)[1]
    ix = np.unique(np.floor(np.asarray(x) / lr_size), return_inverse=True)[1]
    nbx = ix.max() + 1
    return iy[:, None] * nbx + ix[None, :], int((iy.max() + 1) * nbx)


def scf(snow, labels, n, nv_thres, mode=None):
    size = np.bincount(labels.ravel(), minlength=n).astype(float)
    valid = snow <= SNOW
    total = block_sum(np.where(valid, snow, 0), labels, n)
    nv = size - block_sum(valid, labels, n)

    if mode == "min":
        return total / size
    if mode == "max":
        return (total + nv * SNOW) / size

    partial = np.where(nv < nv_thres, total / (size - nv), NO_DATA)
    return np.where(nv == 0, total / size, partial)
```

Cloud filling uses historical scene similarity followed by conditional probability:

```python
cloud = snow == CLOUD
rec = reconstruct_hr(snow, nodata, history, p)
snow[cloud] = rec[cloud]

for i, (lo, hi) in enumerate(SCF_RANGES):
    in_range = ((lr2hr >= lo) if i == 0 else (lr2hr > lo)) & (lr2hr <= hi)
    fill = (
        in_range
        & (out == NO_DATA)
        & ((cp[i] == 0) | (cp[i] == 1))
        & (occ[i] > p["cp_occur_thres"])
        & (snow > SNOW)
    )
    out[fill] = cp[i][fill] * SNOW
```

Clouds inside the available time series are also linearly interpolated:

```python
inside = cloudy & (t > t[ok][0]) & (t < t[ok][-1])
out[inside, j] = np.interp(t[inside], t[ok], out[ok, j])
```

The branch reconstructs all available days instead of defaulting to five:

```python
N_DAYS_TO_RECONSTRUCT = None  # every day in the cube
MODIS_RESOLUTION = 500
SCA_NEIGHBORHOOD_OVERLAP = PIXEL_RATIO
SNOW_FRACTION_THRESHOLD = 15
```

**Impact:** reconstruction is split into inspectable stages, uses larger neighborhood context, and preserves the full selected time window.

## 5. SWE: explicit meteorological downscaling and accumulation model

### Before

SWE downscaling removed temperature and dewpoint/humidity rather than producing a complete, clearly named meteo cube:

```python
# Before: one function combined temperature and humidity
temperature_humidity = downscale_temperature_humidity(
    agera,
    dem,
    geopotential.max_time(),
)

swe = (
    snow_cover_area
    .merge_cubes(temperature_humidity)
    .merge_cubes(shortwave_radiation)
)
```

The coarse temperature scale and reference height were corrected inconsistently, and the shortwave result was not always resampled explicitly onto the DEM grid.

### Now

Temperature, precipitation, and shortwave are built into a named meteo cube:

```python
# Now
precipitation = (
    agera.filter_bands(["total_precipitation"])
    * 0.01
).rename_labels(dimension="bands", target=["precipitation"])

temperature = downscale_temperature(
    agera,
    dem,
    geopotential,
    hemisphere=extent_hemisphere(spatial_extent),
)

shortwave_radiation = downscale_shortwave_radiation(
    agera,
    slope_aspect,
)

return (
    temperature
    .merge_cubes(precipitation)
    .merge_cubes(shortwave_radiation)
)
```

Temperature now uses geopotential height and hemisphere/month lapse rates:

```python
def downscale_temperature(
    agera_cube,
    elevation_cube,
    geopotential_cube,
    hemisphere: str,
):
    temperature = agera_cube.filter_bands(["temperature-mean"]) * 0.01
    height = geopotential_cube / 9.80665

    inputs = (
        temperature.merge_cubes(height)
        .resample_cube_spatial(elevation_cube, method="bilinear")
        .merge_cubes(elevation_cube.max_time())
    )

    udf = UDF.from_file(
        str(TEMPERATURE_DOWNSCALING_UDF),
        context={"hemisphere": hemisphere},
    )
    return inputs.apply_neighborhood(...)
```

```python
# temperature_downscaling_udf.py
out = temperature - 273.15 + lapse_rate * (height - dem)
```

Shortwave units and topology were corrected:

```python
# Before
solar_flux = solar_flux / 1000000

# Now
solar_flux = sw.filter_bands(["solar-radiation-flux"])
solar_flux = solar_flux / 86400  # J/m²/day -> W/m²
solar_flux = solar_flux.resample_cube_spatial(
    slope_aspect,
    method="bilinear",
)
```

The SWE UDF now distinguishes accumulation, melt, cloud, no-data, melt-out, and snow state explicitly:

```python
status = np.where(cold, 1, np.where(warm, 0, np.nan))

# Accumulation vs melt at a snow-covered pixel
swe[i + 1][acc] = (
    swe[i][acc]
    + dsca[acc] * tot_acc[i + 1][acc]
)
swe[i + 1][melting] = swe[i][melting] - melt[i][melting]

# Unknown or no-snow pixels are handled explicitly
swe[i + 1][sca[i + 1] > 100] = np.nan
swe[i + 1][sca[i + 1] == 0] = 0
swe[swe < 0] = 0
```

**Impact:** SWE consumes a complete downscaled meteo cube, has hemisphere-aware temperature downscaling, correct shortwave units, and basin-SCA context for accumulation.

## 6. Tutorial and job workflow

### Before

Multiple validation notebooks and prototype launch paths mixed setup, execution, intermediate products, and validation.

### Now

The root `tutorial.ipynb` is the single guided entry point. It:

- uses a 1.5 × 1.5 km Chilean glacier example;
- separates launch cells from inspection cells;
- exposes job objects such as `scf_job`, `sca_job`, `swe_job`, and `terrain_job`;
- supports reconnecting by `connection.job("j-...")`;
- explicitly documents the missing GAM and limited-window limitations.

The tutorial also uses separate job sizes for heavy, medium, and small workloads.

## 7. Validation coverage

Tests were added or strengthened for:

- representative-pixel selection and classification;
- incidence-angle calculation;
- solar-position/reference-angle behavior;
- temperature downscaling;
- SCA reconstruction; and
- SWE reconstruction.

The repository’s existing full pytest task was run successfully with exit code 0.

## Scope and remaining limitations

- The branch covers the selected six-month reconstruction window, not a complete multi-year hydrological archive.
- Pre-window accumulation is attributed to accumulation days inside the window.
- The original GAM final-filling stage is not ported.
- Remaining unresolved SCA pixels stay `205`; corresponding SWE pixels stay `NaN`.
- The glacier mask currently covers the hosted Andes region and is disabled outside that region.

## Bottom line

The fix is not cosmetic: the branch adds shared preprocessing, terrain-aware inputs, ported reconstruction logic, explicit meteo downscaling, and testable UDF stages. The implementation now represents a coherent openEO pipeline rather than several disconnected prototypes.
