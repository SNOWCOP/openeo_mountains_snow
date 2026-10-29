# /// script
# dependencies = [
#   "threadpoolctl",
#   "scipy",
# ]
# ///

import logging
import xarray
import numpy as np
from scipy import stats
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure
from openeo.metadata import CubeMetadata
from openeo.udf import inspect

REPRESENTATIVE_PIXEL_BAND_NAME = "representative"
logger = logging.getLogger(__name__)


def apply_metadata(metadata: CubeMetadata, context: dict) -> CubeMetadata:
    """Rename the bands by using apply metadata
    :param metadata: Metadata of the input data
    :param context: Context of the UDF
    :return: Renamed labels
    """
    # rename band labels
    return metadata.rename_labels(dimension="bands", target=[REPRESENTATIVE_PIXEL_BAND_NAME])

def apply_datacube(cube :xarray.DataArray, context) -> xarray.DataArray:

    # xarray .sel requires a unique index. Keep first occurrence if duplicates are present.
    band_labels = np.asarray(cube.bands.values)
    if band_labels.size > 0:
        _, first_indices = np.unique(band_labels, return_index=True)
        if first_indices.size != band_labels.size:
            first_indices = np.sort(first_indices)
            cube = cube.isel(bands=first_indices)
            logger.warning("Removed duplicate band labels before selection: bands=%s", list(cube.bands.values))

    total_samples = context.get("total_samples", 500)
    training_seed = context.get("training_seed", 0)

    # Reference SnowFLAKES training_collection.collect_trainings (vpremier/SnowFLAKES_stac).
    # Inputs are reflectances in [0, 1]; diff_B_NIR = B02 - B08.
    solar_incidence_angle = cube.sel(bands="local_solar_incidence_angle").values.astype(float)
    green = cube.sel(bands="B03").values.astype(float)
    used = ["B03", "B04", "B08", "B11", "NDSI", "NDWI", "diff_B_NIR", "local_solar_incidence_angle"]
    valid = np.all(np.isfinite(cube.sel(bands=used).values.astype(float)), axis=0)
    shadow = shadow_mask(cube, valid)

    # Buffer shadow boundaries; morphology over spatial axes only.
    structure_shape = [1] * green.ndim
    for dim in ("y", "x"):
        structure_shape[cube.sel(bands="B03").get_axis_num(dim)] = 3
    structure = generate_binary_structure(2, 1).reshape(structure_shape)
    sun_eroded = binary_erosion(
        binary_dilation(~shadow & valid, structure=structure), structure=structure, iterations=3
    ) & valid

    # TODO exclude glaciers from shadow training once a glacier mask band is available.
    eligible_shadow = valid & shadow

    valid_angles = solar_incidence_angle[valid]
    max_SIA = float(np.max(valid_angles)) if valid_angles.size else 90.0
    if max_SIA <= 90:
        ranges = ((0, 20), (20, 45), (45, 70), (70, np.nextafter(max_SIA, np.inf)))
    else:
        ranges = ((0, 20), (20, 45), (45, 70), (70, 90), (90, 180))
    range_samples = calculate_training_samples(
        np.where(valid, solar_incidence_angle, np.nan), ranges, total_samples
    )

    # labels: 1 = snow, 2 = snow-free
    values = np.zeros(green.shape, dtype=np.float32)

    # Shadow training is independent of the solar-incidence angle.
    snow_shad, snowfree_shad = get_pixels_shadow(cube, eligible_shadow)
    shadow_samples = np.zeros_like(values)
    if np.sum(snow_shad) > 10:
        shadow_samples += sample_histogram_equal(snow_shad, green, int(total_samples / 2), seed=training_seed)
    if np.sum(snowfree_shad) > 10:
        shadow_samples += 2 * sample_histogram_equal(snowfree_shad, green, int(total_samples / 2), seed=training_seed)
    values[eligible_shadow] = shadow_samples[eligible_shadow]

    for curr_range, sample_count in range_samples.items():
        mask_sun = valid & sun_eroded & (solar_incidence_angle >= curr_range[0]) & (solar_incidence_angle < curr_range[1])
        if np.sum(mask_sun) <= 10:
            continue
        snow_sun, snowfree_sun = get_pixels_sun(cube, mask_sun)
        sun_samples = np.zeros_like(values)
        if np.sum(snow_sun) > 10:
            sun_samples += sample_histogram_equal(snow_sun, green, int(sample_count / 2), seed=training_seed)
        if np.sum(snowfree_sun) > 10:
            sun_samples += 2 * sample_histogram_equal(snowfree_sun, green, int(sample_count / 2), seed=training_seed)
        values[mask_sun] = sun_samples[mask_sun]

    if(context.get("classify", False)):
        classify_bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B08", "B11", "B12"]
        curr_bands = cube.sel(bands=[b for b in classify_bands if b in cube.bands.values])
        def reshape(v):
            return np.stack(v, axis=-1)
        try:
            snow_bands = curr_bands.values[:, values == 1]
            snowfree_bands = curr_bands.values[:, values == 2]
            if snow_bands.shape[1] > 0 and snowfree_bands.shape[1] > 0:
                class_array, normalizer, svm, training_array = model_training_svm(reshape(snow_bands), reshape(snowfree_bands))
                values = classify(curr_bands.values, valid, normalizer, svm)
        except Exception as e:
            logger.exception("Error during classification")
            inspect(data=str(e), message=f"Error during classification, not classifying pixels {e}")

    cube.loc['B03'] = 0

    if values is not None and values.shape[0] > 0:
        cube.loc['B03'] = values

    bands_to_drop = list(cube.bands.values)
    bands_to_drop.remove("B03")

    cube = cube.drop_sel(bands=bands_to_drop)
    return cube


def shadow_mask(cube, valid):
    """Cloud OR self OR cast shadow, as SnowFLAKES auxiliary_folder_population.generate_shadow_mask."""
    sia = cube.sel(bands="local_solar_incidence_angle").values.astype(float)
    red = cube.sel(bands="B04").values.astype(float)
    nir = cube.sel(bands="B08").values.astype(float)
    cloud_shadow = cube.sel(bands="cloud_shadow").values == 1
    self_shadow = valid & (sia > 90)
    closdi = (1 - 1.5 * nir - 0.1 * red) * 100 / (1 + 3.5 * nir + 4.9 * red)
    if "closdi_threshold" in cube.bands.values:  # precomputed over the whole AOI
        precomputed = cube.sel(bands="closdi_threshold").values.astype(float)
        threshold = np.nanmax(precomputed) if np.isfinite(precomputed).any() else np.nan
    elif np.any(self_shadow):  # fallback: this chunk only
        threshold = np.nanpercentile(closdi[self_shadow], 10)
    else:
        threshold = np.nan
    with np.errstate(invalid="ignore"):
        cast_shadow = valid & (sia >= 70) & (sia <= 90) & (closdi > threshold)
    return cast_shadow | self_shadow | cloud_shadow


def calculate_training_samples(solar_incidence_angle, ranges, total_samples):
    """Distribute total_samples over SIA ranges proportionally to their pixel counts (largest remainder)."""
    flattened_map = solar_incidence_angle.flatten()
    range_pixel_counts = {r: np.sum((flattened_map >= r[0]) & (flattened_map < r[1])) for r in ranges}

    total_pixels = sum(range_pixel_counts.values())
    range_samples = {r: 0 for r in ranges}
    if total_pixels == 0 or total_samples <= 0:
        return range_samples

    non_empty = [r for r in ranges if range_pixel_counts[r] > 0]
    raw = {r: total_samples * range_pixel_counts[r] / total_pixels for r in non_empty}
    for r in non_empty:
        range_samples[r] = int(np.floor(raw[r]))
    remainder = int(total_samples - sum(range_samples.values()))
    for r in sorted(non_empty, key=lambda item: raw[item] - np.floor(raw[item]), reverse=True)[:remainder]:
        range_samples[r] += 1
    return range_samples


def define_threshold(feature, mask, threshold=(0.08, 0.12), tolerance=0.5):
    """Two-component GMM means; each default mean is replaced only if the fit is within tolerance."""
    from sklearn.mixture import GaussianMixture

    values = np.asarray(feature[mask], dtype=float)
    values = values[np.isfinite(values)]
    default = np.asarray(threshold, dtype=float)
    if values.size < 2:
        return default

    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0,
                          means_init=default.reshape(-1, 1), max_iter=1000, tol=1e-4)
    try:
        gmm.fit(values.reshape(-1, 1))
    except ValueError:
        return default

    means = np.sort(gmm.means_.flatten())
    rel_diff = np.abs(means - default) / default
    return np.where(rel_diff <= tolerance, means, default)


def fit_distribution_and_median(feature, mask, default_median):
    """Median of the best (AIC) Gaussian or lognormal fit; default_median if fewer than 20 values or no fit."""
    values = np.asarray(feature[mask], dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size < 20:
        return default_median

    fits = []
    try:
        mean, std = stats.norm.fit(values)
        if np.isfinite(mean) and np.isfinite(std) and std > 0:
            aic = 4 - 2 * np.sum(stats.norm.logpdf(values, loc=mean, scale=std))
            fits.append((aic, stats.norm.median(loc=mean, scale=std)))
    except (ValueError, RuntimeError, FloatingPointError):
        pass
    positive = values[values > 0]
    if positive.size >= 3:
        try:
            shape, loc, scale = stats.lognorm.fit(positive, floc=0)
            if np.isfinite(shape) and np.isfinite(scale) and shape > 0 and scale > 0:
                aic = 4 - 2 * np.sum(stats.lognorm.logpdf(positive, shape, loc=loc, scale=scale))
                fits.append((aic, stats.lognorm.median(shape, loc=loc, scale=scale)))
        except (ValueError, RuntimeError, FloatingPointError):
            pass
    if not fits:
        return default_median
    return float(min(fits, key=lambda fit: fit[0])[1])


def sample_histogram_equal(mask, values, n_samples, n_bins=20, seed=None):
    """Boolean mask of up to n_samples pixels from mask, sampled approximately uniformly over the histogram of values."""
    rng = np.random.default_rng(seed)

    valid_idx = np.flatnonzero(mask)
    valid_values = np.asarray(values).flat[valid_idx]
    finite = np.isfinite(valid_values)
    valid_idx = valid_idx[finite]
    valid_values = valid_values[finite]
    result = np.zeros(mask.shape, dtype=bool)
    if n_samples <= 0 or valid_values.size == 0:
        return result
    if np.all(valid_values == valid_values[0]):
        result.flat[rng.choice(valid_idx, min(int(n_samples), valid_idx.size), replace=False)] = True
        return result

    bins = np.linspace(valid_values.min(), valid_values.max(), n_bins + 1)
    target = int(np.ceil(n_samples / n_bins))
    selected = []
    for i in range(n_bins):
        upper = valid_values <= bins[i + 1] if i == n_bins - 1 else valid_values < bins[i + 1]
        in_bin = np.where((valid_values >= bins[i]) & upper)[0]
        if len(in_bin) == 0:
            continue
        selected.extend(valid_idx[rng.choice(in_bin, min(target, len(in_bin)), replace=False)])

    selected = np.array(selected, dtype=int)
    if len(selected) > n_samples:
        selected = rng.choice(selected, n_samples, replace=False)
    result.flat[selected] = True
    return result


def get_pixels_shadow(cube, eligible_shadow):
    """Snow / snow-free candidates in shadow from GMM thresholds on green and diff_B_NIR."""
    green = cube.sel(bands="B03").values.astype(float)
    diff_B_NIR = cube.sel(bands="diff_B_NIR").values.astype(float)
    if not np.any(eligible_shadow):
        return np.zeros(green.shape, dtype=bool), np.zeros(green.shape, dtype=bool)

    # Bright pixels are trimmed only for fitting.
    fit_mask = eligible_shadow & (green < np.nanpercentile(green[eligible_shadow], 95))
    green_thresholds = define_threshold(green, fit_mask, threshold=(0.075, 0.1))
    BNIR_thresholds = define_threshold(diff_B_NIR, fit_mask, threshold=(0.08, 0.12))

    snow = eligible_shadow & (green > max(green_thresholds)) & (diff_B_NIR > max(BNIR_thresholds))
    snowfree = eligible_shadow & (green < min(green_thresholds)) & (diff_B_NIR < min(BNIR_thresholds))
    return snow, snowfree


def get_pixels_sun(cube, mask_sun):
    """Snow / snow-free candidates in sun: GMM threshold on NDSI, then fitted medians of green and SWIR."""
    NDSI = cube.sel(bands="NDSI").values.astype(float)
    NDWI = cube.sel(bands="NDWI").values.astype(float)
    green = cube.sel(bands="B03").values.astype(float)
    swir = cube.sel(bands="B11").values.astype(float)

    NDSI_thresholds = define_threshold(NDSI, mask_sun, threshold=(0.2, 0.7))
    mask_snow = mask_sun & (NDWI < 0.1) & (NDSI > max(NDSI_thresholds))
    mask_sf = mask_sun & (NDSI < min(NDSI_thresholds))

    snow = (mask_snow
            & (green > fit_distribution_and_median(green, mask_snow, default_median=0.6))
            & (swir < fit_distribution_and_median(swir, mask_snow, default_median=0.2)))
    snowfree = (mask_sf
                & (green < fit_distribution_and_median(green, mask_sf, default_median=0.5))
                & (swir > fit_distribution_and_median(swir, mask_sf, default_median=0.1)))
    return snow, snowfree

def hyp_disatance(svmModel, svmMatrix):
    return svmModel.decision_function(svmMatrix)

def classify(bands, valid_mask, normalizer, svm, Nprocesses = 8):

    min_score_ns = -1
    max_score_s = 1
    Image_array_to_classify = bands[:, valid_mask].transpose()
    Samples_to_classify = normalizer.transform(Image_array_to_classify)
    # Divide Samples_to_classify into blocks for parallel processing
    samplesBlocks = np.array_split(Samples_to_classify, Nprocesses, axis=0)
    # Calculate the score
    from concurrent.futures import ThreadPoolExecutor
    #use threadpoolexecutor, scikit learn will release the GIL
    with ThreadPoolExecutor(max_workers=Nprocesses) as executor:
        scoreImage_arrayBlocks = list(executor.map(lambda block: hyp_disatance(svm, block), samplesBlocks))

    scoreImage_array = np.concatenate(scoreImage_arrayBlocks, axis=0)
    Score_map = 255 * np.ones(np.shape(valid_mask)).astype(float)
    Score_map[valid_mask] = scoreImage_array.flatten()
    scoreImage_array[scoreImage_array < min_score_ns] = min_score_ns
    scoreImage_array[scoreImage_array > max_score_s] = max_score_s
    SCF_Image_array = (scoreImage_array * 50 + 50).astype('uint8')
    # Create the SCF map
    SCF_map = 255 * np.ones(np.shape(valid_mask))
    SCF_map[valid_mask] = SCF_Image_array.flatten()

    return SCF_map

def model_training_svm(snow_training, no_snow_training, gamma=None):
    from sklearn import preprocessing
    from sklearn.svm import SVC
    from sklearn.metrics.pairwise import rbf_kernel

    training_array = np.concatenate((snow_training, no_snow_training), axis=0)
    class_array = np.concatenate((np.ones(snow_training.shape[0]), np.zeros(no_snow_training.shape[0])), axis=0)
    # Rescale: standardization between 0 and 1
    normalizer = preprocessing.StandardScaler().fit(training_array)
    Samples_train_normalized = normalizer.transform(training_array)
    if gamma == None:
        # Gamma selection by examining the kernel
        gamma_range = np.logspace(-2, 2, 100)
        std_list = []
        for curr_gamma in gamma_range:
            rbf = rbf_kernel(Samples_train_normalized, Samples_train_normalized, gamma=curr_gamma)
            std_list.append(rbf.std())

        idx_max_std = np.argmax(std_list)
        best_gamma = gamma_range[idx_max_std]

        logger.info("Selected gamma for SVM: %s", best_gamma)

    else:
        best_gamma = gamma
    svm = SVC(C=2000000, kernel='rbf', gamma=best_gamma, probability=False,
              decision_function_shape='ovo', cache_size=8000)
    svm.fit(Samples_train_normalized, class_array)
    #pred = svm.predict(Samples_train_normalized)
    return class_array, normalizer, svm, training_array
