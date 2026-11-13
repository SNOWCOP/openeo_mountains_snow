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
from sklearn.mixture import GaussianMixture

REPRESENTATIVE_PIXEL_BAND_NAME = "representative"
CLASSIFICATION_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B08", "B11", "B12"]
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

    solar_incidence_angle = cube.sel(bands="local_solar_incidence_angle").values.astype(float)
    green = cube.sel(bands="B03").values.astype(float)
    diff_B_NIR = cube.sel(bands="diff_B_NIR").values.astype(float)
    shadow_mask = cube.sel(bands="shadow_mask").values
    ndsi = cube.sel(bands="NDSI").values.astype(float)
    ndwi = cube.sel(bands="NDWI").values.astype(float)
    swir = cube.sel(bands="B11").values.astype(float)

    valid_sun = (
        np.isfinite(shadow_mask) & np.isfinite(solar_incidence_angle)
        & np.isfinite(green) & np.isfinite(swir)
        & np.isfinite(ndsi) & np.isfinite(ndwi)
    )
    # Buffer shadow boundaries as in the original training collection. Apply
    # morphology only over spatial axes, independently for each timestamp.
    spatial_axes = [cube.sel(bands="B03").get_axis_num(dim) for dim in ("y", "x")]
    structure_shape = [1] * green.ndim
    for axis in spatial_axes:
        structure_shape[axis] = 3
    structure = generate_binary_structure(2, 1).reshape(structure_shape)
    eligible_sun = (
        binary_erosion(
            binary_dilation((shadow_mask == 0) & valid_sun, structure=structure),
            structure=structure,
            iterations=3,
        ) & valid_sun & (shadow_mask == 0)
    )

    # Add glacier exclusions here when a glacier mask is available.
    eligible_shadow = (
        np.isfinite(shadow_mask) & (shadow_mask != 0)
        & np.isfinite(green) & np.isfinite(diff_B_NIR)
    )

    # define solar incidence angle ranges
    valid_angles = solar_incidence_angle[valid_sun]
    max_SIA = float(np.max(valid_angles)) if valid_angles.size else 90.0
    if max_SIA <= 90:
        ranges = ((0, 20), (20, 45), (45, 70), (70, np.nextafter(max_SIA, np.inf)))
    else:
        ranges = ((0, 20), (20, 45), (45, 70), (70, 90), (90, 180))

    # get a number of training proportional to the area belonging to that range
    range_samples = calculate_training_samples(
        np.where(valid_sun, solar_incidence_angle, np.nan), ranges, total_samples
    )

    # initialize training array: 1 = snow, 2 = snow-free
    values = np.zeros_like(green, dtype=np.float32)

    representative_pixels_mask_snow = np.full_like(green, 0, dtype=np.float32)
    representative_pixels_mask_noSnow = np.full_like(green, 0, dtype=np.float32)

    # get trainings in shadow: independent from the angle
    snow_shad, snowfree_shad = get_pixels_shadow(green, diff_B_NIR, eligible_shadow)

    if np.sum(snow_shad) > 10:
        representative_pixels_mask_snow = sample_histogram_equal(
            snow_shad, green, int(total_samples / 2), n_bins=20, seed=training_seed
        )
    if np.sum(snowfree_shad) > 10:
        representative_pixels_mask_noSnow = sample_histogram_equal(
            snowfree_shad, green, int(total_samples / 2), n_bins=20, seed=training_seed
        ) * 2

    representative_pixels_mask = (
        representative_pixels_mask_noSnow + representative_pixels_mask_snow
    )
    values[eligible_shadow] = representative_pixels_mask[eligible_shadow]

    for curr_range, sample_count in range_samples.items():
        eligible_sun_range = (
            eligible_sun
            & (solar_incidence_angle >= curr_range[0])
            & (solar_incidence_angle < curr_range[1])
        )
        if sample_count <= 0 or np.count_nonzero(eligible_sun_range) <= 10:
            continue

        snow_sun, snowfree_sun = get_pixels_sun(ndsi, ndwi, green, swir, eligible_sun_range)
        sun_samples = np.zeros_like(values)
        if np.count_nonzero(snow_sun) > 10:
            sun_samples += sample_histogram_equal(
                snow_sun, green, int(sample_count / 2), n_bins=20, seed=training_seed
            )
        if np.count_nonzero(snowfree_sun) > 10:
            sun_samples += 2 * sample_histogram_equal(
                snowfree_sun, green, int(sample_count / 2), n_bins=20, seed=training_seed
            )
        values[eligible_sun_range] = sun_samples[eligible_sun_range]

    if(context.get("classify", False)):
        spectral = [b for b in CLASSIFICATION_BANDS if b in cube.bands.values]
        curr_bands = cube.sel(bands=spectral)

        def reshape(v):
            return np.stack(v, axis=-1)
        try:
            snow_representative_pixel_bands = curr_bands.values[:, values == 1]
            snowfree_representative_pixel_bands = curr_bands.values[:, values == 2]
            valid_pixels = np.all(np.isfinite(curr_bands.values), axis=0)

            if (snow_representative_pixel_bands.shape[1] > 0
                    and snowfree_representative_pixel_bands.shape[1] > 0):
                class_array, normalizer, svm, training_array = model_training_svm(
                    reshape(snow_representative_pixel_bands),
                    reshape(snowfree_representative_pixel_bands),
                )

                SCF_map = classify(curr_bands.values, valid_pixels, normalizer, svm)
                values = SCF_map
        except Exception as e:
            logger.exception("Error during classification")
            inspect(data=str(e), message=f"Error during classification: {e}")

    out = cube.sel(bands=["B03"])
    return out.copy(data=values.astype(cube.dtype).reshape(out.shape))

def calculate_training_samples(solar_incidence_angle, ranges, total_samples):
    """
    Calculate the number of training samples for each angle range proportional to the pixel distribution.

    Parameters:
        solar_incidence_angle (np.ndarray): 2D array representing the solar incidence angle map.
        ranges (list of tuple): List of angle ranges (start, end).
        total_samples (int): Total number of training samples to distribute.

    Returns:
        dict: A dictionary with ranges as keys and the number of training samples as values.
    """
    # Flatten the angle map for easier processing
    flattened_map = solar_incidence_angle.flatten()

    # Initialize a dictionary to store the count for each range
    range_pixel_counts = {r: 0 for r in ranges}

    # Count pixels in each range
    for r in ranges:
        range_pixel_counts[r] = np.sum((flattened_map >= r[0]) & (flattened_map < r[1]))

    total_pixels = sum(range_pixel_counts.values())
    range_samples = {r: 0 for r in ranges}
    if total_pixels == 0 or total_samples <= 0:
        return range_samples

    non_empty = [r for r in ranges if range_pixel_counts[r] > 0]
    raw = {
        r: total_samples * range_pixel_counts[r] / total_pixels
        for r in non_empty
    }
    for r in non_empty:
        range_samples[r] = int(np.floor(raw[r]))
    remainder = int(total_samples - sum(range_samples.values()))
    for r in sorted(non_empty, key=lambda item: raw[item] - np.floor(raw[item]), reverse=True)[:remainder]:
        range_samples[r] += 1
    return range_samples


def define_threshold(feature,
                     mask,
                     threshold=(0.08, 0.12),
                     tolerance=0.5):
    """
    Parameters
    ----------
    feature : ndarray
    mask : ndarray(bool)
    threshold : tuple
        Default Gaussian means (low, high)
    tolerance : float
        Relative difference allowed before replacing defaults.
        0.15 = 15%
    """

    values = np.asarray(feature[mask], dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.asarray(threshold)

    values = values.reshape(-1, 1)

    # Fit two-component GMM
    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=0,
        means_init=np.array(threshold).reshape(-1, 1),
        max_iter=1000,
        tol = 1e-4
    )

    try:
        gmm.fit(values)
    except ValueError:
        return np.asarray(threshold)

    means = gmm.means_.flatten()

    # Sort from low to high
    means = means[np.argsort(means)]

    default = np.asarray(threshold)

    rel_diff = np.abs(means - default) / default

    final_means = default.copy()

    for i in range(2):
        if rel_diff[i] <= tolerance:
            final_means[i] = means[i]

    logger.debug("Default means %s, GMM means %s, relative diff %s%%, using %s",
                 default, means, 100 * rel_diff, final_means)

    return final_means


def sample_histogram_equal(mask, values, n_samples, n_bins=20, seed=None):
    """
    Sample pixels approximately uniformly over the histogram.

    Parameters
    ----------
    mask : 2D bool array
        Pixels eligible for sampling (e.g. snow).
    values : 2D array
        Variable whose histogram should be sampled (e.g. green_corr).
    n_samples : int
        Total number of pixels to return.
    n_bins : int
        Number of histogram bins.
    seed : int or None
        Random seed.

    Returns
    -------
    sample_mask : bool array
        Boolean mask of selected pixels.
    """
    rng = np.random.default_rng(seed)

    # Eligible pixels
    valid_idx = np.flatnonzero(mask)
    valid_values = np.asarray(values).flat[valid_idx]
    finite = np.isfinite(valid_values)
    valid_idx = valid_idx[finite]
    valid_values = valid_values[finite]
    if n_samples <= 0 or valid_values.size == 0:
        return np.zeros(mask.shape, dtype=bool)
    if np.all(valid_values == valid_values[0]):
        selected = rng.choice(
            valid_idx, min(int(n_samples), valid_idx.size), replace=False
        )
        result = np.zeros(mask.shape, dtype=bool)
        result.flat[selected] = True
        return result

    # Histogram bins
    bins = np.linspace(valid_values.min(), valid_values.max(), n_bins + 1)

    # Desired samples per bin
    target = int(np.ceil(n_samples / n_bins))

    selected = []

    for i in range(n_bins):
        if i == n_bins - 1:
            in_bin = np.where((valid_values >= bins[i]) &
                              (valid_values <= bins[i+1]))[0]
        else:
            in_bin = np.where((valid_values >= bins[i]) &
                              (valid_values < bins[i+1]))[0]

        if len(in_bin) == 0:
            continue

        n = min(target, len(in_bin))
        chosen = rng.choice(in_bin, n, replace=False)
        selected.extend(valid_idx[chosen])

    selected = np.array(selected)

    # If too many samples, randomly reduce
    if len(selected) > n_samples:
        selected = rng.choice(selected, n_samples, replace=False)

    # Build output mask
    sample_mask = np.zeros(mask.shape, dtype=bool)
    sample_mask.flat[selected] = True

    return sample_mask


def get_pixels_shadow(green, diff_B_NIR, eligible_shadow):
    """Select snow and snow-free candidates within the supplied eligibility mask."""
    if not np.any(eligible_shadow):
        return np.zeros(green.shape, dtype=bool), np.zeros(green.shape, dtype=bool)

    # Trim bright pixels only for fitting; candidate selection uses all
    # eligible shadow pixels, including those above the percentile.
    threshold_fit_mask = eligible_shadow & (
        green < np.nanpercentile(green[eligible_shadow], 95)
    )

    green_thresholds = define_threshold(green, threshold_fit_mask, threshold=(0.075, 0.1))
    BNIR_thresholds = define_threshold(diff_B_NIR, threshold_fit_mask, threshold=(0.08, 0.12))

    snow = np.logical_and.reduce((eligible_shadow,
                                  green>max(green_thresholds),
                                  diff_B_NIR>max(BNIR_thresholds)))

    snowfree = np.logical_and.reduce((eligible_shadow,
                                  green<min(green_thresholds),
                                  diff_B_NIR<min(BNIR_thresholds)))

    return snow, snowfree


def get_pixels_sun(NDSI, NDWI, green, swir, eligible_sun):
    """Select candidates within the prepared valid sun and solar-angle mask."""
    if not np.any(eligible_sun):
        return np.zeros(green.shape, dtype=bool), np.zeros(green.shape, dtype=bool)

    NDSI_thresholds = define_threshold(NDSI, eligible_sun, threshold=(0.2, 0.7))

    # fixed conditions for being a snow pixel
    mask_snow = np.logical_and.reduce((eligible_sun,
                                        NDWI<0.1,
                                        NDSI>max(NDSI_thresholds)))

    # fixed conditions for being a snowfree pixel
    mask_sf = np.logical_and.reduce((eligible_sun,
                                     NDSI<min(NDSI_thresholds)))

    # find dynamic thresholds
    green_threshold_snow = fit_distribution_and_median(green, mask_snow, default_median=0.6)['fitted_median']
    green_threshold_sf = fit_distribution_and_median(green, mask_sf, default_median=0.5)['fitted_median']
    swir_threshold_snow = fit_distribution_and_median(swir, mask_snow, default_median=0.2)['fitted_median']
    swir_threshold_sf = fit_distribution_and_median(swir, mask_sf, default_median=0.1)['fitted_median']

    snow = np.logical_and.reduce((mask_snow,
                                  green > green_threshold_snow,
                                  swir < swir_threshold_snow))

    snowfree = np.logical_and.reduce((mask_sf,
                                      green < green_threshold_sf,
                                      swir > swir_threshold_sf))

    return snow, snowfree


def fit_distribution_and_median(feature, mask, distribution="auto", default_median=np.nan):
    """
    Fit a Gaussian or lognormal distribution to masked feature values.

    If there are not enough valid values, or the distribution cannot be fitted,
    return a dictionary containing default values.

    Parameters
    ----------
    feature : ndarray
        Input feature array.
    mask : ndarray of bool
        Boolean mask with the same shape as feature.
    distribution : {"gaussian", "lognormal", "auto"}
        Distribution to fit; "auto" keeps the lowest AIC.
    default_median : float
        Median returned when fitting is not possible.

    Returns
    -------
    result : dict
        Distribution fit information.
    """

    def make_default_result(default_median, n_samples=0, reason=None):
        return {
            "distribution": "gaussian",
            "empirical_median": default_median,
            "fitted_median": default_median,
            "parameters": {
                "mean": default_median,
                "std": 0.0,
            },
            "log_likelihood": np.nan,
            "aic": np.nan,
            "n_samples": int(n_samples),
            "fit_success": False,
            "reason": reason,
        }

    feature = np.asarray(feature)
    mask = np.asarray(mask, dtype=bool)

    if feature.shape != mask.shape:
        raise ValueError(
            "feature and mask must have the same shape. "
            f"Received {feature.shape} and {mask.shape}."
        )

    # Extract finite masked values
    values = feature[mask].astype(np.float64).ravel()
    values = values[np.isfinite(values)]

    # Return defaults when there are insufficient valid observations
    if values.size < 20:
        return make_default_result(
            default_median=default_median,
            n_samples=values.size,
            reason="Fewer than 20 finite masked values are available."
        )

    empirical_median = float(np.median(values))

    def fit_gaussian(data):
        mean, std = stats.norm.fit(data)

        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise ValueError(
                "Gaussian fit produced invalid parameters or zero variance."
            )

        log_likelihood = np.sum(stats.norm.logpdf(data, loc=mean, scale=std))

        return {
            "distribution": "gaussian",
            "parameters": {
                "mean": float(mean),
                "std": float(std),
            },
            "fitted_median": float(stats.norm.median(loc=mean, scale=std)),
            "log_likelihood": float(log_likelihood),
            "aic": float(2 * 2 - 2 * log_likelihood),
            "n_fit_samples": int(data.size),
        }

    def fit_lognormal(data):
        positive_data = data[data > 0]

        if positive_data.size < 3:
            raise ValueError(
                "A lognormal fit requires at least three positive values."
            )

        shape, loc, scale = stats.lognorm.fit(positive_data, floc=0)

        if (
            not np.isfinite(shape)
            or not np.isfinite(loc)
            or not np.isfinite(scale)
            or shape <= 0
            or scale <= 0
        ):
            raise ValueError(
                "Lognormal fit produced invalid parameters."
            )

        log_likelihood = np.sum(stats.lognorm.logpdf(positive_data, shape, loc=loc, scale=scale))

        return {
            "distribution": "lognormal",
            "parameters": {
                "std": float(shape),
                "loc": float(loc),
                "scale": float(scale),
            },
            "fitted_median": float(stats.lognorm.median(shape, loc=loc, scale=scale)),
            "log_likelihood": float(log_likelihood),
            "aic": float(2 * 2 - 2 * log_likelihood),
            "n_fit_samples": int(positive_data.size),
        }

    distribution = distribution.lower()

    if distribution not in {"gaussian", "lognormal", "auto"}:
        raise ValueError(
            "distribution must be 'gaussian', 'lognormal', or 'auto'."
        )

    # Fit the requested distribution while safely handling failures
    try:
        if distribution == "gaussian":
            selected_fit = fit_gaussian(values)

        elif distribution == "lognormal":
            selected_fit = fit_lognormal(values)

        else:
            fits = []

            try:
                fits.append(fit_gaussian(values))
            except (ValueError, RuntimeError, FloatingPointError):
                pass

            try:
                fits.append(fit_lognormal(values))
            except (ValueError, RuntimeError, FloatingPointError):
                pass

            if not fits:
                return make_default_result(
                    default_median=default_median,
                    n_samples=values.size,
                    reason="Neither Gaussian nor lognormal fitting succeeded."
                )

            selected_fit = min(fits, key=lambda fit: fit["aic"])

    except (ValueError, RuntimeError, FloatingPointError) as error:
        result = make_default_result(
            default_median=default_median,
            n_samples=values.size,
            reason=str(error)
        )

        # The empirical median is still available
        result["empirical_median"] = empirical_median

        return result

    selected_fit.update({
        "empirical_median": empirical_median,
        "n_samples": int(values.size),
        "fit_success": True,
        "reason": None,
    })

    return selected_fit

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
