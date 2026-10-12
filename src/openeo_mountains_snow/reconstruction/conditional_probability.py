"""Conditional probability helpers for reconstruction."""

from openeo import DataCube


def compute_snow_cover_conditional_probabilities(
    all_masks: DataCube,
    snow_cover_fraction_band_labels: list[str],
) -> tuple[DataCube, DataCube]:
    """Compute snow-cover conditional probability and occurrence bands from SCF masks."""

    def merge_masks(mask_cube):
        return mask_cube.and_(mask_cube.array_element(label="snow")) * 1.0

    mask_conditional_probability_snow = all_masks.apply(process=merge_masks)
    mask_conditional_probability_snow = mask_conditional_probability_snow.filter_bands(
        bands=snow_cover_fraction_band_labels
    )
    sum_conditional_probability_snow = mask_conditional_probability_snow.reduce_dimension(
        reducer="sum", dimension="t"
    )

    occurrences = all_masks.reduce_dimension(reducer="sum", dimension="t")
    occurrences = occurrences.filter_bands(bands=snow_cover_fraction_band_labels)
    occurrences = occurrences.rename_labels(
        dimension="bands", target=[f"occ_{b}" for b in snow_cover_fraction_band_labels]
    )

    conditional_probability = sum_conditional_probability_snow / occurrences
    conditional_probability = conditional_probability.rename_labels(
        dimension="bands",
        target=[f"cp_{b}" for b in snow_cover_fraction_band_labels],
    )
    return conditional_probability, occurrences


def compute_conditional_probabilities(
    all_masks: DataCube,
    labels_scf: list[str],
) -> tuple[DataCube, DataCube]:
    """Backward-compatible alias for compute_snow_cover_conditional_probabilities."""
    return compute_snow_cover_conditional_probabilities(all_masks, labels_scf)
