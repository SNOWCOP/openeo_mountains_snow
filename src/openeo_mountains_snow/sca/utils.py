"""
Utility helpers for SCA (snow cover area) reconstruction.

Functions extracted from the legacy gap-filling module.
"""

import openeo
from openeo.processes import array_create


def get_scf_ranges(delta: int, epsilon: int) -> dict:
    """
    Generate SCF (Snow Cover Fraction) ranges for conditional probability computation.

    Creates ranges with a buffer (epsilon) around base delta-sized intervals.

    Args:
        delta: Step size for SCF ranges (e.g. 10 for 10% steps)
        epsilon: Security buffer around each range

    Returns:
        Dict mapping range keys (e.g. "0_20") to (scf1, scf2) tuples.
    """
    SCF_1 = list(range(0, 100, delta))
    SCF_2 = list(range(delta, 100 + delta, delta))

    scf_range_dic = {}
    for scf1, scf2 in zip(SCF_1, SCF_2):
        scf_l = max(0, scf1 - epsilon)
        scf_u = min(100, scf2 + epsilon)
        scf_range_dic[f"{scf_l}_{scf_u}"] = (scf1, scf2)
    return scf_range_dic


def create_mask(snow: openeo.DataCube) -> openeo.DataCube:
    """
    Create valid and snow masks from classified snow data.

    Args:
        snow: Snow data cube (0: no snow, 100: snow, 205: clouds)

    Returns:
        Data cube with two bands: ["valid", "snow"].
    """
    def valid_snow(bands):
        snow_band = bands["snow"]
        mask_valid = (snow_band <= 100) * 1.0
        mask_snow = (snow_band == 100) * 1.0
        return array_create([mask_valid, mask_snow])

    return (
        snow.apply_dimension(dimension="bands", process=valid_snow)
            .rename_labels(dimension="bands", target=["valid", "snow"])
    )
