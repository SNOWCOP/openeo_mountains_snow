"""Unused snow helpers kept for reference (NDSI / Salomonson SCF, binarization, SCL cloud mask).

Not called by the pipeline.
"""

from typing import Optional

import openeo
from openeo.processes import if_, is_nan


def calculate_cloud_mask(scl: openeo.DataCube) -> openeo.DataCube:
    """Binary cloud mask from SCL classes 7, 8, 9."""
    classification = scl.band("SCL")
    binary = (classification == 7) | (classification == 8) | (classification == 9)
    return binary.add_dimension(name="bands", label="clouds", type="bands")


def calculate_ndsi(s2: openeo.DataCube) -> openeo.DataCube:
    """NDSI = (B03 - B11) / (B03 + B11)."""
    b03, b11 = s2.band("B03"), s2.band("B11")
    return ((b03 - b11) / (b03 + b11)).add_dimension(name="bands", label="ndsi", type="bands")


def salomonson(ndsi: openeo.DataCube) -> openeo.DataCube:
    """Salomonson & Appel (2006): SCF = (-0.01 + 1.45 * NDSI) * 100, clipped to [0, 100]."""
    scf = (-0.01 + 1.45 * ndsi.band("ndsi")) * 100
    scf = scf.mask(scf > 100, replacement=100)
    scf = scf.mask(scf < 0, replacement=0)
    scf = scf.apply(lambda x: if_(is_nan(x), 205, x))
    return scf.add_dimension(name="bands", label="scf", type="bands")


def binarize(scf: openeo.DataCube, snowT: Optional[int] = 20) -> openeo.DataCube:
    """SCF to 0 (no snow) / 100 (snow); 205 (cloud) is kept."""
    scf = scf.band("scf")
    scf = scf.mask((scf >= snowT) & (scf <= 100) & (scf != 205), replacement=100)
    scf = scf.mask((scf < snowT) & (scf <= 100) & (scf != 205), replacement=0)
    return scf.add_dimension(name="bands", label="scf", type="bands")
