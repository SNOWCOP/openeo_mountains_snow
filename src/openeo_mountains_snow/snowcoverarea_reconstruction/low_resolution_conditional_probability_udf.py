#calculate conditional probability maps at low resolution
import numpy as np
import xarray as xr
from typing import Dict
import logging

logger = logging.getLogger(__name__)

SCF_RANGES = [
    (0, 20), (0, 30), (10, 40), (20, 50), (30, 60),
    (40, 70), (50, 80), (60, 90), (70, 100), (80, 100)
]

CLOUD_VALUE = 205


def process_single_timestep(cube: xr.DataArray) -> xr.DataArray:
    """
    Process a single time step: expects (bands, y, x) or (t, bands, y, x).
    """

    logger.debug(f"received shape {cube.shape}, dims {cube.dims}")

    # Remove time dimension if present
    if "t" in cube.dims:
        cube = cube.squeeze("t", drop=True)

    # Ensure correct dimension order
    if "bands" in cube.dims:
        cube = cube.transpose("bands", "y", "x")

    # Extract bands
    hr_snow = cube.sel(bands="snow").values
    hr_scf = cube.sel(bands="scf").values

    y_dim, x_dim = hr_snow.shape
    num_ranges = len(SCF_RANGES)

    snow_in_range_all = np.full(
        (num_ranges, y_dim, x_dim), np.nan, dtype=np.float32
    )
    range_mask_all = np.zeros(
        (num_ranges, y_dim, x_dim), dtype=np.float32
    )

    # Snow mask (nan = cloud/invalid)
    snow_mask = np.where(
        hr_snow == CLOUD_VALUE,
        np.nan,
        np.where(hr_snow == 100, 1.0, 0.0),
    )

    # Process SCF ranges
    for idx, (low, high) in enumerate(SCF_RANGES):

        if low == 0:
            in_range = (hr_scf >= low) & (hr_scf <= high)
        else:
            in_range = (hr_scf > low) & (hr_scf <= high)

        valid = hr_scf <= 100
        in_range = in_range & valid

        range_mask_all[idx] = in_range.astype(np.float32)
        snow_in_range_all[idx] = snow_mask * in_range

    combined = np.concatenate(
        [snow_in_range_all, range_mask_all], axis=0
    )

    range_keys = [f"{l}_{h}" for l, h in SCF_RANGES]

    return xr.DataArray(
        combined,
        dims=["bands", "y", "x"],
        coords={
            "bands": range_keys + [f"occ_{k}" for k in range_keys],
            "y": cube.coords["y"],
            "x": cube.coords["x"],
        },
    )


def apply_datacube(cube: xr.DataArray, context: Dict) -> xr.DataArray:
    """
    Main UDF entrypoint.
    """

    logger.info(f"received shape {cube.shape}, dims {cube.dims}")

    # ---- Case: time series ----
    if "t" in cube.dims:

        logger.info(f"processing {cube.sizes['t']} timesteps")

        time_step_results = cube.groupby("t").map(process_single_timestep)

        num_ranges = len(SCF_RANGES)

        snow_masks = time_step_results.isel(
            bands=slice(0, num_ranges)
        ).values

        range_masks = time_step_results.isel(
            bands=slice(num_ranges, 2 * num_ranges)
        ).values

        total_snow = np.nansum(snow_masks, axis=0)
        total_occurrences = np.nansum(range_masks, axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            probabilities = np.where(
                total_occurrences > 0,
                total_snow / total_occurrences,
                np.nan,
            )

        occurrences = total_occurrences.astype(np.int32)

        final = np.concatenate(
            [probabilities, occurrences], axis=0
        )

        range_keys = [f"{l}_{h}" for l, h in SCF_RANGES]

        return xr.DataArray(
            final,
            dims=["bands", "y", "x"],
            coords={
                "bands": range_keys + [f"occ_{k}" for k in range_keys],
                "y": time_step_results.coords["y"],
                "x": time_step_results.coords["x"],
            },
        )

    # ---- Case: single timestep ----
    else:

        single = process_single_timestep(cube)

        num_ranges = len(SCF_RANGES)

        snow_masks = single.isel(
            bands=slice(0, num_ranges)
        ).values

        range_masks = single.isel(
            bands=slice(num_ranges, 2 * num_ranges)
        ).values

        with np.errstate(divide="ignore", invalid="ignore"):
            probabilities = np.where(
                range_masks > 0,
                snow_masks,
                np.nan,
            )

        occurrences = range_masks.astype(np.int32)

        final = np.concatenate(
            [probabilities, occurrences], axis=0
        )

        range_keys = [f"{l}_{h}" for l, h in SCF_RANGES]

        return xr.DataArray(
            final,
            dims=["bands", "y", "x"],
            coords={
                "bands": range_keys + [f"occ_{k}" for k in range_keys],
                "y": single.coords["y"],
                "x": single.coords["x"],
            },
        )