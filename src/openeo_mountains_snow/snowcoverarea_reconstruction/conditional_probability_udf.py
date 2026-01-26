# conditional_probability_udf.py
import numpy as np
import xarray as xr
from typing import Dict
import logging

logger = logging.getLogger(__name__)

def generate_ranges(delta: int, epsilon: int):
    """Generate SCF range definitions."""
    range_definitions = []
    range_keys = []

    for base_low in range(0, 100, delta):
        base_high = min(base_low + delta, 100)
        eps_low = max(base_low - epsilon, 0)
        eps_high = min(base_high + epsilon, 100)

        key = f"{eps_low}_{eps_high}"
        range_definitions.append((key, eps_low, eps_high))
        range_keys.append(key)

    return range_definitions, range_keys

def process_single_timestep(cube: xr.DataArray, context: Dict) -> xr.DataArray:
    """
    Process a single time step.
    """
    logger.debug(f"process_single_timestep received shape: {cube.shape}, dims: {cube.dims}")
    
    # Remove time dimension if present
    if 't' in cube.dims:
        cube = cube.squeeze('t', drop=True)
        logger.debug(f"After squeezing t: shape: {cube.shape}, dims: {cube.dims}")
    
    # Now we should have (bands, y, x) - check and transpose if needed
    if 'bands' in cube.dims:
        # Ensure bands is first dimension
        cube = cube.transpose("bands", "y", "x")
    
    # Get parameters from context
    epsilon = context.get('epsilon', 10)
    delta = context.get('delta', 10)
    
    ranges, range_keys = generate_ranges(delta, epsilon)
    num_ranges = len(ranges)
    
    # Get data for this single time step
    hr_snow = cube.sel(bands='snow').values  # Shape: (y, x)
    hr_scf = cube.sel(bands='scf').values    # Shape: (y, x)
    
    # Get dimensions
    y_dim, x_dim = hr_snow.shape
    
    # Initialize arrays for this time step
    # For a single time step:
    # - snow_in_range: snow mask within range (1=snow, 0=no snow, nan=cloud/invalid)
    # - range_mask: whether pixel is in SCF range (1=in range, 0=not in range)
    snow_in_range_all = np.full((num_ranges, y_dim, x_dim), np.nan, dtype=np.float32)
    range_mask_all = np.zeros((num_ranges, y_dim, x_dim), dtype=np.float32)
    
    # Create snow mask for this time step
    snow_mask = np.where(
        hr_snow == 205,  # Invalid/cloud
        np.nan,
        np.where(hr_snow == 100, 1.0, 0.0)  # 100=snow, 0=no snow
    )
    
    # Process each SCF range for this time step
    for range_idx, (key, eps_low, eps_high) in enumerate(ranges):
        # Create mask for this SCF range
        if eps_low == 0:
            range_mask = (hr_scf >= eps_low) & (hr_scf <= eps_high)
        else:
            range_mask = (hr_scf > eps_low) & (hr_scf <= eps_high)
        
        # Exclude invalid SCF values (205)
        valid_scf = hr_scf <= 100
        range_mask = range_mask & valid_scf
        
        # Store range mask
        range_mask_all[range_idx] = range_mask.astype(np.float32)
        
        # Mask snow data by range
        snow_in_range = snow_mask * range_mask  # nan where cloud/invalid, 0 or 1 where valid
        snow_in_range_all[range_idx] = snow_in_range
    
    # Combine results for this time step
    combined_results = np.concatenate([snow_in_range_all, range_mask_all], axis=0)
    
    return xr.DataArray(
        combined_results,
        dims=['bands', 'y', 'x'],
        coords={
            'bands': range_keys + [f"occ_{k}" for k in range_keys],
            'y': cube.coords['y'],
            'x': cube.coords['x']
        }
    )

def apply_datacube(cube: xr.DataArray, context: Dict) -> xr.DataArray:
    """
    Main UDF function - flexible for both historical and daily processing.
    """
    # Handle NaN values
    cube = cube.fillna(255)
    
    logger.info(f"conditional probability udf received shape: {cube.shape}, dims: {cube.dims}")
    
    # Check if we have a time dimension
    if 't' in cube.dims:
        logger.info(f"conditional probability udf: Processing {cube.sizes['t']} time steps using groupby")
        
        def process_timestep(da):
            return process_single_timestep(da, context)
        
        # Process each time step
        time_step_results = cube.groupby('t').map(process_timestep)
        
        # time_step_results has shape: (t, bands, y, x)
        # We need to aggregate across time for historical probabilities
        
        # Get parameters for ranges
        epsilon = context.get('epsilon', 10)
        delta = context.get('delta', 10)
        ranges, range_keys = generate_ranges(delta, epsilon)
        num_ranges = len(ranges)
        
        # Separate snow masks and range masks
        # First num_ranges bands are snow masks (1=snow, 0=no snow, nan=cloud/invalid)
        snow_masks = time_step_results.isel(bands=slice(0, num_ranges)).values  # (t, num_ranges, y, x)
        # Next num_ranges bands are range masks (1=in range, 0=not in range)
        range_masks = time_step_results.isel(bands=slice(num_ranges, 2 * num_ranges)).values  # (t, num_ranges, y, x)
        
        # Sum across time dimension (ignoring nan)
        total_snow = np.nansum(snow_masks, axis=0)  # (num_ranges, y, x)
        total_occurrences = np.nansum(range_masks, axis=0)  # (num_ranges, y, x)
        
        # Compute conditional probabilities
        with np.errstate(divide='ignore', invalid='ignore'):
            probabilities = np.divide(
                total_snow,
                total_occurrences,
                where=(total_occurrences > 0)
            )
            probabilities = np.where(total_occurrences > 0, probabilities, np.nan)
        
        # Convert occurrences to int
        total_occurrences_int = total_occurrences.astype(np.int32)
        
        # Combine results
        final_results = np.concatenate([probabilities, total_occurrences_int], axis=0)
        
        # Create output DataArray
        output = xr.DataArray(
            final_results,
            dims=['bands', 'y', 'x'],
            coords={
                'bands': range_keys + [f"occ_{k}" for k in range_keys],
                'y': time_step_results.coords['y'],
                'x': time_step_results.coords['x']
            }
        )
        
        return output
        
    else:
        logger.info("No time dimension, processing directly")
        # For a single time step, compute probabilities directly
        
        # Process the single time step
        single_result = process_single_timestep(cube, context)
        
        # Get parameters for ranges
        epsilon = context.get('epsilon', 10)
        delta = context.get('delta', 10)
        ranges, range_keys = generate_ranges(delta, epsilon)
        num_ranges = len(ranges)
        
        # Get snow masks and range masks for this single time step
        snow_masks = single_result.isel(bands=slice(0, num_ranges)).values  # (num_ranges, y, x)
        range_masks = single_result.isel(bands=slice(num_ranges, 2 * num_ranges)).values  # (num_ranges, y, x)
        
        # For a single time step:
        # - snow_masks contains: 1 for snow, 0 for no snow, nan for cloud/invalid
        # - range_masks contains: 1 for in range, 0 for not in range
        
        # Compute probabilities: snow / occurrences
        with np.errstate(divide='ignore', invalid='ignore'):
            probabilities = np.where(
                range_masks > 0,
                snow_masks,  # This is already 1, 0, or nan
                np.nan
            )
        
        # Convert range masks to int occurrences
        occurrences = range_masks.astype(np.int32)
        
        # Combine results
        final_results = np.concatenate([probabilities, occurrences], axis=0)
        
        # Create output DataArray
        output = xr.DataArray(
            final_results,
            dims=['bands', 'y', 'x'],
            coords={
                'bands': range_keys + [f"occ_{k}" for k in range_keys],
                'y': single_result.coords['y'],
                'x': single_result.coords['x']
            }
        )
        
        return output