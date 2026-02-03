#itterative reconstruction
import xarray as xr
import numpy as np
import logging
from openeo.metadata import CubeMetadata


logger = logging.getLogger(__name__)

# Constants (matching your existing UDFs)
CLOUD = 205
NO_DATA = 255
SNOW = 100

# SCF ranges from your existing UDF
SCF_RANGES = [
    (0, 20), (0, 30), (10, 40), (20, 50), (30, 60),
    (40, 70), (50, 80), (60, 90), (70, 100), (80, 100)
]

# Maximum iterations for the while loop
MAX_ITERATIONS = 3

MAX_HR_ITERATIONS = 5  # Maximum HR attempts per main iteration
MIN_PIXEL_CHANGED = 10  # Stop if fewer than X pixels change in HR reconstruction


def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:
    """
    Main UDF function to apply iterative historical reconstruction."""    
    # Fill NaN
    cube = cube.fillna(NO_DATA)
    logger.info(f"Reconstructing cube with shape {cube.shape}")
    logger.info(f"Cube dimensions: {cube.dims}")
    logger.info(f"Cube coordinates: {cube.coords}")
    

    n_days = context.get("n_days_to_reconstruct", 10)
    logger.info(f"Reconstructing last {n_days} days")

    # Check if we have enough data
    total_days = cube.shape[0]
    if total_days <= n_days:
        logger.warning(f"Not enough data: {total_days} total, {n_days} requested")
        # Return original last N days
        return cube.isel(t=slice(-n_days, None))
    
    # Split data
    hist_end = total_days - n_days
    historical_cube = cube.isel(t=slice(0, hist_end))
    reconstruction_cube = cube.isel(t=slice(hist_end, total_days))
    
    logger.info(f"Historical days for CP maps: {historical_cube.shape[0]}")
    logger.info(f"Reconstruction days: {reconstruction_cube.shape[0]}")
    
    # PRE-COMPUTE CP MAPS ONCE from historical period
    historical_snow = historical_cube.isel(bands=0).values
    historical_scf = historical_cube.isel(bands=1).values
    
    historical_cp_maps, historical_occ_maps = compute_historical_cp_maps(
        historical_snow=historical_snow,
        historical_scf=historical_scf
    )


    # Reconstruct each target day
    reconstructed_days = []

    for day_idx in range(n_days):
        logger.info(f"Processing day {day_idx+1}/{n_days}")

        snow_map = reconstruction_cube.isel(t=day_idx, bands=0).values
        scf_map = reconstruction_cube.isel(t=day_idx, bands=1).values


         # Reconstruct this day
        reconstructed_day = hist_rec_iterative(
            snow_map=snow_map,
            scf_map=scf_map,
            hist_snow= historical_snow,
            hist_cp_maps=historical_cp_maps,
            hist_occ_maps=historical_occ_maps,
            scf_ranges=SCF_RANGES
        )
        
        reconstructed_days.append(reconstructed_day)

    reconstructed_snow = np.stack(reconstructed_days, axis=0)
    
    
    # Prepare output - maintain same dimensions as input
    if 't' in cube.dims:
        result = xr.DataArray(
            np.expand_dims(reconstructed_snow, axis=1),
            dims=("t", "bands", "y", "x"),
            coords={
                "t": cube.coords["t"].values[-n_days:],
                "bands": ["reconstructed_snow"],
                "y": cube.coords["y"],
                "x": cube.coords["x"]
            }
        )
    else:
        result = xr.DataArray(
            np.expand_dims(reconstructed_snow, axis=0),
            dims=("bands", "y", "x"),
            coords={
                "bands": ["reconstructed_snow"],
                "y": cube.coords["y"],
                "x": cube.coords["x"]
            }
        )
    
    return result


def hist_rec_iterative(snow_map, scf_map, hist_snow, hist_cp_maps, hist_occ_maps, scf_ranges):
    """
    Iterative reconstruction following the original hist_rec pattern.
    Calls HR and SCF reconstruction functions in a loop.
    """
    iteration = 0

    
    # Main while loop with max iterations
    while iteration < MAX_ITERATIONS:
        iteration += 1
        logger.info(f"Iteration {iteration}")
        
        cloud_mask = snow_map == CLOUD
        
        # Check if we have clouds to process
        if not cloud_mask.any():
            logger.info("No clouds remaining - stopping iterations")
            break
        
        # ----- Step 1: HR cloud reconstruction -----
        # Run HR reconstruction multiple times until little change
        hr_iteration = 0

        
        while hr_iteration < MAX_HR_ITERATIONS:
            hr_iteration += 1
            logger.info(f"  HR sub-iteration {hr_iteration}")
            
            cloud_mask_before = snow_map == CLOUD
            clouds_before = np.sum(cloud_mask_before)
            
            if clouds_before == 0:
                break
            
            # Run HR reconstruction
            reconstructed_hr = hr_reconstruction_single(
                snow_map,
                hist_snow,
                similarity_threshold=0.05,
                min_similar_scenes=5
            )
            
            # Update snow map
            update_mask = cloud_mask_before & (reconstructed_hr != CLOUD)
            pixels_changed = np.sum(update_mask)
            snow_map[update_mask] = reconstructed_hr[update_mask]
            
            logger.info(f"HR changed {pixels_changed} pixels")
            
            # Check if HR is still making progress
            if pixels_changed < MIN_PIXEL_CHANGED:
                logger.info(f"HR convergence - only {pixels_changed} pixels changed")
                break
        
        # Update cloud mask after HR reconstruction
        cloud_mask = snow_map == CLOUD
        
        # ----- Step 2: SCF-based reconstruction -----
        if cloud_mask.any():
            logger.info(" Running SCF-based reconstruction")
            
            # Call your scf_reconstruction function
            reconstructed_scf = scf_reconstruction_single(
                snow_map,
                scf_map,
                hist_cp_maps,
                hist_occ_maps,
                scf_ranges
            )
            
            # Update snow map with SCF reconstruction
            update_mask_scf = cloud_mask & (reconstructed_scf != CLOUD)
            snow_map[update_mask_scf] = reconstructed_scf[update_mask_scf]
        
        # Check for convergence (no clouds left)
        cloud_mask = snow_map == CLOUD
        if not cloud_mask.any():
            logger.info("All clouds processed - stopping iterations")
            break
    
    logger.info(f" Completed after {iteration} iterations")
    return snow_map

#check gap filling; this is also done it in a loop.
# this is also an itteration in a loop based on this daily date thing
def hr_reconstruction_single(current_map, historical_maps, similarity_threshold=0.05, min_similar_scenes=5):
    """
    Single-map version of your HR reconstruction function.
    Adapted from your hr_reconstruction function.
    """
    cloud_mask = current_map == CLOUD
    logger.info(f"HR reconstruction: {np.sum(cloud_mask)} cloudy pixels to process")
    
    if not cloud_mask.any(): #SHould come down to the same critery for if date
        return current_map
    
    # Calculate current SCA from clear pixels
    clear_mask = ~cloud_mask
    clear_pixels = current_map[clear_mask]
    current_sca = np.mean(clear_pixels == SNOW) if len(clear_pixels) > 0 else 0
    
    # Find similar historical scenes
    h_snow_in_clear = (historical_maps == SNOW) & clear_mask
    h_valid_in_clear = (historical_maps <= 100) & clear_mask
    
    h_snow_counts = np.sum(h_snow_in_clear, axis=(1, 2))
    h_valid_counts = np.sum(h_valid_in_clear, axis=(1, 2))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        h_scas = h_snow_counts / h_valid_counts
        h_scas = np.nan_to_num(h_scas, nan=-1.0)
    
    similar_indices = np.where(np.abs(h_scas - current_sca) < similarity_threshold)[0]
    
    if len(similar_indices) < min_similar_scenes:
        logger.info("Not enough similar scenes, skipping HR reconstruction")
        return current_map
    
    logger.info(f"Found {len(similar_indices)} similar scenes for HR reconstruction")
    
    # Reconstruct using similar scenes
    similar_scenes = historical_maps[similar_indices]
    snow_counts = np.sum(similar_scenes == SNOW, axis=0).astype(np.uint16)
    valid_counts = np.sum(similar_scenes <= 100, axis=0).astype(np.uint16)
    
    reconstructed = current_map.copy()
    
    # Apply reconstruction only where historical data is consistent
    is_always_snow = (snow_counts == valid_counts) & (valid_counts > 0)
    is_always_clear = (snow_counts == 0) & (valid_counts > 0)
    
    reconstructed[cloud_mask & is_always_snow] = SNOW
    reconstructed[cloud_mask & is_always_clear] = 0
    
    return reconstructed


#Need to check the selection criteria between jumping form HR and LR
#compute CP dynamically with new cloud masks and see if we can calculate new information. Uncertain if required (MVP) at some point the gap does not become much smaller. 
#track decrease in cliud mask or so or how many pixels ar echanged per iteration and put a treshold on that.
def scf_reconstruction_single(snow_map, scf_map, hist_snow, hist_occ, scf_ranges):
    """
    Single-map version of your SCF reconstruction function.
    Adapted from your scf_reconstruction function.
    """
    cloud_mask = snow_map == CLOUD
    
    if not cloud_mask.any():
        return snow_map
    
    # Calculate min/max SCF
    s_min = get_scf_minmax(snow_map, mode='min')
    s_max = get_scf_minmax(snow_map, mode='max')
    
    # Adjust MODIS SCF to be within bounds
    scf_adj = np.clip(scf_map, s_min, s_max)
    
    # Initialize reconstruction map
    reconstructed = np.full(snow_map.shape, NO_DATA, dtype=np.uint8)
    reconstructed[scf_adj == 100] = SNOW
    reconstructed[scf_adj == 0] = 0
    
    # For each SCF range, fill remaining cloudy pixels
    for i, (r_min, r_max) in enumerate(scf_ranges):
        # Logical range check
        in_range = (scf_adj >= r_min) if i == 0 else (scf_adj > r_min)
        in_range &= (scf_adj <= r_max)
        
        # Historical conditions
        h_snow = hist_snow[i]
        h_occ = hist_occ[i]
        
        # Combine conditions
        update_mask = in_range & (reconstructed == NO_DATA) & (h_occ > 10) & (snow_map > 100)
        
        # Apply historical snow patterns
        reconstructed[update_mask & (h_snow == SNOW)] = SNOW
        reconstructed[update_mask & (h_snow == 0)] = 0
    
    return reconstructed

def get_scf_minmax(snow_map, mode='min'):
    """Your existing function - kept as is"""
    valid_mask = snow_map <= 100
    n_valid = np.sum(valid_mask)
    
    if n_valid < 10:
        return 0.0 if mode == 'min' else 100.0
    
    n_snow = np.sum(snow_map[valid_mask] == SNOW)
    
    if mode == 'min':
        return (n_snow / n_valid) * 100
    else:
        n_cloud = np.sum(snow_map == CLOUD)
        return ((n_snow + n_cloud) / (n_valid + n_cloud)) * 100
    
def compute_historical_cp_maps(historical_snow, historical_scf):
    """
    Compute conditional probability maps ONCE from historical data.
    Returns snow classification (100 or 0) based on probability threshold.
    """
    n_ranges = len(SCF_RANGES)
    n_days = historical_snow.shape[0]
    y_size = historical_snow.shape[1]
    x_size = historical_snow.shape[2]
    
    logger.info(f"Computing CP maps from {n_days} historical days")
    
    # Initialize arrays
    snow_counts = np.zeros((n_ranges, y_size, x_size), dtype=np.uint32)
    occ_counts = np.zeros((n_ranges, y_size, x_size), dtype=np.uint32)
    
    # Accumulate counts over all historical days
    for day_idx in range(n_days):
        scf_day = historical_scf[day_idx]
        snow_day = historical_snow[day_idx]
        
        # Only consider valid pixels (not cloud/no data)
        valid_mask = (scf_day <= 100) & (snow_day <= 100)
        
        for i, (r_min, r_max) in enumerate(SCF_RANGES):
            # Create mask for this SCF range
            if r_min == 0:
                in_range = (scf_day >= r_min) & (scf_day <= r_max)
            else:
                in_range = (scf_day > r_min) & (scf_day <= r_max)
            
            combined_mask = in_range & valid_mask
            
            # Accumulate occurrence count
            occ_counts[i][combined_mask] += 1
            
            # Accumulate snow count
            is_snow = snow_day == SNOW
            snow_counts[i][combined_mask & is_snow] += 1
    
    # Compute probabilities and convert to snow/no-snow classification
    hist_cp_maps = np.zeros((n_ranges, y_size, x_size), dtype=np.uint8)
    hist_occ_maps = occ_counts.astype(np.uint16)
    
    for i in range(n_ranges):
        with np.errstate(divide='ignore', invalid='ignore'):
            probability = np.divide(snow_counts[i], occ_counts[i])
            probability = np.nan_to_num(probability, nan=0.0)
        
        # Convert probability to snow classification (threshold at 50%)
        # 100 = snow, 0 = no snow
        hist_cp_maps[i] = (probability > 0.5) * SNOW
    
    logger.info("CP maps computation completed")
    return hist_cp_maps, hist_occ_maps