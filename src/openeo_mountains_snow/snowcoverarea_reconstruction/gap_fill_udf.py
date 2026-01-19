import xarray as xr
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)

CLOUD = 205
NO_DATA = 255
SNOW = 100

SCF_RANGES = [
        (0, 20),    # 0_20
        (0, 30),    # 0_30  
        (10, 40),   # 10_40
        (20, 50),   # 20_50
        (30, 60),   # 30_60
        (40, 70),   # 40_70
        (50, 80),   # 50_80
        (60, 90),   # 60_90
        (70, 100),  # 70_100
        (80, 100)   # 80_100
    ]
    

def apply_datacube(cube: xr.DataArray, context: Dict) -> xr.DataArray:
    """
    MVP for snow cloud reconstruction with two modes.
    
    Parameters:
    - mode: 'hr' for historical reconstruction, 'scf' for SCF-based reconstruction
    """
    
    # Get parameters with sensible defaults
    mode = context.get("mode", "hr")  # 'hr' or 'scf'
    
    # Fill NaN
    cube = cube.fillna(NO_DATA)
    
    # Get dimensions
    n_times = cube.sizes["t"]
    y_size = cube.sizes["y"]
    x_size = cube.sizes["x"]
    logger.info(f"Data cube dimensions - time: {n_times}, y: {y_size}, x: {x_size}")
    
    # Extract all data needed
    # Historical snow maps (bands 1-10 from first timestep)
    hist_snow_bands = cube.isel(time=0, band=slice(2, 12)).values  # (10, y, x)
    hist_occ_bands = cube.isel(time=0, band=slice(12, 22)).values  # (10, y, x)
    
    # Current snow maps for all timesteps
    snow_current = cube.isel(band=0).values  # HR snow maps
    scf_current = cube.isel(band=1).values   # MODIS SCF
    
   # Choose reconstruction mode
    if mode == 'hr':
        results = hr_reconstruction(
            snow_current, hist_snow_bands, 
            similarity_threshold=0.05, min_similar_scenes=5
        )
    else:  # 'scf'
        results = scf_reconstruction(
            snow_current, scf_current, hist_snow_bands, hist_occ_bands,
            SCF_RANGES
        )
    
    return xr.DataArray(
        results,
        dims=("t", "y", "x"),
        coords={
            "t": cube.coords["t"],
            "y": cube.coords["y"],
            "x": cube.coords["x"]
        }
    )


def hr_reconstruction(current_maps, historical_maps, similarity_threshold=0.05, min_similar_scenes=5):
    """HR reconstruction using historical similarity."""
    n_times = current_maps.shape[0]
    results = np.empty_like(current_maps)
    
    for t in range(n_times):
        logger.info(f"HR reconstruction timestep {t+1}/{n_times}")
        
        current = current_maps[t]
        cloud_mask = current == CLOUD
        
        if not cloud_mask.any():
            results[t] = current
            continue
        
        # Calculate current SCA from clear pixels
        clear_mask = ~cloud_mask
        clear_pixels = current[clear_mask]
        current_sca = np.mean(clear_pixels == SNOW) if len(clear_pixels) > 0 else 0
        
        # Find similar historical scenes
        similar_scenes = []
        
        for hist_idx, hist_map in enumerate(historical_maps):
            # Calculate historical SCA over same clear area
            hist_clear = hist_map[clear_mask]
            valid_hist = hist_clear[hist_clear <= 100]
            
            if len(valid_hist) == 0:
                continue
                
            hist_sca = np.mean(valid_hist == SNOW)
            
            # Check similarity
            if abs(hist_sca - current_sca) < similarity_threshold:
                similar_scenes.append(hist_map)
        
        # Check if enough similar scenes
        if len(similar_scenes) < min_similar_scenes:
            results[t] = current
            continue
        
        # Stack scenes and compute persistence
        scene_stack = np.stack(similar_scenes, axis=0)
        snow_counts = np.sum(scene_stack == SNOW, axis=0)
        valid_counts = np.sum(scene_stack <= 100, axis=0)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            persistence = np.where(valid_counts > 0, snow_counts / valid_counts, 0)
        
        # Fill cloudy pixels
        reconstructed = current.copy()
        reconstructed[cloud_mask & (persistence == 1)] = SNOW
        reconstructed[cloud_mask & (persistence == 0)] = 0
        
        results[t] = reconstructed
    
    return results


def scf_reconstruction(snow_current, scf_current, hist_snow, hist_occ, scf_ranges):
    """
    SCF reconstruction 
    
    Steps:
    1. Calculate min/max SCF from current snow map
    2. Adjust MODIS SCF to be within min/max bounds
    3. For each SCF range, fill cloudy pixels using historical snow maps
    """
    n_times = snow_current.shape[0]
    results = np.empty_like(snow_current)
    
    for t in range(n_times):
        logger.info(f"SCF reconstruction timestep {t+1}/{n_times}")
        
        snow_map = snow_current[t].copy()
        scf_hr = scf_current[t]  # Already at HR resolution
        
        # Create masks
        cloud_mask = snow_map == CLOUD
        
        if not cloud_mask.any():
            results[t] = snow_map
            continue
        
        # Step 1: Calculate min/max SCF from current snow map
        scf_min = calculate_scf_min(snow_map)
        scf_max = calculate_scf_max(snow_map)
        
        # Step 2: Adjust MODIS SCF to be within bounds
        scf_adjusted = scf_hr.copy()
        
        # Ensure MODIS SCF is within min/max bounds
        valid_scf = np.logical_and(scf_adjusted >= 0, scf_adjusted <= 100)
        below_min = np.logical_and(scf_adjusted < scf_min, valid_scf)
        above_max = np.logical_and(scf_adjusted > scf_max, valid_scf)
        
        scf_adjusted[below_min] = scf_min[below_min]
        scf_adjusted[above_max] = scf_max[above_max]
        
        # Step 3: Initialize reconstruction map
        reconstructed = np.full_like(snow_map, NO_DATA, dtype=np.float32)
        
        # Step 4: Apply rule for 0 and 100 SCF
        reconstructed[scf_adjusted == 100] = SNOW
        reconstructed[scf_adjusted == 0] = 0
        
        # Step 5: For each SCF range, fill remaining cloudy pixels
        for i, (range_min, range_max) in enumerate(scf_ranges):
            # Get historical snow map for this SCF range
            hist_snow_map = hist_snow[i]
            occurrence = hist_occ[i]
            
            # Create condition for this SCF range
            if i == 0:  # 0_20 (inclusive)
                in_range = np.logical_and(scf_adjusted >= range_min, scf_adjusted <= range_max)
            else:  # Other ranges (exclusive lower bound)
                in_range = np.logical_and(scf_adjusted > range_min, scf_adjusted <= range_max)
            
            not_filled = reconstructed == NO_DATA

            definitive_snow = hist_snow_map == SNOW  # Always snow historically
            definitive_no_snow = hist_snow_map == 0   # Never snow historically
            definitive_behavior = np.logical_or(definitive_snow, definitive_no_snow)
            
            sufficient_occurrence = occurrence > 10
            
            # c5 = snowMap > 100 (pixel is cloud or invalid in current)
            is_invalid = snow_map > 100  # Cloud (205) or other invalid values
            
            # Combine all conditions
            condition = np.logical_and.reduce((
                in_range,
                not_filled,
                definitive_behavior,
                sufficient_occurrence,
                is_invalid
            ))
            
            # Apply reconstruction: use historical snow value
            # Where condition is True and historical says snow, set to SNOW
            snow_condition = condition & (hist_snow_map == SNOW)
            reconstructed[snow_condition] = SNOW
            
            # Where condition is True and historical says no-snow, set to 0
            no_snow_condition = condition & (hist_snow_map == 0)
            reconstructed[no_snow_condition] = 0
        
        # Update snow map with reconstructed values
        update_mask = reconstructed != NO_DATA
        snow_map[update_mask] = reconstructed[update_mask]
        
        results[t] = snow_map
    
    return results


def calculate_scf_min(snow_map, nv_thres):
    """
    Calculate MINIMUM SCF from snow map.
    
    Based on the original logic: scf(snowMap, ..., scf_min=True)
    The minimum possible SCF given the observed snow pattern.
    """
    # Create mask for valid pixels (not cloud, not nodata)
    valid_mask = snow_map <= 100
    
    # Count valid pixels in the entire processing block
    n_valid = np.sum(valid_mask)
    
    if n_valid < nv_thres:
        # Not enough valid pixels
        return np.full_like(snow_map, 0, dtype=np.float32)
    
    # Count snow pixels among valid pixels
    snow_pixels = snow_map[valid_mask] == SNOW
    n_snow = np.sum(snow_pixels)
    
    # Minimum SCF: assume all uncertain/cloudy pixels are NO-SNOW
    # So minimum SCF is just the observed snow fraction
    min_scf_value = (n_snow / n_valid) * 100 if n_valid > 0 else 0
    
    # Return constant value across the block (OpenEO processes in blocks)
    return np.full_like(snow_map, min_scf_value, dtype=np.float32)


def calculate_scf_max(snow_map, nv_thres):
    """
    Calculate MAXIMUM SCF from snow map.
    
    Based on the original logic: scf(snowMap, ..., scf_max=True)
    The maximum possible SCF given the observed snow pattern.
    """
    # Create mask for valid pixels (not cloud, not nodata)
    valid_mask = snow_map <= 100
    
    # Count valid pixels in the entire processing block
    n_valid = np.sum(valid_mask)
    
    if n_valid < nv_thres:
        # Not enough valid pixels
        return np.full_like(snow_map, 100, dtype=np.float32)
    
    # Count snow pixels among valid pixels
    snow_pixels = snow_map[valid_mask] == SNOW
    n_snow = np.sum(snow_pixels)
    
    # Count cloudy/invalid pixels that could potentially be snow
    # In the original, this would consider the pixel_ratio aggregation
    # Since we're at HR and processing in blocks, we'll use a conservative approach
    
    # Cloud mask
    cloud_mask = snow_map == CLOUD
    n_cloud = np.sum(cloud_mask)
    
    # Maximum SCF: assume all cloudy pixels could be SNOW
    # So maximum snow count = observed snow + all clouds
    max_snow_possible = n_snow + n_cloud
    total_pixels_possible = n_valid + n_cloud
    
    max_scf_value = (max_snow_possible / total_pixels_possible) * 100 if total_pixels_possible > 0 else 100
    
    # Return constant value across the block
    return np.full_like(snow_map, max_scf_value, dtype=np.float32)