import xarray as xr
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)

CLOUD = 205
SNOW = 100

def apply_datacube(cube: xr.DataArray, context: Dict) -> xr.DataArray:
    """
    MVP for snow cloud reconstruction with two modes.
    
    Parameters:
    - mode: 'hr' for historical reconstruction, 'scf' for SCF-based reconstruction
    - similarity_threshold: Max SCA difference (default: 5%)
    - min_similar_scenes: Minimum similar historical scenes (default: 5)
    - consensus_threshold: For SCF mode (default: 0.8)
    """
    
    # Get parameters with sensible defaults
    mode = context.get("mode", "hr")  # 'hr' or 'scf'
    similarity_threshold = context.get("similarity_threshold", 0.05)
    min_similar_scenes = context.get("min_similar_scenes", 5)
    consensus_threshold = context.get("consensus_threshold", 0.8)
    
    # Fill NaN
    cube = cube.fillna(CLOUD)
    
    # Get dimensions
    n_times = cube.sizes["time"]
    y_size = cube.sizes["y"]
    x_size = cube.sizes["x"]
    logger.info(f"Data cube dimensions - time: {n_times}, y: {y_size}, x: {x_size}")
    
    # Extract all data needed
    # Historical snow maps (bands 1-10 from first timestep)
    historical_maps = cube.isel(time=0, band=slice(1, 11)).values  # (10, y, x)
    
    # Current snow maps for all timesteps
    current_maps = cube.isel(band=0).values  # (time, y, x)
    
    # Choose reconstruction mode
    if mode == 'hr':
        reconstructed = highres_reconstruction(
            current_maps, historical_maps, similarity_threshold, min_similar_scenes
        )
    else:  # 'scf'
        reconstructed = scf_reconstruction(
            current_maps, historical_maps, consensus_threshold
        )
    
    return xr.DataArray(
        reconstructed,
        dims=("time", "y", "x"),
        coords={
            "time": cube.coords["time"],
            "y": cube.coords["y"],
            "x": cube.coords["x"]
        }
    )


def highres_reconstruction(current_maps, historical_maps, similarity_threshold, min_similar_scenes):
    """
    HR reconstruction using historical similarity.
    
    Args:
        current_maps: (time, y, x) array of current snow maps
        historical_maps: (10, y, x) array of historical snow maps
        similarity_threshold: Max allowed SCA difference
        min_similar_scenes: Minimum similar scenes needed
    """
    logger.info("Starting historical reconstruction")
    n_times = current_maps.shape[0]
    results = np.empty_like(current_maps)
    
    # Precompute cloud masks for all timesteps
    cloud_masks = current_maps == CLOUD  # (time, y, x)
    clear_masks = ~cloud_masks
    
    # Calculate SCA for each timestep in vectorized way
    # We need to compute SCA per timestep: mean of snow pixels in clear areas
    current_scas = np.zeros(n_times)
    for t in range(n_times):
        clear_pixels = current_maps[t][clear_masks[t]]
        if len(clear_pixels) > 0:
            current_scas[t] = np.mean(clear_pixels == SNOW)
    
    # Process each timestep
    for t in range(n_times):
        logger.info(f"Reconstructing timestep {t+1}/{n_times}")
        current = current_maps[t]
        cloud_mask = cloud_masks[t]
        
        # If no clouds, keep as is
        if not cloud_mask.any():
            results[t] = current
            continue
        
        current_sca = current_scas[t]
        
        # We need to compare each historical map with current clear area
        clear_area = clear_masks[t]
        
        if not clear_area.any():
            logger.info("No clear pixels in current map, using simple consensus")
            # No clear pixels to compare, use simple consensus
            snow_counts = np.sum(historical_maps == SNOW, axis=0)
            valid_counts = np.sum(historical_maps <= 100, axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                snow_prob = np.where(valid_counts > 0, snow_counts / valid_counts, 0)
        else:
            # Calculate historical SCAs over current clear area
            # This is the bottleneck, but we vectorize over historical maps
            historical_scas = np.zeros(historical_maps.shape[0])
            for h in range(historical_maps.shape[0]):
                logger.debug(f"Calculating SCA for historical map {h+1}/10")
                hist_clear = historical_maps[h][clear_area]
                valid_hist = hist_clear[hist_clear <= 100]
                if len(valid_hist) > 0:
                    historical_scas[h] = np.mean(valid_hist == SNOW)
            
            # Find similar scenes
            sca_diffs = np.abs(historical_scas - current_sca)
            similar_indices = np.where(sca_diffs < similarity_threshold)[0]
            
            if len(similar_indices) >= min_similar_scenes:
                logger.info(f"Found {len(similar_indices)} similar historical scenes")
                # Use only similar scenes
                similar_maps = historical_maps[similar_indices]
                snow_counts = np.sum(similar_maps == SNOW, axis=0)
                valid_counts = np.sum(similar_maps <= 100, axis=0)
                with np.errstate(divide='ignore', invalid='ignore'):
                    snow_prob = np.where(valid_counts > 0, snow_counts / valid_counts, 0)
            else:
                logger.info(f"Only {len(similar_indices)} similar scenes found, using all historical maps")
                # Not enough similar scenes, use all
                snow_counts = np.sum(historical_maps == SNOW, axis=0)
                valid_counts = np.sum(historical_maps <= 100, axis=0)
                with np.errstate(divide='ignore', invalid='ignore'):
                    snow_prob = np.where(valid_counts > 0, snow_counts / valid_counts, 0)
        
        # Fill cloudy pixels based on snow probability
        reconstructed = current.copy()
        
        # Conservative: only fill when unanimous
        always_snow = (snow_prob == 1)
        never_snow = (snow_prob == 0)
        
        reconstructed[cloud_mask & always_snow] = SNOW
        reconstructed[cloud_mask & never_snow] = 0
        
        results[t] = reconstructed
    
    return results


def scf_reconstruction(current_maps, historical_maps, consensus_threshold):
    """
    Simple SCF-based reconstruction using historical consensus.
    
    Args:
        current_maps: (time, y, x) array of current snow maps
        historical_maps: (10, y, x) array of historical snow maps
        consensus_threshold: Min consensus to fill (default: 0.8)
    """
    logger.info("Starting SCF-based reconstruction")
    n_times = current_maps.shape[0]
    results = np.empty_like(current_maps)
    
    # Calculate historical consensus once (across all historical maps)
    snow_counts = np.sum(historical_maps == SNOW, axis=0)
    valid_counts = np.sum(historical_maps <= 100, axis=0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        snow_frequency = np.where(valid_counts > 0, snow_counts / valid_counts, 0)
    
    # Precompute masks for faster iteration
    always_snow_mask = snow_frequency >= consensus_threshold
    never_snow_mask = snow_frequency <= (1 - consensus_threshold)
    
    # Process each timestep
    for t in range(n_times):
        logger.info(f"Reconstructing timestep {t+1}/{n_times}")
        current = current_maps[t].copy()
        cloud_mask = current == CLOUD
        
        # Apply consensus-based filling
        current[cloud_mask & always_snow_mask] = SNOW
        current[cloud_mask & never_snow_mask] = 0
        
        results[t] = current
    
    return results