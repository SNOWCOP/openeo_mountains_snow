import xarray as xr
import numpy as np
import logging

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
MAX_ITERATIONS = 2

def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:
    """
    Complete hist_rec implementation that:
    1. Calls HR reconstruction (cloud gap-filling)
    2. Calls SCF reconstruction (downscaling)
    3. Repeats in a while loop until convergence or max iterations
    """
    logger.info(f"Starting hist_rec with cube shape: {cube.shape}, dims: {cube.dims}")
    
    # Fill NaN (your standard approach)
    cube = cube.fillna(NO_DATA).astype(np.uint8)
    
    # Extract bands following your pattern
    snow_current = cube.sel(bands=cube.bands[0]).values
    scf_current = cube.sel(bands=cube.bands[1]).values
    
    # Historical CP and occurrence maps, time index is fictive here
    hist_cp_maps = cube.isel(bands=slice(2, 12), t=0).values.squeeze()  # 10 CP maps
    hist_occ_maps = cube.isel(bands=slice(12, 22), t=0).values.squeeze()  # 10 occurrence maps
    
    # Process each timestep (try to parallelize with apply_meighborhoud P1D)
    results = []
    for t in range(snow_current.shape[0]):
        logger.info(f"Processing timestep {t+1}/{snow_current.shape[0]}")
        
        snow_map = snow_current[t].copy()
        scf_map = scf_current[t].copy()
        
        # Run iterative reconstruction
        reconstructed_snow = hist_rec_iterative(
            snow_map=snow_map,
            scf_map=scf_map,
            hist_cp_maps=hist_cp_maps,
            hist_occ_maps=hist_occ_maps,
            scf_ranges=SCF_RANGES
        )
        
        results.append(reconstructed_snow)
    
    # Expand results to include band dimension
    results = np.stack(results, axis=0)
    results = np.expand_dims(results, axis=1)
    
    return xr.DataArray(
        results,
        dims=("t", "bands", "y", "x"),
        coords={
            "t": cube.coords["t"],
            "bands": ["reconstructed_snow"],
            "y": cube.coords["y"],
            "x": cube.coords["x"]
        }
    )

def hist_rec_iterative(snow_map, scf_map, hist_cp_maps, hist_occ_maps, scf_ranges):
    """
    Iterative reconstruction following the original hist_rec pattern.
    Calls HR and SCF reconstruction functions in a loop.
    """
    iteration = 0
    
    # Main while loop with max iterations
    while iteration < MAX_ITERATIONS:
        iteration += 1
        logger.info(f"  Iteration {iteration}")
        
        cloud_mask = snow_map == CLOUD
        
        # Check if we have clouds to process
        if not cloud_mask.any():
            logger.info("    No clouds remaining - stopping iterations")
            break
        
        # ----- Step 1: HR cloud reconstruction -----
        logger.info("    Running HR cloud reconstruction")
        
        # Create a single timestep version of your hr_reconstruction
        reconstructed_hr = hr_reconstruction_single(
            snow_map, 
            hist_cp_maps,
            similarity_threshold=0.05,  # Your default value
            min_similar_scenes=5        # Your default value
        )
        
        # Update snow map with HR reconstruction
        update_mask = cloud_mask & (reconstructed_hr != CLOUD)
        snow_map[update_mask] = reconstructed_hr[update_mask]
        
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
            update_mask_scf = cloud_mask & (reconstructed_scf != NO_DATA)
            snow_map[update_mask_scf] = reconstructed_scf[update_mask_scf]
        
        # Check for convergence (no clouds left)
        cloud_mask = snow_map == CLOUD
        if not cloud_mask.any():
            logger.info(" All clouds processed - stopping iterations")
            break
    
    logger.info(f" Completed after {iteration} iterations")
    return snow_map

def hr_reconstruction_single(current_map, historical_maps, similarity_threshold=0.05, min_similar_scenes=5):
    """
    Single-map version of your HR reconstruction function.
    Adapted from your hr_reconstruction function.
    """
    cloud_mask = current_map == CLOUD
    
    if not cloud_mask.any():
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
        logger.info("    Not enough similar scenes, skipping HR reconstruction")
        return current_map
    
    logger.info(f"    Found {len(similar_indices)} similar scenes for HR reconstruction")
    
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