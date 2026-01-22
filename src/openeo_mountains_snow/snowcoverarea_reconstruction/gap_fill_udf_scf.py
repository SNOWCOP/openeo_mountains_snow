import xarray as xr
import numpy as np
import logging

logger = logging.getLogger(__name__)

CLOUD = 205
NO_DATA = 255
SNOW = 100

SCF_RANGES = [
    (0, 20), (0, 30), (10, 40), (20, 50), (30, 60),
    (40, 70), (50, 80), (60, 90), (70, 100), (80, 100)
]
    

def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:
    """
    MVP for snow cloud reconstruction with two modes.
    
    Parameters:
   'scf' for SCF-based reconstruction
    """
    
    # Get parameters with sensible defaults
    logger.info(f"Received data with shape: {cube.shape}, dims: {cube.dims}")
    logger.info(f"Context: {context}")

    # Fill NaN
    cube = cube.fillna(NO_DATA).astype(np.uint8)
    
    snow_current = cube.sel(bands=cube.bands[0]).values
    scf_current = cube.sel(bands=cube.bands[1]).values
    hist_snow_bands = cube.isel(bands=slice(2, 12), t=0).values.squeeze()
    hist_occ_bands = cube.isel(bands=slice(12, 22), t=0).values.squeeze()

    
    results = scf_reconstruction(
        snow_current, scf_current, hist_snow_bands, hist_occ_bands,
        SCF_RANGES
    )

    # Expand results to include band dimension
    results = np.expand_dims(results, axis=1)  # shape becomes (t, bands, y, x)
    return xr.DataArray(
        results,
        dims=("t", "bands", "y", "x"),
        coords={
            "t": cube.coords["t"],
            "bands": [0],
            "y": cube.coords["y"],
            "x": cube.coords["x"]
        }
    )


def scf_reconstruction(snow_current, scf_current, hist_snow, hist_occ, scf_ranges):
    """
    SCF reconstruction 
    
    Steps:
    1. Calculate min/max SCF from current snow map
    2. Adjust MODIS SCF to be within min/max bounds
    3. For each SCF range, fill cloudy pixels using historical snow maps
    """
    results = np.empty_like(snow_current)
    
    for t in range(snow_current.shape[0]):
        logger.info(f"SCF reconstruction timestep {t+1}/{snow_current.shape[0]}")
        
        snow_map = snow_current[t].copy()
        scf_hr = scf_current[t]
        cloud_mask = (snow_map == CLOUD)
        
        
        if not cloud_mask.any():
            results[t] = snow_map
            continue
        
        # Step 1: Calculate min/max SCF from current snow map
        s_min = get_scf_minmax(snow_map, mode='min')
        s_max = get_scf_minmax(snow_map, mode='max')
        
        # Step 2: Adjust MODIS SCF to be within bounds
        scf_adj = np.clip(scf_hr, s_min, s_max)
        
        # Step 3: Initialize reconstruction map
        reconstructed = np.full(snow_map.shape, NO_DATA, dtype=np.uint8)
        reconstructed[scf_adj == 100] = SNOW
        reconstructed[scf_adj == 0] = 0
        
        # Step 5: For each SCF range, fill remaining cloudy pixels
        for i, (r_min, r_max) in enumerate(scf_ranges):
            # Logical range check
            in_range = (scf_adj >= r_min) if i == 0 else (scf_adj > r_min)
            in_range &= (scf_adj <= r_max)
            
            # Historical conditions
            h_snow = hist_snow[i]
            h_occ = hist_occ[i]
            
            # Combine conditions to modify only necessary pixels
            update_mask = in_range & (reconstructed == NO_DATA) & (h_occ > 10) & (snow_map > 100)
            
            # Apply historical snow patterns
            reconstructed[update_mask & (h_snow == SNOW)] = SNOW
            reconstructed[update_mask & (h_snow == 0)] = 0
        
        # Final merge
        mask = reconstructed != NO_DATA
        snow_map[mask] = reconstructed[mask]
        results[t] = snow_map
        
    return results


def get_scf_minmax(snow_map, mode='min'):
    """Returns a single float value instead of a full array."""
    valid_mask = snow_map <= 100
    n_valid = np.sum(valid_mask)
    
    if n_valid < 10: #preset threshold
        return 0.0 if mode == 'min' else 100.0
    
    n_snow = np.sum(snow_map[valid_mask] == SNOW)
    
    if mode == 'min':
        return (n_snow / n_valid) * 100
    else:
        n_cloud = np.sum(snow_map == CLOUD)
        return ((n_snow + n_cloud) / (n_valid + n_cloud)) * 100