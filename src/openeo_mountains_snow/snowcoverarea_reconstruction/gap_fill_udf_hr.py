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
    - mode: 'hr' for historical reconstruction, 
    """
    
    # Get parameters with sensible defaults
    logger.info(f"Received data with shape: {cube.shape}, dims: {cube.dims}")
    
    # Fill NaN
    cube = cube.fillna(NO_DATA).astype(np.uint8)
    
    snow_current = cube.sel(bands=cube.bands[0]).values
    hist_snow_bands = cube.isel(bands=slice(2, 12), t=0).values.squeeze()
    
    results = hr_reconstruction(
        snow_current, hist_snow_bands, 
        similarity_threshold=0.05, min_similar_scenes=5
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


def hr_reconstruction(current_maps, historical_maps, similarity_threshold=0.05, min_similar_scenes=5):
    """HR reconstruction using historical similarity."""
    results = np.empty_like(current_maps)
    
    # Ensure historical_maps is 3D (scenes, y, x) 
    # This prevents broadcasting errors with the 2D 'current' map


    for t in range(current_maps.shape[0]):
        logger.info(f"HR reconstruction timestep {t+1}/{current_maps.shape[0]}")
        
        current = current_maps[t] # Shape: (y, x)
        cloud_mask = (current == CLOUD)
        
        if not cloud_mask.any():
            results[t] = current
            continue
        
        # Calculate current SCA from clear pixels
        clear_mask = ~cloud_mask
        clear_pixels = current[clear_mask]
        current_sca = np.mean(clear_pixels == SNOW) if len(clear_pixels) > 0 else 0
        
        # --- FIX: Finding Similar Scenes ---
        # We broadcast the 2D clear_mask across all historical scenes
        h_snow_in_clear = (historical_maps == SNOW) & clear_mask
        h_valid_in_clear = (historical_maps <= 100) & clear_mask
        
        # Sum across spatial axes (1, 2) to get 1D counts for the 10 maps
        h_snow_counts = np.sum(h_snow_in_clear, axis=(1, 2))
        h_valid_counts = np.sum(h_valid_in_clear, axis=(1, 2))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            h_scas = h_snow_counts / h_valid_counts
            h_scas = np.nan_to_num(h_scas, nan=-1.0)
            
        similar_indices = np.where(np.abs(h_scas - current_sca) < similarity_threshold)[0]
            
        if len(similar_indices) < min_similar_scenes:
            logger.info(f"Not enough similar scenes found for timestep {t+1}, skipping.")
            results[t] = current
            continue

        # --- FIX: Localized Persistence ---
        # Slice only the scenes that were similar
        similar_scenes = historical_maps[similar_indices]

        # Aggregate the similar scenes into a 2D map (axis 0 is the scene axis)
        snow_counts = np.sum(similar_scenes == SNOW, axis=0).astype(np.uint16)
        valid_counts = np.sum(similar_scenes <= 100, axis=0).astype(np.uint16)

        reconstructed = current.copy()
        
        # Only fill where there are clouds AND historical data is consistent
        # Use bitwise & for masks
        is_always_snow = (snow_counts == valid_counts) & (valid_counts > 0)
        is_always_clear = (snow_counts == 0) & (valid_counts > 0)

        reconstructed[cloud_mask & is_always_snow] = SNOW
        reconstructed[cloud_mask & is_always_clear] = 0
            
        results[t] = reconstructed
        
    return results