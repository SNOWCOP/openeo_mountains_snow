

import xarray as xr
import logging
import numpy as np
import pandas as pd
import gc



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
MAX_ITERATIONS = 1

logger = logging.getLogger(__name__)


def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:
    

    logger.info(f"Reconstructing cube with shape {cube.shape}")

    n_days = context.get("n_days_to_reconstruct", 10)
    n_ranges = len(SCF_RANGES)
    total_days = cube.shape[0]

    if total_days < 1:
        raise ValueError("Cube must contain at least one time step ")


    n_days_actual = min(n_days, total_days - 1)
    if n_days_actual < n_days:
        logger.warning(f"Only {n_days_actual} of {n_days} requested days can be reconstructed since the patch has only {total_days} time steps")
    
    hist_end = total_days - n_days_actual

    historical_cp_maps = cube.isel(t=0, bands=slice(2, 2 + n_ranges)).values.astype(np.uint8)
    historical_occ_maps = cube.isel(t=0, bands=slice(2 + n_ranges, 2 + 2 * n_ranges)).values.astype(np.uint8)
    historical_snow = cube.isel(bands=0).values.astype(np.uint8)

    np.nan_to_num(historical_cp_maps, nan=NO_DATA, copy=False)
    np.nan_to_num(historical_occ_maps, nan=NO_DATA, copy=False)
    np.nan_to_num(historical_snow, nan=NO_DATA, copy=False)

    
    coords_t = cube.coords["t"].values[hist_end:hist_end + n_days_actual]
    coords_y = cube.coords["y"].values
    coords_x = cube.coords["x"].values


    reconstructed_days  = []
    for day_idx in range(n_days_actual):

        snow_map  = cube.isel(t=hist_end + day_idx, bands=0).values.astype(np.uint8)
        scf_map  = cube.isel(t=hist_end + day_idx, bands=1).values.astype(np.uint8)
        
        np.nan_to_num(snow_map, nan=NO_DATA, copy=False)
        np.nan_to_num(scf_map,  nan=NO_DATA, copy=False)

         # Run reconstruction (modifies snow_map in-place and returns it)
        reconstructed = hist_rec_iterative(
            snow_map=snow_map,
            scf_map=scf_map,
            hist_snow=historical_snow,
            hist_cp_maps=historical_cp_maps,
            hist_occ_maps=historical_occ_maps,
            scf_ranges=SCF_RANGES,
            day_idx=day_idx
        )
        reconstructed_days.append(reconstructed)  # reconstructed is already uint8

        # Free per-day arrays (they will be overwritten next iteration)
        # but explicit deletion helps if you're in a tight loop
        del snow_map, scf_map, reconstructed
        gc.collect()  

    del cube
    gc.collect()

    # Build result
    reconstructed_snow = np.stack(reconstructed_days, axis=0)
    reconstructed_snow = np.expand_dims(reconstructed_snow, axis=1)
    del reconstructed_days
    gc.collect()

     # (n_days, 1, y, x)

    result = xr.DataArray(
        reconstructed_snow,
        dims=("t", "bands", "y", "x"),
        coords={
            "t": coords_t,
            "bands": ["reconstructed_snow"],
            "y": coords_y,
            "x": coords_x
        }
    )

    return result


def hist_rec_iterative(snow_map, scf_map, hist_snow, hist_cp_maps, hist_occ_maps, scf_ranges, day_idx = None):
    """
    Iterative reconstruction following the original hist_rec pattern.
    Calls HR and SCF reconstruction functions in a loop.
    """
    iteration = 0
    
    # Main while loop with max iterations
    while iteration < MAX_ITERATIONS:
        iteration += 1

        logger.info(f"Iteration {iteration}")
        
        cloud_mask = (snow_map == CLOUD)
        logger.info(f"Cloudy pixels to process: {np.sum(cloud_mask)}")
        
        # Check if we have clouds to process
        if not cloud_mask.any():
            logger.info("No clouds remaining - stopping iterations")
            break
        
        # ----- Step 1: HR cloud reconstruction -----
        # TODO run in itterations>
        # Run HR reconstruction
        reconstructed_hr = hr_reconstruction_single(
            snow_map,
            hist_snow,
            similarity_threshold=0.005,
            min_similar_scenes=5
        )
        
        # Update snow map
        update_mask_hr = cloud_mask & (reconstructed_hr != NO_DATA)
        snow_map[update_mask_hr] = reconstructed_hr[update_mask_hr]
        logger.info(f"HR update non NAN {np.sum((reconstructed_hr != NO_DATA))} pixels")

        del reconstructed_hr
        del update_mask_hr
        gc.collect()
                
        # Update cloud mask after HR reconstruction
        cloud_mask = (snow_map == CLOUD)
        
        # ----- Step 2: SCF-based reconstruction -----
        if not cloud_mask.any():
            logger.info("No clouds remaining - stopping iterations")
            break
            
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
        
        del reconstructed_scf
        del update_mask_scf
        gc.collect()
        
    logger.info(f" Completed after {iteration} iterations")
    return snow_map

#check gap filling; this is also done it in a loop.
# this is also an itteration in a loop based on this daily date thing
def hr_reconstruction_single(snow_map,
                             historical_maps,
                             similarity_threshold=0.05,
                             min_similar_scenes=5,
                             cloud_thres=0.3):
    """
    HR reconstruction with:
    fractional SCA computation
    cloud contamination rejection
    """

    cloud_mask = snow_map == CLOUD
    logger.info(f"HR reconstruction: {np.sum(cloud_mask)} cloudy pixels to process")

    if not cloud_mask.any():
        return snow_map

    # -----------------------------
    # VALID PIXELS (partial_SCA logic)
    # -----------------------------
    valid_mask = (snow_map <= 100) & (~cloud_mask)

    N_total = np.sum(~cloud_mask)
    N_valid = np.sum(valid_mask)

    if N_total == 0 or N_valid == 0:
        return snow_map

    cloud_fraction = (N_total - N_valid) / N_total

    # currently skipping as this is quite agressive patch based
    #if cloud_fraction >= cloud_thres:
    #    logger.info("Too cloudy - skipping HR reconstruction")
    #    return current_map

    current_sca = np.sum(snow_map[valid_mask]) / N_valid

    h_valid = (historical_maps <= 100) & valid_mask
    h_valid_counts = np.sum(h_valid, axis=(1, 2))

    h_sca_sums = np.sum(
        np.where(h_valid, historical_maps, 0), #replace invalid by 0
        axis=(1, 2)
    )

    with np.errstate(divide='ignore', invalid='ignore'):
        h_scas = h_sca_sums / h_valid_counts

    similar_indices = np.where((h_valid_counts > 0) & (np.sum(np.abs(h_scas - current_sca)) < similarity_threshold))[0] #TODO probably need agglomerated statistiscs on the difference

    if len(similar_indices) < min_similar_scenes:
        logger.info("Not enough similar scenes, skipping HR reconstruction")
        return snow_map

    logger.info(f"Found {len(similar_indices)} similar scenes")

    similar_scenes = historical_maps[similar_indices]

    snow_counts = np.sum(similar_scenes == SNOW, axis=0)
    valid_counts = np.sum(similar_scenes <= 100, axis=0)

    reconstructed = np.full(snow_map.shape, NO_DATA)
    
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

    hr_valid_mask = snow_map != 255
    scf_adj[hr_valid_mask] = snow_map[hr_valid_mask]
    
    # Initialize reconstruction map
    reconstructed = np.full(snow_map.shape, NO_DATA)
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
        update_mask = in_range & (reconstructed == CLOUD) & (h_occ > 10) & (snow_map > 100)
        
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
