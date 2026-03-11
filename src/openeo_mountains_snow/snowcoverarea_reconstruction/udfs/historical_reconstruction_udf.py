

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


# HR parameters
HR_SIMILARITY_THRESHOLD = 0.5
HR_MIN_SIMILAR_SCENES = 5
HR_CLOUD_THRESHOLD = 0.5


# MODIS BASED SCF RECONSTRUCTION PARAMETERS

SCF_OCC_THRESHOLD = 10

SCF_RANGES = [
    (0, 20), (0, 30), (10, 40), (20, 50), (30, 60),
    (40, 70), (50, 80), (60, 90), (70, 100), (80, 100)
]



logger = logging.getLogger(__name__)


def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:
    

    logger.info(f"Reconstructing cube with shape {cube.shape}")

    n_days = context.get("n_days_to_reconstruct", 10)
    n_ranges = len(SCF_RANGES)
    total_days = cube.shape[0]

    if total_days < 2:
        logger.error("Insufficient data: need at least 2 time steps to perform any reconstruction. Returning empty cube.")
        return xr.DataArray(
            np.empty((0, 1, len(cube.y), len(cube.x)), dtype=np.uint8),
            dims=("t", "bands", "y", "x"),
            coords={
                "t": [],
                "bands": ["reconstructed_snow"],
                "y": cube.coords["y"].values,
                "x": cube.coords["x"].values,
            }
        )


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
        reconstructed = hist_rec(
            snow_map=snow_map,
            scf_map=scf_map,
            hist_snow=historical_snow,
            hist_cp_maps=historical_cp_maps,
            hist_occ_maps=historical_occ_maps,
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


def hist_rec(snow_map, scf_map, hist_snow, hist_cp_maps, hist_occ_maps):
    """
    Iterative reconstruction following the original hist_rec pattern.
    Calls HR and SCF reconstruction functions in a loop.
    """
    
    cloud_mask = (snow_map == CLOUD)
    logger.info(f"Cloudy pixels to process: {np.sum(cloud_mask)}")
    
    # Check if we have clouds to process
    if not cloud_mask.any():
        logger.error("No clouds remaining - stopping reconstruction")
        return snow_map
    
    # ----- Step 1: HR cloud reconstruction -----
    # TODO run in itterations>
    # Run HR reconstruction
    reconstructed_hr = hr_reconstruction_single(
        snow_map,
        hist_snow,
        similarity_threshold=HR_SIMILARITY_THRESHOLD,
        min_similar_scenes=HR_MIN_SIMILAR_SCENES,
        cloud_thres = HR_CLOUD_THRESHOLD
    
    )
    
    # Update snow map
    update_mask_hr = cloud_mask & (reconstructed_hr != NO_DATA)
    snow_map[update_mask_hr] = reconstructed_hr[update_mask_hr]
    logger.info(f"HR updated {np.sum(update_mask_hr)} pixels")

    del reconstructed_hr
    del update_mask_hr
    gc.collect()
            
    # Update cloud mask after HR reconstruction
    cloud_mask = (snow_map == CLOUD)
    
    # ----- Step 2: SCF-based reconstruction -----
    if not cloud_mask.any():
        logger.error("No clouds remaining - stopping reconstruction")
        return snow_map
        
    # Call your scf_reconstruction function
    reconstructed_scf = scf_reconstruction_single(
        snow_map,
        scf_map,
        hist_cp_maps,
        hist_occ_maps,
        scf_ranges=SCF_RANGES,
        occ_threshold=SCF_OCC_THRESHOLD
    )
    
    # Update snow map with SCF reconstruction
    update_mask_scf = cloud_mask & (reconstructed_scf != NO_DATA) 
    snow_map[update_mask_scf] = reconstructed_scf[update_mask_scf]
    logger.info(f"SCF updated {np.sum(update_mask_scf)} pixels")

    
    del reconstructed_scf
    del update_mask_scf
    gc.collect()
        
    return snow_map

#check gap filling; this is also done it in a loop.
# this is also an itteration in a loop based on this daily date thing
def hr_reconstruction_single(snow_map,
                             historical_maps,
                             similarity_threshold=0.1,
                             min_similar_scenes=5,
                             cloud_thres=0.3):
    """
    HR reconstruction with:
    fractional SCA computation
    cloud contamination rejection
    """
    logger.info("Starting HR reconstruction")
    cloud_mask = snow_map == CLOUD

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
    if cloud_fraction >= cloud_thres:
        logger.error("Too cloudy - skipping HR reconstruction")
        return snow_map
 

    current_sca = np.sum(snow_map[valid_mask]) / N_valid

    h_valid = (historical_maps <= 100) & valid_mask


    h_valid_counts = np.sum(h_valid, axis=(1, 2))

    h_sca_sums = np.sum(
        np.where(h_valid, historical_maps, 0), #replace invalid by 0
        axis=(1, 2)
    )

    with np.errstate(divide='ignore', invalid='ignore'):
        h_scas = h_sca_sums / h_valid_counts


    similar_indices = np.where((h_valid_counts > 0) & ((np.abs(h_scas - current_sca)/len(h_scas)) < similarity_threshold))[0] #TODO probably need agglomerated statistiscs on the difference

    if len(similar_indices) < min_similar_scenes:
        return snow_map

    similar_scenes = historical_maps[similar_indices]

    snow_counts = np.sum(similar_scenes == SNOW, axis=0)
    valid_counts = np.sum(similar_scenes <= 100, axis=0)
    
    is_always_snow = (snow_counts == valid_counts) & (valid_counts > 0)
    is_always_clear = (snow_counts == 0) & (valid_counts > 0)


    reconstructed = np.full(snow_map.shape, NO_DATA)
    reconstructed[cloud_mask & is_always_snow] = SNOW
    reconstructed[cloud_mask & is_always_clear] = 0
    logger.info(f"HR reconstruction updated {np.sum(cloud_mask & (reconstructed != NO_DATA))} pixels")



    return reconstructed


#Need to check the selection criteria between jumping form HR and LR
#compute CP dynamically with new cloud masks and see if we can calculate new information. Uncertain if required (MVP) at some point the gap does not become much smaller. 
#track decrease in cliud mask or so or how many pixels ar echanged per iteration and put a treshold on that.
def scf_reconstruction_single(snow_map, scf_map, hist_snow, hist_occ, scf_ranges, occ_threshold=5):
    """
    Single-map version of your SCF reconstruction function.
    Adapted from your scf_reconstruction function.
    """
    logger.info("Starting SCF-based reconstruction")
    cloud_mask = snow_map == CLOUD

    
    if not cloud_mask.any():
        return snow_map
    
    # Calculate min/max SCF
    s_min = get_scf_minmax(snow_map, mode='min')
    s_max = get_scf_minmax(snow_map, mode='max')
    
    # Adjust MODIS SCF to be within bounds
    scf_adj = np.clip(scf_map, s_min, s_max)
    scf_adj = scf_adj*100 # scale to percent

    hr_valid_mask = snow_map != CLOUD
    
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
        h_snow = hist_snow[i]*100
        h_occ = hist_occ[i]
        
        # Combine conditions
        update_mask = in_range & (reconstructed == NO_DATA) & (h_occ > occ_threshold) & (snow_map > 100)
        logger.info(f"Range {r_min}-{r_max}%: Updating {np.sum(update_mask)} pixels based on historical occurrence > {occ_threshold}")

        # Apply historical snow patterns
        reconstructed[update_mask & (h_snow == SNOW)] = SNOW
        reconstructed[update_mask & (h_snow == 0)] = 0


    
    return reconstructed

def get_scf_minmax(snow_map, mode="min"):
    
    snow_mask = snow_map == SNOW
    cloud_mask = snow_map == CLOUD

    n_snow = np.sum(snow_mask)
    n_cloud = np.sum(cloud_mask)
    total_pixels = snow_map.size

    if mode == "min":
        return n_snow / total_pixels
    
    elif mode == "max":
        return (n_snow + n_cloud) / total_pixels