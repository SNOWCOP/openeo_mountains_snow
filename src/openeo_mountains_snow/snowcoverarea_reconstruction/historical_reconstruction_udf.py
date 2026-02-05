#itterative reconstruction
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
MAX_ITERATIONS = 10


def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:
    """
    Main UDF function to apply iterative historical reconstruction."""    
    # Fill NaN
    cube = cube.fillna(NO_DATA)
    logger.info(f"Reconstructing cube with shape {cube.shape}")
    logger.info(f"Cube dimensions: {cube.dims}")
    logger.info(f"Cube coordinates: {cube.coords}")
    

    n_days = context.get("n_days_to_reconstruct", 10)
    n_ranges = len(SCF_RANGES)

    logger.info(f"Reconstructing last {n_days} days")

    # Check if we have enough data
    total_days = cube.shape[0]
    if total_days <= n_days:
        logger.warning(f"Not enough data: {total_days} total, {n_days} requested")
        # Return original last N days
        return cube.isel(t=slice(-n_days, None))
    
    # Split data TODO avoid split
    hist_end = total_days - n_days
    reconstruction_cube = cube.isel(t=slice(hist_end, total_days))
    
    historical_snow = cube.isel(bands=0).values    
    historical_cp_maps = cube.isel(t=0,bands=slice(2, 2 + n_ranges -1)).values
    historical_occ_maps = cube.isel(t=0,bands=slice(2 + n_ranges, 2 + 2 * n_ranges)).values()

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
            similarity_threshold=0.05,
            min_similar_scenes=5
        )
        
        # Update snow map
        update_mask = cloud_mask & (reconstructed_hr != NO_DATA)
        snow_map[update_mask] = reconstructed_hr[update_mask]
        logger.info(f"HR update non NAN {np.sum((reconstructed_hr != NO_DATA))} pixels")
                
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
        logger.info(f"SCF update non NAN {np.sum((reconstructed_scf != NO_DATA))} pixels")
        
        
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

    similar_indices = np.where((h_valid_counts > 0) & (np.abs(h_scas - current_sca) < similarity_threshold))[0]

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
    
