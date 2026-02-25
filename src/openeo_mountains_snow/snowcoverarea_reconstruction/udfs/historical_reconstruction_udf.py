

import xarray as xr
import boto3
import logging
import tempfile
import os
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

class Checkpoint:
    
    def __init__(self, access_key, secret_key, token, bucket, prefix, endpoint):
        self.bucket = bucket
        self.prefix = prefix.rstrip('/')
        self.y_coords = None
        self.x_coords = None
        self.prefix = f"{self.prefix}"

        
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_session_token=token,
            endpoint_url=endpoint
        )
        logger.info(f"S3 checkpoint ready: {bucket}/{prefix}")
    
    def set_reference(self, cube):
        """Store reference coordinates as plain numpy arrays (safe even after cube is closed)."""
        if 'y' in cube.coords:
            self.y_coords = cube.y.values
        if 'x' in cube.coords:
            self.x_coords = cube.x.values
    
    def save(self, data, **tags):
        """Save data to S3 as NetCDF. Works with numpy arrays or xarray DataArrays."""

        if isinstance(data, np.ndarray):
            if self.y_coords is None or self.x_coords is None:
                logger.warning("No reference coordinates. Call set_reference() first.")
                return
            data = xr.DataArray(
                data,
                dims=('y', 'x'),
                coords={'y': self.y_coords, 'x': self.x_coords}
            )
        

        if isinstance(data, xr.DataArray):
            data = data.load()
        else:
            raise TypeError(f"Expected numpy.ndarray or xr.DataArray, got {type(data)}")
        
        parts = []
        for key in sorted(tags.keys()):
            val = tags[key]
            if isinstance(val, int):
                parts.append(f"{key}{val:03d}")
            else:
                parts.append(f"{key}_{val}")
        
        if 'x' in data.coords and 'y' in data.coords:
            x_min = int(data.x.min())
            y_min = int(data.y.min())
            parts.append(f"x{x_min}_y{y_min}")
        
        if 't' in data.coords and len(data.t) > 0:
            t_str = pd.to_datetime(data.t.values[0]).strftime('%Y%m%d')
            parts.append(f"t{t_str}")
        
        parts.append("_".join(f"{d}{s}" for d, s in data.sizes.items()))
        filename = "_".join(parts) + ".nc"
        key = f"{self.prefix}/{filename}"
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp:
                tmp_path = tmp.name
            
                data.to_dataset(name='data').to_netcdf(
                    tmp_path,
                    format='NETCDF4',
                    engine='netcdf4'
                )
    
            # Upload to S3 using boto3
            self.s3.upload_file(
                Filename=tmp_path,
                Bucket=self.bucket,
                Key=key
            )
            
            # Clean up
            os.unlink(tmp_path)
            logger.info(f"✓ {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"✗ {filename}: {e}")
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return None



def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:

    cfg = context.get("checkpoint_config", {})
    
    chk = Checkpoint(
        access_key=cfg.get('access_key', ''),
        secret_key=cfg.get('secret_key', ''),
        token=cfg.get('token', ''),
        bucket=cfg.get('bucket', 'openeo-artifacts-waw3-1'),
        prefix=cfg.get('prefix', ''),
        endpoint=cfg.get('endpoint', 'https://s3.waw3-1.openeo.v1.dataspace.copernicus.eu'),
    )
    chk.set_reference(cube)

    logger.info(f"Reconstructing cube with shape {cube.shape}")

    n_days = context.get("n_days_to_reconstruct", 10)
    n_ranges = len(SCF_RANGES)
    total_days = cube.shape[0]

    if total_days <= n_days:
        logger.warning(f"Not enough data: {total_days} total, {n_days} requested")
        return cube.isel(t=slice(-n_days, None))

    hist_end = total_days - n_days

    historical_cp_maps = cube.isel(t=0, bands=slice(2, 2 + n_ranges)).values.astype(np.uint8)
    historical_occ_maps = cube.isel(t=0, bands=slice(2 + n_ranges, 2 + 2 * n_ranges)).values.astype(np.uint8)
    historical_snow = cube.isel(bands=0).values.astype(np.uint8)

    np.nan_to_num(historical_cp_maps, nan=NO_DATA, copy=False)
    np.nan_to_num(historical_occ_maps, nan=NO_DATA, copy=False)
    np.nan_to_num(historical_snow, nan=NO_DATA, copy=False)

    coords_y = cube.coords["y"].values
    coords_x = cube.coords["x"].values
    coords_t = cube.coords["t"].values[hist_end:hist_end + n_days]


    reconstructed_days  = []
    for day_idx in range(n_days):

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
            checkpoint=chk,
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
    del reconstructed_days
    gc.collect()

    result = xr.DataArray(
        np.expand_dims(reconstructed_snow, axis=1),
        dims=("t", "bands", "y", "x"),
        coords={
            "t": coords_t,
            "bands": ["reconstructed_snow"],
            "y": coords_y,
            "x": coords_x
        }
    )
    return result


def hist_rec_iterative(snow_map, scf_map, hist_snow, hist_cp_maps, hist_occ_maps, scf_ranges, day_idx = None, checkpoint=None):
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

    

        #if checkpoint:
            #checkpoint.save(reconstructed_hr, name="reconstructed_hr", day=day_idx, iter=iteration, stage="hr")

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

        #if checkpoint:
            #checkpoint.save(reconstructed_scf, name="reconstructed_scf", day=day_idx, iter=iteration, stage="scf")
        
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
