#%%

"""
Main execution script for historical snow cover reconstruction.

Orchestrates the entire pipeline: loading data, computing conditional probabilities,
reconstructing snow cover, downscaling climate data, and executing batch jobs.
"""

import openeo

from config import (
    BACKEND, TEMPORAL_EXTENT, SPATIAL_EXTENT, JOB_OPTIONS, 
    N_DAYS_TO_RECONSTRUCT, NEIGHBORHOOD_SIZE, AGERA_TEMPORAL_EXTENT, 
    MODIS_TEMPORAL_EXTENT, SCA_RECONSTRUCTION_UDF, SWE_RECONSTRUCTION_UDF
)
from scf_processing import compute_scf_masks, create_modis_scf_cube
from downscale_variables import downscale_shortwave_radiation, downscale_temperature_humidity, preprocess_low_resolution_agera
from s3_utils import S3Manager
from utils_gapfilling import calculate_snow


def main():
    """Execute the full historical reconstruction pipeline."""
    
    # ==============================
    # Authentication & Setup
    # ==============================
    
    eoconn = openeo.connect(BACKEND, auto_validate=False)
    eoconn.authenticate_oidc()
    
    # ==============================
    # 1. Compute SCF Masks & Conditional Probabilities
    # ==============================
    
    all_masks, labels_scf = compute_scf_masks(eoconn)
    
    # ==============================
    # 2. Compute Conditional Probabilities
    # ==============================
    
    def merge_masks(all_masks):
        """Multiply masks with snow band."""
        return all_masks.and_(all_masks.array_element(label="snow")) * 1.0

    mask_cp_snow = all_masks.apply(process=merge_masks)
    mask_cp_snow = mask_cp_snow.filter_bands(bands=labels_scf)

    sum_cp_snow = mask_cp_snow.reduce_dimension(reducer="sum", dimension="t")

    # Mask of all SCF occurrences over time
    occurences = all_masks.reduce_dimension(reducer="sum", dimension="t")
    occurences = occurences.filter_bands(bands=labels_scf)
    occurences = occurences.rename_labels(
        dimension="bands", target=[f"occ_{b}" for b in labels_scf]
    )

    # Conditional probabilities
    cp = sum_cp_snow / occurences
    cp = cp.rename_labels(dimension="bands", target=[f"cp_{b}" for b in labels_scf])

    # ==============================
    # 3. Load High-Resolution Data
    # ==============================
    
    
    # HR Sentinel-2 snow
    hr_snow = calculate_snow(
        eoconn, TEMPORAL_EXTENT, SPATIAL_EXTENT
    ).rename_labels(dimension="bands", target=["snow"])

    # HR MODIS SCF
    hr_scf = create_modis_scf_cube(
        eoconn, MODIS_TEMPORAL_EXTENT, SPATIAL_EXTENT
    ).rename_labels(dimension="bands", target=["scf"])

    # Add time dimension to cp and occurences
    first_date = hr_snow.metadata.temporal_dimension.extent[0]

    cp_with_time = cp.add_dimension(
        name='time',
        label=first_date,
        type='temporal'
    )

    occurences_with_time = occurences.add_dimension(
        name='time',
        label=first_date,
        type='temporal'
    )
    
    sca = (hr_snow.merge_cubes(hr_scf)
                     .merge_cubes(cp_with_time)
                     .merge_cubes(occurences_with_time))

    # ==============================
    # 4. Historical Reconstruction via UDF
    # ==============================
    
    
    
    sca_udf = openeo.UDF.from_file(
        str(SCA_RECONSTRUCTION_UDF),
        context={
            "n_days_to_reconstruct": N_DAYS_TO_RECONSTRUCT,
        }
    )
    
    sca = sca.apply_neighborhood(
        process=sca_udf,
        size=[
            {"dimension": "x", "value": NEIGHBORHOOD_SIZE, "unit": "px"},
            {"dimension": "y", "value": NEIGHBORHOOD_SIZE, "unit": "px"},
        ]
    )
    
    if sca.metadata.has_band_dimension():
        sca = sca.rename_labels(dimension="bands", target=["sca"])
    else:
        sca = sca.add_dimension(name="bands", label="sca", type="bands")

    # ==============================
    # 5. Load and Downscale Climate Data
    # ==============================
    
    dem = eoconn.load_collection("COPERNICUS_30", spatial_extent=SPATIAL_EXTENT)
    if dem.metadata.has_temporal_dimension():
        dem = dem.reduce_dimension(dimension="t", reducer="max")

    dem = dem.add_dimension(
        name='t',
        label=first_date,
        type='temporal'
    )

    agera = eoconn.load_stac(
        "https://stac.openeo.vito.be/collections/agera5_daily",
        spatial_extent=SPATIAL_EXTENT,
        temporal_extent=AGERA_TEMPORAL_EXTENT, 
    )
    agera = agera.filter_bands(bands=["2m_temperature_mean", "dewpoint_temperature_mean", "solar_radiation_flux"])
    agera = agera.rename_labels(dimension="bands", target=["temperature-mean", "dewpoint-temperature", "solar-radiation-flux"])

    geopotential = eoconn.load_stac(
        "https://artifactory.vgt.vito.be/artifactory/auxdata-public/geopotential.json",
        spatial_extent=SPATIAL_EXTENT,
        bands=["geopotential"]
    )
    geopotential.metadata = geopotential.metadata.add_dimension(
        "t", label=first_date, type="temporal"
    )
    
    agera_downscaled = downscale_temperature_humidity(agera, dem, geopotential.max_time())


    # ==============================
    # 6. Downscale Shortwave Radiation
    # ==============================
    
    aspect = eoconn.load_stac(
        "https://stac.openeo.vito.be/collections/DEM_aspec_30m",
        spatial_extent=SPATIAL_EXTENT
    ).reduce_dimension(dimension='t', reducer='mean')

    slope = eoconn.load_stac(
        "https://stac.openeo.vito.be/collections/DEM_slope_30m",
        spatial_extent=SPATIAL_EXTENT
    ).reduce_dimension(dimension='t', reducer='mean')

    slope_aspect = aspect.merge_cubes(slope).rename_labels(
        dimension="bands", target=["aspect", "slope"]
    )

    shortwave_rad_cube = downscale_shortwave_radiation(agera, slope_aspect)
   
    # ==============================
    # 7. Merge All Results
    # ==============================


    total_cube = sca.merge_cubes(agera_downscaled).merge_cubes(shortwave_rad_cube)

    # ==============================
    # 7. Merge All Results
    # ==============================

    swe_udf = openeo.UDF.from_file(
        str(SWE_RECONSTRUCTION_UDF),
    )
    
    swe = total_cube.apply_neighborhood(
        process=swe_udf,
        size=[
            {"dimension": "x", "value": NEIGHBORHOOD_SIZE, "unit": "px"},
            {"dimension": "y", "value": NEIGHBORHOOD_SIZE, "unit": "px"},
            
        ]
    )

    swe = swe.rename_labels(dimension="bands", target=["swe"])
    # ==============================
    # 9. Execute Batch Job
    # ==============================
    
    
    swe.execute_batch(
        title="swe",
        job_options=JOB_OPTIONS
    )
    

if __name__ == "__main__":
    main()
#%%

# %%

import requests

url = "https://stac.openeo.vito.be/search"

params = {
    "collections": "agera5_daily",
    "bbox": "10.723280484152271,46.676079331215135,10.776587456380184,46.70836189336941",
    "datetime": "2023-07-01T00:00:00Z/2023-07-04T23:59:59.999000Z",
    "limit": 20,
}

r = requests.get(url, params=params)
r.raise_for_status()

data = r.json()
print(data["type"], len(data["features"]))





#%%

import re
import os
import tempfile
from collections import defaultdict

import s3fs
import xarray as xr
import pandas as pd

# --- Configuration (same as in your UDF) ---
bucket = "openeo-artifacts-waw3-1"
prefix = "7b9a7a133160c440a4fa6586e3a3de183e1df18e/2026/02/26/104619/"
endpoint = "https://s3.waw3-1.openeo.v1.dataspace.copernicus.eu"

s3_manager = S3Manager()
s3_manager.authenticate()
s3_manager.print_credentials_info()
checkpoint_config = s3_manager.get_checkpoint_config()
# ----------------------------------------------------------------------
# CONFIGURATION – set your S3 credentials and prefix here
# ----------------------------------------------------------------------
S3_CONFIG = {
    'key': checkpoint_config['access_key'],           # from your auth code
    'secret': checkpoint_config['secret_key'],    # from your auth code
    'token': checkpoint_config['token'],        # from your auth code
    'endpoint': "https://s3.waw3-1.openeo.v1.dataspace.copernicus.eu",
    'bucket': 'openeo-artifacts-waw3-1',
    'prefix': prefix,                # from your auth code
}

OUTPUT_DIR = 'merged_swe_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Connect to S3
# ----------------------------------------------------------------------
fs = s3fs.S3FileSystem(
    key=S3_CONFIG['key'],
    secret=S3_CONFIG['secret'],
    token=S3_CONFIG['token'],
    client_kwargs={'endpoint_url': S3_CONFIG['endpoint']}
)

# ----------------------------------------------------------------------
# List all SWE band NetCDF files
# ----------------------------------------------------------------------
s3_path = f"{S3_CONFIG['bucket']}/{S3_CONFIG['prefix']}"
all_files = fs.ls(s3_path)

# Filter for SWE band files only
swe_files = [f for f in all_files if f.endswith('.nc') and 'band_swe' in f]
print(f"Found {len(swe_files)} SWE band NetCDF files.")

# ----------------------------------------------------------------------
# Extract metadata from filename for better grouping
# ----------------------------------------------------------------------
def parse_swe_filename(s3_key):
    """Extract metadata from SWE filename."""
    fname = s3_key.split('/')[-1]
    # Pattern for swe_band_swe_date_20230701_stage_output_time_idx000_x631810_y5168530_t20230701_y64_x64.nc
    pattern = r"swe_band_swe_date_(?P<date>\d{8})_stage_output_time_idx(?P<time_idx>\d+)_x(?P<x>\d+)_y(?P<y>\d+)_t(?P<t_date>\d{8})_y(?P<ny>\d+)_x(?P<nx>\d+).nc"
    match = re.search(pattern, fname)
    if match:
        return {
            'date': match.group('date'),
            'time_idx': int(match.group('time_idx')),
            'x_start': int(match.group('x')),
            'y_start': int(match.group('y')),
            't_date': match.group('t_date'),
            'ny': int(match.group('ny')),
            'nx': int(match.group('nx')),
            'base_key': f"swe_band_swe_date_{match.group('date')}_stage_output",  # Group key
            's3_key': s3_key
        }
    return None

# Parse all files
file_metadata = [parse_swe_filename(f) for f in swe_files if parse_swe_filename(f)]

# Group by base_key (same date)
groups = defaultdict(list)
for meta in file_metadata:
    groups[meta['base_key']].append(meta)

print(f"Found {len(groups)} time steps to merge.")

# ----------------------------------------------------------------------
# Process each time step: merge all tiles
# ----------------------------------------------------------------------
for base_key, metadata_list in groups.items():
    print(f"\nProcessing {base_key} ({len(metadata_list)} tiles)...")
    
    datasets = []
    temp_files = []  # Track temp files for cleanup
    
    for meta in metadata_list:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp:
            tmp_path = tmp.name
            temp_files.append(tmp_path)  # Track for cleanup
        
        try:
            # Download from S3
            fs.get(meta['s3_key'], tmp_path)
            
            # Open with netCDF4 engine
            ds = xr.open_dataset(tmp_path, engine='netcdf4')
            
            # Quick shape check
            data_var = list(ds.data_vars)[0]  # Usually 'data'
            print(f"  Tile x{meta['x_start']}_y{meta['y_start']}: {ds[data_var].shape}")
            
            datasets.append(ds)
            
        except Exception as e:
            print(f"  Error loading {meta['s3_key']}: {e}")
            # Clean up this temp file immediately if there's an error
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                    temp_files.remove(tmp_path)
                except:
                    pass
    
    if not datasets:
        print(f"  No datasets loaded for {base_key}, skipping.")
        # Clean up any remaining temp files
        for tmp_path in temp_files:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        continue
    
    # Merge all tiles for this time step
    try:
        # Combine by coordinates (automatically handles x,y grid)
        combined = xr.combine_by_coords(
            datasets, 
            combine_attrs='drop_conflicts'
        )
        
        # Save merged file
        date_str = base_key.split('_')[-1]  # Extract date from base_key
        out_path = os.path.join(OUTPUT_DIR, f"swe_merged_{date_str}.nc")
        combined.to_netcdf(out_path)
        
        print(f"  ✓ Saved merged file: {out_path}")
        print(f"    Final shape: {combined.dims}")
        
        # Quick validation
        data_var = list(combined.data_vars)[0]
        print(f"    Data shape: {combined[data_var].shape}")
        
        # IMPORTANT: Close all datasets before deleting temp files
        for ds in datasets:
            ds.close()
        
        # Now safe to delete temp files
        for tmp_path in temp_files:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                    print(f"    Cleaned up: {tmp_path}")
                except Exception as e:
                    print(f"    Warning: Could not delete {tmp_path}: {e}")
        
    except Exception as e:
        print(f"  ✗ Merge failed for {base_key}: {e}")
        
        # Still need to close datasets even on failure
        for ds in datasets:
            ds.close()
        
        # Clean up temp files
        for tmp_path in temp_files:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass

print("\nAll SWE time steps processed.")

#%%

\
    )