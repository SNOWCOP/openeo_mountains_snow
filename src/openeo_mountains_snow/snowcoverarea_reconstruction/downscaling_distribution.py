#%%

import openeo

from openeo import DataCube
from openeo.processes import (if_, and_, or_, eq, gte, lte, gt , 
                             array_create , ProcessBuilder)
import numpy as np
from typing import Dict, Tuple, List


#%%

# openEO backend
BACKEND = 'https://openeo.dataspace.copernicus.eu/'

# Spatial extent (Senales catchment example)
WEST = 631800.0
SOUTH = 5167700.0
EAST = 641800.0
NORTH = 5177700.0
CRS = 32632  # UTM zone 32N

# Temporal extents
HIST_START = '2022-01-01'  # Historical period for distribution
HIST_END = '2022-12-31'
DAILY_START = '2023-01-01'  # Period to downscale
DAILY_END = '2023-01-07'

# Resolution
HR_RESOLUTION = 20.0  # meters (Sentinel-2)
LR_RESOLUTION = 500.0  # meters (MODIS)

# Snow parameters
SNOW_THRESHOLD = 50  # Threshold for binary snow classification
CLOUD_VALUE = 205  # Code for cloud/invalid pixels
WATER_VALUE = 210  # Code for water pixels
CLOUD_PROB_THRESHOLD = 50  # Cloud probability threshold (%)

# Distribution parameters
DELTA = 10  # SCF range step size
EPSILON = 5  # Buffer for SCF ranges
PIXEL_RATIO = int(LR_RESOLUTION / HR_RESOLUTION)  # 500m / 20m = 25

# Downscaling thresholds
MIN_OCCURRENCES = 10  # Minimum historical occurrences
SNOW_PROB_THRESHOLD = 0.9  # Probability threshold for snow
NO_SNOW_PROB_THRESHOLD = 0.1  # Probability threshold for no-snow

#%%

def create_sentinel2_snow_cube(connection: openeo.Connection,
                                      temporal_extent: List[str],
                                      spatial_extent: Dict) -> DataCube:
    """
    Transforms Sentinel-2 L2A probability bands into a binary snow classification.
    
    Process:
    1. Loads SNW (snow probability) and CLD (cloud probability) bands
    2. Applies thresholds to classify each pixel as: snow, no-snow, or cloud
    3. Returns a single-band cube with the classification
    
    Classification rules:
    - Cloud: CLD ≥ CLOUD_PROB_THRESHOLD 
    - Snow: Not cloudy AND SNW ≥ SNOW_PROB_THRESHOLD 
    - No snow: Not cloudy AND SNW < SNOW_PROB_THRESHOLD
    
    Args:
        connection: Authenticated openEO connection
        temporal_extent: [start_date, end_date] for data loading
        spatial_extent: Dictionary with keys: west, south, east, north, crs
        
    Returns:
        DataCube with single band named 'snow' containing values:
        100 (snow), 0 (no snow), 205 (cloud)
    """

    # Load Sentinel-2 L2A with SNW and CLD probability bands
    s2_cube = connection.load_collection(
        "SENTINEL2_L2A",
        temporal_extent=temporal_extent,
        spatial_extent=spatial_extent,
        bands=["SNW", "CLD"]  # Use Snow and Cloud Probability bands
    )

    def classify_snow_pixel(pixel: ProcessBuilder) -> ProcessBuilder:
        """Alternative with explicit band name mapping."""
        # Create a small dictionary-like access
        bands = ["SNW", "CLD"]
        
        # Access by name reference (conceptual - actual implementation uses index)
        snow_prob = pixel.array_element(index=bands.index("SNW"))
        cloud_prob = pixel.array_element(index=bands.index("CLD"))
        
        # Same logic as before
        is_cloudy = cloud_prob >= CLOUD_PROB_THRESHOLD
        is_snowy = snow_prob >= SNOW_PROB_THRESHOLD
        
        return if_(is_cloudy, CLOUD_VALUE, if_(is_snowy, 100.0, 0.0))

    snow_cube = s2_cube.apply(classify_snow_pixel)
    return snow_cube.rename_labels(dimension="bands", target=["snow"])


def create_modis_scf_cube(connection: openeo.Connection,
                          temporal_extent: List[str],
                          spatial_extent: Dict) -> DataCube:
    """
    Load and prepare MODIS snow cover fraction (SCF) data.
    
    Args:
        connection: Active openEO connection
        temporal_extent: [start_date, end_date]
        spatial_extent: Dictionary with bbox and crs
        
    Returns:
        DataCube with single band 'scf' containing:
        - 0-100: Valid snow cover fraction percentage
        - 205: Invalid/cloud/water pixels
    """
    
    
    # Load MODIS from the STAC endpoint
    modis_cube = connection.load_stac(
        url="https://stac.eurac.edu/collections/MOD10A1v61",
        temporal_extent=temporal_extent,
        spatial_extent=spatial_extent,
        bands=["SCF"]  # MODIS Snow Cover Fraction band
    )
    
    def clean_modis_scf(pixel: ProcessBuilder) -> ProcessBuilder:
        """
        Clean MODIS SCF values by handling invalid data codes.
        
        MODIS SCF values:
        - 0-100: Valid snow cover fraction (%)
        - 205: Cloud (preserved as is)
        - 254: Water
        - 255: No data
        
        Args:
            pixel: ProcessBuilder with single SCF band value
            
        Returns:
            ProcessBuilder with cleaned value
        """
        scf_value = pixel.array_element(index=0)  # SCF band
        
        # Check for invalid values that need to be replaced
        is_water = eq(scf_value, 254)
        is_no_data = eq(scf_value, 255)
        is_other_invalid = gt(scf_value, 100)  # Values > 100 are invalid
        
        should_replace = or_(
            is_water,
            or_(
                is_no_data,
                is_other_invalid
            )
        )
        
        # Replace invalid values with CLOUD_VALUE, keep others as-is
        return if_(
            should_replace,
            CLOUD_VALUE,  # 205 for invalid/water/no-data
            scf_value     # Original value for valid SCF (0-100, 205)
        )
    
    # Apply cleaning to all pixels
    cleaned_cube = modis_cube.apply(clean_modis_scf)
    
    # Resample to consistent resolution (500m for MODIS)
    cleaned_cube = cleaned_cube.resample_spatial(
        resolution=LR_RESOLUTION,
        projection=CRS,
        method="near"
    )
    
    # Rename the output band for clarity
    final_cube = cleaned_cube.rename_labels(dimension="bands", target=["scf"])
    
    return final_cube



#%%
connection = openeo.connect(BACKEND)
connection.authenticate_oidc()
    
# Define spatial extent
spatial_extent = {
    "west": WEST,
    "south": SOUTH,
    "east": EAST,
    "north": NORTH,
    "crs": f"EPSG:{CRS}"
}

hr_snow_cube = create_modis_scf_cube(connection,
                           [HIST_START, HIST_END],
                           spatial_extent)

hr_snow_cube.execute_batch()

#%%


def compute_distribution(connection: openeo.Connection,
                         temporal_extent: List[str],
                         spatial_extent: Dict) -> Tuple[ProcessBuilder, ProcessBuilder, Dict]:
    """
    Compute conditional probability distribution from historical data.
    
    Returns: (cp_cube, occur_cube, scf_range_dict)
    """

    
    # Create input cubes
    hr_snow = create_sentinel2_snow_cube(connection, temporal_extent, spatial_extent)
    lr_scf = create_modis_scf_cube(connection, temporal_extent, spatial_extent)
    
    # Create SCF ranges for analysis
    scf_range_dict = {}
    cp_bands = []
    occur_bands = []
    
    # Generate ranges with delta and epsilon
    for base_low in range(0, 100, DELTA):
        base_high = min(base_low + DELTA, 100)
        eps_low = max(base_low - EPSILON, 0)
        eps_high = min(base_high + EPSILON, 100)
        
        key = f"{eps_low}_{eps_high}"
        scf_range_dict[key] = (base_low, base_high)
        
        
        # Create mask for this SCF range
        if eps_low == 0:
            range_mask = and_(gte(lr_scf, eps_low), lte(lr_scf, eps_high))
        else:
            range_mask = and_(gt(lr_scf, eps_low), lte(lr_scf, eps_high))
        
        # Upsample mask to HR resolution
        range_mask_hr = range_mask.resample_spatial(
            resolution=HR_RESOLUTION,
            projection=CRS,
            method="near"
        )
        
        # Count occurrences
        occurrences = range_mask_hr.reduce_dimension(dimension="t", reducer="sum")
        
        # Create snow mask (1=snow, 0=no snow, nan=invalid)
        snow_mask = if_(
            hr_snow > 100,  # Invalid
            np.nan,
            if_(
                hr_snow >= SNOW_THRESHOLD,
                1.0,  # Snow
                0.0   # No snow
            )
        )
        
        # Compute snow occurrences in this range
        snow_in_range = snow_mask * range_mask_hr
        snow_sum = snow_in_range.reduce_dimension(dimension="t", reducer="sum")
        
        # Conditional probability: P(snow | SCF in range)
        cp = if_(
            occurrences > 0,
            snow_sum / occurrences,
            np.nan
        )
        
        cp_bands.append(cp)
        occur_bands.append(occurrences)
    
    # Create multi-band cubes
    range_keys = list(scf_range_dict.keys())
    cp_cube = array_create(cp_bands).add_dimension("bands", "bands", range_keys)
    occur_cube = array_create(occur_bands).add_dimension("bands", "bands", range_keys)
    
    print(f"\nDistribution computed for {len(range_keys)} SCF ranges")
    return cp_cube, occur_cube, scf_range_dict

def downscale_daily(connection: openeo.Connection,
                    cp_cube: ProcessBuilder,
                    occur_cube: ProcessBuilder,
                    scf_range_dict: Dict,
                    temporal_extent: List[str],
                    spatial_extent: Dict) -> ProcessBuilder:
    """
    Downscale daily MODIS data to high resolution.
    """
    print(f"\nDownscaling MODIS data ({temporal_extent[0]} to {temporal_extent[1]})")
    
    # Load daily MODIS data
    daily_modis = create_modis_scf_cube(connection, temporal_extent, spatial_extent)
    
    # Upsample MODIS to HR
    modis_hr = daily_modis.resample_spatial(
        resolution=HR_RESOLUTION,
        projection=CRS,
        method="near"
    )
    
    # Initialize result with basic classification
    result = if_(
        modis_hr > 100,  # Invalid
        CLOUD_VALUE,
        if_(
            modis_hr >= SNOW_THRESHOLD,
            100.0,  # Snow
            0.0     # No snow
        )
    )
    
    # Apply downscaling for each SCF range
    for key in scf_range_dict.keys():
        eps_low, eps_high = map(int, key.split('_'))
        
        # Get probability and occurrence for this range
        cp = cp_cube.band(key)
        occur = occur_cube.band(key).resample_spatial(
            resolution=HR_RESOLUTION,
            projection=CRS,
            method="near"
        )
        
        # Check if MODIS value is in this range
        if eps_low == 0:
            in_range = and_(gte(modis_hr, eps_low), lte(modis_hr, eps_high))
        else:
            in_range = and_(gt(modis_hr, eps_low), lte(modis_hr, eps_high))
        
        # Conditions for confident classification
        has_enough_data = occur >= MIN_OCCURRENCES
        is_confident_snow = cp >= SNOW_PROB_THRESHOLD
        is_confident_no_snow = cp <= NO_SNOW_PROB_THRESHOLD
        
        # Apply snow classification
        should_be_snow = and_(in_range, has_enough_data, is_confident_snow)
        should_be_no_snow = and_(in_range, has_enough_data, is_confident_no_snow)
        
        result = if_(
            should_be_snow,
            100.0,  # Snow
            if_(
                should_be_no_snow,
                0.0,  # No snow
                result  # Keep current value
            )
        )
    
    # Preserve pure MODIS pixels (0% and 100%)
    result = if_(
        eq(modis_hr, 0),
        0.0,  # Definitely no snow
        if_(
            eq(modis_hr, 100),
            100.0,  # Definitely snow
            result
        )
    )
    
    return result






