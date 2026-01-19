#%%

import openeo

from openeo import DataCube
from openeo.processes import (if_, and_, or_, eq, gte, lte, gt, 
                              ProcessBuilder)
from typing import Dict, List

# openEO backend
BACKEND = 'https://openeo.dataspace.copernicus.eu/'

# Spatial extent (Senales catchment example)
WEST = 636800.0
SOUTH = 5152700.0
EAST = 641800.0
NORTH = 5177700.0
CRS = 32632  # UTM zone 32N

# Temporal extents
HIST_START = '2022-12-01'  # Historical period for distribution
HIST_END = '2022-12-31'

DAILY_START = '2023-01-01'  # Period to downscale
DAILY_END = '2023-01-13'

# Resolution
HR_RESOLUTION = 20.0  # meters (Sentinel-2)
LR_RESOLUTION = 500.0  # meters (MODIS)

# Snow parameters
CLOUD_VALUE = 205  # Code for cloud/invalid pixels
WATER_VALUE = 210  # Code for water pixels
CLOUD_PROB_THRESHOLD = 50  # Cloud probability threshold (%)

# Distribution parameters
DELTA = 10  # SCF range step size
EPSILON = 10  # Buffer for SCF ranges
PIXEL_RATIO = int(LR_RESOLUTION / HR_RESOLUTION)  # 500m / 20m = 25

# Downscaling thresholds
MIN_OCCURRENCES = 10  # Minimum historical occurrences
SNOW_PROB_THRESHOLD = 0.9  # Probability threshold for snow
NO_SNOW_PROB_THRESHOLD = 0.1  # Probability threshold for no-snow


def generate_ranges(delta: int, epsilon: int):
    range_definitions = []
    range_keys = []

    for base_low in range(0, 100, delta):
        base_high = min(base_low + delta, 100)
        eps_low = max(base_low - epsilon, 0)
        eps_high = min(base_high + epsilon, 100)

        key = f"{eps_low}_{eps_high}"
        range_definitions.append((key, eps_low, eps_high))
        range_keys.append(key)

    return range_definitions, range_keys

range_definitions, range_keys = generate_ranges(DELTA, EPSILON)

#%%


def create_sentinel2_snow_cube(connection: openeo.Connection,
                                      temporal_extent: List[str],
                                      spatial_extent: Dict) -> DataCube:
    
    
    """
    Calculates snow from Sentinel-2 L2A scene classification.
    This method is less reliable because snow and clouds are mixed up in scene classification.

    """
    # ==============================
    # Load collections
    # ==============================

    scl = connection.load_collection(
        "SENTINEL2_L2A",
        temporal_extent=temporal_extent,
        spatial_extent=spatial_extent,
        bands=['SCL'],
        max_cloud_cover=CLOUD_PROB_THRESHOLD,
    )

    scl = scl.resample_spatial(resolution=20,
                          projection=32632,
                          method="near")

    # ==============================
    # Get the snow cover information
    # ==============================

    def snow_callback(scl_data: openeo.processes.ProcessBuilder):
        classification: openeo.processes.ProcessBuilder = scl_data["SCL"]
        snow_pixel = (classification == 11) * 100.0
        return if_((classification == 7).or_(classification == 8).or_(classification == 9), CLOUD_VALUE, snow_pixel)

    snow = scl.apply_dimension(dimension="bands", process=snow_callback)
    snow = snow.rename_labels(dimension="bands", target=["snow"])


    return snow


def create_modis_scf_cube(connection: openeo.Connection,

                          temporal_extent: List[str],
                          spatial_extent: Dict) -> DataCube:
    """
    Load and prepare MODIS snow cover fraction (SCF) data.
    
    Args:
        connection: openEO connection
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


#%% TODO this workflow needs to run itteratively 


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

temporal_extent_history = [HIST_START, HIST_END]
temporal_extent_daily = [DAILY_START, DAILY_END]

# Create input history DataCubes
hr_snow = create_sentinel2_snow_cube(connection, temporal_extent_history, spatial_extent)
lr_scf = create_modis_scf_cube(connection, temporal_extent_history, spatial_extent)

# Upsample MODIS to Sentinel-2 resolution 
hr_scf = lr_scf.resample_spatial(
    resolution=HR_RESOLUTION,
    projection=CRS,
    method="near"
)

historic_cube = hr_snow.merge_cubes(hr_scf)

context_cp = {
    "epsilon": EPSILON,
    "delta": DELTA
}

# Step 4: Define the UDF that computes everything
udf = openeo.UDF.from_file(
    "C:\Git_projects\openeo_mountains_snow\src\openeo_mountains_snow\snowcoverarea_reconstruction\conditional_probability_udf.py",
    context=context_cp)

    # Apply UDF
historic_cube = historic_cube.reduce_dimension(
    reducer=udf,
    dimension="t"
)

historic_cube = historic_cube.rename_labels(dimension="bands", target=range_keys + [f"occ_{k}" for k in range_keys])


daily_scf = create_modis_scf_cube(connection, temporal_extent_daily, spatial_extent)
daily_hr_snow = create_sentinel2_snow_cube(connection, temporal_extent_daily, spatial_extent)
    
daily_hr_scf = daily_scf.resample_spatial(
    resolution=HR_RESOLUTION,
    projection=CRS,
    method="near"
)
    
daily_input = daily_hr_snow.merge_cubes(daily_hr_scf)
reconstruction_cube = daily_input.merge_cubes(historic_cube)

reconstruct_udf = openeo.UDF.from_file(
    "C:\Git_projects\openeo_mountains_snow\src\openeo_mountains_snow\snowcoverarea_reconstruction\gap_fill_udf.py",
)

filled_cube = reconstruction_cube.apply_dimension(
    dimension="t",
    process=reconstruct_udf
)

filled_cube

#%%
job_options = {
    "executor-memoryOverhead": "6G",
    "python-memory": "disable", #the default is in fact to derive this from executor-memoryOverhead


}
filled_cube.execute_batch()

#%%
