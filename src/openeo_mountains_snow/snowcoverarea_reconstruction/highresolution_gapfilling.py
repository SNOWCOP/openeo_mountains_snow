"""
High resolution gap-filling utilities.

Functions for calculating and processing high-resolution snow cover data.
"""

import openeo
from openeo.processes import if_


def calculate_snow_from_scl(connection, temporal_extent, spatial_extent, cloud_prob=80.0) -> openeo.DataCube:
    """
    Calculate snow cover from Sentinel-2 L2A scene classification (SCL).
    
    Note: This method is less reliable as snow and clouds are often mixed up
    in scene classification data.
    
    Args:
        connection: openEO Connection object
        temporal_extent: Temporal extent as [start_date, end_date]
        spatial_extent: Spatial extent dictionary with crs and bounds
        cloud_prob: Maximum cloud probability threshold (default: 80%)
        
    Returns:
        DataCube with snow classification (0: no snow, 100: snow, 205: clouds)
    """
    scl = connection.load_collection(
        "SENTINEL2_L2A",
        temporal_extent=temporal_extent,
        spatial_extent=spatial_extent,
        bands=['SCL'],
        max_cloud_cover=cloud_prob,
    )

    scl = scl.resample_spatial(resolution=20, projection=32632, method="near")

    def snow_callback(scl_data: openeo.processes.ProcessBuilder):
        """Classify pixels as snow (100), clouds (205), or no snow (0)."""
        classification = scl_data["SCL"]
        snow_pixel = (classification == 11) * 100.0
        return if_((classification == 7).or_(classification == 8).or_(classification == 9),
                   205.0, snow_pixel)

    snow = scl.apply_dimension(dimension="bands", process=snow_callback)
    snow = snow.rename_labels(dimension="bands", target=["snow"])

    return snow