
import openeo
from openeo.processes import if_


def calculate_snow_from_scl(connection,temporal_extent,spatial_extent, cloud_prob = 80.0) -> openeo.DataCube:
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
        max_cloud_cover=cloud_prob,
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
        return if_((classification == 7).or_(classification == 8).or_(classification == 9), 205.0, snow_pixel)

    snow = scl.apply_dimension(dimension="bands", process=snow_callback)
    snow = snow.rename_labels(dimension="bands", target=["snow"])


    return snow