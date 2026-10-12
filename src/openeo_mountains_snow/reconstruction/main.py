#%%

"""
Main execution script for historical snow cover reconstruction.

Orchestrates the entire pipeline: loading data, computing conditional probabilities,
reconstructing snow cover, downscaling climate data, and executing batch jobs.
"""

import openeo

from openeo_mountains_snow.reconstruction.config import (
    BACKEND, TEMPORAL_EXTENT, SPATIAL_EXTENT, JOB_OPTIONS, 
    N_DAYS_TO_RECONSTRUCT, NEIGHBORHOOD_SIZE, AGERA_TEMPORAL_EXTENT, 
    MODIS_TEMPORAL_EXTENT, SCA_RECONSTRUCTION_UDF, SWE_RECONSTRUCTION_UDF
)
from openeo_mountains_snow.reconstruction.conditional_probability import compute_snow_cover_conditional_probabilities
from openeo_mountains_snow.reconstruction.snow_cover_fraction_processing import (
    compute_snow_cover_fraction_masks,
    create_modis_snow_cover_fraction_cube,
)
from openeo_mountains_snow.snow_water_equivalent.downscale_variables import downscale_shortwave_radiation, downscale_temperature_humidity, preprocess_low_resolution_agera
from openeo_mountains_snow.reconstruction.utils_gapfilling import calculate_snow


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
    
    all_masks, snow_cover_fraction_band_labels = compute_snow_cover_fraction_masks(eoconn)
    
    # ==============================
    # 2. Compute Conditional Probabilities
    # ==============================
    
    conditional_probability_cube, snow_cover_occurrences_cube = compute_snow_cover_conditional_probabilities(
        all_masks,
        snow_cover_fraction_band_labels,
    )

    # ==============================
    # 3. Load High-Resolution Data
    # ==============================
    
    
    # HR Sentinel-2 snow
    hr_snow = calculate_snow(
        eoconn, TEMPORAL_EXTENT, SPATIAL_EXTENT
    ).rename_labels(dimension="bands", target=["snow"])

    # HR MODIS SCF
    high_resolution_snow_cover_fraction = create_modis_snow_cover_fraction_cube(
        eoconn, MODIS_TEMPORAL_EXTENT, SPATIAL_EXTENT
    ).rename_labels(dimension="bands", target=["scf"])

    # Add time dimension to cp and occurrences
    first_date = hr_snow.metadata.temporal_dimension.extent[0]

    conditional_probability_cube_with_time = conditional_probability_cube.add_dimension(
        name='time',
        label=first_date,
        type='temporal'
    )

    snow_cover_occurrences_cube_with_time = snow_cover_occurrences_cube.add_dimension(
        name='time',
        label=first_date,
        type='temporal'
    )
    
    snow_cover_area_cube = (
        hr_snow
        .merge_cubes(high_resolution_snow_cover_fraction)
        .merge_cubes(conditional_probability_cube_with_time)
        .merge_cubes(snow_cover_occurrences_cube_with_time)
    )

    # ==============================
    # 4. Historical Reconstruction via UDF
    # ==============================
    
    
    
    snow_cover_area_udf = openeo.UDF.from_file(
        str(SCA_RECONSTRUCTION_UDF),
        context={
            "n_days_to_reconstruct": N_DAYS_TO_RECONSTRUCT,
        }
    )
    
    snow_cover_area_cube = snow_cover_area_cube.apply_neighborhood(
        process=snow_cover_area_udf,
        size=[
            {"dimension": "x", "value": NEIGHBORHOOD_SIZE, "unit": "px"},
            {"dimension": "y", "value": NEIGHBORHOOD_SIZE, "unit": "px"},
        ]
    )
    

    snow_cover_area_cube = snow_cover_area_cube.add_dimension(name="bands", label="sca", type="bands")

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


    total_cube = snow_cover_area_cube.merge_cubes(agera_downscaled).merge_cubes(shortwave_rad_cube)

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
    















