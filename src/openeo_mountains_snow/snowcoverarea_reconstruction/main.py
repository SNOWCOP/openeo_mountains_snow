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
from downscale_variables import downscale_precipitation, downscale_shortwave_radiation, downscale_temperature_humidity, preprocess_low_resolution_agera
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
    
    sca_input = (hr_snow.merge_cubes(hr_scf)
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
    
    sca = sca_input.apply_neighborhood(
        process=sca_udf,
        size=[
            {"dimension": "x", "value": NEIGHBORHOOD_SIZE, "unit": "px"},
            {"dimension": "y", "value": NEIGHBORHOOD_SIZE, "unit": "px"},
        ],
        overlap=[
            {"dimension": "x", "value": NEIGHBORHOOD_SIZE//2, "unit": "px"},
            {"dimension": "y", "value": NEIGHBORHOOD_SIZE//2, "unit": "px"},
        ],
        
    )
    

    sca = sca.rename_labels(dimension="bands", target=["sca"])

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
    temperature_downscaled =agera_downscaled.filter_bands(bands=["temperature_downscaled"])


    # ==============================
    # 6. Downscale precipitation
    # ==============================



    precip = eoconn.load_stac(
                    "https://stac.openeo.vito.be/collections/agera5_daily",                     # or "ERA5-LAND"
                    spatial_extent=SPATIAL_EXTENT,
                    temporal_extent=AGERA_TEMPORAL_EXTENT,
                    bands=["total_precipitation"]          # typical band name for daily total
                    )
    
    precip = precip.reduce_dimension(dimension='t', reducer='mean') #TODO enable multiple moths with varying precep
    
 
    gamma_june = 0.00025   # m⁻¹

    precip_downscaled = downscale_precipitation(precip, dem, geopotential.max_time(), gamma_june)


    # ==============================
    # 7. Downscale Shortwave Radiation
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
    # 7. Merge All Results for SWE Computation
    # ==============================
    # Merge cubes in specific order to provide exactly 4 bands to SWE UDF:
    # Band 0: sca (1 band)
    # Band 1: temperature_downscaled (from agera_downscaled)
    # Band 2: precipitation 
    # Band 3: shortwave-radiation-flux-downscaled (from shortwave_rad_cube)
    # No additional bands should be included

    total_cube = sca.merge_cubes(temperature_downscaled).merge_cubes(precip_downscaled).merge_cubes(shortwave_rad_cube)

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
    sca_input = sca_input.save_result(format="netCDF")



    # ==============================
    # 9. Execute Batch Job
    # ==============================

    
    sca_input.execute_batch(
        title="swe_input",
        job_options=JOB_OPTIONS
    )
    

if __name__ == "__main__":
    main()

# %%

import rasterio
import matplotlib.pyplot as plt
import numpy as np

path1 = r"C:\Users\VROMPAYH\Downloads\openEO_2024-08-23Z.tif"
path2 = r"C:\Users\VROMPAYH\Downloads\openEO_2023-07-01Z.tif"

# Read band 1 from first file
with rasterio.open(path1) as src:
    sca = src.read(1)

# Read bands 2–4 from second file
with rasterio.open(path2) as src:
    temp = src.read(2)
    rh = src.read(3)
    sw = src.read(4)

data = [sca, temp, rh, sw]

titles = [
    "SCA",
    "Temperature",
    "Precipitation",
    "Shortwave Radiation Flux"
]

units = ["", "°C", "%", "W m$^{-2}$"]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for i, ax in enumerate(axes.flat):

    vmin, vmax = np.percentile(data[i], (2, 98))
    im = ax.imshow(data[i], cmap="viridis", vmin=vmin, vmax=vmax)

    ax.set_title(titles[i])
    ax.axis("off")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if units[i]:
        cbar.set_label(units[i])

plt.tight_layout()
plt.show()


#%%

