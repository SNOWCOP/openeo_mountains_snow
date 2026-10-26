#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:07:46 2024

@author: rbarella
"""
import numpy as np
import os
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from scipy.ndimage import binary_erosion, binary_dilation
from sklearn.mixture import GaussianMixture

from SnowFLAKES.fit_distribution import fit_distribution_and_median

from SnowFLAKES.utilities import (
    load_map,
    build_valid_scene,
    get_sensor,
    define_bands,
    scene_valid_mask,
    scene_no_data_value,
    create_folder,
)













def save_histogram(
    data,
    output_path,
    bins=50,
    xlabel="Value",
    ylabel="Count",
    title=None,
    density=False,
    alpha=0.6
):
    """
    Save one or more histograms.

    Parameters
    ----------
    data : dict
        Dictionary {label: values}. For example:
        {
            "Snow": green[snow_mask],
            "No snow": green[nosnow_mask]
        }

    output_path : str
        Output image path.

    bins : int
        Number of histogram bins.

    xlabel : str
        X-axis label.

    ylabel : str
        Y-axis label.

    title : str or None
        Figure title.

    density : bool
        Plot probability density instead of counts.

    alpha : float
        Histogram transparency.
    """


    plt.figure()
    plt.hist(
        data,
        bins=bins,
        alpha=alpha
    )

    plt.xlabel(xlabel)
    plt.ylabel("Density" if density else ylabel)

    if title is not None:
        plt.title(title)

    if len(data) > 1:
        plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    
    





    

    
    
def get_pixels_sun(bands, curr_aux_folder, mask_sun, curr_range):
    
    NDSI = load_map(curr_aux_folder, '*NDSI.tif')
    NDWI = load_map(curr_aux_folder, '*NDWI.tif')

    green = bands["GREEN"]
    swir = bands["SWIR"]
    
    NDSI_thresholds = define_threshold(
        NDSI,
        mask_sun,
        f"NDSI_sun_{curr_range[0]}-{curr_range[1]}",
        curr_aux_folder,
        threshold=(0.2, 0.7),
    )

    
    
    # fixed conditions for being a snow pixel
    mask_snow = np.logical_and.reduce((mask_sun, 
                                        NDWI<0.1, 
                                        NDSI>max(NDSI_thresholds)))
    
    # fixed conditions for being a snowfree pixel
    mask_sf = np.logical_and.reduce((mask_sun, 
                                     NDSI<min(NDSI_thresholds)))
    

    # find dynamic thresholds
    fit_green_snow = fit_distribution_and_median(green, 
                                                 f"green_sun_snow_{curr_range[0]}-{curr_range[1]}", 
                                                 mask_snow, 
                                                 curr_aux_folder,
                                                 default_median=0.6)
    
    fit_green_sf = fit_distribution_and_median(green, 
                                               f"green_sun_sf_{curr_range[0]}-{curr_range[1]}", 
                                               mask_sf, 
                                               curr_aux_folder,
                                               default_median=0.5)
    
    fit_swir_snow = fit_distribution_and_median(swir, 
                                                f"swir_sun_snow_{curr_range[0]}-{curr_range[1]}", 
                                                mask_snow, 
                                                curr_aux_folder,
                                                default_median=0.2)
    
    fit_swir_sf = fit_distribution_and_median(swir, 
                                             f"swir_sun_sf_{curr_range[0]}-{curr_range[1]}",
                                             mask_sf, 
                                             curr_aux_folder,
                                             default_median=0.1)
    
    green_threshold_snow = fit_green_snow['fitted_median'] #- fit_green_snow['parameters']['std']
    
    green_threshold_sf = fit_green_sf['fitted_median'] #+ fit_green_sf['parameters']["std"]
    
    swir_threshold_snow = fit_swir_snow['fitted_median'] #+ fit_swir_snow['parameters']["std"]
    
    swir_threshold_sf = fit_swir_sf['fitted_median'] #- fit_swir_sf['parameters']["std"]


    # conditions of val
    snow = np.logical_and.reduce((mask_snow,
                                  green > green_threshold_snow,
                                  swir < swir_threshold_snow))
    
    # snowfree = snowfree_1 | snowfree_2
    snowfree = np.logical_and.reduce((mask_sf,
                                      green < green_threshold_sf,
                                      swir > swir_threshold_sf))

    return snow, snowfree








    
    


def collect_trainings(data, scene_id, config, total_samples=500):
    
    # load information for current scene
    sensor = get_sensor(scene_id)
    bands = define_bands(data, sensor)


      
    

    
    # enlarge shadow - sun masks to create a buffer where training collection
    # is avoided

    sun_mask_eroded = binary_erosion(
        binary_dilation((shadow_mask == 0) & curr_scene_valid),
        iterations=3
    ) & curr_scene_valid
    




    #######################



    empty = np.zeros(curr_scene_valid.shape, dtype='uint8')
    illumination = np.zeros(curr_scene_valid.shape, dtype='uint8')
    




    
    # collect training for each SIA range 
    for curr_range, sample_count in range_samples.items():

        

        
    
        print(f"SIA range: {curr_range}")

        # # SUN --------------------------------------------------------------

        # mask angles and sun
        mask_sun = curr_angle_valid & sun_mask_eroded
        
        pixel_perc_sun = int(
            np.sum(mask_sun) * 100 / max(np.sum(curr_scene_valid), 1)
        )



        if np.sum(mask_sun) > 10:
            
            # initialize empty masks
            representative_pixels_mask_snow = np.zeros(empty.shape, dtype='uint8')
            representative_pixels_mask_noSnow = np.zeros(empty.shape, dtype='uint8')
            
            print('Collecting trainings in sun')


            snow_sun, snowfree_sun = get_pixels_sun(
                bands, curr_aux_folder, mask_sun,
                curr_range
            )

    
            if np.sum(snow_sun) > 10:
                representative_pixels_mask_snow  = sample_histogram_equal(snow_sun, green, int(sample_count / 2), n_bins=20, seed=training_seed)
    
                
                save_histogram(
                    green[representative_pixels_mask_snow],
                    os.path.join(curr_aux_folder, f"hist_snow_selected_{curr_range[0]}-{curr_range[1]}.png"),
                    bins=50,
                    xlabel="Value",
                    ylabel="Count"
                )
                
      
    
            if np.sum(snowfree_sun) > 10:
                representative_pixels_mask_noSnow  = sample_histogram_equal(snowfree_sun, green, int(sample_count / 2), n_bins=20, seed=training_seed) * 2
 
                
                save_histogram(
                    green[representative_pixels_mask_noSnow==2],
                    os.path.join(curr_aux_folder, f"hist_sf_selected_{curr_range[0]}-{curr_range[1]}.png"),
                    bins=50,
                    xlabel="Value",
                    ylabel="Count"
                )
                
      
            
       
    
            # merge the two masks
            representative_pixels_mask = representative_pixels_mask_noSnow + representative_pixels_mask_snow
            empty[mask_sun] = representative_pixels_mask[mask_sun]
            
            # mark selected training pixels as sun
            illumination[representative_pixels_mask > 0] = 1

            print(str(np.sum(representative_pixels_mask_snow.flatten())) + ' SNOW PIXELS')
            print(str(np.sum(representative_pixels_mask_noSnow.flatten() / 2)) + ' NO SNOW PIXELS')

            

    # Convert points where result == 1 or 2 to a shapefile
    points = []
    values = []
    illum_values = []
    with rasterio.open(cloud_path) as src:
        cloud_crs = src.crs
        for row, col in zip(*np.where((empty == 1) | (empty == 2))):
            x, y = src.xy(row, col)
            points.append(Point(x, y))
            values.append(empty[row, col])
            illum_values.append(illumination[row, col])
            
    gdf = gpd.GeoDataFrame({"value": values, 
                            "illum": illum_values}, 
                            geometry=points, crs=cloud_crs)

    if not points:
        print(f"No representative training pixels found for {scene_id}")
        return None


    shapefile_path = os.path.join(curr_aux_folder, 'representative_pixels_for_training_samples.shp')
    gdf.to_file(shapefile_path, driver="ESRI Shapefile")


    return shapefile_path


    
    
    
        
        
    
    
