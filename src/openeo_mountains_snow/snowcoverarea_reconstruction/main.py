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

# %%

import xarray as xr
import matplotlib.pyplot as plt
import math

# Load dataset
path = r"C:\Users\VROMPAYH\Downloads\openEO (22).nc"
ds = xr.open_dataset(path)

# Keep only spatial variables (exclude CRS)
variables = [
    v for v in ds.data_vars
    if ds[v].dims == ("t", "y", "x")
]

n_vars = len(variables)

cols = 3
rows = math.ceil(n_vars / cols)

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
axes = axes.flatten()

for i, var in enumerate(variables):
    
    # First variable -> timestep 5
    if i == 0:
        data = ds[var].isel(t=5)
        timestep_used = 5
    else:
        data = ds[var].isel(t=0)
        timestep_used = 0

    # Compute min/max ignoring NaNs
    vmin = float(data.min(skipna=True))
    vmax = float(data.max(skipna=True))

    im = axes[i].imshow(data, vmin=vmin, vmax=vmax)
    axes[i].set_title(f"{var} (t={timestep_used})")
    axes[i].axis("off")

    plt.colorbar(im, ax=axes[i])

# Hide unused axes
for j in range(n_vars, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()
# %%

import numpy as np
import xarray as xr
import pandas as pd 


def get_status_and_delta(ta, era5, temp_thres=1.0, prec_thres=1.0):
    """
    Compute:
      (1) Boolean accumulation mask: True = accumulation, False = melting/other
      (2) Fraction of precipitation contributing to SWE accumulation (per timestep)

    Parameters
    ----------
    SCA : xarray.DataArray
        Snow cover area classification (dims: time,x,y)
    ta : xarray.DataArray or Dataset
        Air temperature time series (°C) with variable 't2m'
    era5 : xarray.Dataset or DataArray
        ERA5-Land dataset containing variable 'tp' (precipitation, meters)
    temp_thres : float
        Temperature threshold for accumulation (default = 1°C)
    prec_thres : float
        Precipitation threshold for accumulation (default = 1 mm/day)

    Returns
    -------
    status : xarray.DataArray (bool)
        True where accumulation conditions are met
        False where melting or no-precipitation
    delta : xarray.DataArray (float32)
        Fraction of total accumulation at each timestep
        Sum over time per pixel ≈ 1 (where accumulation occurs)
    pr_reprojected : xarray.DataArray
        Precipitation reprojected to SCA grid
    """

    # ERA5 tp : precipitation pr_reprojected should already be correct after merge_cube

   

    # Boolean accumulation mask
    status = xr.where(
                    (ta < temp_thres) & (era5 > prec_thres),
                    1,
                    -1
                ).astype('int8')
    
    # Masked precipitation
    masked_pr = era5.where(status == 1)
    
    # Total accumulation per pixel (lazy reduction)
    sum_pr = masked_pr.sum(dim='t')

    # Safe denominator → avoid division by zero
    safe_sum_pr = sum_pr.where(sum_pr > 0)

    # Redistribute accumulation precipitation fractionally
    delta = masked_pr / safe_sum_pr

    # No accumulation → 0
    delta = delta.fillna(0).astype('float32')

    return status, delta

def get_melt_pomeroy(SCA, ta, era5, SW, status, TF=1.2, SRF=0.2256):
    """
    Compute snowmelt over time and space using Pomeroy albedo scheme.

    Parameters
    ----------
    SCA : xarray.DataArray
        Snow-covered area dataset, must contain variable 'SCA' with dimensions (time, y, x).
    ta : xarray.DataArray
        Air temperature dataset (°C) with same dimensions as SCA.
    era5 : xarray.DataArray
        Precipitation (mm/day) with a 'time' dimension matching SCA.
    SW : xarray.DataArray
        Incoming shortwave radiation dataset (W/m² or MJ/m²/day).
    status : xarray.DataArray
        Binary mask (1/ -1) defining active snow/melt areas per timestep.
    TF : float, optional
        Temperature factor for degree-day melt component (default = 1.2).
    SRF : float, optional
        Shortwave radiation factor for radiative melt component (default = 0.2256).

    Returns
    -------
    melt_da : xarray.DataArray
        Melt (same units as TF/SRF * input fields), dimensions (time, y, x),
        with the same coordinates as the input `SCA`.
    """

    # --- Parameters ---
    d_wet = 0.005 * 24 
    d_dry = 0.0003 * 24 
    asmn = 0.6  # min albedo
    asmx = 0.9  # max albedo
    Salb = 10  # 10 mm

    # --- Initialize output arrays ---
    dim = tuple(SCA.shape)  # (t, y, x)
    albs = np.zeros(dim, dtype=np.float32) + 0.9
    melt = np.zeros(dim, dtype=np.float32)

    time = SCA['t']

    # --- Time loop ---
    for i in range(len(time) - 1):
        # Previous albedo
        alb_prev = albs[i, :, :].copy()

        # Current timestep variables
        sca_curr = SCA.isel(t=i+1).values
        status_curr = status.isel(t=i+1).values
        ta_curr = ta.isel(t=i+1).values
        SW_curr = SW.isel(t=i+1).values

        # Precipitation for current day
        pr_curr = era5.isel(t=i+1).values.copy()
        pr_curr = np.where(status_curr == 1, pr_curr, 0)

        # Compute only where snow cover > 0 
        mask = (sca_curr > 0) & (sca_curr != 205) #TODO avoid clouds?
        
        alb_dry = alb_prev - d_dry
        alb_wet = ((alb_prev - asmn) * np.exp(-d_wet)) + asmn
        
        alb_t = alb_dry.copy()
        alb_t = np.where(ta_curr < 0, alb_dry, alb_wet)
        
        alb_curr = alb_prev.copy()
        alb_curr[mask] = alb_t[mask] + (asmx - alb_t[mask]) * (pr_curr[mask] / Salb)
        
        # --- Clip albedo and save ---
        alb_curr = np.clip(alb_curr, asmn, asmx)
        albs[i + 1, :, :] = alb_curr
        
        # --- Compute current melt with updated albedo ---
        melt_curr = np.zeros_like(ta_curr)
        melt_curr[mask] = (
            TF * np.maximum(ta_curr[mask], 0) + SRF * SW_curr[mask] * (1 - alb_curr[mask])
        )
        melt[i + 1, :, :] = melt_curr

    # --- Convert melt array to xarray.DataArray ---
    melt_da = xr.DataArray(
        melt,
        dims=("t", "y", "x"),
        coords={"t": SCA['t'], "y": SCA.y, "x": SCA.x},
        name="melt",
        attrs={
            "units": "mm water equivalent/day",
            "description": "Snowmelt computed dynamically using evolving albedo."
        },
    )

    return melt_da

def compute_state_and_accumulation(SCA, melt, status, delta):
    """
    Compute cumulative snow accumulation (as a fraction of total precipitation,
                                      ie. the deltas)
    and total melt energy that is assumed to correspond to the total 
    accumulation (mass conservation) for each snow period between melt-out 
    events.

    The function loops over daily Snow Cover Area (SCA) maps and tracks
    accumulation and melting through time for each pixel. Each "snow period"
    (from the first accumulation until complete melt-out) is treated as a 
    self-contained event. The cumulative precipitation fraction (sca_sum)
    and total melt energy (tot_acc) are both reset whenever the pixel becomes
    snow-free.

    Parameters
    ----------
    SCA : xr.DataArray
        Snow classification (SCA) over time with dims ('time','x','y')
        Values: 0 = no snow, 100 = snow, 205 = cloud/no data
    melt : xr.DataArray
        Melt energy (e.g., degree-day or energy balance proxy) with same time and 
        spatial dimensions as SCA.
    status : xr.DataArray (int)
        Accumulation/melting state mask, with values:
            1  = accumulation (cold + precipitation)
           -1  = melting / no accumulation
    delta : xr.DataArray (float)
        Fraction of daily precipitation contributing to accumulation, 
        such that the sum over time during the hydrological year ≈ 1 per pixel.

    Returns
    -------
    sca_sum_xr : xr.DataArray
        Cumulative fraction of precipitation per snow period.
    tot_acc_xr : xr.DataArray
        Total melt energy accumulated during each snow period (same logic as sca_sum_xr).
    
    Notes
    -----
    - The `changes` array is used internally to track pixel transitions:
        +2 → first day of snow accumulation (snow onset)
        +1 → snow-covered and accumulating
         0 → snow-free (after the date of snow end)
        -1 → snow-covered but melting
        -2 → last melt-out
    - Both `sca_sum` and `tot_acc` are reset to zero whenever the pixel becomes snow-free.
    - Cloud/no-data pixels (205) are treated as snow-covered for continuity.
    """

    # --- Initialize dimensions and arrays ---
    time = SCA['t']
    dim = tuple(SCA.shape)  # (t, y, x)
    
    sca_sum = np.zeros(dim, dtype=np.float32)
    tot_acc = np.zeros(dim, dtype=np.float32)
    changes = np.zeros(dim, dtype=np.float32)
    
    # === Time iteration over SCA ===
    for i in range(len(time) - 1):
    
        # Snow cover for previous and current day
        snow_prev = SCA.isel(t=i).values
        snow_curr = SCA.isel(t=i+1).values
        
        # Melt for the current day
        melt_curr = melt.isel(t=i).values.copy()

        # --- Assign snow state transitions ---
        mask_snow = np.logical_or(snow_curr == 100, snow_curr == 205)
        changes[i+1, :, :][mask_snow] = status.isel(t=i).values[mask_snow]

        # Start of new snow period
        mask_snow_start = np.logical_and(snow_curr == 100, snow_prev == 0)
        changes[i+1, :, :][mask_snow_start] = 2
        
        # End of snow period (melt-out)
        mask_snow_end = np.logical_and(snow_curr == 0, snow_prev == 100)
        changes[i+1, :, :][mask_snow_end] = -2

        # --- Compute total accumulation (or total melt) ---
        melt_curr[changes[i+1, :, :] > 0] = 0  # skip accumulation pixels

        tot_acc[i+1, :, :] = tot_acc[i, :, :] + melt_curr
        tot_acc[i+1, :, :][changes[i+1, :, :] == 0] = 0  # reset where snow-free
        tot_acc[i+1, :, :][mask_snow_start] = melt_curr[mask_snow_start]

        # --- Compute cumulative accumulation fraction (precipitation delta) ---
        delta_sca = delta.isel(t=i+1).values.copy()
        delta_sca[status.isel(t=i+1).values != 1] = 0

        sca_sum[i+1, :, :] = sca_sum[i, :, :] + delta_sca
        sca_sum[i+1, :, :][changes[i+1, :, :] == 0] = 0  # reset where snow-free
        sca_sum[i+1, :, :][mask_snow_start] = delta_sca[mask_snow_start]

    # --- Final masking: keep only the values when melt-out ---
    sca_sum[changes != -2] = 0
    tot_acc[changes != -2] = 0

    # --- Convert to xarray and interpolate missing periods ---
    sca_sum_xr = xr.DataArray(
        sca_sum, 
        dims=('t', 'y', 'x'),
        coords={'t': SCA['t'], 'y': SCA.y, 'x': SCA.x}
    )
    tot_acc_xr = xr.DataArray(
        tot_acc, 
        dims=('t', 'y', 'x'),
        coords={'t': SCA['t'], 'y': SCA.y, 'x': SCA.x}
    )

    changes = xr.DataArray(
        changes, 
        dims=('t', 'y', 'x'),
        coords={'t': SCA['t'], 'y': SCA.y, 'x': SCA.x}
    )

    # Fill missing (zero) values backward in time within snow events
    sca_sum_xr = sca_sum_xr.where(sca_sum_xr != 0).bfill(dim='t')
    tot_acc_xr = tot_acc_xr.where(tot_acc_xr != 0).bfill(dim='t')

    return sca_sum_xr, tot_acc_xr
 
def get_swe(SCA, melt, status, delta, sca_sum_xr, tot_acc_xr):
    """
    Compute Snow Water Equivalent (SWE) time series using:
    - status_xr: accumulation/melting mask (+1/-1)
    - delta: fractional precipitation contributions
    - sca_sum_xr: fractional snow accumulation
    - tot_acc_xr: total accumulation energy

    Parameters
    ----------
    SCA : xr.DataArray
        Snow classification (SCA) over time with dims ('time','x','y')
        Values: 0 = no snow, 100 = snow, 205 = cloud/no data
    melt : xr.DataArray
        Melt energy for each timestep
    status : xr.DataArray
        Accumulation status mask (+1 accumulation, -1 melting)
    delta : xr.DataArray
        Fractional precipitation available for SWE accumulation
    sca_sum_xr : xr.DataArray
        Fractional snow accumulation contributions
    tot_acc_xr : xr.DataArray
        Thermal energy available for melting

    Returns
    -------
    swe : np.ndarray
        Snow Water Equivalent array (band, time, y, x)
    """

    dim = tuple(SCA.shape)  # (t, y, x)
    swe = np.zeros(dim, dtype=np.float32)

    for i in range(len(SCA['t']) - 1):
        date = pd.Timestamp(SCA['t'][i + 1].values).strftime("%Y-%m-%d")

        melt_curr = melt.isel(t=i).values.copy()
        snow_curr = SCA.isel(t=i+1).values

        # Masks
        mask_acc = status.isel(t=i+1).values == 1
        mask_melt = status.isel(t=i+1).values == -1

        # Spatial increment for accumulation
        dsca = delta.isel(t=i+1).values.copy() / sca_sum_xr.isel(t=i+1).values
        dsca[mask_melt] = 0  # zero where not accumulating

        # Update SWE
        swe[i+1][mask_acc] = swe[i][mask_acc] + dsca[mask_acc] * tot_acc_xr.isel(t=i+1).values[mask_acc]
        swe[i+1][mask_melt] = swe[i][mask_melt] - melt_curr[mask_melt]

        # Invalid snow codes (cloud / no data)
        swe[i+1][snow_curr > 100] = np.nan
        swe[i+1][snow_curr == 0] = 0

    # Remove negative SWE values (true melt-out)
    swe[swe < 0] = 0

    swe_xr = xr.DataArray(
        swe, 
        dims=('t', 'y', 'x'),
        coords={'t': SCA['t'], 'y': SCA.y, 'x': SCA.x}
    )

    return swe_xr


# Extract bands (assuming this order - adjust indices if needed)
sca  = ds["sca"]
ta   = ds["temperature_downscaled"].isel(t=0)
era5 = ds["relative_humidity"].isel(t=0)
sw   = ds["solar-radiation-flux"].isel(t=0)

ta   = ta.expand_dims(t=sca.t).transpose("t","y","x")
era5 = era5.expand_dims(t=sca.t).transpose("t","y","x")
sw   = sw.expand_dims(t=sca.t).transpose("t","y","x")

ta = ta - 273.15  # Convert from K to °C if needed
sw = sw / 1000000  # Convert from MJ/m^2/day to W/m^2 if needed

status, delta = get_status_and_delta(ta, era5)

TF = 1.2  # melt factor mm / (°C day)
SRF = 0.2256  # radiation melt factor
melt = get_melt_pomeroy(sca, ta, era5, sw, status, TF=TF, SRF=SRF)
sca_sum_xr, tot_acc_xr = compute_state_and_accumulation(sca, melt, status, delta)
swe = get_swe(sca, melt, status, delta, sca_sum_xr, tot_acc_xr)


import matplotlib.pyplot as plt


for i in range(sca.sizes['t']):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot changes
    swe.isel(t=i).plot(ax=axes[0])
    axes[0].set_title(f"swe – timestep {i}")

    # Plot sca
    sca.isel(t=i).plot(ax=axes[1])
    axes[1].set_title(f"sca – timestep {i}")

    plt.tight_layout()
    plt.show()


#%%
# Expand them back to match time dimension
ta   = ta.expand_dims(t=sca.t).transpose("t","y","x")
era5 = era5.expand_dims(t=sca.t).transpose("t","y","x")
sw   = sw.expand_dims(t=sca.t).transpose("t","y","x")


# Ensure precipitation is in mm/day (if ERA5 tp in meters, convert)
# pr = pr * 1000  # uncomment if needed

# Step 1: Get status and delta
status, delta = get_status_and_delta(ta, era5)

# Step 2: Compute melt using Pomeroy scheme
TF = 1.2  # melt factor mm / (°C day)
SRF = 0.2256  # radiation melt factor
melt = get_melt_pomeroy(sca, ta, era5, sw, status, TF=TF, SRF=SRF)

# Step 3: Compute cumulative state and accumulation
sca_sum_xr, tot_acc_xr = compute_state_and_accumulation(sca, melt, status, delta)

# Step 4: Compute SWE
swe = get_swe(sca, melt, status, delta, sca_sum_xr, tot_acc_xr)

if swe is None or swe.size == 0:
    # Create a dummy result with correct shape to avoid breaking the pipeline
    swe = np.ones((len(cube.t), len(cube.y), len(cube.x)), dtype=np.uint8)*255


swe = np.expand_dims(swe, axis=1)  # Add bands dimension at position 1
    















