import xarray as xr
import numpy as np
import pandas as pd
import logging
import boto3
import tempfile
import os


logger = logging.getLogger(__name__)

class Checkpoint:
    def __init__(self, access_key, secret_key, token, bucket, prefix, endpoint):
        self.bucket = bucket
        self.prefix = prefix.rstrip('/')
        self.t_coords = None
        self.y_coords = None
        self.x_coords = None

        self.s3 = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_session_token=token,
            endpoint_url=endpoint
        )
        logger.info(f"S3 checkpoint ready: {bucket}/{prefix}")

    def set_reference(self, cube):
        """Store reference coordinates from the input cube."""
        if 't' in cube.coords:
            self.t_coords = cube.t.values
        if 'y' in cube.coords:
            self.y_coords = cube.y.values
        if 'x' in cube.coords:
            self.x_coords = cube.x.values

    def save(self, data, name, **tags):
        """
        Save data to S3 as NetCDF.
        Works with numpy arrays or xarray DataArrays.
        If numpy, uses stored coordinates to build a DataArray.
        """
        # Convert numpy to DataArray if possible
        if isinstance(data, np.ndarray):
            if self.y_coords is None or self.x_coords is None:
                logger.warning("No reference coordinates. Call set_reference() first.")
                return
            # If 3D (t,y,x) and we have t coords
            if data.ndim == 3 and self.t_coords is not None:
                data = xr.DataArray(
                    data,
                    dims=('t', 'y', 'x'),
                    coords={'t': self.t_coords, 'y': self.y_coords, 'x': self.x_coords}
                )
            elif data.ndim == 2:
                data = xr.DataArray(
                    data,
                    dims=('y', 'x'),
                    coords={'y': self.y_coords, 'x': self.x_coords}
                )
            else:
                logger.error(f"Cannot save numpy array with shape {data.shape} – expected 2D or 3D")
                return

        if not isinstance(data, xr.DataArray):
            logger.error(f"Expected DataArray, got {type(data)}")
            return

        # Build filename from tags, coordinates, and shape
        parts = [name]
        for key in sorted(tags.keys()):
            val = tags[key]
            if isinstance(val, int):
                parts.append(f"{key}{val:03d}")
            else:
                parts.append(f"{key}_{val}")

        # Add spatial anchor (top-left corner) for quick identification
        if 'x' in data.coords and 'y' in data.coords:
            x_min = int(data.x.values.min())
            y_min = int(data.y.values.min())
            parts.append(f"x{x_min}_y{y_min}")

        # Add temporal anchor if present
        if 't' in data.coords and len(data.t) > 0:
            t_str = pd.to_datetime(data.t.values[0]).strftime('%Y%m%d')
            parts.append(f"t{t_str}")

        # Add shape
        shape_str = "_".join(f"{d}{s}" for d, s in zip(data.dims, data.shape))
        parts.append(shape_str)

        filename = "_".join(parts) + ".nc"
        key = f"{self.prefix}/{filename}"

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp:
                tmp_path = tmp.name
                data.to_dataset(name='data').to_netcdf(tmp_path, format='NETCDF4', engine='netcdf4')

            self.s3.upload_file(Filename=tmp_path, Bucket=self.bucket, Key=key)
            os.unlink(tmp_path)
            logger.info(f"✓ {filename}")
            return filename
        except Exception as e:
            logger.error(f"✗ {filename}: {e}")
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return None

def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:
    """
    Compute SWE from a merged cube containing:
    - band 0: SCA (snow cover area, with values 0, 100, 205)
    - band 1: temperature (from agera, °C)
    - band 2: humidity (from agera, %)
    - band 3: shortwave radiation (W/m² or MJ/m²/day)
    
    The cube is expected to have dimensions (time, bands, y, x)
    """

    cfg = context.get("checkpoint_config")
    if cfg:
        chk = Checkpoint(
            access_key=cfg.get('access_key', ''),
            secret_key=cfg.get('secret_key', ''),
            token=cfg.get('token', ''),
            bucket=cfg.get('bucket', 'openeo-artifacts-waw3-1'),
            prefix=cfg.get('prefix', ''),
            endpoint=cfg.get('endpoint', 'https://s3.waw3-1.openeo.v1.dataspace.copernicus.eu'),
        )
        chk.set_reference(cube)
        # Save input cube metadata (full shape and coordinates)
        chk.save(cube.isel(bands=0).drop_vars('bands'), name="input_sca", stage="input")
    else:
        chk = None
        logger.info("No checkpoint config – running without S3 saves")
    
    # Log full details about this tile
    logger.info(f"Input cube shape: {cube.shape}, dims: {cube.dims}")
    
    
    # Extract bands (assuming this order - adjust indices if needed)
    sca = cube.isel(bands=0)  # SCA band
    ta = cube.isel(bands=1)   # temperature band
    era5 = cube.isel(bands=2)    # humidity band
    sw = cube.isel(bands=3)    # shortwave radiation band
    
    
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
    swe = np.expand_dims(swe, axis=1)  # Add bands dimension at position 1
    
    # Create result with same dimension order as input
    result = xr.DataArray(
        swe,
        dims=("t", "bands", "y", "x"),  # Match input dimension order
        coords={
            "t": cube.coords["t"].values,
            "bands": ["swe"],  # Single band
            "y": cube.coords["y"].values,
            "x": cube.coords["x"].values
        },
        name="swe",
        attrs={
            "units": "mm",
            "description": "Snow Water Equivalent",
            "long_name": "Snow Water Equivalent"
        }
    )

    if chk:
        # Loop over bands
        for band_idx, band_name in enumerate(result.bands.values):
            # Loop over time
            for t_idx in range(result.shape[0]):
                # Extract single timestep for this band
                data_slice = result.isel(t=t_idx, bands=band_idx)  # Shape: (y, x)
                
                # Get date string
                date_str = pd.Timestamp(result.t.values[t_idx]).strftime('%Y%m%d')
                
                # Save with band name and time in filename
                chk.save(
                    data_slice,
                    name=band_name,
                    stage="output",
                    band=band_name,
                    time_idx=t_idx,
                    date=date_str
                )

    logger.info(f"Output SWE cube shape: {result.shape}, dims: {result.dims}")
    return result


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
        date = pd.Timestamp(time[i + 1].values).strftime("%Y-%m-%d")
    
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

    return swe


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
        date = pd.Timestamp(time[i + 1].values).strftime("%Y-%m-%d")

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
        mask = sca_curr > 0
        
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