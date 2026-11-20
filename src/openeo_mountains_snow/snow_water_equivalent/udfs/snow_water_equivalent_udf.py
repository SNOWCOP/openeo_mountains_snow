"""Snow water equivalent (SWE) reconstruction.

Port of ``get_status_and_delta`` / ``get_melt_pomeroy`` /
``compute_state_and_accumulation`` / ``get_swe`` from the original
SWE_reconstruction code. Melt of each snow period is summed until melt-out and
redistributed over the accumulation days in proportion to their precipitation.

Input bands (per t): 0 = SCA (0/100/205, NaN = no data), 1 = air temperature (°C),
2 = precipitation (mm/day), 3 = incoming shortwave radiation (W/m²).
Output band ``swe`` in mm, rounded to whole mm; NaN where SCA is cloud/no data
TODO; current mitigation for sparse MODIS offering on CDSEThe snowpack is carried through those days: the SCA is gap-filled with the last known
value (the first known value before the first observation).

Context ``basin_sca`` ({"YYYY-MM-DD": percent}): basin-wide SCA series, as in the original.
Without it, the SCA of the neighborhood is used.
"""
import numpy as np
import xarray as xr

DEFAULTS = {
    "temperature_threshold": 2.0,     # °C (original: 275.15 K)
    "precipitation_threshold": 5.0,   # mm/day
    "sca_increase_threshold": 10.0,   # basin SCA increase, percentage points
    "TF": 0.24,                       # temperature factor, mm / (°C day)
    "SRF": 0.15,                      # shortwave radiation factor
    "no_melt_days": 0,                # original forces zero melt for the first 122 days
}


def get_status_and_delta(sca, ta, pr, p, basin_sca=None):
    """1 = accumulation, -1 = melting, 0 = neutral; delta = share of period precipitation."""
    if basin_sca is None:
        basin_sca = np.array([np.nanmean(np.where(s <= 100, s, np.nan)) if (s <= 100).any() else np.nan
                              for s in sca])
    sca_increase = np.concatenate([[0.0], np.diff(basin_sca)])
    wet = pr > p["precipitation_threshold"]
    cold = ta < p["temperature_threshold"]
    warm = ta > p["temperature_threshold"]
    increase = (sca_increase > p["sca_increase_threshold"])[:, None, None]
    acc = wet & (cold | (warm & increase))
    status = np.where(acc, 1, np.where(warm, -1, 0)).astype(np.int8)

    masked_pr = np.where(status == 1, pr, np.nan)
    total = np.nansum(masked_pr, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = masked_pr / np.where(total > 0, total, np.nan)
    return status, np.nan_to_num(delta, nan=0.0)


def get_melt_pomeroy(sca, ta, pr, sw, status, p):
    d_wet, d_dry = 0.005 * 24, 0.0003 * 24
    asmn, asmx, salb = 0.6, 0.9, 10
    t_thresh = p["temperature_threshold"]
    albs = np.full(sca.shape, 0.9, dtype=np.float32)
    melt = np.zeros(sca.shape, dtype=np.float32)
    for i in range(sca.shape[0] - 1):
        alb_prev = albs[i]
        pr_curr = np.where(status[i + 1] == 1, pr[i + 1], 0)
        mask = sca[i + 1] > 0
        alb_dry = alb_prev - d_dry
        alb_wet = (alb_prev - asmn) * np.exp(-d_wet) + asmn
        alb_t = np.where(ta[i + 1] < t_thresh, alb_dry, alb_wet)
        alb = alb_prev.copy()
        alb[mask] = alb_t[mask] + (asmx - alb_t[mask]) * (pr_curr[mask] / salb)
        albs[i + 1] = np.clip(alb, asmn, asmx)

        warm = ta[i + 1] > t_thresh
        melt[i + 1] = np.minimum(p["TF"] * np.where(warm, ta[i + 1], 0)
                                 + p["SRF"] * np.where(warm, sw[i + 1], 0) * (1 - albs[i + 1]), 60)
        if i < p["no_melt_days"]:
            melt[i + 1] = 0
    return melt


def compute_state_and_accumulation(sca, melt, status, delta):
    sca_sum = np.zeros(sca.shape, dtype=np.float32)
    tot_acc = np.zeros(sca.shape, dtype=np.float32)
    changes = np.zeros(sca.shape, dtype=np.float32)
    for i in range(sca.shape[0] - 1):
        prev, curr = sca[i], sca[i + 1]
        melt_curr = melt[i].copy()

        mask_snow = (curr == 100) | (curr == 205)
        changes[i + 1][mask_snow] = status[i][mask_snow]
        start = (curr == 100) & (prev == 0)
        changes[i + 1][start] = 2
        end = (curr == 0) & (prev == 100)
        changes[i + 1][end] = -2

        melt_curr[status[i] != -1] = 0
        reset = ~mask_snow & ~end
        tot_acc[i + 1] = tot_acc[i] + melt_curr
        tot_acc[i + 1][reset] = 0
        tot_acc[i + 1][start] = melt_curr[start]

        delta_sca = delta[i + 1].copy()
        delta_sca[status[i + 1] != 1] = 0
        sca_sum[i + 1] = sca_sum[i] + delta_sca
        sca_sum[i + 1][reset] = 0
        sca_sum[i + 1][start] = delta_sca[start]

    sca_sum[changes != -2] = 0
    tot_acc[changes != -2] = 0

    def bfill(a):
        a = np.where(a != 0, a, np.nan)
        for i in range(a.shape[0] - 2, -1, -1):
            a[i] = np.where(np.isnan(a[i]), a[i + 1], a[i])
        return a

    return bfill(sca_sum), bfill(tot_acc)


def get_swe(sca, melt, status, delta, sca_sum, tot_acc):
    swe = np.zeros(sca.shape, dtype=np.float32)
    swe[:1][sca[:1] > 100] = np.nan
    for i in range(sca.shape[0] - 1):
        swe[i + 1] = swe[i]
        acc = status[i + 1] == 1
        melting = status[i + 1] == -1
        with np.errstate(divide="ignore", invalid="ignore"):
            dsca = delta[i + 1] / sca_sum[i + 1]
        swe[i + 1][acc] = swe[i][acc] + dsca[acc] * tot_acc[i + 1][acc]
        swe[i + 1][melting] = swe[i][melting] - melt[i][melting]
        swe[i + 1][sca[i + 1] > 100] = np.nan
        swe[i + 1][sca[i + 1] == 0] = 0
    swe[swe < 0] = 0
    return swe


def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:
    p = {**DEFAULTS, **(context or {})}
    #TODO define a better fix
    observed = cube.isel(bands=0).values.astype(np.float32)
    unknown = ~(observed <= 100)  # cloud (205), no data (NaN) or other flags
    sca = np.where(unknown, np.nan, observed)
    for i in range(1, len(sca)):  # forward fill
        sca[i] = np.where(np.isnan(sca[i]), sca[i - 1], sca[i])
    for i in range(len(sca) - 2, -1, -1):  # back fill before the first observation
        sca[i] = np.where(np.isnan(sca[i]), sca[i + 1], sca[i])
    sca = np.nan_to_num(sca, nan=255)  # never observed
    ta = cube.isel(bands=1).values.astype(np.float32)
    pr = cube.isel(bands=2).values.astype(np.float32)
    sw = cube.isel(bands=3).values.astype(np.float32)
    basin_sca = None
    if p.get("basin_sca"):
        dates = np.datetime_as_string(cube.t.values, unit="D")
        basin_sca = np.array([p["basin_sca"].get(d, np.nan) for d in dates], dtype=float)

    status, delta = get_status_and_delta(sca, ta, pr, p, basin_sca)
    melt = get_melt_pomeroy(sca, ta, pr, sw, status, p)
    sca_sum, tot_acc = compute_state_and_accumulation(sca, melt, status, delta)
    swe = np.round(get_swe(sca, melt, status, delta, sca_sum, tot_acc))
    swe[unknown] = np.nan

    out = cube.isel(bands=slice(0, 1)).copy(data=swe[:, None])
    return out.assign_coords(bands=["swe"])
