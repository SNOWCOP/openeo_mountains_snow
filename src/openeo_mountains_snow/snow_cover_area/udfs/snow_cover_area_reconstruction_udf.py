"""Daily high-resolution snow cover area gap filling.

Port of ``SCA_reconstruction_hist`` / ``reconstruct_HR`` / ``run_hist_block``
from the original dailyHR_SCA code (vpremier/dailyHR_SCA):

1. block 1: historical gap filling of the HR map, followed by one
   SCF-conditional-probability downscaling step driven by the MODIS SCF;
2. block 2: the corrected LR SCF of block 1 is linearly interpolated in time
   where it was cloudy, clamped to the SCF min/max, and the reconstruction is
   repeated.

Input bands (per t): 0 = HR snow (0/100/205/255; NaN = no image that day -> 205,
NaN on every day -> 255),
1 = MODIS SCF on the HR grid (0..100, 205 cloud, NaN = no data),
2..2+N = conditional probability (0..1) per SCF range at t=0,
2+N..2+2N = occurrences per SCF range at t=0.
LR pixels are derived from the coordinates: floor(coord / lr_pixel_size).
"""
import numpy as np
import xarray as xr

CLOUD = 205
NO_DATA = 255
SNOW = 100

# Base SCF ranges, same order as the CP/occurrence bands.
SCF_RANGES = [(lo, lo + 10) for lo in range(0, 100, 10)]

DEFAULTS = {
    "n_days_to_reconstruct": None,  # None/0 = every day in the cube
    "lr_pixel_size": 500,
    "scf_thres": 5,       # max SCA% difference for a similar scene
    "sim_thres": 5,       # max pixel disagreement % for a similar scene
    "occur_thres": 10,    # min number of similar scenes
    "cloud_thres": 0.1,   # max cloud fraction of a historical scene
    "cp_occur_thres": 10, # min occurrences for a usable conditional probability
    "nv_thres": 40,       # max HR invalid pixels per LR pixel for a valid SCF
    "rule0100": True,     # LR SCF 0/100 forces all HR pixels to 0/100
}


def lr_labels(y, x, lr_size):
    iy = np.unique(np.floor(np.asarray(y) / lr_size), return_inverse=True)[1].ravel()
    ix = np.unique(np.floor(np.asarray(x) / lr_size), return_inverse=True)[1].ravel()
    nbx = ix.max() + 1
    return iy[:, None] * nbx + ix[None, :], int((iy.max() + 1) * nbx)


def block_sum(values, labels, n):
    return np.bincount(labels.ravel(), weights=values.ravel().astype(float), minlength=n)


def scf(snow, labels, n, nv_thres, mode=None):
    """Per LR pixel SCF of a binary HR map (original ``scf``)."""
    size = np.bincount(labels.ravel(), minlength=n).astype(float)
    valid = snow <= SNOW
    total = block_sum(np.where(valid, snow, 0), labels, n)
    nv = size - block_sum(valid, labels, n)
    if mode == "min":
        return total / size
    if mode == "max":
        return (total + nv * SNOW) / size
    with np.errstate(divide="ignore", invalid="ignore"):
        partial = np.where(nv < nv_thres, total / (size - nv), NO_DATA)
    return np.where(nv == 0, total / size, partial)


def modis_blocks(modis, labels, n):
    """MODIS SCF per LR pixel: mean if fully valid, CLOUD if cloudy, else NO_DATA."""
    size = np.bincount(labels.ravel(), minlength=n)
    valid = np.isfinite(modis) & (modis <= SNOW)
    cloudy = block_sum(np.isfinite(modis) & (modis > SNOW), labels, n) > 0
    mean = block_sum(np.where(valid, modis, 0), labels, n) / size
    full = block_sum(valid, labels, n) == size
    return np.where(full, mean, np.where(cloudy, CLOUD, NO_DATA))


def reconstruct_hr(snow, nodata, history, p):
    """Original ``reconstruct_HR``: fill clouds that are always 0 or 100 in similar scenes."""
    cloud = snow == CLOUD
    out = np.full(snow.shape, CLOUD, dtype=float)
    area = ~nodata & ~cloud
    n_area = area.sum()
    if n_area == 0 or not cloud.any() or len(history) == 0:
        return out
    curr = snow[area]
    sca_now = curr.sum() / n_area * 100 / SNOW
    h = history[:, area]
    hv = h <= SNOW
    nv = hv.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sca = np.where(hv, h, 0).sum(axis=1) / nv * 100 / SNOW
        diff = np.where(hv, np.abs(h - curr), 0).sum(axis=1) / nv * 100 / SNOW
    similar = ((n_area - nv) / n_area < p["cloud_thres"]) & \
              (np.abs(sca - sca_now) < p["scf_thres"]) & (diff < p["sim_thres"])
    if similar.sum() <= p["occur_thres"]:
        return out
    sim = history[similar]
    v = (sim <= SNOW) & cloud
    cnt = v.sum(axis=0)
    total = np.where(v, sim, 0).sum(axis=0)
    out[cloud & (cnt > 0) & (total == SNOW * cnt)] = SNOW
    out[cloud & (cnt > 0) & (total == 0)] = 0
    return out


def hist_rec(snow, nodata, scf_init, modis_lr, history, cp, occ, labels, n, p):
    """Original ``SCA_reconstruction_hist`` for one day.

    Returns the reconstructed HR map, the corrected LR SCF and the LR SCF min/max.
    """
    snow = snow.copy()
    scf_hr = scf_init.copy()
    scf_rec = True
    while True:
        cloud = snow == CLOUD
        rec = reconstruct_hr(snow, nodata, history, p)
        snow[cloud] = rec[cloud]
        scf_hr = np.where(scf_hr == NO_DATA, scf(snow, labels, n, p["nv_thres"]), scf_hr)
        scf_min = scf(snow, labels, n, p["nv_thres"], "min")
        scf_max = scf(snow, labels, n, p["nv_thres"], "max")

        if scf_rec:
            # NaN != NO_DATA, so NaN in scf_hr (block 2 interpolation gaps) propagates.
            scf_lr = np.where(scf_hr != NO_DATA, scf_hr, modis_lr)
            valid = (scf_lr >= 0) & (scf_lr <= SNOW)
            scf_lr = np.where(valid & (scf_lr < scf_min), scf_min, scf_lr)
            scf_lr = np.where(valid & (scf_lr > scf_max), scf_max, scf_lr)
            lr2hr = scf_lr[labels]

            out = np.full(snow.shape, NO_DATA, dtype=float)
            if p["rule0100"]:
                out[lr2hr == SNOW] = SNOW
                out[lr2hr == 0] = 0
            for i, (lo, hi) in enumerate(SCF_RANGES):
                in_range = ((lr2hr >= lo) if i == 0 else (lr2hr > lo)) & (lr2hr <= hi)
                fill = in_range & (out == NO_DATA) & ((cp[i] == 0) | (cp[i] == 1)) \
                    & (occ[i] > p["cp_occur_thres"]) & (snow > SNOW)
                out[fill] = cp[i][fill] * SNOW
            snow = np.where(out != NO_DATA, out, snow)
            snow[nodata] = NO_DATA
            scf_hr = scf_lr
            scf_rec = False

        if not (rec[cloud] != CLOUD).any():
            return snow, scf_hr, scf_min, scf_max


def interpolate_scf(scf_days, scf_min, scf_max):
    """Original ``interpolate_scf``: linear in time over cloudy (205) LR values, clamped.

    Cloudy values without valid values on both sides become NaN. Deviation from the
    original: 255 is not used as an interpolation anchor.
    """
    out = scf_days.astype(float)
    t = np.arange(out.shape[0])
    for j in range(out.shape[1]):
        ok = (out[:, j] >= 0) & (out[:, j] <= SNOW)
        cloudy = out[:, j] == CLOUD
        out[cloudy, j] = np.nan
        if ok.sum() > 1:
            inside = cloudy & (t > t[ok][0]) & (t < t[ok][-1])
            out[inside, j] = np.interp(t[inside], t[ok], out[ok, j])
    valid = (out >= 0) & (out <= SNOW)
    out = np.where(valid & (out < scf_min), scf_min, out)
    return np.where(valid & (out > scf_max), scf_max, out)


def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:
    p = {**DEFAULTS, **(context or {})}
    n_ranges = len(SCF_RANGES)
    if cube.sizes["t"] < 2:
        return cube.isel(bands=slice(0, 1)).assign_coords(bands=["reconstructed_snow"])

    labels, n = lr_labels(cube.y.values, cube.x.values, p["lr_pixel_size"])
    # NaN on a day = no HR image = cloud (original); NaN on every day = outside the data (no data).
    snow_all = cube.isel(bands=0).values.astype(float)
    never = np.isnan(snow_all).all(axis=0)
    snow_all = np.where(np.isnan(snow_all), CLOUD, snow_all)
    snow_all[:, never] = NO_DATA
    modis = cube.isel(bands=1).values.astype(float)
    cp = cube.isel(t=0, bands=slice(2, 2 + n_ranges)).values.astype(float)
    occ = cube.isel(t=0, bands=slice(2 + n_ranges, 2 + 2 * n_ranges)).values.astype(float)

    n_t = snow_all.shape[0]
    n_days = p["n_days_to_reconstruct"]
    days = list(range(max(0, n_t - int(n_days)) if n_days else 0, n_t))
    nodata = snow_all == NO_DATA

    def history(i):
        return np.delete(snow_all, i, axis=0)

    # Block 1
    modis_lr, b1 = {}, {}
    for i in days:
        modis_lr[i] = modis_blocks(modis[i], labels, n)
        b1[i] = hist_rec(snow_all[i], nodata[i], modis_lr[i], modis_lr[i],
                         history(i), cp, occ, labels, n, p)

    # Block 2: earlier days in the chunk act as interpolation anchors, built like block 1's
    # initial LR SCF (MODIS, HR aggregate where MODIS is no data) without reconstruction.
    pre = range(days[0])

    def raw_lr(i):
        lr = modis_blocks(modis[i], labels, n)
        lr = np.where(lr == NO_DATA, scf(snow_all[i], labels, n, p["nv_thres"]), lr)
        return (lr, scf(snow_all[i], labels, n, p["nv_thres"], "min"),
                scf(snow_all[i], labels, n, p["nv_thres"], "max"))

    stacks = [raw_lr(i) for i in pre] + [b1[i][1:] for i in days]
    scf_interp = interpolate_scf(*(np.stack([s[k] for s in stacks]) for k in range(3)))[len(pre):]
    result = snow_all.copy()
    for k, i in enumerate(days):
        result[i] = hist_rec(b1[i][0], nodata[i], scf_interp[k], modis_lr[i],
                             history(i), cp, occ, labels, n, p)[0]

    out = cube.isel(t=days, bands=slice(0, 1)).copy(data=result[days][:, None].astype(np.uint8))
    return out.assign_coords(bands=["reconstructed_snow"])
