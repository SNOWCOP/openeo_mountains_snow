#%%

"""
Historical snow cover reconstruction pipeline.

Orchestrates: loading data, computing conditional probabilities,
reconstructing snow cover, downscaling climate data, and executing batch jobs.
"""

from pathlib import Path

import openeo
from omegaconf import DictConfig, OmegaConf

from openeo_mountains_snow.snow_cover_fraction import snow_cover_fraction_cube
from openeo_mountains_snow.snowcoverarea_reconstruction.scf_processing import (
    compute_scf_masks, create_modis_scf_cube,
)
from openeo_mountains_snow.snowcoverarea_reconstruction.downscale_variables import (
    downscale_shortwave_radiation, downscale_temperature_humidity,
)

_UDF_DIR = Path(__file__).parent / "udfs"
SCA_RECONSTRUCTION_UDF = _UDF_DIR / "historical_reconstruction_udf.py"
SWE_RECONSTRUCTION_UDF = _UDF_DIR / "swe_udf.py"


def run_reconstruction(cfg: DictConfig, eoconn: openeo.Connection, aoi) -> None:
    """Execute the full historical reconstruction pipeline."""

    exp = cfg.experiment
    proc = cfg.processing
    recon = cfg.reconstruction

    spatial_extent = OmegaConf.to_container(exp.aoi, resolve=True)
    temporal_extent = list(exp.temporal_extent)
    modis_temporal_extent = list(exp.modis_temporal_extent)
    agera_temporal_extent = list(exp.agera_temporal_extent)

    # ==============================
    # 1. Compute SCF Masks & Conditional Probabilities
    # ==============================

    all_masks, labels_scf = compute_scf_masks(eoconn, cfg, spatial_extent, temporal_extent)

    # ==============================
    # 2. Compute Conditional Probabilities
    # ==============================

    def merge_masks(all_masks):
        return all_masks.and_(all_masks.array_element(label="snow")) * 1.0

    mask_cp_snow = all_masks.apply(process=merge_masks)
    mask_cp_snow = mask_cp_snow.filter_bands(bands=labels_scf)
    sum_cp_snow = mask_cp_snow.reduce_dimension(reducer="sum", dimension="t")

    occurences = all_masks.reduce_dimension(reducer="sum", dimension="t")
    occurences = occurences.filter_bands(bands=labels_scf)
    occurences = occurences.rename_labels(
        dimension="bands", target=[f"occ_{b}" for b in labels_scf]
    )

    cp = sum_cp_snow / occurences
    cp = cp.rename_labels(dimension="bands", target=[f"cp_{b}" for b in labels_scf])

    # ==============================
    # 3. Load High-Resolution Data
    # ==============================

    # HR Sentinel-2 snow cover fraction (spectral indices + representative pixels)
    hr_snow = snow_cover_fraction_cube(
        aoi=spatial_extent,
        time_period=temporal_extent,
        c=eoconn,
        cfg=cfg,
    ).rename_labels(dimension="bands", target=["snow"])

    # HR MODIS SCF
    hr_scf = create_modis_scf_cube(
        eoconn, cfg, modis_temporal_extent, spatial_extent
    ).rename_labels(dimension="bands", target=["scf"])

    first_date = hr_snow.metadata.temporal_dimension.extent[0]

    cp_with_time = cp.add_dimension(name="time", label=first_date, type="temporal")
    occurences_with_time = occurences.add_dimension(name="time", label=first_date, type="temporal")

    sca = (
        hr_snow.merge_cubes(hr_scf)
        .merge_cubes(cp_with_time)
        .merge_cubes(occurences_with_time)
    )

    # ==============================
    # 4. Historical Reconstruction via UDF
    # ==============================

    sca_udf = openeo.UDF.from_file(
        str(SCA_RECONSTRUCTION_UDF),
        context={"n_days_to_reconstruct": recon.n_days},
    )

    sca = sca.apply_neighborhood(
        process=sca_udf,
        size=[
            {"dimension": "x", "value": recon.neighborhood_size, "unit": "px"},
            {"dimension": "y", "value": recon.neighborhood_size, "unit": "px"},
        ],
    )
    sca = sca.add_dimension(name="bands", label="sca", type="bands")

    # ==============================
    # 5. Load and Downscale Climate Data
    # ==============================

    dem = eoconn.load_collection("COPERNICUS_30", spatial_extent=spatial_extent)
    if dem.metadata.has_temporal_dimension():
        dem = dem.reduce_dimension(dimension="t", reducer="max")
    dem = dem.add_dimension(name="t", label=first_date, type="temporal")

    agera = eoconn.load_stac(
        cfg.agera5.stac_url,
        spatial_extent=spatial_extent,
        temporal_extent=agera_temporal_extent,
    )
    agera = agera.filter_bands(bands=list(cfg.agera5.bands))
    agera = agera.rename_labels(dimension="bands", target=list(cfg.agera5.band_aliases))

    geopotential = eoconn.load_stac(
        cfg.geopotential.stac_url,
        spatial_extent=spatial_extent,
        bands=["geopotential"],
    )
    geopotential.metadata = geopotential.metadata.add_dimension(
        "t", label=first_date, type="temporal"
    )

    agera_downscaled = downscale_temperature_humidity(agera, dem, geopotential.max_time())

    # ==============================
    # 6. Downscale Shortwave Radiation
    # ==============================

    aspect = eoconn.load_stac(
        cfg.dem.aspect_stac_url, spatial_extent=spatial_extent
    ).reduce_dimension(dimension="t", reducer="mean")

    slope = eoconn.load_stac(
        cfg.dem.slope_stac_url, spatial_extent=spatial_extent
    ).reduce_dimension(dimension="t", reducer="mean")

    slope_aspect = aspect.merge_cubes(slope).rename_labels(
        dimension="bands", target=["aspect", "slope"]
    )

    shortwave_rad_cube = downscale_shortwave_radiation(agera, slope_aspect)

    # ==============================
    # 7. Merge All & Compute SWE
    # ==============================

    total_cube = sca.merge_cubes(agera_downscaled).merge_cubes(shortwave_rad_cube)

    swe_udf = openeo.UDF.from_file(str(SWE_RECONSTRUCTION_UDF))

    swe = total_cube.apply_neighborhood(
        process=swe_udf,
        size=[
            {"dimension": "x", "value": recon.neighborhood_size, "unit": "px"},
            {"dimension": "y", "value": recon.neighborhood_size, "unit": "px"},
        ],
    )
    swe = swe.rename_labels(dimension="bands", target=["swe"])

    # ==============================
    # 8. Execute Batch Job
    # ==============================

    job_options = OmegaConf.to_container(exp.job_options, resolve=True)
    swe.execute_batch(
        title=exp.title_prefix or "swe",
        job_options=job_options,
    )




