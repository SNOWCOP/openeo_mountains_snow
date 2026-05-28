#%%

"""
Historical snow cover reconstruction pipeline.

Orchestrates the full reconstruction by calling focused modules:
- scf_processing: SCF masks, conditional probabilities, MODIS data
- downscale_variables: climate data loading and downscaling
- UDFs: historical reconstruction and SWE computation
"""

from pathlib import Path

import openeo
from omegaconf import DictConfig, OmegaConf

from openeo_mountains_snow.scf.snow_cover_fraction import snow_cover_fraction_cube
from openeo_mountains_snow.sca.scf_processing import (
    compute_scf_masks, compute_conditional_probabilities, create_modis_scf_cube,
)
from openeo_mountains_snow.swe.downscale_variables import (
    load_climate_data,
)

_PKG_DIR = Path(__file__).parent
SCA_RECONSTRUCTION_UDF = _PKG_DIR / "sca" / "udfs" / "historical_reconstruction_udf.py"
SWE_RECONSTRUCTION_UDF = _PKG_DIR / "swe" / "udfs" / "swe_udf.py"


def run_reconstruction(cfg: DictConfig, eoconn: openeo.Connection, spatial_extent: dict) -> None:
    """Execute the full historical reconstruction pipeline."""

    exp = cfg.experiment
    recon = cfg.reconstruction

    temporal_extent = list(exp.temporal_extent)
    modis_temporal_extent = list(exp.modis_temporal_extent)
    agera_temporal_extent = list(exp.agera_temporal_extent)

    # ==============================
    # 1. SCF Masks & Conditional Probabilities
    # ==============================

    all_masks, labels_scf = compute_scf_masks(eoconn, cfg, spatial_extent, temporal_extent)
    cp, occurences = compute_conditional_probabilities(all_masks, labels_scf)

    # ==============================
    # 2. Load High-Resolution Data
    # ==============================

    hr_snow = snow_cover_fraction_cube(
        spatial_extent=spatial_extent,
        time_period=temporal_extent,
        c=eoconn,
        cfg=cfg,
    ).rename_labels(dimension="bands", target=["snow"])

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
    # 3. Historical Reconstruction via UDF
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
    sca = sca.rename_labels(dimension="bands", target=["sca"])

    # ==============================
    # 4. Downscale Climate Data
    # ==============================

    agera_downscaled, shortwave_rad_cube = load_climate_data(
        eoconn, cfg, spatial_extent, agera_temporal_extent, first_date
    )

    # ==============================
    # 5. Merge All & Compute SWE
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
    # 6. Execute Batch Job
    # ==============================

    job_options = OmegaConf.to_container(exp.job_options, resolve=True)
    swe.execute_batch(
        title=exp.title_prefix or "swe",
        job_options=job_options,
    )


# %%