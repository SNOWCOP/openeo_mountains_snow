"""Step 2: daily snow cover area (SCA) reconstruction.

Combines Sentinel-2 snow, MODIS snow cover fraction and historical conditional probabilities,
then fills clouds with the reconstruction UDF. Output band: `sca` (0 / 100 / 205).
"""

import openeo

from openeo_mountains_snow.config import (
    MODIS_RESOLUTION, N_DAYS_TO_RECONSTRUCT, NEIGHBORHOOD_SIZE, SCA_NEIGHBORHOOD_OVERLAP, SCA_RECONSTRUCTION_UDF,
    TEMPORAL_EXTENT,
)
from openeo_mountains_snow.snow_cover_area.conditional_probability import (
    compute_conditional_probability,
    compute_snow_cover_fraction_range_masks,
    create_modis_snow_cover_fraction_cube,
)


def conditional_probability_inputs(
    connection: openeo.Connection,
    high_resolution_snow: openeo.DataCube,
    spatial_extent: dict,
    temporal_extent: list[str] = TEMPORAL_EXTENT,
) -> openeo.DataCube:
    """`cp_*` (conditional probability) and `occ_*` (occurrence) bands, without a time dimension."""
    all_masks, labels = compute_snow_cover_fraction_range_masks(
        connection, high_resolution_snow, temporal_extent, spatial_extent
    )
    conditional_probability, _, occurrences = compute_conditional_probability(all_masks, labels)
    return conditional_probability.rename_labels(
        dimension="bands", target=[f"cp_{b}" for b in labels]
    ).merge_cubes(occurrences.rename_labels(dimension="bands", target=[f"occ_{b}" for b in labels]))


def snow_cover_area_inputs(
    connection: openeo.Connection,
    high_resolution_snow: openeo.DataCube,
    spatial_extent: dict,
    temporal_extent: list[str] = TEMPORAL_EXTENT,
) -> openeo.DataCube:
    """Input cube of the reconstruction UDF: snow, MODIS scf, cp_* and occ_* (cp/occ at the first date).

    `high_resolution_snow` is a 0/100/205 `snow` cube, normally
    `snow_map_from_snow_cover_fraction(...)` of the step-1 representative-pixels output.
    """
    cp_occ = conditional_probability_inputs(connection, high_resolution_snow, spatial_extent, temporal_extent)
    modis_snow_cover_fraction = create_modis_snow_cover_fraction_cube(
        connection, temporal_extent, spatial_extent
    )

    first_date = f"{high_resolution_snow.metadata.temporal_dimension.extent[0][:10]}T00:00:00Z"
    return (
        high_resolution_snow
        .merge_cubes(modis_snow_cover_fraction)
        .merge_cubes(cp_occ.add_dimension(name="time", label=first_date, type="temporal"))
    )


def snow_cover_area_cube(
    connection: openeo.Connection,
    high_resolution_snow: openeo.DataCube,
    spatial_extent: dict,
    temporal_extent: list[str] = TEMPORAL_EXTENT,
) -> openeo.DataCube:
    """Build the reconstructed snow cover area cube."""
    sca_inputs = snow_cover_area_inputs(connection, high_resolution_snow, spatial_extent, temporal_extent)

    reconstruction_udf = openeo.UDF.from_file(
        str(SCA_RECONSTRUCTION_UDF),
        context={"n_days_to_reconstruct": N_DAYS_TO_RECONSTRUCT, "lr_pixel_size": MODIS_RESOLUTION},
    )
    sca = sca_inputs.apply_neighborhood(
        process=reconstruction_udf,
        size=[
            {"dimension": "x", "value": NEIGHBORHOOD_SIZE, "unit": "px"},
            {"dimension": "y", "value": NEIGHBORHOOD_SIZE, "unit": "px"},
        ],
        overlap=[
            {"dimension": "x", "value": SCA_NEIGHBORHOOD_OVERLAP, "unit": "px"},
            {"dimension": "y", "value": SCA_NEIGHBORHOOD_OVERLAP, "unit": "px"},
        ],
    )
    return sca
