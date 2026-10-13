"""Step 2: daily snow cover area (SCA) reconstruction.

Combines Sentinel-2 snow, MODIS snow cover fraction and historical conditional probabilities,
then fills clouds with the reconstruction UDF. Output band: `sca` (0 / 100 / 205).
"""

import openeo

from openeo_mountains_snow.config import (
    N_DAYS_TO_RECONSTRUCT, NEIGHBORHOOD_SIZE, SCA_RECONSTRUCTION_UDF, TEMPORAL_EXTENT,
)
from openeo_mountains_snow.snow_cover_area.conditional_probability import (
    compute_conditional_probability,
    compute_snow_cover_fraction_range_masks,
    create_modis_snow_cover_fraction_cube,
)


def snow_cover_area_cube(
    connection: openeo.Connection,
    high_resolution_snow: openeo.DataCube,
    spatial_extent: dict,
) -> openeo.DataCube:
    """Build the reconstructed snow cover area cube.

    `high_resolution_snow` is a 0/100/205 `snow` cube, normally
    `snow_map_from_snow_cover_fraction(...)` of the step-1 representative-pixels output.
    """
    all_masks, labels = compute_snow_cover_fraction_range_masks(
        connection, high_resolution_snow, TEMPORAL_EXTENT, spatial_extent
    )
    conditional_probability, _, occurrences = compute_conditional_probability(all_masks, labels)
    conditional_probability = conditional_probability.rename_labels(
        dimension="bands", target=[f"cp_{b}" for b in labels]
    )
    occurrences = occurrences.rename_labels(dimension="bands", target=[f"occ_{b}" for b in labels])

    modis_snow_cover_fraction = create_modis_snow_cover_fraction_cube(
        connection, TEMPORAL_EXTENT, spatial_extent
    )

    first_date = high_resolution_snow.metadata.temporal_dimension.extent[0]
    sca_inputs = (
        high_resolution_snow
        .merge_cubes(modis_snow_cover_fraction)
        .merge_cubes(conditional_probability.add_dimension(name="time", label=first_date, type="temporal"))
        .merge_cubes(occurrences.add_dimension(name="time", label=first_date, type="temporal"))
    )

    reconstruction_udf = openeo.UDF.from_file(
        str(SCA_RECONSTRUCTION_UDF),
        context={"n_days_to_reconstruct": N_DAYS_TO_RECONSTRUCT},
    )
    sca = sca_inputs.apply_neighborhood(
        process=reconstruction_udf,
        size=[
            {"dimension": "x", "value": NEIGHBORHOOD_SIZE, "unit": "px"},
            {"dimension": "y", "value": NEIGHBORHOOD_SIZE, "unit": "px"},
        ],
    )
    return sca
