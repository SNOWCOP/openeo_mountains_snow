import json
from pathlib import Path

import hydra
import openeo
from omegaconf import DictConfig
from openeo.api.process import Parameter
from openeo.rest.udp import build_process_dict

from openeo_mountains_snow.snow_cover_fraction import snow_cover_fraction_cube


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def generate_udp(cfg : DictConfig) -> None:

    spatial_extent = Parameter.spatial_extent()
    temporal_extent = Parameter.temporal_interval()

    c = openeo.connect(cfg.connection.endpoint).authenticate_oidc()

    scf_cube = snow_cover_fraction_cube(spatial_extent, temporal_extent, c, cfg)

    job_options = {
        "executor-memory": "3G",
        "python-memory": "disable",
        "executor-memoryOverhead": "4G"
    }

    returns = {
        "description": "A data cube with the newly computed values.\n\nAll dimensions stay the same, except for the dimensions specified in corresponding parameters. There are three cases how the dimensions can change:\n\n1. The source dimension is the target dimension:\n   - The (number of) dimensions remain unchanged as the source dimension is the target dimension.\n   - The source dimension properties name and type remain unchanged.\n   - The dimension labels, the reference system and the resolution are preserved only if the number of values in the source dimension is equal to the number of values computed by the process. Otherwise, all other dimension properties change as defined in the list below.\n2. The source dimension is not the target dimension. The target dimension exists with a single label only:\n   - The number of dimensions decreases by one as the source dimension is 'dropped' and the target dimension is filled with the processed data that originates from the source dimension.\n   - The target dimension properties name and type remain unchanged. All other dimension properties change as defined in the list below.\n3. The source dimension is not the target dimension and the latter does not exist:\n   - The number of dimensions remain unchanged, but the source dimension is replaced with the target dimension.\n   - The target dimension has the specified name and the type other. All other dimension properties are set as defined in the list below.\n\nUnless otherwise stated above, for the given (target) dimension the following applies:\n\n- the number of dimension labels is equal to the number of values computed by the process,\n- the dimension labels are incrementing integers starting from zero,\n- the resolution changes, and\n- the reference system is undefined.",
        "schema": {
            "type": "object",
            "subtype": "datacube"
        }
    }
    udp = build_process_dict(scf_cube, "sentinel2_snow_cover_fraction",
                             "Computes snow cover fraction at 10m resolution, with clouded areas set to nodata.",
                             description="Computes snow cover fraction at 10m resolution, with clouded areas set to nodata. The implementation is based on the SnowFLAKES algorithm by EURAC.",
                             links=[{"rel": "about", "href": "https://github.com/bare92/SnowFLAKES"}],
                             categories=["snow"],
                             parameters=[spatial_extent], returns=returns,
                             default_job_options=job_options)

    with open(Path(__file__).parent / "sentinel2_snow_cover_fraction.json", "w+") as f:
        json.dump(udp, f, indent=2)

if "__main__" == __name__:
    generate_udp()