import json
from pathlib import Path

import geopandas
import hydra
import openeo
from omegaconf import DictConfig, OmegaConf
from openeo.internal.graph_building import PGNode
from openeo.rest.mlmodel import MlModel
from shapely.geometry import mapping
import typer



@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_openeo(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    c = openeo.connect("openeo-dev.vito.be").authenticate_oidc()

    job_options = {
        "executor-memory": "2G",
        "executor-memoryOverhead": "4G",
        "executor-cores": 1,
        "image-name": "openeo-docker-ci.artifactory.vgt.vito.be/openeo-yarn:20251014-4204"
        #"stac-version-experimental": "1.1"
    }

    aoi = json.load(open(Path(__file__).parent.parent.parent / "test" / "snowflakes_openeo_tests" / "andes_area1.geojson"))

    date = ["2023-02-22", "2023-02-23"]
    if ( cfg.corsa_job is not None):
        #inference_result = c.load_stac_from_job(cfg.corsa_job)
        inference_result = c.load_stac("https://openeo.vito.be/openeo/1.2/jobs/j-2510131112144492a4e614f7f3027882/results/ZGZhNjc4Y2I5YWIxN2Y2NWQ0ZjAyNWUzMGZhYzVlMGQ5MDExNjE3NmU0NGZkMTdkNzAzNDE5MzIyNzQ3Y2JiZEBlZ2kuZXU=/5183e3f996b282cdf6926e20f2fc3d1a?expires=1760979029")
    else:
        inference_result = c.datacube_from_process(
            process_id="corsa_compression",
            spatial_extent=aoi,
            namespace="vito",
            temporal_extent=date)

    #model_job = c.job("j-25101315033849049a4a67ea6d24b7ca")
    #links = model_job.get_results().get_metadata()['links']
    #ml_model_metadata_url = [link for link in links if 'ml_model_metadata.json' in link['href']][0]['href']
    #print(ml_model_metadata_url)
    model = c.load_ml_model(id="https://openeo.vito.be/openeo/1.2/jobs/j-25101315033849049a4a67ea6d24b7ca/results/items/ZGZhNjc4Y2I5YWIxN2Y2NWQ0ZjAyNWUzMGZhYzVlMGQ5MDExNjE3NmU0NGZkMTdkNzAzNDE5MzIyNzQ3Y2JiZEBlZ2kuZXU=/4bab4b92c835f56661fc3090c78285c0/ml_model_metadata.json?expires=1760979031")


    reducer = PGNode(
        process_id="predict_probabilities", data={"from_parameter": "data"}, model={"from_parameter": "context"}
    )
    inference_result.reduce_dimension(dimension="bands", reducer=reducer, context=model).execute_batch(title="snow inference", job_options=job_options)


if "__main__" == __name__:
    run_openeo()