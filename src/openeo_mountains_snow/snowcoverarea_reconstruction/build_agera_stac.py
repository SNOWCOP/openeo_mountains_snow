
#%%
import openeo
import json
from pathlib import Path

from config import AGERA_TEMPORAL_EXTENT

BACKEND = "https://openeo.vito.be/"
eoconn = openeo.connect(BACKEND).authenticate_oidc()



path_to_geojson = Path(__file__).parent / "andes.json"
assert path_to_geojson.exists(), f"GeoJSON file not found at {path_to_geojson}"
geoms = json.loads(path_to_geojson.read_text())


agera = eoconn.load_stac(
    "stac.openeo.vito.be/collections/agera5_daily",
    temporal_extent=AGERA_TEMPORAL_EXTENT,
    bands=["temperature-mean", "dewpoint-temperature", "solar-radiation-flux"]
)

agera = agera.filter_spatial(geometries=geoms)


save_result_options = {
        "separate_asset_per_band": True,
    }
    
agera = agera.save_result(
    format="GTiff",
    options=save_result_options,
)

job = agera.create_job(title="AGERA5 test", description="Testing AGERA5 collection")


job_options = {
        "executor_memory": "4G",
        "driver_memory": "4G",
        "executor_memoryOverhead": "4G",
        "omit-derived-from-links": True,
        "stac-version-experimental": "1.1",
    }

job = agera.create_job(
    job_options=job_options,
    title="AGERA5  Job",
    )

job.start_and_wait()