"""Client-side Sentinel-2 sun angle lookup (CDSE STAC), passed as context to the solar position UDF."""

import time

import pandas as pd
import requests
import shapely
from pyproj import Transformer


def aoi_shape(aoi):
    """Shapely geometry (EPSG:4326) from a shapely geometry or GeoJSON (Feature)Collection."""
    if isinstance(aoi, dict):
        features = aoi.get("features", [aoi])
        return shapely.unary_union([shapely.geometry.shape(f.get("geometry", f)) for f in features])
    return aoi


def s2_tile_sun_angles(aoi, time_period, collection="sentinel-2-l1c"):
    """Per date, ``[lon, lat, sun_zenith, sun_azimuth]`` at each S2 tile centre, from the CDSE STAC.

    S2 sun angles are tile means, valid at the tile centre: the solar position UDF
    recovers the overpass time there instead of at the (offset) chunk centre.
    """
    aoi = aoi_shape(aoi)
    url = "https://stac.dataspace.copernicus.eu/v1/search"
    tiles = {}
    for page in _stac_pages(url, collection, list(aoi.bounds), time_period):
        for item in page["features"]:
            props = item["properties"]
            asset = next(a for a in item["assets"].values() if "proj:bbox" in a)
            x0, y0, x1, y1 = asset["proj:bbox"]
            lon, lat = Transformer.from_crs(asset["proj:code"], "EPSG:4326", always_xy=True).transform(
                (x0 + x1) / 2, (y0 + y1) / 2)
            tiles.setdefault(props["datetime"][:10], []).append(
                [lon, lat, 90 - props["view:sun_elevation"], props["view:sun_azimuth"]])
    return tiles


def _stac_pages(url, collection, bbox, time_period, tries=5):
    """STAC search pages, one query per month: the CDSE replica cancels long queries
    (SerializationError) and rate-limits (429); both are retried with backoff."""
    start, end = pd.Timestamp(time_period[0]), pd.Timestamp(time_period[1])
    while start <= end:
        stop = min(start + pd.offsets.MonthEnd(0), end)
        body = {"collections": [collection], "bbox": bbox, "limit": 1000,
                "datetime": f"{start:%Y-%m-%d}T00:00:00Z/{stop:%Y-%m-%d}T23:59:59Z"}
        while body:
            for k in range(tries):
                response = requests.post(url, json=body, timeout=60)
                rate_limited = response.status_code == 429
                if response.ok or not (rate_limited or "SerializationError" in response.text) or k == tries - 1:
                    break
                time.sleep((10 if rate_limited else 1) * 2 ** k)
            response.raise_for_status()
            page = response.json()
            yield page
            body = next((link.get("body") for link in page.get("links", []) if link["rel"] == "next"), None)
        start = stop + pd.Timedelta(days=1)
