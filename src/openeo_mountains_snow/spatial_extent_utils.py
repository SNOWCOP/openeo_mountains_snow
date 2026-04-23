"""
Spatial extent helpers.

The standard spatial extent format throughout this project is
a **bbox dict**: ``{west, south, east, north, crs}``.

This module provides:
- ``resolve_aoi(experiment_cfg)``  -- turn any config format into a bbox dict
- ``bbox_to_geometry(bbox)``       -- convert bbox to GeoJSON Polygon (for aggregate_spatial)
"""

import json
from pathlib import Path
from typing import Any, Dict

import shapely
import shapely.geometry
import shapely.ops
from pyproj import Transformer


def resolve_aoi(experiment_cfg) -> Dict[str, Any]:
    """
    Resolve an experiment config's AOI into a bbox dict.

    Handles every format used in experiment YAML files:
      - aoi: null                          -> default senales GeoJSON -> bbox
      - aoi: [west, south, east, north]    -> list with optional aoi_crs
      - aoi: {west, south, east, north}    -> bbox dict (returned as-is)
    """
    from omegaconf import DictConfig, OmegaConf

    aoi = experiment_cfg.aoi

    # null -> default GeoJSON file
    if aoi is None:
        geojson = json.loads((Path(__file__).parent / "senales_wgs84.geojson").read_text())
        return _geojson_to_bbox(geojson)

    # List [west, south, east, north]
    if isinstance(aoi, (list, tuple)) or (isinstance(aoi, DictConfig) and OmegaConf.is_list(aoi)):
        coords = list(aoi)
        crs = getattr(experiment_cfg, "aoi_crs", None) or "EPSG:4326"
        return {"west": coords[0], "south": coords[1], "east": coords[2], "north": coords[3], "crs": crs}

    # Dict-like (OmegaConf or plain)
    if isinstance(aoi, DictConfig):
        aoi = OmegaConf.to_container(aoi, resolve=True)
    if isinstance(aoi, dict):
        if "type" in aoi:  # GeoJSON
            return _geojson_to_bbox(aoi)
        if "west" in aoi:  # already a bbox
            aoi.setdefault("crs", "EPSG:4326")
            return aoi

    raise ValueError(f"Cannot interpret experiment aoi: {aoi!r}")


def bbox_to_geometry(bbox: Dict[str, Any]) -> dict:
    """
    Convert a bbox dict to a GeoJSON Polygon in EPSG:4326.

    Use this for openEO calls that require a geometry, e.g. ``aggregate_spatial``.
    """
    box = shapely.box(bbox["west"], bbox["south"], bbox["east"], bbox["north"])
    crs = bbox.get("crs", "EPSG:4326")
    if crs and crs != "EPSG:4326":
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        from shapely.ops import transform
        box = transform(transformer.transform, box)
    return shapely.geometry.mapping(box)


def _geojson_to_bbox(geojson: dict) -> Dict[str, Any]:
    """Extract bounding box from a GeoJSON geometry/Feature/FeatureCollection."""
    t = geojson.get("type")
    if t == "FeatureCollection":
        shapes = [shapely.geometry.shape(f["geometry"]) for f in geojson["features"]]
        shape = shapely.ops.unary_union(shapes)
    elif t == "Feature":
        shape = shapely.geometry.shape(geojson["geometry"])
    else:
        shape = shapely.geometry.shape(geojson)
    bounds = shape.bounds  # (minx, miny, maxx, maxy)
    return {"west": bounds[0], "south": bounds[1], "east": bounds[2], "north": bounds[3], "crs": "EPSG:4326"}


def bbox_to_wgs84(bbox: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a bbox dict to EPSG:4326. Returns a new bbox dict.

    If already in EPSG:4326, returns a copy. Use this for STAC sources
    that only support WGS84 spatial filtering (e.g. geopotential).
    """
    crs = bbox.get("crs", "EPSG:4326")
    if crs == "EPSG:4326":
        return {k: v for k, v in bbox.items()}
    geojson = bbox_to_geometry(bbox)  # already converts to WGS84
    shape = shapely.geometry.shape(geojson)
    bounds = shape.bounds
    return {"west": bounds[0], "south": bounds[1], "east": bounds[2], "north": bounds[3], "crs": "EPSG:4326"}
