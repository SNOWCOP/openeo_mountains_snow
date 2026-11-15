"""Cloud, water and glacier masks (openEO process graphs)."""

import openeo
from omegaconf import DictConfig
from openeo.processes import any, if_


def omnicloudmask_classes(region, time_period, conn: openeo.Connection, cfg: DictConfig, dilation_iterations: int = 3):
    """OmniCloudMask on L1C via predict_onnx: 0 clear, 1 thick cloud, 2 thin cloud, 3 shadow, 255 nodata.

    Thick cloud is dilated like SnowFLAKES `create_omnicloudmask` (scipy binary_dilation, cross
    structure, `dilation_iterations` times == diamond of that radius).
    """
    rgn = conn.load_collection(
        cfg.sentinel2_l1c.collection,
        spatial_extent=region,
        temporal_extent=time_period,
        bands=["B04", "B03", "B8A"])
    rgn = rgn.process("convert_data_type", data=rgn, data_type="float32")
    predicted = rgn.process("predict_onnx", data=rgn, model=cfg.omnicloudmask.model_url)
    classes = predicted.reduce_dimension(dimension="bands", reducer=lambda x: x.array_element(0))

    r = dilation_iterations
    kernel = [[1 if abs(i) + abs(j) <= r else 0 for j in range(-r, r + 1)] for i in range(-r, r + 1)]
    thick_dilated = (classes.apply(lambda x: if_(x == 1, 1, 0))
                     .apply_kernel(kernel)
                     .apply(lambda x: x > 0))
    return classes.mask(thick_dilated, replacement=1)


def glacier_mask(region, conn: openeo.Connection, cfg: DictConfig):
    """True on glacier pixels from the hosted glacier STAC collection, None if not configured."""
    if cfg.glacier_mask.stac_url is None:
        return None
    glacier = conn.load_stac(cfg.glacier_mask.stac_url, spatial_extent=region, bands=[cfg.glacier_mask.band])
    return glacier.max_time().reduce_dimension(dimension="bands", reducer=lambda x: x[0] == 1)


def cloud_water_mask(region, time_period, conn: openeo.Connection, cfg:DictConfig):
    if cfg.omnicloudmask.model_url is not None:
        cloud_mask = omnicloudmask_classes(region, time_period, conn, cfg).apply(
            lambda x: (x == 1).or_(x == 2))
    else:
        scl = conn.load_collection(
            cfg.sentinel2_l2a.collection,
            spatial_extent=region,
            temporal_extent=time_period,

            bands=[cfg.sentinel2_l2a.scl_band])

        cloud_mask = scl.reduce_dimension(dimension="bands", reducer=lambda x: any([ x[0] == cloud_value for cloud_value in cfg.sentinel2_l2a.cloud_values]))

    water = conn.load_collection(
        cfg.water_mask.collection,
        spatial_extent=region,
        bands=[cfg.water_mask.band]).max_time()

    water_mask = water.reduce_dimension(dimension="bands", reducer=lambda x: any(
        [x[0] == water_value for water_value in cfg.water_mask.water_values]))

    return cloud_mask | water_mask
