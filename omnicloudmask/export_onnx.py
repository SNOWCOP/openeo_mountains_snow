# /// script
# requires-python = ">=3.10"
# dependencies = ["omnicloudmask>=1.7", "onnx", "onnxruntime"]
# ///
# %% [markdown]
# Export the OmniCloudMask v4 ensemble to ONNX plus an MLM STAC item for openEO `predict_onnx`
# (approach of openeo-community-examples PR #111). Run the cells top to bottom.
#
# Upload both files from `OUT_DIR` to public storage (same place as `MODEL_HREF`), then set
# `omnicloudmask.model_url` in conf/config.yaml to the public URL of the JSON item.
#
# Model contract: float32 (1, 3, H, W) with bands [red B04, green B03, NIR B8A], any reflectance scale,
# NaN or all-zero = nodata. Output float32 (1, 1, H, W): 0 clear, 1 thick cloud, 2 thin cloud,
# 3 cloud shadow, 255 nodata. Per-patch channel normalisation and ensemble-mean logits replicate
# `omnicloudmask.predict_from_array`; unlike it, predict_onnx tiles without overlap/blending.

# %% Parameters
from pathlib import Path

MODEL_HREF = "https://s3.waw3-1.cloudferro.com/project_dependencies/snowcop/omnicloudmask_v4.onnx"
OUT_DIR = Path("build")  # relative to the working directory (the file's folder in the interactive window)
PATCH_SIZE = 512  # fixed tile size; multiple of 32

# %% Imports and model wrapper
import json

import numpy as np
import onnx
import onnxruntime as ort
import torch
from omnicloudmask import predict_from_array
from omnicloudmask.download_models import get_models
from omnicloudmask.model_utils import load_model_from_weights

MODEL_VERSION = 4
NODATA_CLASS = 255


class OmniCloudMaskOnnx(torch.nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0)
        # omnicloudmask.model_utils.channel_norm: per band, mean/std over valid (non-nodata) pixels.
        w = (x != 0).to(x.dtype)
        n = w.sum(dim=(2, 3), keepdim=True).clamp(min=1)
        mean = (x * w).sum(dim=(2, 3), keepdim=True) / n
        std = ((((x - mean) * w) ** 2).sum(dim=(2, 3), keepdim=True) / n).sqrt()
        std = torch.where(std == 0, torch.ones_like(std), std)
        x = (x - mean) / std * w

        logits = torch.stack([model(x) for model in self.models]).mean(dim=0)
        classes = logits.argmax(dim=1, keepdim=True).to(x.dtype)
        valid = w.amax(dim=1, keepdim=True) > 0
        return torch.where(valid, classes, torch.full_like(classes, NODATA_CLASS))


def mlm_item(model_href: str, patch_size: int) -> dict:
    dim_order = ["batch", "bands", "height", "width"]
    return {
        "stac_version": "1.0.0",
        "stac_extensions": ["https://stac-extensions.github.io/ml-model/v1.0.0/schema.json"],
        "type": "Feature",
        "id": f"omnicloudmask_v{MODEL_VERSION}",
        "geometry": None,
        "bbox": None,
        "properties": {
            "datetime": None,
            "mlm:name": f"OmniCloudMask v{MODEL_VERSION} ensemble",
            "mlm:input": [{"name": "input", "data_type": "float32", "shape": [1, 3, patch_size, patch_size], "dim_order": dim_order}],
            "mlm:output": [{"name": "output", "data_type": "float32", "shape": [1, 1, patch_size, patch_size], "dim_order": dim_order}],
        },
        "links": [],
        "assets": {"model": {"href": model_href, "roles": ["mlm:model"]}},
        "collection": "ml-models",
    }

#%%

# %% Load models and export to ONNX
OUT_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device("cpu")
models = [
    load_model_from_weights(
        model_name=m["timm_model_name"],
        weights_path=m["Path"],
        model_library=m["model_library"],
        device=device,
    )
    for m in get_models(model_version=MODEL_VERSION)
]
wrapper = OmniCloudMaskOnnx(models).eval()

onnx_path = OUT_DIR / f"omnicloudmask_v{MODEL_VERSION}.onnx"
dummy = torch.rand(1, 3, PATCH_SIZE, PATCH_SIZE)
torch.onnx.export(wrapper, dummy, onnx_path, input_names=["input"], output_names=["output"], opset_version=18)
# Newer torch writes weights to a sidecar .onnx.data file; predict_onnx needs one self-contained file.
onnx.save(onnx.load(onnx_path), onnx_path, save_as_external_data=False)
onnx_path.with_suffix(".onnx.data").unlink(missing_ok=True)
print(f"Wrote {onnx_path}")

# %% Parity check: ONNX vs omnicloudmask on one patch without overlap
rng = np.random.default_rng(0)
sample = rng.uniform(0, 0.6, size=(3, PATCH_SIZE, PATCH_SIZE)).astype(np.float32)
sample[:, :50, :50] = 0
onnx_out = ort.InferenceSession(str(onnx_path)).run(None, {"input": sample[None]})[0][0]
ref = predict_from_array(sample, patch_size=PATCH_SIZE, patch_overlap=0, inference_device="cpu",
                         model_version=MODEL_VERSION, no_data_value=0)
valid = onnx_out != NODATA_CLASS
agreement = float((onnx_out[valid] == ref[valid]).mean())
print(f"ONNX vs omnicloudmask agreement on valid pixels: {agreement:.4f}")
assert agreement >= 0.99, "Parity check failed"

# %% Write MLM STAC item
item_path = OUT_DIR / f"omnicloudmask_v{MODEL_VERSION}.json"
item_path.write_text(json.dumps(mlm_item(MODEL_HREF, PATCH_SIZE), indent=2))
print(f"Wrote {item_path}")
