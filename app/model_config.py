from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class SegmentationModelConfig:
    """Typed model metadata used for preprocessing and output decoding."""
    input_size: int
    mean: List[float]
    std: List[float]
    background_class_id: int
    input_name: str
    output_name: str


DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


def load_segmentation_config(path: str, fallback_input_size: int) -> SegmentationModelConfig:
    """Load model config JSON and fill missing fields with safe defaults."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Cast values explicitly so malformed JSON types fail early.
    input_size = int(data.get("input_size", fallback_input_size))
    mean = list(map(float, data.get("mean", DEFAULT_MEAN)))
    std = list(map(float, data.get("std", DEFAULT_STD)))
    background_class_id = int(data.get("background_class_id", 0))
    # ONNX tensor names default to common SegFormer export names.
    input_name = data.get("input_name", "pixel_values")
    output_name = data.get("output_name", "logits")

    return SegmentationModelConfig(
        input_size=input_size,
        mean=mean,
        std=std,
        background_class_id=background_class_id,
        input_name=input_name,
        output_name=output_name,
    )
