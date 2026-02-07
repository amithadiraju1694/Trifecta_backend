from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from huggingface_hub import snapshot_download

from .config import AppConfig
from .model_config import SegmentationModelConfig, load_segmentation_config
from .ort_pool import OrtSessionPool, build_session

@dataclass
class LoadedModels:
    """Container for loaded model pools and the segmentation config."""
    segmentation_pool: OrtSessionPool
    segmentation_config: SegmentationModelConfig
    facemask_pool: Optional[OrtSessionPool]
    textmask_pool: Optional[OrtSessionPool]


@dataclass
class ModelPaths:
    """Resolved filesystem paths for model artifacts in a snapshot."""
    segmentation_onnx: str
    segmentation_config: str
    facemask_onnx: str
    textmask_onnx: str


def _download_repo(config: AppConfig) -> str:
    """Download model files from the configured Hugging Face repo snapshot."""
    patterns = [
        config.seg_onnx_filename,
        config.seg_config_filename,
        config.facemask_onnx_filename,
        config.textmask_onnx_filename,
    ]
    # Only fetch the required files to minimize download and startup time.
    snapshot_dir = snapshot_download(
        repo_id=config.hf_repo_id,
        revision=config.hf_revision,
        token=config.hf_token,
        allow_patterns=patterns,
        local_files_only=config.hf_local_files_only,
    )
    return snapshot_dir


def _build_paths(snapshot_dir: str, config: AppConfig) -> ModelPaths:
    """Build absolute paths to model assets inside the snapshot directory."""
    return ModelPaths(
        segmentation_onnx=os.path.join(snapshot_dir, config.seg_onnx_filename),
        segmentation_config=os.path.join(snapshot_dir, config.seg_config_filename),
        facemask_onnx=os.path.join(snapshot_dir, config.facemask_onnx_filename),
        textmask_onnx=os.path.join(snapshot_dir, config.textmask_onnx_filename),
    )


def _create_pool(onnx_path: str, config: AppConfig) -> OrtSessionPool:
    """Create an ONNX Runtime session pool on a single device."""
    session = build_session(onnx_path, device_id=config.gpu_id, use_cuda=config.use_cuda)
    return OrtSessionPool([session], max_concurrency_per_gpu=config.max_concurrency_per_gpu)


def _warmup(pool: OrtSessionPool, input_name: str, output_name: str, input_size: int) -> None:
    """Warm the pool with a dummy inference to trigger lazy initialization."""
    # Allocate a single dummy tensor sized to the model's expected input.
    dummy = np.zeros((1, 3, input_size, input_size), dtype=np.float32)
    pool.warmup_sync({input_name: dummy}, [output_name])



def load_models(config: AppConfig) -> LoadedModels:
    """Download, validate, and load model artifacts into runtime pools."""
    # Fetch snapshot first so paths resolve to local files.
    snapshot_dir = _download_repo(config)
    paths = _build_paths(snapshot_dir, config)

    # Fail fast on required assets to avoid partial initialization.
    if not os.path.exists(paths.segmentation_onnx):
        raise FileNotFoundError(f"Missing segmentation model at {paths.segmentation_onnx}")
    if not os.path.exists(paths.segmentation_config):
        raise FileNotFoundError(f"Missing segmentation config at {paths.segmentation_config}")

    seg_config = load_segmentation_config(paths.segmentation_config, config.input_size)
    seg_pool = _create_pool(paths.segmentation_onnx, config)
    # Warmup reduces first-request latency by initializing kernels and memory.
    _warmup(seg_pool, seg_config.input_name, seg_config.output_name, seg_config.input_size)

    facemask_pool = None
    if os.path.exists(paths.facemask_onnx):
        # Optional model; only load if present in the snapshot.
        facemask_pool = _create_pool(paths.facemask_onnx, config)
    textmask_pool = None
    if os.path.exists(paths.textmask_onnx):
        # Optional model; only load if present in the snapshot.
        textmask_pool = _create_pool(paths.textmask_onnx, config)

    return LoadedModels(
        segmentation_pool=seg_pool,
        segmentation_config=seg_config,
        facemask_pool=facemask_pool,
        textmask_pool=textmask_pool,
    )
