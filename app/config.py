from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import yaml


def _env(name: str, default: str | None = None) -> str | None:
    """Read an environment variable as a string with a fallback value."""
    # os.getenv returns None when a variable does not exist.
    val = os.getenv(name)
    if val is None:
        return default
    return val


def _env_bool(name: str, default: bool = False) -> bool:
    """Read an environment variable and convert common truthy strings to bool."""
    val = os.getenv(name)
    if val is None:
        return default
    # Normalize whitespace/case so values like " TRUE " still work.
    return val.strip().lower() in {"1", "true", "yes", "y"}


def _env_int(name: str, default: int) -> int:
    """Read an environment variable as an integer with safe fallback."""
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val)
    # Invalid numbers should not crash service startup.
    except ValueError:
        return default


def _env_csv_int(name: str, default: List[int]) -> List[int]:
    """Read a comma-separated integer list from an environment variable."""
    val = os.getenv(name)
    if not val:
        return default
    # Split by comma and ignore empty entries.
    items = [v.strip() for v in val.split(",") if v.strip()]
    out: List[int] = []
    for item in items:
        try:
            out.append(int(item))
        # Skip invalid entries instead of failing the whole list.
        except ValueError:
            continue
    return out if out else default


def _yaml_bool(value: Any, default: bool = False) -> bool:
    """Convert YAML scalar values to bool with tolerant parsing."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _yaml_opt_str(value: Any) -> str | None:
    """Convert YAML scalar values to optional string."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return str(value)


def _yaml_str(value: Any, default: str) -> str:
    """Convert YAML scalar values to string with default fallback."""
    parsed = _yaml_opt_str(value)
    return parsed if parsed is not None else default


def _load_hf_yaml_config() -> dict[str, Any]:
    """Load HF repository source configuration from YAML."""
    config_path = Path(__file__).with_name("hf_model_config.yaml")
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing HF YAML config at {config_path}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"HF YAML config at {config_path} must be a mapping")

    source = raw.get("source", raw)
    if not isinstance(source, dict):
        raise ValueError("HF YAML 'source' section must be a mapping")

    return {
        "hf_repo_id": source.get("repo_id", source.get("hf_repo_id")),
        "hf_revision": source.get("revision", source.get("hf_revision")),
        "hf_token": source.get("token", source.get("hf_token")),
        "hf_local_files_only": source.get(
            "local_files_only", source.get("hf_local_files_only", False)
        ),
    }


@dataclass(frozen=True)
class AppConfig:
    """Immutable runtime configuration loaded from YAML and environment variables."""
    hf_repo_id: str
    hf_revision: str | None
    hf_token: str | None
    hf_local_files_only: bool

    seg_onnx_filename: str
    seg_config_filename: str
    facemask_onnx_filename: str
    textmask_onnx_filename: str

    input_size: int
    max_batch_size: int
    batch_timeout_ms: int
    max_concurrency_per_gpu: int
    gpu_ids: List[int]
    use_cuda: bool

    output_format: str
    allow_webdataset: bool

    log_level: str

def load_config() -> AppConfig:
    """Build and return application configuration from YAML and environment variables."""
    hf_yaml = _load_hf_yaml_config()

    # Model artifact source configuration from YAML.
    hf_repo_id = _yaml_str(hf_yaml.get("hf_repo_id"), "AmithAdiraju1694/Video_Summary")
    hf_revision = _yaml_opt_str(hf_yaml.get("hf_revision"))
    hf_token = _yaml_opt_str(hf_yaml.get("hf_token"))
    hf_local_files_only = _yaml_bool(hf_yaml.get("hf_local_files_only"), False)

    # File names expected inside the model repository.
    seg_onnx_filename = _env("SEG_ONNX_FILENAME", "segmentation.onnx")
    seg_config_filename = _env("SEG_CONFIG_FILENAME", "segmentation_config.json")
    facemask_onnx_filename = _env("FACEMASK_ONNX_FILENAME", "facemask.onnx")
    textmask_onnx_filename = _env("TEXTMASK_ONNX_FILENAME", "textmask.onnx")

    # Inference and batching parameters.
    input_size = _env_int("INPUT_SIZE", 512)
    max_batch_size = _env_int("MAX_BATCH_SIZE", 4)
    batch_timeout_ms = _env_int("BATCH_TIMEOUT_MS", 4)
    max_concurrency_per_gpu = _env_int("MAX_CONCURRENCY_PER_GPU", 2)
    gpu_ids = _env_csv_int("GPU_IDS", [0, 1])
    use_cuda = _env_bool("USE_CUDA", True)

    output_format = (_env("OUTPUT_FORMAT", "packbits") or "packbits").lower()
    allow_webdataset = _env_bool("ALLOW_WEBDATASET", True)

    log_level = _env("LOG_LEVEL", "info")

    # Return one strongly-typed object used throughout the service.
    return AppConfig(
        hf_repo_id=hf_repo_id,
        hf_revision=hf_revision,
        hf_token=hf_token,
        hf_local_files_only=hf_local_files_only,
        seg_onnx_filename=seg_onnx_filename,
        seg_config_filename=seg_config_filename,
        facemask_onnx_filename=facemask_onnx_filename,
        textmask_onnx_filename=textmask_onnx_filename,
        input_size=input_size,
        max_batch_size=max_batch_size,
        batch_timeout_ms=batch_timeout_ms,
        max_concurrency_per_gpu=max_concurrency_per_gpu,
        gpu_ids=gpu_ids,
        use_cuda=use_cuda,
        output_format=output_format,
        allow_webdataset=allow_webdataset,
        log_level=log_level,
    )
