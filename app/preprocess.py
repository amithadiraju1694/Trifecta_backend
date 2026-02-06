from __future__ import annotations

import gzip
import io
import tarfile
from typing import Iterable, List, Tuple

import cv2
import numpy as np


class ImageDecodeError(ValueError):
    """Raised when raw bytes cannot be decoded into a valid image."""
    pass


def maybe_decompress(data: bytes, content_encoding: str | None) -> bytes:
    """Decompress request bytes when `Content-Encoding` indicates gzip."""
    if not content_encoding:
        return data
    encoding = content_encoding.lower().strip()
    if encoding == "gzip":
        # gzip.decompress returns the original raw image/tar bytes.
        return gzip.decompress(data)
    return data


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode encoded image bytes into an RGB `numpy.ndarray`."""
    # Build a uint8 view over bytes for OpenCV decoding.
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageDecodeError("Failed to decode image bytes")
    # OpenCV decodes BGR by default; convert to RGB for model preprocessing.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_webdataset_images(data: bytes) -> List[np.ndarray]:
    """Read a TAR archive and decode every file that looks like an image."""
    images: List[np.ndarray] = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            fileobj = tar.extractfile(member)
            if fileobj is None:
                continue
            content = fileobj.read()
            try:
                images.append(decode_image_bytes(content))
            # Skip non-image files in the TAR instead of failing all inputs.
            except ImageDecodeError:
                continue
    return images


def resize_and_normalize(
    img: np.ndarray,
    input_size: int,
    mean: Iterable[float],
    std: Iterable[float],
) -> np.ndarray:
    """Resize an RGB image and normalize it into CHW float tensor format."""
    # Resize each image to the fixed square resolution expected by the model.
    resized = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    # Convert to [0,1] float values before channel-wise normalization.
    tensor = resized.astype(np.float32) / 255.0
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)
    tensor = (tensor - mean_arr) / std_arr
    # Rearrange from height-width-channel to channel-height-width.
    tensor = np.transpose(tensor, (2, 0, 1))
    return tensor


def prepare_batch(
    images: List[np.ndarray],
    input_size: int,
    mean: Iterable[float],
    std: Iterable[float],
) -> np.ndarray:
    """Convert a list of RGB images into one batched float32 tensor."""
    # Preprocess each image independently, then stack into shape [B, C, H, W].
    batch = [resize_and_normalize(img, input_size, mean, std) for img in images]
    return np.stack(batch, axis=0).astype(np.float32)
