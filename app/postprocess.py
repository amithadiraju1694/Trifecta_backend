from __future__ import annotations

import struct
from typing import List

import numpy as np


def background_mask_from_logits(logits: np.ndarray, background_class_id: int) -> np.ndarray:
    """Return a binary background mask from per-pixel class logits.

    This selects the argmax class per pixel and marks pixels equal to the
    provided background class id as 1; all others are 0.
    """
    # Argmax over class dimension to get per-pixel predicted class ids.
    # logits: [B, C, H, W]
    pred = np.argmax(logits, axis=1)
    # Compare to background id and store as uint8 for compactness.
    mask = (pred == background_class_id).astype(np.uint8)
    return mask

def packbits_mask(mask: np.ndarray) -> bytes:
    """Pack a single 2D binary mask into a compact bytes payload.

    The payload layout is:
    - 3 big-endian uint16 header: (height, width, row_stride_bytes)
    - followed by bit-packed mask rows in big-endian bit order.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be HxW")
    h, w = mask.shape
    # Number of packed bytes needed to store one row of width `w`.
    row_stride = (w + 7) // 8
    # Bit-pack each row to minimize size; "big" matches the header convention.
    packed = np.packbits(mask, axis=1, bitorder="big")
    header = struct.pack(">HHH", h, w, row_stride
    )  # big-endian uint16s
    return header + packed.tobytes()


def packbits_batch(masks: np.ndarray) -> bytes:
    """Pack a batch of masks into a length-prefixed byte stream.

    Each mask payload is produced by packbits_mask and is preceded by a
    big-endian uint32 byte length, allowing efficient concatenation/decoding.
    """
    # masks: [B, H, W]
    if masks.ndim != 3:
        raise ValueError("masks must be BxHxW")
    chunks: List[bytes] = []
    for idx in range(masks.shape[0]):
        # Prefix each packed payload with its length for framing.
        payload = packbits_mask(masks[idx])
        chunks.append(struct.pack(">I", len(payload)))
        chunks.append(payload)
    return b"".join(chunks)
