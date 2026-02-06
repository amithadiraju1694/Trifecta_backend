from __future__ import annotations

import asyncio
import time
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response

from .config import AppConfig, load_config
from .batcher import AsyncBatcher
from .model_loader import LoadedModels, load_models
from .postprocess import background_mask_from_logits, packbits_batch, packbits_mask
from .preprocess import (
    ImageDecodeError,
    decode_image_bytes,
    load_webdataset_images,
    maybe_decompress,
    prepare_batch,
)


app = FastAPI(title="Trifecta Image ML Service", version="0.1.0")


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize config, models, and shared batching worker at server startup."""
    config = load_config()
    models = load_models(config)
    # Create one batcher shared by all incoming segmentation requests.
    seg_batcher = AsyncBatcher(
        pool=models.segmentation_pool,
        input_name=models.segmentation_config.input_name,
        output_name=models.segmentation_config.output_name,
        max_batch_size=config.max_batch_size,
        batch_timeout_ms=config.batch_timeout_ms,
    )
    seg_batcher.start()
    app.state.config = config
    app.state.models = models
    app.state.seg_batcher = seg_batcher


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Stop background workers cleanly when the process is shutting down."""
    batcher: AsyncBatcher = app.state.seg_batcher
    await batcher.stop()


def _load_images_from_request(
    body: bytes,
    content_type: Optional[str],
    content_encoding: Optional[str],
    allow_webdataset: bool,
) -> List[np.ndarray]:
    """Parse request bytes and return a list of RGB images as numpy arrays."""
    if content_type:
        ct = content_type.lower()
    else:
        ct = ""

    # WebDataset requests are TAR archives containing multiple image files.
    if allow_webdataset and ("application/x-tar" in ct or "application/x-tar+gzip" in ct):
        # Some clients also mark gzip in Content-Encoding; handle that first.
        if content_encoding and content_encoding.lower().strip() == "gzip":
            body = maybe_decompress(body, content_encoding)
        return load_webdataset_images(body)

    # Regular image request: optional decompression + decode one image.
    data = maybe_decompress(body, content_encoding)
    return [decode_image_bytes(data)]


@app.get("/healthz")
async def healthz() -> dict:
    """Return a simple readiness/liveness signal for health checks."""
    return {"ok": True}


@app.post("/run_segmentation")
async def run_segmentation(request: Request, format: Optional[str] = None) -> Response:
    """Run segmentation and return masks in packbits or PNG format."""
    # Track total server processing time for observability.
    t0 = time.perf_counter()
    config: AppConfig = app.state.config
    models: LoadedModels = app.state.models

    # Read the full request body once; it may be compressed/binary.
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty request body")

    content_type = request.headers.get("content-type")
    content_encoding = request.headers.get("content-encoding")

    try:
        # Decode image bytes off the event loop because decoding is CPU-bound.
        images = await asyncio.to_thread(
            _load_images_from_request,
            body=body,
            content_type=content_type,
            content_encoding=content_encoding,
            allow_webdataset=config.allow_webdataset,
        )
    except ImageDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Validate image count before preprocessing or inference.
    if not images:
        raise HTTPException(status_code=400, detail="No decodable images found")
    if len(images) > config.max_batch_size:
        raise HTTPException(status_code=413, detail="Batch size exceeds MAX_BATCH_SIZE")

    # Preprocess into normalized NCHW tensor expected by the model.
    seg_cfg = models.segmentation_config
    batch = await asyncio.to_thread(
        prepare_batch, images, seg_cfg.input_size, seg_cfg.mean, seg_cfg.std
    )

    batcher: AsyncBatcher = app.state.seg_batcher

    # Enqueue for dynamic batching so multiple requests can share one inference call.
    logits = await batcher.enqueue(batch)

    # Convert model logits into binary background masks.
    masks = background_mask_from_logits(logits, seg_cfg.background_class_id)

    out_format = (format or config.output_format or "packbits").lower()
    headers = {
        "X-Mask-Format": out_format,
        "X-Batch-Size": str(len(images)),
    }

    # Encode response payload in the selected output format.
    if out_format == "packbits":
        if masks.shape[0] == 1:
            payload = packbits_mask(masks[0])
        else:
            payload = packbits_batch(masks)
        headers["Content-Type"] = "application/octet-stream"
        resp = Response(content=payload, media_type="application/octet-stream", headers=headers)
    
    elif out_format == "png":
        if masks.shape[0] != 1:
            raise HTTPException(status_code=400, detail="PNG format supports batch size 1 only")
        # Import only in this branch to avoid unnecessary module load cost.
        import cv2

        # Convert {0,1} mask to 8-bit grayscale pixels before PNG encoding.
        mask_img = (masks[0] * 255).astype(np.uint8)
        ok, enc = cv2.imencode(".png", mask_img)
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to encode PNG")
        resp = Response(content=enc.tobytes(), media_type="image/png", headers=headers)
    
    else:
        raise HTTPException(status_code=400, detail="Unsupported format")

    # Include measured server compute time for client-side diagnostics.
    t1 = time.perf_counter()
    resp.headers["X-Server-Time-Ms"] = f"{(t1 - t0) * 1000:.2f}"
    return resp


@app.post("/run_facemask")
async def run_facemask() -> Response:
    """Placeholder endpoint for facemask model integration."""
    return Response(status_code=204)


@app.post("/run_textmask")
async def run_textmask() -> Response:
    """Placeholder endpoint for textmask model integration."""
    return Response(status_code=204)
