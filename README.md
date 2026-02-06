# Trifecta Image ML Microservice

High‑throughput, low‑latency semantic segmentation service using ONNX Runtime on multi‑GPU.

## Model Layout (HF Repo)
The service downloads these files at startup from `HF_REPO_ID` (default: `AmithAdiraju1694/Video_Summary`).

Required:
- `segmentation.onnx`
- `segmentation_config.json`

Optional (future):
- `facemask.onnx`
- `textmask.onnx`

`segmentation_config.json` schema:
```json
{
  "input_size": 512,
  "mean": [0.485, 0.456, 0.406],
  "std": [0.229, 0.224, 0.225],
  "background_class_id": 0,
  "input_name": "pixel_values",
  "output_name": "logits"
}
```

## Export ONNX (dummy model)
This creates `segmentation.onnx` and `segmentation_config.json` from an open‑source model.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python tools/export_to_onnx.py --output-dir ./exported
```

Upload the two files to your HF repo (e.g., `AmithAdiraju1694/Video_Summary`).

## Run
```bash
docker build -t trifecta-seg .

docker run --gpus all -p 8080:8080 \
  -e HF_REPO_ID=AmithAdiraju1694/Video_Summary \
  -e GPU_IDS=0,1 \
  trifecta-seg
```

## API
### POST `/run_segmentation`
Accepts image bytes or WebDataset tar.

- Single image: `Content-Type: image/jpeg` (or png/webp) or `application/octet-stream`
- Gzip: add `Content-Encoding: gzip`
- WebDataset: `Content-Type: application/x-tar` (or `application/x-tar+gzip`)

Response (default `packbits`):
- `application/octet-stream`
- `X-Mask-Format: packbits`
- `X-Batch-Size: N`

`packbits` payload:
- For a single image: header `H,W,row_stride` (big‑endian uint16) + packed bytes.
- For a batch: each mask is length‑prefixed by a 4‑byte big‑endian size.

Node.js decode sketch:
```js
const h = buf.readUInt16BE(0);
const w = buf.readUInt16BE(2);
const rowStride = buf.readUInt16BE(4);
const packed = buf.subarray(6);
// unpack bits into HxW mask on client
```

To request PNG output for a single image:
```
POST /run_segmentation?format=png
```

### POST `/run_facemask`
Returns `204 No Content` (stub for future).

### POST `/run_textmask`
Returns `204 No Content` (stub for future).

## Performance Notes
- Set `INPUT_SIZE` to a smaller value (e.g., 384 or 256) for lower latency.
- Use `GPU_IDS=0,1` to enable multi‑GPU session pools.
- `MAX_CONCURRENCY_PER_GPU` caps in‑flight requests per GPU.
- Keep models in HF repo to load at startup and stay hot in GPU memory.

## Environment Variables
- `HF_REPO_ID` (default: `AmithAdiraju1694/Video_Summary`)
- `HF_REVISION`
- `HF_TOKEN`
- `HF_LOCAL_FILES_ONLY` (true/false)
- `SEG_ONNX_FILENAME` (default: `segmentation.onnx`)
- `SEG_CONFIG_FILENAME` (default: `segmentation_config.json`)
- `FACEMASK_ONNX_FILENAME` (default: `facemask.onnx`)
- `TEXTMASK_ONNX_FILENAME` (default: `textmask.onnx`)
- `GPU_IDS` (default: `0,1`)
- `USE_CUDA` (default: true)
- `MAX_BATCH_SIZE` (default: 4)
- `BATCH_TIMEOUT_MS` (default: 4)
- `MAX_CONCURRENCY_PER_GPU` (default: 2)
- `INPUT_SIZE` (default: 512)
- `OUTPUT_FORMAT` (default: packbits)
- `ALLOW_WEBDATASET` (default: true)
