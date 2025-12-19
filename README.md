# Korean OCR Pipeline

An OCR inference pipeline that detects text regions with YOLOv8 and recognizes
the cropped regions with PARSeq. It can run locally for inspection or deploy as
a GPU-backed FastAPI endpoint on Modal.

## Pipeline

1. Resize and normalize the input image for YOLO.
2. Detect text bounding boxes and sort them in reading order.
3. Crop detected regions and batch them for PARSeq.
4. Decode and filter recognized text by detection and OCR confidence.
5. Return individual results and line-merged text.

## Requirements

- Python 3.9 or newer
- The ONNX model files listed below
- A CUDA-capable environment for GPU inference, or an ONNX Runtime setup that
  supports the desired execution provider

Install the Python dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Model files

Model weights are intentionally not tracked in Git because they are large
binary artifacts and may have separate redistribution terms. Place these files
under `models/` before running the project:

```text
models/
├── parseq_ar_mem.onnx
├── parseq_encoder.onnx
├── parseq_refine.onnx
└── yolov8s_best.onnx
```

If the weights may be redistributed, store them with Git LFS or attach a
versioned archive to a GitHub release and document its checksum.

## Local inference

Place a non-sensitive input image at `test_image.jpg`, then run:

```bash
python test.py
```

The script prints recognition results and opens a Matplotlib visualization.
The local test image is ignored by Git to prevent accidental publication of
personal or vehicle-identifying information.

## Modal deployment

Authenticate the Modal CLI, then deploy the endpoint defined in `main.py`:

```bash
modal setup
modal deploy main.py
```

The endpoint accepts an uploaded image in a `file` multipart field and returns
JSON containing per-box confidence values and merged text.

## Project layout

```text
.
├── config.py       # Character set, thresholds, and model paths
├── main.py         # Modal application and FastAPI endpoint
├── models.py       # ONNX Runtime inference engines
├── utils.py        # Preprocessing, decoding, and postprocessing
├── test.py         # Local end-to-end inference and visualization
└── examples/       # Standalone model inference references
```
