# OCR models

Place the local ONNX files in this directory before running inference:

- `yolov8s_best.onnx`: detects text regions.
- `parseq_encoder.onnx`: encodes cropped text images.
- `parseq_ar_mem.onnx`: autoregressive PARSeq decoder.
- `parseq_refine.onnx`: refines PARSeq predictions.

The model binaries are ignored by Git because of their size and distribution
terms. See `config.py` for the paths used by the inference service.