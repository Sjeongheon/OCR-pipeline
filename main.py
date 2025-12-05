import modal
import cv2
import numpy as np
from fastapi import UploadFile, File

from config import *
from models import YOLOEngine, ParseqEngine
from utils import (
    preprocess_yolo,
    preprocess_parseq_batch,
    sort_boxes,
    get_crop_images,
    tokenizer,
    softmax,
    merge_items_to_text,
    filter_ocr_items
)

cuda_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .entrypoint([])
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "numpy",
        "opencv-python-headless",
        "onnxruntime-gpu==1.20.0",
        "fastapi[standard]",
        "python-multipart",
    )
    .add_local_dir(".", remote_path="/root")
)

app = modal.App("ocr-serverless-split")

@app.cls(gpu="T4", image=cuda_image, scaledown_window=600, min_containers=2)
class OCRService:
    @modal.enter()
    def load_models(self):
        self.parseq = ParseqEngine(PARSEQ_ENC_PATH, PARSEQ_AR_PATH, PARSEQ_REF_PATH)
        self.yolo = YOLOEngine(YOLO_PATH)

    @modal.fastapi_endpoint(method="POST")
    async def analyze(self, file: UploadFile = File(...)):
        # A. 파일 바이트 직접 읽기 (base64 X)
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if original_img is None:
            return {"status": "error", "message": "Invalid image"}

        # B. YOLO Detect
        yolo_in = preprocess_yolo(original_img)
        yolo_out = self.yolo.run(yolo_in)

        preds = np.squeeze(yolo_out[0])  # (300, 6)
        valid_preds = preds[preds[:, 4] > CONF_THRESHOLD]

        # C. Sort & Crop
        sorted_preds = sort_boxes(valid_preds.tolist())
        crops, meta_infos = get_crop_images(original_img, sorted_preds)

        if not crops:
            return {"status": "success", "count": 0, "results": []}

        # D. PARSeq Recognize
        ocr_in = preprocess_parseq_batch(crops)
        logits = self.parseq.predict_batch(ocr_in)

        # E. Decode
        probs = softmax(logits)
        texts, ocr_scores = tokenizer.decode(probs)

        # F. yolo_score / ocr_score 기반 필터링
        items = filter_ocr_items(
            texts=texts,
            meta_infos=meta_infos,
            ocr_scores=ocr_scores,
            ocr_min=0.5,
            yolo_min=0.1,
        )

        if not items:
            return {
                "status": "success",
                "count": 0,
                "results": [],
                "merged_text": "",
            }

        # G. 같은 줄 / 다음 줄 구분해서 merged_text 만들기
        merged_text = merge_items_to_text(items)

        # H. 박스별 결과도 같이 리턴 (필터링 된 것만)
        results = []
        for item in items:
            results.append(
                {
                    "id": item["idx"] + 1,
                    "text": item["text"],
                    "confidence": {
                        "yolo": item["yolo"],
                        "ocr": item["ocr"],
                    },
                }
            )

        return {
            "status": "success",
            "count": len(results),
            "results": results,
            "merged_text": merged_text
        }