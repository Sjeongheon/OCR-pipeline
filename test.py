import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# 우리가 만든 모듈들 임포트
from config import *
from models import YOLOEngine, ParseqEngine
from utils import (
    preprocess_yolo,
    preprocess_parseq_batch,
    sort_boxes,
    get_crop_images,
    tokenizer,
    softmax
)

# ==========================================
# 0. 테스트 설정
# ==========================================
TEST_IMG_PATH = "test_image.jpg"

# 로컬 테스트용 모델 경로 설정
LOCAL_YOLO_PATH = "./models/yolov8s_best.onnx"
LOCAL_ENC_PATH  = "./models/parseq_encoder.onnx"
LOCAL_AR_PATH   = "./models/parseq_ar_mem.onnx"
LOCAL_REF_PATH  = "./models/parseq_refine.onnx"

def draw_results(img, results, meta_infos, boxes):
    """결과 시각화 함수 (OpenCV + Matplotlib)"""
    draw_img = img.copy()

    # 원본 이미지 크기 및 스케일 비율 계산
    orig_h, orig_w = img.shape[:2]
    scale_x = orig_w / YOLO_IMG_SIZE
    scale_y = orig_h / YOLO_IMG_SIZE

    # 색상 생성
    np.random.seed(42)
    colors = np.random.uniform(0, 255, size=(len(boxes), 3)).astype(np.uint8)

    print("\n[상세 인식 결과]")
    print("-" * 60)
    print(f"{'No':<4} | {'Text':<20} | {'OCR Conf':<10} | {'YOLO Conf':<10}")
    print("-" * 60)

    for i, (text, box) in enumerate(zip(results, boxes)):
        # 1. 좌표 가져오기 (YOLO 960x960 기준 Float 값)
        x1, y1, x2, y2 = box[0:4]

        # 2. [핵심 수정] 원본 크기로 변환(Scaling) 및 정수형(Int) 변환
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        # 3. 이미지 범위 벗어나지 않게 클리핑 (선택 사항)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)

        # 색상 가져오기
        color = colors[i].tolist()

        # 콘솔 출력
        ocr_score = meta_infos[i]['ocr_score']
        yolo_score = meta_infos[i]['yolo_score']
        print(f"{i+1:<4} | {text:<20} | {ocr_score:.4f}     | {yolo_score:.4f}")

        # 4. 이미지에 그리기 (이제 좌표가 int라서 에러 안 남)
        cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, 2)

        # 텍스트 라벨 (번호 표시)
        label = f"{i+1}"
        cv2.putText(draw_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

    print("-" * 60)
    return draw_img

def main():
    # 1. 파일 존재 확인
    if not os.path.exists(TEST_IMG_PATH):
        print(f"오류: 테스트 이미지({TEST_IMG_PATH})가 없습니다.")
        return

    # 2. 모델 로드
    print(">>> 모델 로딩 시작...")
    try:
        yolo = YOLOEngine(LOCAL_YOLO_PATH)
        parseq = ParseqEngine(LOCAL_ENC_PATH, LOCAL_AR_PATH, LOCAL_REF_PATH)
        print(">>> 모델 로딩 완료!")
    except Exception as e:
        print(f"모델 로딩 실패: {e}")
        print("ONNX 파일들이 현재 폴더에 있는지 확인해주세요.")
        return

    # 3. 이미지 읽기
    original_img = cv2.imread(TEST_IMG_PATH)

    # 4. 전체 파이프라인 실행 및 시간 측정
    start_time = time.time()

    # --- Step A: YOLO ---
    print(">>> 1. YOLO 추론 중...")
    yolo_input = preprocess_yolo(original_img)
    yolo_out = yolo.run(yolo_input)

    preds = np.squeeze(yolo_out[0])
    valid_preds = preds[preds[:, 4] > CONF_THRESHOLD]
    sorted_preds = sort_boxes(valid_preds.tolist()) # 정렬

    print(f"    탐지된 박스 수: {len(sorted_preds)}개")

    if not sorted_preds:
        print("탐지된 텍스트가 없습니다.")
        return

    # --- Step B: Crop ---
    crops, meta_infos = get_crop_images(original_img, sorted_preds)

    # --- Step C: PARSeq (Batch) ---
    print(">>> 2. PARSeq(OCR) 배치 추론 중...")
    ocr_input = preprocess_parseq_batch(crops)
    logits = parseq.predict_batch(ocr_input) # (Batch, T, V)

    # --- Step D: Decode ---
    probs = softmax(logits)
    texts, ocr_scores = tokenizer.decode(probs)

    end_time = time.time()
    print(f">>> 전체 처리 시간: {end_time - start_time:.4f}초")

    # --- Step E: 결과 병합 및 메타 정보 업데이트 ---
    for i, score in enumerate(ocr_scores):
        meta_infos[i]['ocr_score'] = score

    # 5. 시각화 및 결과 출력
    result_img = draw_results(original_img, texts, meta_infos, sorted_preds)

    # 6. 이미지 보여주기 (Colab/Jupyter용)
    plt.figure(figsize=(12, 12))
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("OCR Result")
    plt.show()

if __name__ == "__main__":
    main()