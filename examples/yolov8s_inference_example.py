import cv2
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt

# ==========================================
# 1. 설정 및 모델 로드
# ==========================================
onnx_model_path = "yolov8s_best.onnx"
img_path = "/content/자동차등록증.jpg" # 테스트 이미지 경로
conf_threshold = 0.25  # 시각화할 신뢰도 기준 (0.25 이상만 표시)

# 랜덤 색상 생성 (시각화용)
np.random.seed(42)
colors = np.random.uniform(0, 255, size=(len(1) + 10, 3)).astype(np.uint8)

# 세션 생성
session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]

# ==========================================
# 2. 이미지 전처리
# ==========================================
original_img = cv2.imread(img_path)
if original_img is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {img_path}")

# 원본 크기 저장 (나중에 좌표 복원할 때 필요)
orig_h, orig_w = original_img.shape[:2]

# 리사이즈 (960x960)
target_size = 960
img = cv2.resize(original_img, (target_size, target_size))

# 전처리: BGR -> RGB, Normalize, Transpose, Batch Add
img_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_input = img_input.astype(np.float32) / 255.0
img_input = img_input.transpose(2, 0, 1)
input_tensor = np.expand_dims(img_input, axis=0)

# ==========================================
# 3. 추론 실행
# ==========================================
outputs = session.run(output_names, {input_name: input_tensor})

# ==========================================
# 4. 후처리 및 시각화 (Scaling 포함)
# ==========================================
# [수정됨] outputs[0]은 (1, 300, 6)이므로 [0]을 한번 더 붙여서 (300, 6)으로 만듭니다.
predictions = np.squeeze(outputs[0])

# 신뢰도가 낮은 것과 Padding(0) 제거
valid_predictions = predictions[predictions[:, 4] > conf_threshold]

print(f"원본 크기: {orig_w}x{orig_h}")
print(f"탐지된 객체 수: {len(valid_predictions)}개")

# 좌표 복원을 위한 스케일 비율 계산
scale_x = orig_w / target_size
scale_y = orig_h / target_size

# 그리기용 이미지 복사
draw_img = original_img.copy()

for det in valid_predictions:
    # 960 크기 기준 좌표
    x1, y1, x2, y2 = det[0:4]
    score = det[4]
    class_id = int(det[5])

    # [핵심] 원본 이미지 크기로 좌표 변환 (Scaling)
    x1 = int(x1 * scale_x)
    y1 = int(y1 * scale_y)
    x2 = int(x2 * scale_x)
    y2 = int(y2 * scale_y)

    # 좌표가 이미지 밖으로 나가지 않게 클리핑
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(orig_w, x2), min(orig_h, y2)

    # 시각화 설정
    color = colors[class_id % len(colors)].tolist()
    label_text = class_names[class_id] if class_id < len(class_names) else f"ID {class_id}"
    label = f"{label_text}: {score:.2f}"

    # 박스 그리기
    cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, 2)

    # 라벨 그리기
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(draw_img, (x1, y1 - 20), (x1 + tw, y1), color, -1)
    cv2.putText(draw_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1)

# ==========================================
# 5. 결과 저장 및 출력
# ==========================================
output_path = "/content/result_visualized.jpg"
cv2.imwrite(output_path, draw_img)
print(f"결과가 저장되었습니다: {output_path}")

plt.figure(figsize=(12, 12))
plt.imshow(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()