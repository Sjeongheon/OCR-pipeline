import numpy as np
import cv2
from config import CHARSET_TRAIN, BOS_ID, EOS_ID, PAD_ID, YOLO_IMG_SIZE, PARSEQ_W, PARSEQ_H, LINE_TOLERANCE

# ==========================================
# 1. Tokenizer
# ==========================================
class Tokenizer:
    def __init__(self):
        self.charset = list(CHARSET_TRAIN)
        self.pad_id = PAD_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        # id -> 문자 매핑 리스트 생성 (순서 중요: EOS -> Charset -> BOS -> PAD)
        # 0: <eos>, 1~N: charset, N+1: <bos>, N+2: <pad> (실제 ID값과 리스트 인덱스는 다를 수 있음 주의)
        # 하지만 제공된 코드 로직상 id2char 리스트를 만들고 인덱싱하는 방식을 사용
        self.id2char = ['<eos>'] + self.charset + ['<bos>', '<pad>']

    def decode(self, probs):
        """
        probs: (Batch, Seq_Len, Class) - Logits or Softmax
        returns: List[str] texts, List[float] confidences
        """
        token_ids = np.argmax(probs, axis=-1) # (Batch, Seq)
        max_probs = np.max(probs, axis=-1)    # (Batch, Seq)

        # Softmax 근사 계산 (Logits 상태라면 확률값으로 변환)
        # (이미 max_probs를 뽑았으므로 여기서는 대략적인 신뢰도용)

        batch_size = token_ids.shape[0]
        texts = []
        confs = []

        for b in range(batch_size):
            chars = []
            char_confs = []
            for idx, score in zip(token_ids[b], max_probs[b]):
                if idx == self.eos_id: break
                if idx == self.pad_id or idx == self.bos_id: continue

                # 안전하게 인덱스 범위 확인
                if 0 <= idx < len(self.id2char):
                    chars.append(self.id2char[idx])
                    char_confs.append(score)

            texts.append("".join(chars))
            # 단순 평균 신뢰도 (Logits 값이면 정확한 확률은 아니지만 상대적 비교 가능)
            confs.append(float(np.mean(char_confs)) if char_confs else 0.0)

        return texts, confs

tokenizer = Tokenizer()

# ==========================================
# 2. Preprocessing
# ==========================================
def preprocess_yolo(img):
    img_resized = cv2.resize(img, (YOLO_IMG_SIZE, YOLO_IMG_SIZE))
    img_input = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_input = img_input.astype(np.float32) / 255.0
    img_input = img_input.transpose(2, 0, 1)
    return np.expand_dims(img_input, axis=0)

def preprocess_parseq_batch(crop_imgs):
    """(Batch, 3, 32, 128) 생성 및 Norm(-1~1)"""
    batch_input = []
    for img in crop_imgs:
        img = cv2.resize(img, (PARSEQ_W, PARSEQ_H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = img * 2.0 - 1.0 # [중요] PARSeq 정규화
        batch_input.append(img)
    return np.array(batch_input)

# ==========================================
# 3. Helpers
# ==========================================
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def sort_boxes(boxes):
    # y // 20 으로 줄 맞춤 -> x 정렬
    return sorted(boxes, key=lambda x: (int(x[1] // LINE_TOLERANCE), x[0]))

def get_crop_images(original_img, boxes):
    crops, meta_infos = [], []
    orig_h, orig_w = original_img.shape[:2]
    scale_x = orig_w / YOLO_IMG_SIZE
    scale_y = orig_h / YOLO_IMG_SIZE

    for box in boxes:
        x1, y1, x2, y2 = box[0:4]
        score = box[4]

        # Scaling
        x1, y1 = int(x1 * scale_x), int(y1 * scale_y)
        x2, y2 = int(x2 * scale_x), int(y2 * scale_y)

        # Clamping
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)

        if x2 <= x1 or y2 <= y1: continue

        crops.append(original_img[y1:y2, x1:x2])
        meta_infos.append({
            "yolo_score": float(score),
            "bbox": (x1, y1, x2, y2)
        })

    return crops, meta_infos


# ==========================================
# 4. Postprocessing
# ==========================================
def filter_ocr_items(
    texts,
    meta_infos,
    ocr_scores,
    ocr_min,
    yolo_min
):
    """
    YOLO / OCR 스코어 기반으로 텍스트 박스 필터링.
    bbox가 없는 항목은 merged_text를 위해 건너뜀.

    반환: [{
        "idx": int,
        "text": str,
        "yolo": float,
        "ocr": float,
        "bbox": (x1, y1, x2, y2),
    }, ...]
    """
    items = []

    for i, text in enumerate(texts):
        text_clean = text.strip()
        if not text_clean:
            continue

        yolo_score = meta_infos[i].get("yolo_score", 0.0)
        ocr_score = ocr_scores[i]

        if yolo_min is not None and yolo_score < yolo_min:
            continue
        if ocr_score < ocr_min:
            continue

        bbox = meta_infos[i].get("bbox")
        if bbox is None:
            # 줄 머징을 위해 bbox 없는 건 스킵
            continue

        x1, y1, x2, y2 = bbox

        items.append(
            {
                "idx": i,
                "text": text_clean,
                "yolo": float(yolo_score),
                "ocr": float(ocr_score),
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
            }
        )

    return items

def merge_items_to_text(items, line_threshold: float = 0.8) -> str:
    """
    YOLO 박스 기반 OCR 결과(items)를 줄 단위 텍스트로 합쳐서 반환.
    items: [{ "text": str, "bbox": (x1, y1, x2, y2), ... }, ...]
           이미 좌→우, 상→하 정렬되어 있다고 가정(sort_boxes 이후).
    line_threshold: 같은 줄로 볼 y 중심 오차 비율 (기본 0.8배).
    """
    if not items:
        return ""

    lines = []
    current_line = []
    current_center_y = None
    current_height = None

    for item in items:
        x1, y1, x2, y2 = item["bbox"]
        h = y2 - y1
        cy = (y1 + y2) / 2.0

        if not current_line:
            # 첫 박스 → 새 줄 시작
            current_line = [item["text"]]
            current_center_y = cy
            current_height = h
            continue

        # 같은 줄인지 판단 (y 중심 차이가 높이의 line_threshold 배 이하면 같은 줄로 봄)
        if abs(cy - current_center_y) <= current_height * line_threshold:
            current_line.append(item["text"])
            # 라인의 대표 y/height를 조금 갱신 (선택)
            current_center_y = (current_center_y + cy) / 2.0
            current_height = max(current_height, h)
        else:
            # 다음 줄로 넘어감
            lines.append(" ".join(current_line))
            current_line = [item["text"]]
            current_center_y = cy
            current_height = h

    if current_line:
        lines.append(" ".join(current_line))

    return "\n".join(lines)