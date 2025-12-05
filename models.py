import onnxruntime as ort
import numpy as np
import sys
from config import BOS_ID, EOS_ID, REFINE_ITERS

class YOLOEngine:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def run(self, x):
        return self.session.run(None, {self.input_name: x})

class ParseqEngine:
    def __init__(self, enc_path, ar_path, ref_path):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        print("Loading PARSeq modules...")

        # 1. Encoder
        self.enc_sess = ort.InferenceSession(enc_path, providers=providers)
        self.enc_in = self.enc_sess.get_inputs()[0].name

        # 2. AR Decoder
        self.ar_sess = ort.InferenceSession(ar_path, providers=providers)
        self.ar_in = self.ar_sess.get_inputs()[0].name

        # 3. Refine Decoder
        self.ref_sess = ort.InferenceSession(ref_path, providers=providers)
        self.rf_in_tgt = self.ref_sess.get_inputs()[0].name    # tgt_in
        self.rf_in_mem = self.ref_sess.get_inputs()[1].name    # memory
        self.rf_in_mask = self.ref_sess.get_inputs()[2].name   # tgt_padding_mask

        print("PARSeq modules loaded.")

    def predict_batch(self, images):
        """
        images: (Batch, 3, 32, 128) numpy array
        returns: logits (Batch, Seq_Len, Vocab_Size)
        """
        batch_size = images.shape[0]

        # ------------------------------------
        # Step 1: Encoder
        # ------------------------------------
        # Output: memory (Batch, S, D)
        memory = self.enc_sess.run(None, {self.enc_in: images})[0]

        # ------------------------------------
        # Step 2: AR Decoder
        # ------------------------------------
        # Output: logits (Batch, T, V)
        logits = self.ar_sess.run(None, {self.ar_in: memory})[0]

        # ------------------------------------
        # Step 3: Refine Loop (if needed)
        # ------------------------------------
        if REFINE_ITERS > 0:
            # 배치 처리를 위한 BOS 컬럼 생성 (Batch, 1)
            bos_col = np.full((batch_size, 1), BOS_ID, dtype=np.int64)

            for _ in range(REFINE_ITERS):
                # Argmax로 토큰 ID 추출
                pred_ids = np.argmax(logits, axis=-1) # (Batch, T)

                # Target 생성: [BOS] + preds[:, :-1]
                tgt_full = np.concatenate([bos_col, pred_ids[:, :-1]], axis=1).astype(np.int64)

                # Mask 생성: EOS 이후는 True
                # (Batch, T) boolean mask
                eos_mask = (tgt_full == EOS_ID).astype(np.int32)
                cumsum = np.cumsum(eos_mask, axis=1)
                tgt_padding_mask = (cumsum > 0).astype(bool) # ONNX Runtime expects bool for mask

                # Refine 실행
                logits = self.ref_sess.run(None, {
                    self.rf_in_tgt: tgt_full,
                    self.rf_in_mem: memory,
                    self.rf_in_mask: tgt_padding_mask
                })[0]

        return logits