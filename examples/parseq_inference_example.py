import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image

charset_train = " !#%'()*+,-./0123456789:<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz~₩「」ㄹ가각간갇갈감갑값갓갔강갖같개객갯갱거걱건걷걸검겁것겅게겠겨격겪견결겸겹경계고곡곤곧골곰곱곳공과곽관괄광괜괴굉교구국군굳굴굵굶굽궁권궐귀귓규균그극근글긁금급긋긍기긴길김깅깊까깍깎깐깔깜깝깥깨꺼꺾껍껏껑께꼬꼭꼼꼽꽂꽃꾸꾼꿀꿈뀌뀨끄끈끊끌끓끔끗끝끼낄낌나낙낚난날남납낫났낭낮내낸냄냉냐냥너넉넌널넓넘넣네넥넬넴넷녀녁년념녕노녹논놀놈농높놓뇌뇨뇽누눈눞뉴늄느늑는늘능늦늬니닌닐님닝다닥닦단닫달닭담답닷당닿대댁댐더덕던덜덤덥덧덩덮데델뎡도독돈돌돕돗동돼됐되된될됨됩두둑둔둘둠둥뒤뒷드득든듣들듬듭듯등디딘딜딩딪따딱딸땀땅때땜떠떡떤떨떴떻떼뗑또똑똔뚊뚜뚝뚠뚫뚱뛰뜨뜩뜰뜻띄띠띤띵라락란람랍랐랑랗래랙랜램랫략량러럭런럼럽럿렁렇레렉렌려력련렬렴렵령례로록론롤롬롭롯뢰료룡루룩룬류률륨륭르른를름릇릉리릭린릴림립릿링마막만많말맑맘맛망맞맡맣매맥맨머먹먼멀멋멍메멘며면멸명몇모목몬몰몸못몽묘무묵묶문묻물뭇뭉뭐뭔므미믹민믿밀밉밌및밑바박밖반받발밝밟밤밥방밭배백밸뱌버번벌범법벗베벤벨벼벽변별병보복볶본볼봄봅봇봉봤부북분불붐붓붕붙뷰브븎블비빅빈빌빔빗빙빚빛빠빨빵빼뺏뼈뽑뿌뿐뿜쀼쁘사삭산살삶삼삽샀상새색샌생샤샷서석섞선설섬섭섯성세센셈셋셔션셨소속손솔솜송솥쇄쇠쇼수숙순술숨숭숲쉬쉽슈스슨슬슴습승시식신실싫심십싱싶싸싹쌀쌍쌓써썩썰쎄쏘쏟쑤쓰쓴쓸씀씌씨씩씬씹씻아악안앉않알앎암압앗았앙앞앟애액앨앱앴앵야약얀얇양얕얘어억언얻얼엄업없엇었엉에엔엘엠여역연열염엽였영옆예옛오옥온올옮옷옹옾와완왔왕왜왠외왼요욕용우욱운울움웃웅워원월웨웬웹위윗유육윤율융으윾은을음읍응의이익읶인일읽잃임입잇있잉잊잎자작잔잘잠잡잣장잦재잼쟁쟤저적전절젊점접젓정젖제져조족존졸좁종좋좌죄죠주죽준줄중쥐즈즉즌즐즘증지직진질짐집짓징짜짝짤짧째쩌쩍쩔쪽쫓쬬찌찍찢차착찬찮찰참창찾채책처척천철첨첩첫청체쳐초촉촌총촬최추축춘출춤춧충취츠측츰층치칙친칠침칫칭카칸칼캄캐캠커컨컴컵컷케켓켜코콘콜콤콩쾌쿠퀴크큰클큼킄키킨킬킹타탁탄탈탑탕태택탤탱터턱턴털텅테텍텐텔템토톤톱통퇴투퉁튀튜트특튼튽틀틈티틱틸팀팅파판팔팡패팩팬팽퍼펌페펜펴편펼평폐포폭폰폴폼표푸풀품풍퓨프플픔피픽핀필핏핑하학한할함합항해핵핸햄햇했행향허헌험헤혀혁현혈협형혜호혹혼홀홈홍화확환활황회획횟횡효후훈훨훼휀휘휴흉흐흑흔흘흙흠흡흥흩희히힘"
charset_test = charset_train

max_len = 25
T = max_len + 1
img_h, img_w = 32, 128

bos_id = 1050
eos_id = 0
pad_id = 1051

from typing import List, Tuple
import torch

class Tokenizer:
    def __init__(self, charset: str):
        """
        charset: 실제 문자 집합 (예: '0123456789abcdefghijklmnopqrstuvwxyz...')
        토큰 구성:
          0: <pad>
          1: <bos>
          2: <eos>
          3~: charset 문자들
        """
        self.charset = list(charset)
        self.pad_id = 1051
        self.bos_id = 1050
        self.eos_id = 0

        # 전체 토큰 수
        self.num_tokens = 3 + len(self.charset)

        # id -> 문자 매핑
        self.id2char: List[str] = ['<eos>'] + self.charset + ['<bos>', '<pad>']

    @torch.no_grad()
    def decode(
        self,
        probs: torch.Tensor,
        raw: bool = False
    ) -> Tuple[
        Union[List[str], List[List[str]]],
        Union[List[torch.Tensor], torch.Tensor]
    ]:
        """
        PARSeq용 greedy 디코더.

        Args:
            probs: softmax가 적용된 확률 값
                   - shape: (N, T, C) 또는 (T, C)
                   - N=batch, T=sequence length, C=vocab size
            raw:  True  -> BOS/EOS/PAD 포함한 토큰 시퀀스를 그대로 돌려줌
                  False -> BOS/EOS/PAD 제거하고 최종 문자열만 돌려줌

        Returns:
            raw=False 일 때:
                texts: List[str]          # 배치마다 문자열
                char_probs: List[Tensor]  # 각 문자열의 글자별 confidence (길이 가변)

            raw=True 일 때:
                raw_tokens: List[List[str]]     # 각 step의 토큰 문자열 (B, E 등 포함)
                raw_conf:   Tensor (N, T)       # 각 step별 max probability
        """
        if probs.dim() == 2:
            # (T, C) -> (1, T, C)
            probs = probs.unsqueeze(0)

        # probs: (N, T, C)
        max_probs, token_ids = probs.max(dim=-1)  # 둘 다 (N, T)

        batch_size, seq_len = token_ids.shape
        device = probs.device
        dtype = probs.dtype

        if raw:
            # raw=True: BOS / EOS / PAD도 표시해서 모두 돌려줌
            # HF 데모처럼 [B], [E] 같은 토큰을 쓰도록 구현
            raw_tokens: List[List[str]] = []

            for b in range(batch_size):
                seq_ids = token_ids[b]        # (T,)
                tokens: List[str] = []

                for idx in seq_ids.tolist():
                    if idx == self.pad_id:
                        tokens.append("")          # PAD는 공백으로 표시(원본도 그냥 비워두는 식)
                    elif idx == self.bos_id:
                        tokens.append("[B]")       # BOS/blank 표시
                    elif idx == self.eos_id:
                        tokens.append("[E]")       # EOS 표시
                    else:
                        tokens.append(self.id2char[idx])

                raw_tokens.append(tokens)

            # max_probs: (N, T) 그대로 반환
            return raw_tokens, max_probs

        # raw=False: 실제 텍스트와 그 텍스트에 해당하는 글자별 confidence
        texts: List[str] = []
        all_char_probs: List[torch.Tensor] = []

        for b in range(batch_size):
            seq_ids = token_ids[b]     # (T,)
            seq_p  = max_probs[b]      # (T,)

            chars: List[str] = []
            char_ps: List[float] = []

            for idx, p in zip(seq_ids.tolist(), seq_p.tolist()):
                # EOS 만나면 거기서 종료 (EOS는 포함 X)
                if idx == self.eos_id:
                    break

                # BOS, PAD는 스킵 (문자로 만들지 않음)
                if idx == self.pad_id or idx == self.bos_id:
                    continue

                chars.append(self.id2char[idx])
                char_ps.append(p)

            texts.append("".join(chars))

            if char_ps:
                all_char_probs.append(torch.tensor(char_ps, device=device, dtype=dtype))
            else:
                # 아무 글자도 없으면 빈 텐서
                all_char_probs.append(torch.tensor([], device=device, dtype=dtype))

        return texts, all_char_probs


tokenizer = Tokenizer(charset_train)

# ============================================================
# 3. ONNX Inference 파이프라인 (encoder → AR → refine_iters번)
# ============================================================

# --- 세션 준비 ---
enc_sess = ort.InferenceSession("parseq_encoder.onnx",
                                providers=["CPUExecutionProvider"])
ar_sess  = ort.InferenceSession("parseq_ar_mem.onnx",
                                providers=["CPUExecutionProvider"])
rf_sess  = ort.InferenceSession("parseq_refine.onnx",
                                providers=["CPUExecutionProvider"])

enc_in  = enc_sess.get_inputs()[0].name     # "images"
enc_out = enc_sess.get_outputs()[0].name    # "memory"

ar_in   = ar_sess.get_inputs()[0].name      # "memory"
ar_out  = ar_sess.get_outputs()[0].name     # "logits"

rf_in_names = [i.name for i in rf_sess.get_inputs()]   # ["tgt_in","memory","tgt_padding_mask"]
rf_out = rf_sess.get_outputs()[0].name                 # "logits"


# # --- 전처리 ---
# def preprocess(img_path: str):
#     img = Image.open(img_path).convert("RGB")
#     img_t = img_transform(img)   # (3,H,W) torch tensor
#     img_np = img_t.unsqueeze(0).cpu().numpy().astype(np.float32)  # (1,3,H,W)
#     return img_np

def preprocess(img_path: str):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((128, 32))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # (C,H,W)
    arr = arr * 2.0 - 1.0               # [-1,1]
    return arr[None, ...]               # (1,3,H,W)


# --- AR만 한 번 돌리기 ---
def onnx_parseq_ar_only(img_path: str):
    img_np = preprocess(img_path)                               # (1,3,H,W)
    memory = enc_sess.run([enc_out], {enc_in: img_np})[0]       # (1,S,D)

    logits_ar = ar_sess.run(
        [ar_out],
        {ar_in: memory}
    )[0]                                                        # (1,T,V)

    return memory, logits_ar


# --- refine step ONNX 여러 번 적용 ---
def onnx_refine(memory: np.ndarray,
                logits: np.ndarray,
                refine_iters: int = 1):
    """
    memory: (1,S,D)
    logits: (1,T,V)  – AR 결과 또는 직전 refine 결과
    """
    batch_size, T_steps, _ = logits.shape
    assert batch_size == 1

    bos_col = np.full((1, 1), bos_id, dtype=np.int64)

    for _ in range(refine_iters):
        # 1) argmax → 토큰
        pred_ids = logits.argmax(axis=-1)          # (1,T)

        # 2) [bos] + preds[:, :-1]
        tgt_full = np.concatenate(
            [bos_col, pred_ids[:, :-1]],
            axis=1
        ).astype(np.int64)                         # (1,T)

        # 3) EOS 이후 padding mask
        eos_mask = (tgt_full == eos_id).astype(np.int32)
        cumsum = eos_mask.cumsum(axis=-1)
        tgt_padding_mask = (cumsum > 0)            # (1,T) bool

        # 4) refine 한 번
        logits = rf_sess.run(
            [rf_out],
            {
                rf_in_names[0]: tgt_full,          # "tgt_in"
                rf_in_names[1]: memory,            # "memory"
                rf_in_names[2]: tgt_padding_mask,  # "tgt_padding_mask"
            }
        )[0]                                       # (1,T,V)

    return logits  # 마지막 refine 결과


# --- logits → 텍스트 디코딩 ---
def decode_logits_to_text(logits_np: np.ndarray):
    logits_t = torch.from_numpy(logits_np)    # CPU tensor
    probs = F.softmax(logits_t, dim=-1)       # (1,T,V)
    preds, _ = tokenizer.decode(probs)
    return preds[0]


# --- 전체 OCR 파이프라인 ---
def ocr_onnx_full(img_path: str, refine_iters: int = 1):
    # 1) AR-only
    memory, logits_ar = onnx_parseq_ar_only(img_path)

    # 2) refine 적용
    if refine_iters > 0:
        logits_final = onnx_refine(memory, logits_ar, refine_iters=refine_iters)
    else:
        logits_final = logits_ar

    # 3) 디코딩
    text = decode_logits_to_text(logits_final)
    return text, logits_final


# ============================================================
# 4. 테스트
# ============================================================

test_img = "/content/test.png"   # 네 이미지 경로로 변경

text, logits_final = ocr_onnx_full(test_img, refine_iters=1)
print("최종 인식 결과:", text)