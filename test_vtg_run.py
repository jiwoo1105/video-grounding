#!/usr/bin/env python3
"""
vtg_run.py 검증 테스트 — GPU 없이 실행 가능.

torch / transformers / qwen_vl_utils / av 를 가짜 모듈로 주입해서
실제 추론 코드 경로를 그대로 통과시킨 뒤, 각 모델이

  · 공식 모델 카드와 동일한 프롬프트를 쓰는가
  · 공식 모델 카드와 동일한 process_vision_info 호출 규약을 쓰는가
  · 픽셀 예산(patch 크기)이 맞는가
  · 출력 파싱이 정확한가

를 확인합니다.  실행:  python3 test_vtg_run.py
"""
import sys
import types
from pathlib import Path

# ==========================================================================
#  가짜 모듈 주입
# ==========================================================================
CALLS = {}          # 각 단계에서 실제로 넘어간 인자를 기록


def _fake_torch():
    t = types.ModuleType("torch")
    t.bfloat16 = "bfloat16"

    class _Cuda:
        @staticmethod
        def is_available(): return True
        @staticmethod
        def get_device_name(i=0): return "FakeGPU"
        @staticmethod
        def get_device_capability(i=0): return (8, 0)
        @staticmethod
        def memory_allocated(*a): return 0
        @staticmethod
        def max_memory_allocated(*a): return 0
        @staticmethod
        def empty_cache(): pass
        @staticmethod
        def reset_peak_memory_stats(): pass
        @staticmethod
        def get_arch_list(): return ["sm_80"]

    t.cuda = _Cuda()
    t.OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
    t.cuda.OutOfMemoryError = t.OutOfMemoryError

    # torch.inference_mode 는 데코레이터(@torch.inference_mode())로도,
    # 컨텍스트 매니저(with torch.inference_mode():)로도 쓰입니다.
    class _InferenceMode:
        def __call__(self, fn=None):
            return fn if fn is not None else self
        def __enter__(self): return self
        def __exit__(self, *a): return False
    t.inference_mode = _InferenceMode()
    return t


class FakeInputs(dict):
    """processor(...) 반환값 흉내. **inputs 언패킹, .input_ids.shape[1], .to() 지원."""
    def __init__(self, n=100):
        super().__init__(input_ids=types.SimpleNamespace(shape=(1, n)))

    @property
    def input_ids(self):
        return self["input_ids"]

    def to(self, device):
        return self


class FakeProcessor:
    def __init__(self, repo, **kw):
        self.repo = repo
        self.init_kwargs = kw
        self.tokenizer = types.SimpleNamespace(padding_side="right")

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        CALLS.setdefault("messages", []).append(messages)
        return "<CHAT_TEMPLATE>"

    def __call__(self, **kw):
        CALLS.setdefault("processor_call", []).append(kw)
        return FakeInputs()

    def batch_decode(self, ids, **kw):
        return [CALLS["fake_output"]]


class FakeModel:
    device = "cuda:0"

    def eval(self): return self

    def generate(self, **kw):
        CALLS.setdefault("generate", []).append(kw)
        return [[0] * 200]


def _fake_transformers():
    m = types.ModuleType("transformers")

    class AutoModelForImageTextToText:
        @staticmethod
        def from_pretrained(repo, **kw):
            if "torch_dtype" in kw:      # 최신 시그니처를 흉내 → dtype 만 허용
                raise TypeError("unexpected keyword argument 'torch_dtype'")
            CALLS.setdefault("model_load", []).append((repo, kw))
            return FakeModel()

    class AutoProcessor:
        @staticmethod
        def from_pretrained(repo, **kw):
            # Qwen2.5-VL 프로세서가 do_resize 를 안 받는 상황을 시뮬레이션
            if FAKE_STATE.get("reject_do_resize") and "do_resize" in kw:
                raise TypeError("unexpected keyword argument 'do_resize'")
            CALLS.setdefault("processor_load", []).append((repo, kw))
            return FakeProcessor(repo, **kw)

    m.AutoModelForImageTextToText = AutoModelForImageTextToText
    m.AutoProcessor = AutoProcessor
    return m


FAKE_STATE = {}


def _fake_qwen_vl_utils():
    m = types.ModuleType("qwen_vl_utils")

    def process_vision_info(messages, image_patch_size=None,
                            return_video_kwargs=False, return_video_metadata=False):
        CALLS.setdefault("pvi", []).append(dict(
            image_patch_size=image_patch_size,
            return_video_kwargs=return_video_kwargs,
            return_video_metadata=return_video_metadata,
        ))
        if return_video_kwargs and return_video_metadata:      # qwen3 경로
            return None, [("VIDEO", "META")], {"fps": [2.0]}
        if return_video_metadata:                              # TimeLens-7B 경로
            return None, [("VIDEO", "META")]
        if return_video_kwargs:                                # 표준 Qwen2.5 경로
            return None, ["VIDEO"], {"fps": [2.0]}
        return None, ["VIDEO"]

    m.process_vision_info = process_vision_info
    return m


def _fake_av():
    m = types.ModuleType("av")

    class _Ctx:
        def __enter__(self):
            s = types.SimpleNamespace(duration=3500, time_base=0.01, frames=70,
                                      average_rate=2.0)
            return types.SimpleNamespace(streams=types.SimpleNamespace(video=[s]),
                                         duration=35_000_000)
        def __exit__(self, *a): return False

    m.open = lambda *a, **k: _Ctx()
    return m


sys.modules["torch"] = _fake_torch()
sys.modules["transformers"] = _fake_transformers()
sys.modules["qwen_vl_utils"] = _fake_qwen_vl_utils()
sys.modules["av"] = _fake_av()

sys.path.insert(0, str(Path(__file__).parent))
import vtg_run as V  # noqa: E402


# ==========================================================================
#  공식 모델 카드에서 그대로 옮겨온 기대값
# ==========================================================================
OFFICIAL_PROMPT_HEAD = {
    # TencentARC/TimeLens-8B 모델 카드
    "timelens-8b":  "Please find the visual event described by the sentence 'Q', "
                    "determining its starting and ending times.",
    # TencentARC/TimeLens-7B 모델 카드 (앞 설명문이 추가로 붙음)
    "timelens-7b":  "You are given a video with multiple frames. The numbers before "
                    "each video frame indicate its sampling timestamp (in seconds). "
                    "Please find the visual event described by the sentence 'Q',",
    # MCG-NJU/TimeLens2-* 모델 카드
    "timelens2-8b": 'Given the query: "Q", return ALL time spans (in seconds) '
                    "where the query is relevant.",
    "timelens2-4b": 'Given the query: "Q", return ALL time spans (in seconds) '
                    "where the query is relevant.",
    # xiaomi-research/time-r1 demo.py
    "time-r1-7b":   'To accurately pinpoint the event "Q" in the video, '
                    "determine the precise time period of the event.",
}

# (image_patch_size, return_video_kwargs, return_video_metadata)
EXPECTED_PVI = {
    "timelens-8b":  (16, True,  True),
    "timelens2-8b": (16, True,  True),
    "timelens2-4b": (16, True,  True),
    "timelens-7b":  (None, False, True),    # 7B 카드는 metadata 만
    "time-r1-7b":   (None, True,  False),   # 표준 Qwen2.5 방식
    "qwen3-vl-8b":  (16, True,  True),
    "qwen2.5-vl-7b": (None, False, True),
}

FAKE_ANSWERS = {
    "timelens-8b":  "The event happens in 12.40 - 18.70 seconds.",
    "timelens-7b":  "The event happens in 3.10 - 9.55 seconds.",
    "timelens2-8b": "[[1.20, 4.50], [10.00, 13.75]]",
    "timelens2-4b": "[[0.00, 3.20]]",
    "time-r1-7b":   "<think>...</think><answer>1.05 to 7.62</answer>",
}
EXPECTED_SPANS = {
    "timelens-8b":  [[12.40, 18.70]],
    "timelens-7b":  [[3.10, 9.55]],
    "timelens2-8b": [[1.20, 4.50], [10.00, 13.75]],
    "timelens2-4b": [[0.00, 3.20]],
    "time-r1-7b":   [[1.05, 7.62]],
}

fails = []


def check(cond, msg):
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        fails.append(msg)


# ==========================================================================
print("=" * 78)
print(" 1. 레지스트리 정합성")
print("=" * 78)
for k, c in V.MODELS.items():
    ok = (c["style"] in V.STYLES
          and c["family"] in ("qwen2_5", "qwen3")
          and ((c["patch"] == 16) == (c["family"] == "qwen3")))
    check(ok, f"{k:<15} family={c['family']:<8} style={c['style']:<10} patch={c['patch']}")
for p, ms in V.PRESETS.items():
    check(all(m in V.MODELS for m in ms), f"프리셋 {p:<9} -> {', '.join(ms)}")

print()
print("=" * 78)
print(" 2. 각 모델 end-to-end 실행 (가짜 GPU)")
print("=" * 78)
for key in ["timelens-8b", "timelens-7b", "timelens2-8b", "timelens2-4b", "time-r1-7b"]:
    CALLS.clear()
    CALLS["fake_output"] = FAKE_ANSWERS[key]
    print(f"\n--- {key} ---")

    g = V.Grounder(key)
    r = g("/tmp/fake.mp4", "Q")

    # ① 프롬프트가 공식 카드와 일치하는가
    msg = CALLS["messages"][0]
    sent = msg[-1]["content"][-1]["text"]
    check(sent.strip().startswith(OFFICIAL_PROMPT_HEAD[key].split("Q")[0].strip()[:60]),
          "프롬프트 앞부분이 공식 카드와 일치")
    check(OFFICIAL_PROMPT_HEAD[key].replace("Q", "Q") in " ".join(sent.split()),
          "프롬프트 전문이 공식 카드와 일치")

    # ② process_vision_info 호출 규약
    pvi = CALLS["pvi"][0]
    exp = EXPECTED_PVI[key]
    got = (pvi["image_patch_size"], pvi["return_video_kwargs"], pvi["return_video_metadata"])
    check(got == exp, f"process_vision_info 규약 {got} == 공식 {exp}")

    # ③ 픽셀 예산 = 토큰수 x (patch*2)^2
    v = msg[-1]["content"][0]
    px = V.MODELS[key]["patch"] * 2
    check(v["total_pixels"] == V.MODELS[key]["tokens"] * px * px,
          f"total_pixels = {V.MODELS[key]['tokens']} x {px}x{px}")

    # ④ qwen3 계열만 video_metadata 를 프로세서에 넘겨야 함
    pk = CALLS["processor_call"][0]
    if V.MODELS[key]["family"] == "qwen3":
        check("video_metadata" in pk and pk["video_metadata"] == ["META"],
              "video_metadata 전달됨 (Qwen3-VL 필수)")
        check("fps" in pk, "video_kwargs(fps) 전달됨")
    else:
        check("video_metadata" not in pk, "video_metadata 미전달 (Qwen2.5-VL)")

    # ⑤ system 메시지는 Time-R1 만
    has_sys = msg[0]["role"] == "system"
    check(has_sys == (key == "time-r1-7b"), f"system 메시지 {'있음' if has_sys else '없음'}")

    # ⑥ 출력 파싱
    check(r["spans"] == EXPECTED_SPANS[key],
          f"파싱 {r['spans']} == {EXPECTED_SPANS[key]}")

    # ⑦ dtype 폴백 (torch_dtype 거부 -> dtype 사용)
    check(CALLS["model_load"][0][1].get("dtype") == "bfloat16",
          "dtype= 인자로 로드 (transformers 5.x 호환)")

    # ⑧ greedy 디코딩
    check(CALLS["generate"][0]["do_sample"] is False, "do_sample=False (재현성)")

print()
print("=" * 78)
print(" 3. do_resize 를 거부하는 프로세서에서의 폴백")
print("=" * 78)
FAKE_STATE["reject_do_resize"] = True
CALLS.clear(); CALLS["fake_output"] = FAKE_ANSWERS["timelens-7b"]
try:
    g = V.Grounder("timelens-7b")
    kw = CALLS["processor_load"][-1][1]
    check("do_resize" not in kw, f"do_resize 없이 재시도 성공 (남은 인자: {list(kw)})")
except Exception as e:
    check(False, f"폴백 실패: {e!r}")
FAKE_STATE.clear()

print()
print("=" * 78)
print(" 4. --total-tokens / --fps 오버라이드")
print("=" * 78)
CALLS.clear(); CALLS["fake_output"] = FAKE_ANSWERS["timelens-8b"]
g = V.Grounder("timelens-8b", total_tokens=8192, fps=1.0)
g("/tmp/fake.mp4", "Q")
v = CALLS["messages"][0][-1]["content"][0]
check(v["total_pixels"] == 8192 * 32 * 32, "total_tokens 오버라이드 반영")
check(v["fps"] == 1.0, "fps 오버라이드 반영")

print()
print("=" * 78)
print(" 5. 파서 단위 테스트")
print("=" * 78)
cases = [
    (V.parse_timelens,  "The event happens in 0.00 - 5.00 seconds.", [[0.0, 5.0]]),
    (V.parse_timelens,  "The event happens in 12.5 - 18.25 seconds", [[12.5, 18.25]]),
    (V.parse_timelens,  "cannot determine", []),
    (V.parse_time_r1,   "<answer>12.54 to 17.83</answer>", [[12.54, 17.83]]),
    (V.parse_time_r1,   "<think>2.10 to 5.00</think><answer>1.05 to 7.62</answer>", [[1.05, 7.62]]),
    (V.parse_time_r1,   "no answer", []),
    (V.parse_timelens2, "[[1.2, 4.5], [10.0, 13.75]]", [[1.2, 4.5], [10.0, 13.75]]),
    (V.parse_timelens2, "```json\n[[0.0, 3.2]]\n```", [[0.0, 3.2]]),
    (V.parse_timelens2, "The span is [5.5, 9.0].", [[5.5, 9.0]]),
    (V.parse_timelens2, "The event happens in 4.0 - 8.0 seconds", [[4.0, 8.0]]),
    (V.parse_timelens2, "none", []),
]
for fn, txt, want in cases:
    got = fn(txt)
    check(got == want, f"{fn.__name__:<16} {str(got):<28} == {want}")

print()
print("=" * 78)
print(f" 결과: {'전부 통과' if not fails else str(len(fails)) + '건 실패'}")
print("=" * 78)
for f in fails:
    print("  FAIL:", f)
sys.exit(1 if fails else 0)
