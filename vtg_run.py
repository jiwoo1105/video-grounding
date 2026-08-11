#!/usr/bin/env python3
"""
Video Temporal Grounding 통합 추론 스크립트

세 계열의 grounding 모델을 하나의 CLI로 돌립니다. 각 모델은
프롬프트 형식 / 출력 형식 / processor 호출 규약이 전부 다르므로
이 스크립트가 그 차이를 흡수합니다.

  Time-R1   (Xiaomi+RUC, NeurIPS'25)  Qwen2.5-VL   <answer>1.05 to 7.62</answer>
  TimeLens  (TencentARC, CVPR 2026)   Qwen2.5/3-VL "The event happens in 1.0 - 7.6 seconds"
  TimeLens2 (MCG-NJU, arXiv 2026-07)  Qwen3-VL     [[1.2, 4.5], [10.0, 13.7]]

사용 예:
    python3 vtg_run.py --list
    python3 vtg_run.py --model timelens-8b --video a.mp4 --query "a person opens the door"
    python3 vtg_run.py --model best --video a.mp4 --query-file queries.txt --plot
    python3 vtg_run.py --model ablation --video a.mp4 --query-file queries.txt --plot
"""
import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

import torch

# --------------------------------------------------------------------------
# 프롬프트 — 각 모델이 학습된 원문 그대로. 절대 섞어 쓰면 안 됩니다.
# --------------------------------------------------------------------------

# 출처: xiaomi-research/time-r1  demo.py  PROMPT_TEMPLATE
P_TIME_R1 = """
To accurately pinpoint the event "{}" in the video, determine the precise time period of the event.

Output your thought process within the <think> </think> tags, including analysis with either specific time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.

Then, provide the start and end times (in seconds, precise to two decimal places) in the format "start time to end time" within the <answer> </answer> tags. For example: "12.54 to 17.83".
"""

# 출처: TencentARC/TimeLens-8B 모델 카드  GROUNDER_PROMPT
P_TIMELENS8 = (
    "Please find the visual event described by the sentence '{}', "
    "determining its starting and ending times. The format should be: "
    "'The event happens in <start time> - <end time> seconds'."
)

# 출처: TencentARC/TimeLens-7B 모델 카드  GROUNDER_PROMPT
# ★ 8B와 다릅니다. 앞에 "프레임 앞의 숫자가 촬영 시각"이라는 설명이 붙습니다.
#   7B는 Qwen2.5-VL 기반이라 타임스탬프를 수동으로 끼워 넣고, 그 사실을 알려줘야 합니다.
#   8B는 Qwen3-VL이 타임스탬프 삽입을 내장하고 있어 이 설명이 필요 없습니다.
P_TIMELENS7 = (
    "You are given a video with multiple frames. The numbers before each video frame "
    "indicate its sampling timestamp (in seconds). "
    "Please find the visual event described by the sentence '{}', "
    "determining its starting and ending times. The format should be: "
    "'The event happens in <start time> - <end time> seconds'."
)

# 출처: MCG-NJU/TimeLens2-4B 모델 카드
P_TIMELENS2 = (
    'Given the query: "{}", return ALL time spans (in seconds) where the query is relevant.\n'
    "Output format MUST be a JSON array of [start, end] pairs.\n"
)


# --------------------------------------------------------------------------
# 출력 파서
# --------------------------------------------------------------------------
def _pairs(text, sep):
    return re.findall(rf"(\d+\.?\d*)\s*(?:{sep})\s*(\d+\.?\d*)", text)


def parse_time_r1(text):
    """<answer>12.54 to 17.83</answer> -> [[12.54, 17.83]]"""
    m = re.search(r"<answer>(.*?)</answer>", text, re.S)
    scope = m.group(1) if m else text
    hits = _pairs(scope, "to|and") or _pairs(text, "to|and")
    return [[float(hits[-1][0]), float(hits[-1][1])]] if hits else []


def parse_timelens(text):
    """'The event happens in 0.00 - 5.00 seconds' -> [[0.0, 5.0]]"""
    hits = _pairs(text, r"-|to")
    return [[float(hits[-1][0]), float(hits[-1][1])]] if hits else []


def parse_timelens2(text):
    """'[[1.2, 4.5], [10.0, 13.7]]' -> 그대로. 다중 구간 지원."""
    for m in re.finditer(r"\[\s*\[.*?\]\s*\]", text, re.S):
        try:
            arr = json.loads(m.group(0))
            out = [[float(a), float(b)] for a, b in arr if float(b) >= float(a)]
            if out:
                return out
        except (ValueError, TypeError):
            continue
    m = re.search(r"\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]", text)
    if m:
        return [[float(m.group(1)), float(m.group(2))]]
    return parse_timelens(text)          # 형식을 벗어난 경우 최후의 보루


STYLES = {
    "time_r1":    dict(prompt=P_TIME_R1,    parse=parse_time_r1,   system=True),
    "timelens8":  dict(prompt=P_TIMELENS8,  parse=parse_timelens,  system=False),
    "timelens7":  dict(prompt=P_TIMELENS7,  parse=parse_timelens,  system=False),
    "timelens2":  dict(prompt=P_TIMELENS2,  parse=parse_timelens2, system=False),
}

# --------------------------------------------------------------------------
# 모델 레지스트리
#
# family : processor 호출 규약. qwen2_5 = patch14(토큰 28x28), qwen3 = patch16(32x32)
# tokens : 영상 전체에 배정할 비주얼 토큰 예산. 각 공식 설정값을 기본으로.
# --------------------------------------------------------------------------
MODELS = {
    # ---- TimeLens2 (MCG-NJU, arXiv 2026-07) — 성능 최상 + 다중 구간 -------
    "timelens2-8b": dict(
        repo="MCG-NJU/TimeLens2-8B", family="qwen3", style="timelens2",
        patch=16, tokens=16384, backbone="Qwen3-VL-8B-Instruct",
        note="7벤치 평균 mIoU 48.0. 백본 대비 +18.1  ★성능 1위",
    ),
    "timelens2-4b": dict(
        repo="MCG-NJU/TimeLens2-4B", family="qwen3", style="timelens2",
        patch=16, tokens=32768, backbone="Qwen3-VL-4B-Instruct",
        note="평균 mIoU 47.7 — 8B와 0.3 차이. ★가성비 1위",
    ),
    # ---- TimeLens (TencentARC, CVPR 2026) — 피어리뷰 완료 ---------------
    "timelens-8b": dict(
        repo="TencentARC/TimeLens-8B", family="qwen3", style="timelens8",
        patch=16, tokens=14336, backbone="Qwen3-VL-8B-Instruct",
        note="CVPR'26 게재. Charades mIoU 55.2  ★인용 안전",
    ),
    "timelens-7b": dict(
        repo="TencentARC/TimeLens-7B", family="qwen2_5", style="timelens7",
        patch=14, tokens=14336, backbone="Qwen2.5-VL-7B-Instruct",
        note="CVPR'26 게재. Charades mIoU 48.8", trust_remote_code=True,
    ),
    # ---- Time-R1 (Xiaomi+RUC, NeurIPS'25) — 피어리뷰 완료 ----------------
    "time-r1-7b": dict(
        repo="Boshenxx/Time-R1-7B", family="qwen2_5", style="time_r1",
        patch=14, tokens=3584, backbone="Qwen2.5-VL-7B-Instruct",
        note="NeurIPS'25. <think> 추론 과정을 출력",
    ),
    "time-r1-3b": dict(
        repo="Boshenxx/Time-R1-3B", family="qwen2_5", style="time_r1",
        patch=14, tokens=3584, backbone="Qwen2.5-VL-3B-Instruct",
        note="경량",
    ),
    # ---- 백본 원본 (학습 전) — 대조군 -----------------------------------
    "qwen3-vl-8b": dict(
        repo="Qwen/Qwen3-VL-8B-Instruct", family="qwen3", style="timelens8",
        patch=16, tokens=14336, backbone="(자기 자신)",
        note="학습 전 원본. Charades-TL R1@0.5 53.4 / mIoU 48.3",
    ),
    "qwen2.5-vl-7b": dict(
        repo="Qwen/Qwen2.5-VL-7B-Instruct", family="qwen2_5", style="timelens7",
        patch=14, tokens=14336, backbone="(자기 자신)",
        note="학습 전 원본. Charades-TL R1@0.5 37.8 / mIoU 39.3",
    ),
}

# 자주 쓰는 조합
PRESETS = {
    # ★ 메인 4종 — 전부 공식 모델 카드에 추론 코드가 공개돼 있는 것들
    "main":     ["timelens-8b", "timelens-7b", "timelens2-8b", "timelens2-4b"],
    "best":     ["timelens2-8b"],                                  # 성능 최상
    "light":    ["timelens2-4b"],                                  # 가성비 / 긴 영상
    "safe":     ["timelens-8b"],                                   # CVPR 게재본
    "cvpr":     ["timelens-7b", "timelens-8b"],                    # CVPR 논문 2종
    "ablation": ["qwen3-vl-8b", "timelens-8b"],                    # 학습 전 vs 후
    "compare":  ["timelens-8b", "timelens2-8b", "time-r1-7b"],     # 세 프로젝트 비교
}


def preflight():
    """실행 환경이 맞는지 먼저 확인하고, 아니면 무엇을 해야 하는지 알려줍니다."""
    problems = []
    if not torch.cuda.is_available():
        problems.append(
            "GPU(CUDA)를 찾을 수 없습니다.\n"
            "    이 스크립트는 **엘리스 클라우드 GPU 인스턴스에서** 실행하는 것입니다.\n"
            "    맥/윈도우 로컬에서는 모델을 돌릴 수 없습니다.\n\n"
            "    로컬에서 확인할 수 있는 건 코드 검증뿐입니다:\n"
            "        python3 test_vtg_run.py\n"
            "        python3 test_e2e.py")
    for mod, why in (("transformers", "모델 로딩"), ("qwen_vl_utils", "영상 전처리"),
                     ("av", "영상 길이 측정")):
        try:
            __import__(mod)
        except ImportError:
            problems.append(f"'{mod}' 가 설치돼 있지 않습니다 ({why}).")

    if problems:
        print("\n" + "=" * 70)
        print(" 실행할 수 없습니다")
        print("=" * 70)
        for p in problems:
            print(f"\n  - {p}")
        print("\n" + "=" * 70)
        print(" 엘리스 인스턴스에서:")
        print("     bash setup_env.sh")
        print("     source ~/vtg-env/bin/activate")
        print("=" * 70)
        sys.exit(1)


def video_duration(path):
    import av
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        if s.duration is not None:
            return float(s.duration * s.time_base)
        return float(c.duration / 1_000_000)


# --------------------------------------------------------------------------
class Grounder:
    def __init__(self, key, total_tokens=None, attn="sdpa", fps=2.0, style=None):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        cfg = MODELS[key]
        self.key, self.cfg = key, cfg
        self.family = cfg["family"]
        self.style = STYLES[style or cfg["style"]]
        self.fps = fps

        px = cfg["patch"] * 2                 # merge 후 토큰 한 변의 픽셀 수
        self.total_pixels = (total_tokens or cfg["tokens"]) * px * px
        self.min_pixels = 64 * px * px

        t0 = time.perf_counter()
        print(f"\n[load] {key}  ({cfg['repo']})", flush=True)
        # transformers 5.x 에서 torch_dtype -> dtype 으로 개명. 양쪽 모두 시도.
        common = dict(device_map="cuda:0", attn_implementation=attn)
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                cfg["repo"], dtype=torch.bfloat16, **common).eval()
        except TypeError:
            self.model = AutoModelForImageTextToText.from_pretrained(
                cfg["repo"], torch_dtype=torch.bfloat16, **common).eval()

        # do_resize / trust_remote_code 는 모델에 따라 필요 여부가 다르고,
        # 안 받는 프로세서에 넘기면 죽으므로 하나씩 떼면서 재시도합니다.
        pkw = dict(padding_side="left", do_resize=False)
        if cfg.get("trust_remote_code"):
            pkw["trust_remote_code"] = True
        for drop in ([], ["do_resize"], ["do_resize", "trust_remote_code"]):
            kw = {k: v for k, v in pkw.items() if k not in drop}
            try:
                self.processor = AutoProcessor.from_pretrained(cfg["repo"], **kw)
                break
            except (TypeError, ValueError) as e:
                last = e
        else:
            raise last

        print(f"[load] {time.perf_counter()-t0:.1f}s / "
              f"VRAM {torch.cuda.memory_allocated()/1024**3:.1f} GB", flush=True)

    # -- 계열별 전처리 -----------------------------------------------------
    # 세 공식 모델 카드의 process_vision_info 호출 규약이 전부 다릅니다.
    #   TimeLens-8B / TimeLens2 (Qwen3-VL)
    #     process_vision_info(msgs, image_patch_size=16,
    #                         return_video_kwargs=True, return_video_metadata=True)
    #     -> 3-tuple, videos 는 (tensor, meta) 쌍이라 zip 으로 분리 후
    #        processor(..., video_metadata=metas, **video_kwargs)
    #   TimeLens-7B (Qwen2.5-VL)
    #     process_vision_info(msgs, return_video_metadata=True)
    #     -> 2-tuple, video_kwargs 없이 processor 에 그대로 전달
    #   Time-R1 (Qwen2.5-VL, 표준 Qwen 방식)
    #     process_vision_info(msgs, return_video_kwargs=True)
    #     -> 3-tuple, processor(..., **video_kwargs)
    # 버전에 따라 인자를 못 받을 수 있어 순차적으로 폴백합니다.
    # ---------------------------------------------------------------------
    def _vision_info(self, messages):
        from qwen_vl_utils import process_vision_info
        patch = self.cfg["patch"]

        if self.family == "qwen3":
            imgs, vids, vkw = process_vision_info(
                messages, image_patch_size=patch,
                return_video_kwargs=True, return_video_metadata=True)
            metas = None
            if vids:
                vids, metas = zip(*vids)
                vids, metas = list(vids), list(metas)
            return imgs, vids, dict(video_metadata=metas, **vkw)

        if self.style["prompt"] is P_TIMELENS7:          # TimeLens-7B 공식 방식
            try:
                imgs, vids = process_vision_info(messages, return_video_metadata=True)
                return imgs, vids, {}
            except (TypeError, ValueError):
                pass                                      # 아래 표준 경로로 폴백

        try:                                              # Time-R1 / 표준 Qwen2.5-VL
            imgs, vids, vkw = process_vision_info(messages, return_video_kwargs=True)
        except TypeError:                                 # 아주 옛 버전
            imgs, vids = process_vision_info(messages)
            vkw = {}
        return imgs, vids, vkw

    def _build_inputs(self, video_path, query):
        vid = {
            "type": "video", "video": Path(video_path).resolve().as_uri(),
            "fps": self.fps,
            "min_pixels": self.min_pixels, "total_pixels": self.total_pixels,
        }
        text_part = {"type": "text", "text": self.style["prompt"].format(query)}

        messages = []
        if self.style["system"]:
            messages.append({"role": "system",
                             "content": [{"type": "text", "text": "You are a helpful assistant."}]})
        messages.append({"role": "user", "content": [vid, text_part]})

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        imgs, vids, extra = self._vision_info(messages)

        return self.processor(
            text=[text], images=imgs, videos=vids,
            padding=True, return_tensors="pt", **extra,
        ).to(self.model.device)

    @torch.inference_mode()
    def __call__(self, video_path, query, max_new_tokens=512):
        inputs = self._build_inputs(video_path, query)
        n_in = inputs.input_ids.shape[1]

        t0 = time.perf_counter()
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        dt = time.perf_counter() - t0

        raw = self.processor.batch_decode(
            [out[0][n_in:]], skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]

        return {
            "model": self.key, "query": query,
            "spans": self.style["parse"](raw),
            "raw": raw.strip(),
            "input_tokens": int(n_in),
            "latency_s": round(dt, 2),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 1),
        }

    def free(self):
        del self.model, self.processor
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# --------------------------------------------------------------------------
def plot(results, duration, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in results if r["spans"]]
    if not rows:
        print("[plot] 그릴 구간이 없습니다."); return

    palette = ["#c96343", "#4c9f70", "#5b7ea8", "#a87ba8", "#c9a343"]
    colors = {}
    for r in rows:
        colors.setdefault(r["model"], palette[len(colors) % len(palette)])

    fig, ax = plt.subplots(figsize=(12, 0.62 * len(rows) + 1.8))
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, duration, height=0.6, color="#eeeeee", zorder=1)
        for s, e in r["spans"]:
            ax.barh(y, max(e - s, duration * 0.004), left=s, height=0.6,
                    color=colors[r["model"]], zorder=2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f'{r["model"]}\n"{r["query"][:38]}"'
                        for r in reversed(rows)], fontsize=7)
    ax.set_xlim(0, duration)
    ax.set_xlabel("time (s)")
    ax.set_title("Video Temporal Grounding — predicted spans", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    print(f"[plot] 저장: {out_png}")


def main():
    ap = argparse.ArgumentParser(
        description="Video temporal grounding inference (TimeLens / TimeLens2 / Time-R1)")
    ap.add_argument("--model", default="timelens-8b",
                    help="모델 키, 프리셋(best/compare/ablation), 또는 콤마 구분 목록")
    ap.add_argument("--video")
    ap.add_argument("--query")
    ap.add_argument("--query-file", help="한 줄에 하나씩 질의를 담은 텍스트 파일")
    ap.add_argument("--total-tokens", type=int, default=None,
                    help="비주얼 토큰 예산. OOM이면 절반으로")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--attn", default="sdpa",
                    choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--style", default=None, choices=list(STYLES),
                    help="프롬프트 형식 강제 지정 (기본: 모델별 학습 형식)")
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        print(f"{'key':<15} {'repo':<30} {'백본':<26} note")
        print("-" * 110)
        for k, v in MODELS.items():
            print(f"{k:<15} {v['repo']:<30} {v['backbone']:<26} {v['note']}")
        print("\n프리셋:")
        for k, v in PRESETS.items():
            print(f"  {k:<10} {', '.join(v)}")
        return

    if not args.video or not (args.query or args.query_file):
        ap.error("--video 와 --query(또는 --query-file)가 필요합니다. 목록은 --list")
    if not Path(args.video).exists():
        ap.error(f"영상을 찾을 수 없습니다: {args.video}")

    preflight()          # GPU / 라이브러리 확인 (--list 는 여기 도달 전에 반환됨)

    queries = ([l.strip() for l in Path(args.query_file).read_text("utf-8").splitlines()
                if l.strip() and not l.startswith("#")]
               if args.query_file else [args.query])

    keys = PRESETS.get(args.model) or [k.strip() for k in args.model.split(",")]
    for k in keys:
        if k not in MODELS:
            ap.error(f"알 수 없는 모델: {k}  (--list 로 확인)")

    dur = video_duration(args.video)
    print(f"video   : {args.video}  ({dur:.2f}s)")
    print(f"queries : {len(queries)}")
    print(f"models  : {', '.join(keys)}")

    results = []
    with open(args.out, "w", encoding="utf-8") as f:
        for k in keys:                       # 순차 로드 -> 해제 (동시 적재 불가)
            g = None
            try:
                g = Grounder(k, args.total_tokens, args.attn, args.fps, args.style)
                for q in queries:
                    r = g(args.video, q, args.max_new_tokens)
                    r["video"], r["duration"] = args.video, round(dur, 2)
                    results.append(r)
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"  {str(r['spans']):<30} {r['latency_s']:>6.2f}s "
                          f"{r['peak_vram_gb']:>5.1f}GB | {q}")
            except torch.cuda.OutOfMemoryError:
                half = (args.total_tokens or MODELS[k]["tokens"]) // 2
                print(f"\n!! OOM ({k}). --total-tokens {half} 로 다시 시도하세요.\n")
            finally:
                if g is not None:
                    g.free()

    print(f"\n저장: {args.out}")
    if args.plot:
        plot(results, dur, str(Path(args.out).with_suffix(".png")))

    if results:
        print("\n" + "=" * 70)
        print(f"원본 출력 예시 ({results[0]['model']}):")
        print("=" * 70)
        print(results[0]["raw"][:1200])


if __name__ == "__main__":
    sys.exit(main())
