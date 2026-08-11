#!/usr/bin/env python3
"""
Video Temporal Grounding — 웹 UI

영상을 고르거나 올리고 문장을 쓰면 해당 장면의 시간 구간을 찾아 잘라서 보여줍니다.

영상 고르는 방법이 두 가지입니다.
  ① 서버에 있는 영상 (인스턴스에서 --download-bench 로 받은 것) -> 드롭다운에서 선택
  ② 내 컴퓨터에 있는 영상 -> 끌어다 놓기

벤치마크 영상을 고르면 **사람이 검수한 질의와 정답 구간**이 같이 표시돼서,
클릭 한 번으로 넣고 결과가 맞는지 바로 확인할 수 있습니다.

실행 (엘리스 인스턴스에서):
    source ~/vtg-env/bin/activate
    python3 app.py                  # http://<인스턴스주소>:7860
    python3 app.py --share          # 임시 공개 링크 (외부에서 접속)
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import vtg_run as V

VIDEO_DIR = Path("videos")

# 로드된 모델을 재사용 (요청마다 다시 올리면 매번 1~2분 낭비)
_STATE = {"key": None, "tokens": None, "fps": None, "g": None}

# grounding 특화 모델 4종. 전부 공식 모델 카드에 추론 코드가 공개된 것들입니다.
# (학습 전 원본 Qwen3-VL 은 뺐습니다. 필요하면 vtg_run.py --model qwen3-vl-8b 로 쓸 수 있습니다.)
CHOICES = [
    ("TimeLens-8B  (CVPR'26, 권장)",         "timelens-8b"),
    ("TimeLens2-8B (최신, 다중 구간)",        "timelens2-8b"),
    ("TimeLens2-4B (가벼움/빠름)",            "timelens2-4b"),
    ("TimeLens-7B  (CVPR'26, Qwen2.5 계열)",  "timelens-7b"),
]
LABEL2KEY = {lbl: key for lbl, key in CHOICES}

NONE = "— 직접 올리기 —"


def is_downloaded(key):
    """HF 캐시에 이미 받아져 있는지. 첫 사용 시 몇 분 기다려야 하는지 미리 알려줍니다."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        repo = V.MODELS[key]["repo"].replace("/", "--")
        return (Path(HF_HUB_CACHE) / f"models--{repo}").exists()
    except Exception:      # noqa: BLE001
        return False


def model_note(label):
    key = LABEL2KEY[label]
    if is_downloaded(key):
        return f"`{V.MODELS[key]['repo']}` — 이미 받아둔 모델입니다."
    return (f"`{V.MODELS[key]['repo']}` — **아직 안 받은 모델입니다.** "
            "처음 실행할 때 다운로드로 몇 분 걸립니다.")


# ==========================================================================
#  서버에 있는 영상 + 준비된 질의
# ==========================================================================
def server_videos():
    """videos/ 안의 mp4 목록. 인스턴스에서 받은 벤치마크 영상이 여기 들어옵니다."""
    return [NONE] + sorted(str(p) for p in VIDEO_DIR.glob("*.mp4"))


def known_queries():
    """videos/bench.json + external.json 에서 영상별 질의/정답을 모읍니다."""
    out = {}
    for f in ("bench.json", "external.json"):
        p = VIDEO_DIR / f
        if not p.exists():
            continue
        try:
            for v in json.loads(p.read_text("utf-8")):
                name = v.get("name")
                if not name:
                    continue
                out[name] = list(zip(v.get("queries") or [],
                                     (v.get("gts") or []) + [None] * len(v.get("queries") or [])))
        except (ValueError, TypeError):
            continue
    return out


def on_pick_video(path):
    """서버 영상을 고르면 영상을 띄우고, 준비된 질의 목록을 채웁니다."""
    import gradio as gr
    if not path or path == NONE:
        return None, gr.update(choices=[], value=None, visible=False), ""

    name = Path(path).stem
    pairs = known_queries().get(name, [])
    if not pairs:
        return path, gr.update(choices=[], value=None, visible=False), ""

    labels = []
    for q, gt in pairs:
        labels.append(f"{q}   [정답 {gt[0]:.0f}~{gt[1]:.0f}초]" if gt else q)
    md = (f"**{name}** — 준비된 질의 {len(pairs)}개. "
          "아래에서 고르면 질의창에 채워집니다.")
    return path, gr.update(choices=labels, value=None, visible=True), md


def on_pick_query(label):
    """질의를 고르면 정답 표시를 떼고 질의창에 넣습니다."""
    return (label or "").split("   [정답")[0].strip()


# ==========================================================================
#  추론
# ==========================================================================
def get_grounder(key, tokens, fps, attn):
    """같은 설정이면 그대로 재사용, 바뀌었으면 갈아끼웁니다."""
    if (_STATE["key"], _STATE["tokens"], _STATE["fps"]) == (key, tokens, fps):
        return _STATE["g"]
    if _STATE["g"] is not None:
        _STATE["g"].free()
        _STATE["g"] = None
    _STATE.update(key=key, tokens=tokens, fps=fps,
                  g=V.Grounder(key, total_tokens=tokens, attn=attn, fps=fps))
    return _STATE["g"]


def budget_for(duration):
    """영상 길이에 맞춰 토큰 예산 / fps 를 자동 결정합니다."""
    if duration <= 60:
        return 32768, 2.0
    if duration <= 180:
        return 16384, 2.0
    if duration <= 600:
        return 16384, 1.0
    return 8192, 1.0


def cut_clip(video, start, end, pad=0.5):
    out = Path(tempfile.mkdtemp()) / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(max(0.0, start - pad)),
         "-to", str(end + pad), "-i", str(video),
         "-c:v", "libx264", "-preset", "veryfast", str(out)], check=True)
    return str(out)


def timeline_png(duration, spans, gt=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(tempfile.mkdtemp()) / "timeline.png"
    fig, ax = plt.subplots(figsize=(11, 2.0 if gt else 1.5))
    ax.barh(0, duration, height=0.45, color="#eeeeee")
    for s, e in spans:
        ax.barh(0, max(e - s, duration * 0.004), left=s, height=0.45, color="#c96343")
        ax.text((s + e) / 2, 0.32, f"{s:.1f}~{e:.1f}", ha="center", fontsize=8)
    if gt:
        ax.barh(-0.6, duration, height=0.45, color="#eeeeee")
        ax.barh(-0.6, max(gt[1] - gt[0], duration * 0.004), left=gt[0],
                height=0.45, color="#4c9f70")
        ax.text((gt[0] + gt[1]) / 2, -0.88, f"{gt[0]:.1f}~{gt[1]:.1f}", ha="center", fontsize=8)
        # matplotlib 기본 폰트에 한글이 없어 라벨은 영문으로 둡니다 (□□ 방지)
        ax.set_yticks([0, -0.6])
        ax.set_yticklabels(["predicted", "ground truth"], fontsize=9)
        ax.set_ylim(-1.1, 0.6)
    else:
        ax.set_yticks([])
        ax.set_ylim(-0.4, 0.6)
    ax.set_xlim(0, duration)
    ax.set_xlabel("time (s)")
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close(fig)
    return str(out)


def tiou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def find_gt(video, query):
    """벤치마크 영상이면 이 질의의 정답 구간을 찾아 돌려줍니다."""
    if not video:
        return None
    for q, gt in known_queries().get(Path(video).stem, []):
        if gt and q.strip().lower() == (query or "").strip().lower():
            return gt
    return None


def run(video, query, model_label, attn, progress=None):
    if not video:
        return "영상을 고르거나 올려주세요.", None, None, ""
    if not (query or "").strip():
        return "찾을 장면을 문장으로 써주세요.", None, None, ""

    key = LABEL2KEY[model_label]
    dur = V.video_duration(video)
    tokens, fps = budget_for(dur)

    if progress:
        progress(0.1, desc=f"모델 준비 중… ({key}, 첫 실행은 다운로드로 몇 분)")
    g = get_grounder(key, tokens, fps, attn)

    if progress:
        progress(0.5, desc=f"영상 분석 중… ({dur:.0f}초, {tokens} 토큰 / {fps}fps)")
    r = g(video, query.strip())
    spans = r["spans"]

    if not spans:
        return ("**구간을 찾지 못했습니다.** 문장을 더 구체적으로 바꿔보세요.\n\n"
                "예: `a person walks` → `a woman in a red jacket opens the door`",
                None, None, r["raw"])

    gt = find_gt(video, query)
    lines = [f"### 결과 — {len(spans)}개 구간", ""]
    for i, (s, e) in enumerate(spans, 1):
        lines.append(f"**{i}. {s:.2f}초 ~ {e:.2f}초**  (길이 {e - s:.2f}초)")
    if gt:
        score = tiou(spans[0], gt)
        mark = "잘 맞음" if score >= 0.5 else ("아쉬움" if score >= 0.3 else "빗나감")
        lines += ["", f"정답: **{gt[0]:.2f}초 ~ {gt[1]:.2f}초**",
                  f"**tIoU = {score:.3f}**  ({mark})"]
    lines += ["", f"영상 {dur:.1f}초 · {r['latency_s']}초 소요 · "
                  f"VRAM {r['peak_vram_gb']}GB · 모델 `{key}`"]

    png = timeline_png(dur, spans, gt)
    try:
        clip = cut_clip(video, spans[0][0], spans[0][1])
    except Exception:            # noqa: BLE001  ffmpeg 없어도 결과는 보여줌
        clip = None
    return "\n".join(lines), png, clip, r["raw"]


# ==========================================================================
def build_ui():
    import gradio as gr

    with gr.Blocks(title="Video Temporal Grounding") as demo:
        gr.Markdown(
            "# 영상에서 장면 찾기\n"
            "영상을 고르고 **찾고 싶은 장면을 문장으로** 쓰면, 몇 초부터 몇 초인지 찾아줍니다.\n"
            "문장은 **영어로** 쓰세요 — 모델이 영어로 학습됐습니다."
        )
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    picker = gr.Dropdown(server_videos(), value=NONE, scale=4,
                                         label="서버에 있는 영상 (인스턴스에서 받은 것)")
                    refresh = gr.Button("새로고침", scale=1)
                video = gr.Video(label="영상 — 위에서 고르거나 여기에 끌어다 놓기", height=280)

                hint = gr.Markdown()
                qpick = gr.Dropdown([], label="준비된 질의 (정답 포함)", visible=False)

                query = gr.Textbox(
                    label="찾을 장면 (영어)",
                    placeholder="a woman in a red jacket opens the refrigerator",
                    lines=2)
                model = gr.Dropdown([c[0] for c in CHOICES], value=CHOICES[0][0],
                                    label="모델")
                mnote = gr.Markdown(model_note(CHOICES[0][0]))
                attn = gr.Radio(["sdpa", "flash_attention_2"], value="sdpa",
                                label="attention (flash 는 설치돼 있을 때만)")
                btn = gr.Button("장면 찾기", variant="primary", size="lg")

                gr.Markdown(
                    "**좋은 문장 쓰는 법**\n"
                    "- 시작·끝을 눈으로 확정할 수 있는 동작으로\n"
                    "- 영상에서 **한 번만** 일어나는 사건으로\n"
                    "- 구체적인 명사 + 명확한 동작\n\n"
                    "❌ `a person is walking` (영상 내내 나옴)\n"
                    "✅ `two people shake hands`\n\n"
                    "> 영상에 **없는** 장면을 물어도 모델은 아무 구간이나 답합니다. "
                    "현재 기술의 알려진 한계입니다."
                )
            with gr.Column(scale=1):
                out_md = gr.Markdown()
                out_png = gr.Image(label="타임라인 (예측 / 정답)", height=200)
                out_clip = gr.Video(label="찾은 구간 (앞뒤 0.5초 여유)", height=300)
                with gr.Accordion("모델 원본 출력", open=False):
                    out_raw = gr.Textbox(lines=6, show_label=False)

        picker.change(on_pick_video, picker, [video, qpick, hint])
        refresh.click(lambda: gr.update(choices=server_videos()), None, picker)
        qpick.change(on_pick_query, qpick, query)
        model.change(model_note, model, mnote)      # 안 받은 모델이면 미리 알려줌

        def _run(v, q, m, a, progress=gr.Progress()):
            return run(v, q, m, a, progress)

        btn.click(_run, [video, query, model, attn],
                  [out_md, out_png, out_clip, out_raw])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="외부 접속용 임시 공개 링크")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    V.preflight()
    try:
        import gradio  # noqa: F401
    except ImportError:
        sys.exit("!! gradio 가 없습니다.  pip install gradio")

    vids = [v for v in server_videos() if v != NONE]
    print(f"서버 영상 {len(vids)}개: {', '.join(Path(v).name for v in vids) or '없음'}")
    build_ui().launch(server_name="0.0.0.0", server_port=args.port,
                      share=args.share, show_error=True)
