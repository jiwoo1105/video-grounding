#!/usr/bin/env python3
"""
Video Temporal Grounding — 웹 UI

영상을 고르거나 올리고 문장을 쓰면 해당 장면의 시간 구간을 찾아 잘라서 보여줍니다.

영상 고르는 방법이 두 가지입니다.
  ① 서버에 있는 영상 (인스턴스에서 --download-bench 로 받은 것) -> 드롭다운에서 선택
  ② 내 컴퓨터에 있는 영상 -> 끌어다 놓기

영상을 고르면 미리 준비해둔 질의 목록이 떠서, 클릭 한 번으로 넣을 수 있습니다.

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
    """아직 안 받은 모델일 때만 경고를 띄웁니다. 다 받아뒀으면 아무것도 안 보입니다."""
    key = LABEL2KEY[label]
    if is_downloaded(key):
        return ""
    return f"⏳ `{V.MODELS[key]['repo']}` — 첫 실행 시 다운로드로 몇 분 걸립니다."


# ==========================================================================
#  서버에 있는 영상 + 준비된 질의
# ==========================================================================
def server_videos():
    """videos/ 안의 mp4 목록. 인스턴스에서 받은 벤치마크 영상이 여기 들어옵니다."""
    return [NONE] + sorted(str(p) for p in VIDEO_DIR.glob("*.mp4"))


def known_queries():
    """videos/bench.json + external.json 에서 영상별 준비된 질의를 모읍니다."""
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

    labels = [q for q, _ in pairs]
    md = (f"**{name}** — 준비된 질의 {len(pairs)}개. "
          "아래에서 고르면 질의창에 채워집니다.")
    return path, gr.update(choices=labels, value=None, visible=True), md


def on_pick_query(label):
    """고른 질의를 질의창에 넣습니다."""
    return (label or "").strip()


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


def timeline_png(duration, spans):
    """영상 전체를 가로 막대로 펼치고, 모델이 찾은 구간을 표시합니다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(tempfile.mkdtemp()) / "timeline.png"
    fig, ax = plt.subplots(figsize=(11, 1.5))
    ax.barh(0, duration, height=0.45, color="#eeeeee")
    for s, e in spans:
        ax.barh(0, max(e - s, duration * 0.004), left=s, height=0.45, color="#c96343")
        ax.text((s + e) / 2, 0.32, f"{s:.1f}~{e:.1f}", ha="center", fontsize=9)
    ax.set_yticks([])
    ax.set_ylim(-0.4, 0.6)
    ax.set_xlim(0, duration)
    ax.set_xlabel("time (s)")
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close(fig)
    return str(out)


def _clip_update(path):
    """클립을 못 만들면(ffmpeg 없음) 칸 자체를 숨깁니다."""
    import gradio as gr
    return gr.update(value=path, visible=bool(path))


def run(video, query, model_label, progress=None):
    if not video:
        return "영상을 고르거나 올려주세요.", None, _clip_update(None), ""
    if not (query or "").strip():
        return "찾을 장면을 문장으로 써주세요.", None, _clip_update(None), ""

    key = LABEL2KEY[model_label]
    dur = V.video_duration(video)
    tokens, fps = budget_for(dur)

    if progress:
        progress(0.1, desc=f"모델 준비 중… ({key}, 첫 실행은 다운로드로 몇 분)")
    g = get_grounder(key, tokens, fps, "sdpa")   # flash-attn 은 별도 설치 필요

    if progress:
        progress(0.5, desc=f"영상 분석 중… ({dur:.0f}초, {tokens} 토큰 / {fps}fps)")
    r = g(video, query.strip())
    spans = r["spans"]

    if not spans:
        return ("**구간을 찾지 못했습니다.** 문장을 더 구체적으로 바꿔보세요.\n\n"
                "예: `a person walks` → `a woman in a red jacket opens the door`",
                None, _clip_update(None), r["raw"])

    lines = [f"### 결과 — {len(spans)}개 구간", ""]
    for i, (s, e) in enumerate(spans, 1):
        lines.append(f"**{i}. {s:.2f}초 ~ {e:.2f}초**  (길이 {e - s:.2f}초)")
    lines += ["", f"영상 {dur:.1f}초 · {r['latency_s']}초 소요 · "
                  f"VRAM {r['peak_vram_gb']}GB · 모델 `{key}`"]

    png = timeline_png(dur, spans)
    try:
        clip = cut_clip(video, spans[0][0], spans[0][1])
    except Exception:            # noqa: BLE001  ffmpeg 없으면 클립 칸을 아예 숨김
        clip = None
    return "\n".join(lines), png, _clip_update(clip), r["raw"]


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
                qpick = gr.Dropdown([], label="준비된 질의", visible=False)

                query = gr.Textbox(
                    label="찾을 장면 (영어)",
                    placeholder="a woman in a red jacket opens the refrigerator",
                    lines=2)
                model = gr.Dropdown([c[0] for c in CHOICES], value=CHOICES[0][0],
                                    label="모델")
                mnote = gr.Markdown(model_note(CHOICES[0][0]))
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
                out_png = gr.Image(label="타임라인 — 모델이 찾은 구간", height=170)
                out_clip = gr.Video(label="찾은 구간 (앞뒤 0.5초 여유)",
                                    height=300, visible=False)
                with gr.Accordion("모델 원본 출력", open=False):
                    out_raw = gr.Textbox(lines=6, show_label=False)

        picker.change(on_pick_video, picker, [video, qpick, hint])
        refresh.click(lambda: gr.update(choices=server_videos()), None, picker)
        qpick.change(on_pick_query, qpick, query)
        model.change(model_note, model, mnote)      # 안 받은 모델이면 미리 알려줌

        def _run(v, q, m, progress=gr.Progress()):
            return run(v, q, m, progress)

        btn.click(_run, [video, query, model],
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
    # allowed_paths: Gradio 는 보안상 허용된 경로의 파일만 브라우저에 내보냅니다.
    #   videos/ 를 넣어야 드롭다운으로 고른 서버 영상이 재생됩니다 (없으면 404).
    # max_file_size: 업로드 상한. 기본값이 낮아 큰 영상이 막히는 경우가 있습니다.
    build_ui().launch(
        server_name="0.0.0.0", server_port=args.port,
        share=args.share, show_error=True,
        allowed_paths=[str(VIDEO_DIR.resolve()), str(Path(tempfile.gettempdir()))],
        max_file_size="500mb",
    )
