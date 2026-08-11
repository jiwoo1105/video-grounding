#!/usr/bin/env python3
"""Video Temporal Grounding 데모 — 단일 파일. 터미널 붙여넣기용.

영상을 올리고 영어 문장을 쓰면 몇 초부터 몇 초인지 찾아 잘라 보여줍니다.
    python3 demo.py          # http://<주소>:7860
    python3 demo.py --share  # 외부 접속 링크
"""
import argparse, json, subprocess, sys, tempfile
from pathlib import Path
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

VD = Path("videos")

# ── 프롬프트: 각 공식 모델 카드 원문 그대로 (섞어 쓰면 성능이 무너집니다) ──
P8 = ("Please find the visual event described by the sentence '{}', determining its "
      "starting and ending times. The format should be: 'The event happens in "
      "<start time> - <end time> seconds'.")
P7 = ("You are given a video with multiple frames. The numbers before each video frame "
      "indicate its sampling timestamp (in seconds). " + P8)
P2 = ('Given the query: "{}", return ALL time spans (in seconds) where the query is '
      'relevant.\nOutput format MUST be a JSON array of [start, end] pairs.\n')

#   키: (저장소, 계열, 프롬프트, 패치, 토큰예산)
M = {
 "TimeLens-8B  (CVPR'26, 권장)":   ("TencentARC/TimeLens-8B", "q3", P8, 16, 14336),
 "TimeLens2-8B (최신)":            ("MCG-NJU/TimeLens2-8B",   "q3", P2, 16, 16384),
 "TimeLens2-4B (가벼움)":          ("MCG-NJU/TimeLens2-4B",   "q3", P2, 16, 32768),
 "TimeLens-7B  (Qwen2.5 계열)":    ("TencentARC/TimeLens-7B", "q2", P7, 14, 14336),
}
S = {"key": None, "m": None, "p": None}


def parse(txt, style):
    import re
    if style is P2:
        for mm in re.finditer(r"\[\s*\[.*?\]\s*\]", txt, re.S):
            try:
                a = json.loads(mm.group(0))
                o = [[float(x), float(y)] for x, y in a if float(y) >= float(x)]
                if o:
                    return o
            except Exception:
                pass
    h = re.findall(r"(\d+\.?\d*)\s*(?:-|to)\s*(\d+\.?\d*)", txt)
    return [[float(h[-1][0]), float(h[-1][1])]] if h else []


def dur_of(p):
    import av
    with av.open(str(p)) as c:
        s = c.streams.video[0]
        return float(s.duration * s.time_base) if s.duration else float(c.duration / 1e6)


def load(label):
    repo, fam, _, patch, _ = M[label]
    if S["key"] == label:
        return
    if S["m"] is not None:
        del S["m"], S["p"]
        import gc; gc.collect(); torch.cuda.empty_cache()
    print(f"[load] {repo} …", flush=True)
    kw = dict(device_map="cuda:0", attn_implementation="sdpa")
    try:
        m = AutoModelForImageTextToText.from_pretrained(repo, dtype=torch.bfloat16, **kw)
    except TypeError:
        m = AutoModelForImageTextToText.from_pretrained(repo, torch_dtype=torch.bfloat16, **kw)
    pk = dict(padding_side="left", do_resize=False)
    if fam == "q2":
        pk["trust_remote_code"] = True
    try:
        pr = AutoProcessor.from_pretrained(repo, **pk)
    except (TypeError, ValueError):
        pk.pop("do_resize", None)
        pr = AutoProcessor.from_pretrained(repo, **pk)
    S.update(key=label, m=m.eval(), p=pr)


@torch.inference_mode()
def find(video, query, label):
    repo, fam, prompt, patch, tok = M[label]
    d = dur_of(video)
    fps = 2.0 if d <= 180 else 1.0
    tok = tok if d <= 60 else (16384 if d <= 600 else 8192)
    load(label)
    m, pr = S["m"], S["p"]
    px = patch * 2
    vid = {"type": "video", "video": Path(video).resolve().as_uri(), "fps": fps,
           "min_pixels": 64 * px * px, "total_pixels": tok * px * px}
    msg = [{"role": "user", "content": [vid, {"type": "text", "text": prompt.format(query)}]}]
    txt = pr.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    if fam == "q3":     # Qwen3-VL: 타임스탬프를 video_metadata 로 별도 전달
        im, vi, vk = process_vision_info(msg, image_patch_size=patch,
                                         return_video_kwargs=True, return_video_metadata=True)
        meta = None
        if vi:
            vi, meta = zip(*vi); vi, meta = list(vi), list(meta)
        inp = pr(text=[txt], images=im, videos=vi, video_metadata=meta,
                 padding=True, return_tensors="pt", **vk)
    else:               # Qwen2.5-VL (TimeLens-7B 공식 방식)
        im, vi = process_vision_info(msg, return_video_metadata=True)
        inp = pr(text=[txt], images=im, videos=vi, padding=True, return_tensors="pt")
    inp = inp.to(m.device)

    n = inp.input_ids.shape[1]
    out = m.generate(**inp, max_new_tokens=512, do_sample=False)
    raw = pr.batch_decode([out[0][n:]], skip_special_tokens=True)[0]
    return parse(raw, prompt), raw.strip(), d


def known(name):
    f = VD / "external.json"
    if not f.exists():
        return []
    try:
        for v in json.loads(f.read_text("utf-8")):
            if v.get("name") == name:
                g = (v.get("gts") or []) + [None] * 9
                return list(zip(v.get("queries") or [], g))
    except Exception:
        pass
    return []


def run(video, query, label, progress=None):
    if not video:
        return "영상을 고르거나 올려주세요.", None, None, ""
    if not (query or "").strip():
        return "찾을 장면을 영어 문장으로 써주세요.", None, None, ""
    if progress:
        progress(0.2, desc=f"{label} 준비 중 (첫 실행은 다운로드로 몇 분)")
    spans, raw, d = find(video, query.strip(), label)
    if not spans:
        return "**구간을 찾지 못했습니다.** 문장을 더 구체적으로 바꿔보세요.", None, None, raw

    gt = next((g for q, g in known(Path(video).stem)
               if g and q.strip().lower() == query.strip().lower()), None)
    md = [f"### 결과 — {len(spans)}개 구간", ""]
    for i, (s, e) in enumerate(spans, 1):
        md.append(f"**{i}. {s:.2f}초 ~ {e:.2f}초**  (길이 {e-s:.2f}초)")
    if gt:
        it = max(0.0, min(spans[0][1], gt[1]) - max(spans[0][0], gt[0]))
        un = max(spans[0][1], gt[1]) - min(spans[0][0], gt[0])
        t = it / un if un > 0 else 0
        md += ["", f"정답: **{gt[0]:.1f}초 ~ {gt[1]:.1f}초**",
               f"**tIoU = {t:.3f}** ({'잘 맞음' if t >= .5 else '아쉬움' if t >= .3 else '빗나감'})"]
    md += ["", f"영상 {d:.1f}초 · 모델 `{M[label][0]}`"]

    png = None
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        png = str(Path(tempfile.mkdtemp()) / "t.png")
        fig, ax = plt.subplots(figsize=(11, 1.9 if gt else 1.4))
        ax.barh(0, d, height=.45, color="#eee")
        for s, e in spans:
            ax.barh(0, max(e - s, d * .004), left=s, height=.45, color="#c96343")
        if gt:
            ax.barh(-.6, d, height=.45, color="#eee")
            ax.barh(-.6, max(gt[1] - gt[0], d * .004), left=gt[0], height=.45, color="#4c9f70")
            ax.set_yticks([0, -.6]); ax.set_yticklabels(["predicted", "ground truth"], fontsize=9)
            ax.set_ylim(-1.1, .5)
        else:
            ax.set_yticks([]); ax.set_ylim(-.4, .5)
        ax.set_xlim(0, d); ax.set_xlabel("time (s)")
        plt.tight_layout(); plt.savefig(png, dpi=140); plt.close(fig)
    except Exception:
        png = None

    clip = None
    try:
        clip = str(Path(tempfile.mkdtemp()) / "c.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                        "-ss", str(max(0, spans[0][0] - .5)), "-to", str(spans[0][1] + .5),
                        "-i", str(video), "-c:v", "libx264", "-preset", "veryfast", clip],
                       check=True)
    except Exception:
        clip = None
    return "\n".join(md), png, clip, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("GPU(CUDA)가 없습니다. 엘리스 인스턴스에서 실행하세요.")
    import gradio as gr

    def vids():
        return ["— 직접 올리기 —"] + sorted(str(p) for p in VD.glob("*.mp4"))

    def pick(p):
        if not p or p.startswith("—"):
            return None, gr.update(choices=[], visible=False)
        ks = known(Path(p).stem)
        if not ks:
            return p, gr.update(choices=[], visible=False)
        return p, gr.update(visible=True, value=None, choices=[
            f"{q}   [정답 {g[0]:.0f}~{g[1]:.0f}초]" if g else q for q, g in ks])

    with gr.Blocks(title="Video Grounding") as demo:
        gr.Markdown("# 영상에서 장면 찾기\n영상을 고르고 **영어 문장**으로 찾을 장면을 쓰세요.")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    pk = gr.Dropdown(vids(), value=vids()[0], scale=4, label="서버 영상")
                    rf = gr.Button("새로고침", scale=1)
                vd = gr.Video(label="영상 (여기에 끌어다 놓아도 됩니다)", height=280)
                qp = gr.Dropdown([], label="준비된 질의", visible=False)
                q = gr.Textbox(label="찾을 장면 (영어)", lines=2,
                               placeholder="a man drinks from a plastic water bottle")
                md = gr.Dropdown(list(M), value=list(M)[0], label="모델")
                btn = gr.Button("장면 찾기", variant="primary", size="lg")
            with gr.Column():
                o1 = gr.Markdown()
                o2 = gr.Image(label="타임라인", height=200)
                o3 = gr.Video(label="찾은 구간", height=300)
                with gr.Accordion("모델 원본 출력", open=False):
                    o4 = gr.Textbox(lines=6, show_label=False)

        pk.change(pick, pk, [vd, qp])
        rf.click(lambda: gr.update(choices=vids()), None, pk)
        qp.change(lambda s: (s or "").split("   [정답")[0].strip(), qp, q)
        btn.click(lambda v, t, m, p=gr.Progress(): run(v, t, m, p), [vd, q, md], [o1, o2, o3, o4])

    demo.launch(server_name="0.0.0.0", server_port=a.port, share=a.share, show_error=True)


if __name__ == "__main__":
    main()
