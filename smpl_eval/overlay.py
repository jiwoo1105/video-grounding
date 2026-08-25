"""tracks.npz 를 영상 위에 그린다 — 게이트 1 및 실패사례 확인용.

MIG 는 OpenGL 을 지원하지 않아 메쉬 렌더링(PyRender)을 쓸 수 없다.
대신 PIL 로 관절·골격·bbox 를 그린다. 포즈가 사람에 제대로 붙었는지
판단하는 데는 이것으로 충분하며, 어디서나 동작한다.
"""
import argparse
import io
import os
import subprocess

import numpy as np

from smpl_eval.conventions import SMPL24

# SMPL 24관절 골격 연결
SMPL_LINKS = [
    (0, 1), (1, 4), (4, 7), (7, 10),          # 왼다리
    (0, 2), (2, 5), (5, 8), (8, 11),          # 오른다리
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),  # 척추·목·머리
    (9, 13), (13, 16), (16, 18), (18, 20), (20, 22),   # 왼팔
    (9, 14), (14, 17), (17, 19), (19, 21), (21, 23),   # 오른팔
]
LEFT_JOINTS = {1, 4, 7, 10, 13, 16, 18, 20, 22}

# 트랙 ID 별 색 (구분이 목적이라 채도 높은 색으로)
PALETTE = [(255, 82, 82), (68, 189, 50), (0, 168, 255), (251, 197, 49),
           (156, 136, 255), (255, 121, 198), (0, 210, 211), (255, 159, 67)]


def read_frame(video_path, idx):
    """ffmpeg 로 특정 프레임 하나를 PNG 바이트로 뽑는다."""
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", video_path,
         "-vf", f"select=eq(n\\,{idx})", "-vframes", "1",
         "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True, check=True).stdout


def draw_frame(video_path, tracks, frame_idx, label_tracks=True):
    """한 프레임에 tracks 를 그려 PIL Image 를 돌려준다."""
    from PIL import Image, ImageDraw

    img = Image.open(io.BytesIO(read_frame(video_path, frame_idx))).convert("RGB")
    d = ImageDraw.Draw(img)
    _draw_rows(d, tracks, np.where(tracks["frame_ids"] == frame_idx)[0])
    return img


def render_video(video_path, tracks_path, out_path, fps=None, max_frames=None,
                 scale=1.0, label=None):
    """전체 구간을 오버레이한 mp4 를 만든다.

    프레임을 한 장씩 뽑아 그리면 ffmpeg 호출이 프레임 수만큼 발생해 느리다.
    영상을 통째로 디코딩해 파이프로 받고, 그린 뒤 다시 파이프로 인코딩한다.
    """
    import shutil
    from PIL import Image, ImageDraw
    from smpl_eval.schema import load_tracks

    tracks, meta = load_tracks(tracks_path)
    fps = fps or meta.get("fps") or 30.0
    W, H = meta.get("resolution", [1920, 1080])
    ow, oh = int(W * scale), int(H * scale)
    ow -= ow % 2
    oh -= oh % 2

    total = int(max_frames or (tracks["frame_ids"].max() + 1))
    by_frame = {}
    for i in range(len(tracks["frame_ids"])):
        by_frame.setdefault(int(tracks["frame_ids"][i]), []).append(i)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", video_path, "-frames:v", str(total),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE)
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{ow}x{oh}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out_path],
        stdin=subprocess.PIPE)

    nbytes = W * H * 3
    n_drawn = 0
    try:
        for f in range(total):
            buf = dec.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            img = Image.frombytes("RGB", (W, H), buf)
            rows = by_frame.get(f, [])
            if rows:
                _draw_rows(ImageDraw.Draw(img), tracks, rows)
                n_drawn += 1
            d2 = ImageDraw.Draw(img)
            cap = f"{label or meta.get('model','')}   frame {f}   tracks {len(rows)}"
            d2.rectangle([0, 0, 10 + 7 * len(cap), 22], fill=(0, 0, 0))
            d2.text((6, 6), cap, fill=(255, 255, 255))
            if scale != 1.0:
                img = img.resize((ow, oh), Image.BILINEAR)
            enc.stdin.write(img.tobytes())
    finally:
        enc.stdin.close()
        enc.wait()
        dec.stdout.close()
        dec.wait()
    return out_path, n_drawn


def _draw_rows(d, tracks, rows):
    """한 프레임의 여러 트랙을 그린다 (draw_frame 과 공유)."""
    for row in rows:
        tid = int(tracks["track_ids"][row])
        col = PALETTE[tid % len(PALETTE)]
        j = tracks["joints2d"][row]
        ok = np.isfinite(j).all(-1)

        b = tracks["bbox"][row]
        if b[2] > b[0]:
            d.rectangle([float(v) for v in b], outline=col, width=3)
            d.text((float(b[0]) + 4, float(b[1]) + 4), f"id {tid}", fill=col)

        for a, c in SMPL_LINKS:
            if ok[a] and ok[c]:
                d.line([tuple(j[a]), tuple(j[c])], fill=col, width=3)
        for k in range(len(j)):
            if ok[k]:
                x, y = float(j[k][0]), float(j[k][1])
                r = 4 if k in LEFT_JOINTS else 3
                d.ellipse([x - r, y - r, x + r, y + r],
                          fill=(255, 255, 255), outline=col)


def render(video_path, tracks_path, out_dir, frames):
    from smpl_eval.schema import load_tracks

    tracks, _meta = load_tracks(tracks_path)
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for f in frames:
        if not (tracks["frame_ids"] == f).any():
            continue
        p = os.path.join(out_dir, f"frame_{f:05d}.png")
        draw_frame(video_path, tracks, f).save(p)
        written.append(p)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="smpl_eval/gate_out/overlay")
    ap.add_argument("--frames", default="0,10,20",
                    help="쉼표로 구분한 프레임 번호 (--video-out 없을 때)")
    ap.add_argument("--video-out", default=None,
                    help="지정하면 전체 구간을 오버레이한 mp4 를 만든다")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--label", default=None)
    a = ap.parse_args(argv)

    if a.video_out:
        p, n = render_video(a.video, a.tracks, a.video_out,
                            max_frames=a.max_frames, scale=a.scale,
                            label=a.label)
        print(f"wrote {p}  ({n} 프레임에 트랙 표시)")
        return 0

    for p in render(a.video, a.tracks, a.out, [int(x) for x in a.frames.split(",")]):
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
