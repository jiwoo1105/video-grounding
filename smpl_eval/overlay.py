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

# 라벨용 트루타입 폰트 후보. 없으면 PIL 기본(11px)으로 떨어진다.
#
# 한글이 들어가는 라벨이 있으므로 **한글 글리프를 가진 폰트를 먼저** 둔다.
# DejaVu 에는 한글이 없어 전부 두부(□)로 렌더링된다 (실측 확인).
_FONT_PATHS = [
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf",
    "/usr/share/fonts/truetype/nanum/NanumSquare_acB.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    # 한글이 없는 폴백
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]
_font_cache = {}


def _font(size):
    """크기별 폰트를 캐시해 돌려준다. 프레임마다 로드하면 느리다."""
    if size in _font_cache:
        return _font_cache[size]
    from PIL import ImageFont
    f = None
    for path in _FONT_PATHS:
        if os.path.isfile(path):
            try:
                f = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
    if f is None:
        f = ImageFont.load_default()
    _font_cache[size] = f
    return f


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
                # 최종 출력에서 읽히도록 축소배율을 미리 보정한다
                fs = max(14, int(round(H / 42 / max(scale, 0.2))))
                _draw_rows(ImageDraw.Draw(img), tracks, rows, font_size=fs)
                n_drawn += 1
            d2 = ImageDraw.Draw(img)
            cap = f"{label or meta.get('model','')}  |  frame {f}  |  tracks {len(rows)}"
            cf = _font(max(16, int(round(H / 34 / max(scale, 0.2)))))
            try:
                cx0, cy0, cx1, cy1 = d2.textbbox((0, 0), cap, font=cf)
                cw, ch = cx1 - cx0, cy1 - cy0
            except AttributeError:
                cw, ch = d2.textsize(cap, font=cf)
            d2.rectangle([0, 0, cw + 20, ch + 16], fill=(0, 0, 0))
            d2.text((10, 8), cap, fill=(255, 255, 255), font=cf)
            if scale != 1.0:
                img = img.resize((ow, oh), Image.BILINEAR)
            enc.stdin.write(img.tobytes())
    finally:
        enc.stdin.close()
        enc.wait()
        dec.stdout.close()
        dec.wait()
    return out_path, n_drawn


def _luma(c):
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def _draw_id_chip(d, x, y, tid, col, font):
    """bbox 위에 트랙 ID 를 대비 높은 칩으로 올린다.

    박스 안에 같은 색으로 쓰면 영상 배경에 묻혀 읽히지 않는다(실측).
    트랙 색을 배경으로 채우고 글자는 명도에 따라 흑/백을 고른다.
    """
    text = str(tid)
    try:
        x0, y0, x1, y1 = d.textbbox((0, 0), text, font=font)
        tw, th = x1 - x0, y1 - y0
    except AttributeError:
        tw, th = d.textsize(text, font=font)
    pad = max(3, th // 4)
    cx0, cy0 = x, max(0, y - th - 2 * pad - 2)
    d.rectangle([cx0, cy0, cx0 + tw + 2 * pad, cy0 + th + 2 * pad],
                fill=col, outline=(0, 0, 0), width=2)
    fg = (0, 0, 0) if _luma(col) > 140 else (255, 255, 255)
    d.text((cx0 + pad, cy0 + pad), text, fill=fg, font=font)


def _draw_rows(d, tracks, rows, font_size=26):
    """한 프레임의 여러 트랙을 그린다 (draw_frame 과 공유)."""
    font = _font(font_size)
    for row in rows:
        tid = int(tracks["track_ids"][row])
        col = PALETTE[tid % len(PALETTE)]
        j = tracks["joints2d"][row]
        ok = np.isfinite(j).all(-1)

        b = tracks["bbox"][row]
        if b[2] > b[0]:
            d.rectangle([float(v) for v in b], outline=col, width=3)

        for a, c in SMPL_LINKS:
            if ok[a] and ok[c]:
                d.line([tuple(j[a]), tuple(j[c])], fill=col, width=3)
        for k in range(len(j)):
            if ok[k]:
                x, y = float(j[k][0]), float(j[k][1])
                r = 4 if k in LEFT_JOINTS else 3
                d.ellipse([x - r, y - r, x + r, y + r],
                          fill=(255, 255, 255), outline=col)

        # 라벨은 골격 위에 그려야 가려지지 않는다
        if b[2] > b[0]:
            _draw_id_chip(d, float(b[0]), float(b[1]), tid, col, font)


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
