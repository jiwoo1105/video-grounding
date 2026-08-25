"""ID 변경을 눈에 보이게 그린다.

일반 오버레이는 '모델이 부여한 트랙 ID' 를 보여주므로, 번호가 그대로인 채
가리키는 사람만 바뀌는 실패(정체 혼동)를 눈으로 잡을 수 없다.

이 도구는 **GT 사람을 기준으로** 그린다.
  - **박스**: GT 사람 고유 색·번호 P. 프레임 내내 불변 (정답이므로 성능과 무관)
  - **골격**: 그 사람에게 현재 배정된 **예측 트랙 T 의 색**.
             같은 사람 위에서 골격 색이 바뀌면 그것이 ID 변경이다.
  - 예측 ID 가 직전 프레임과 달라지면 화면에 경고를 띄운다
  - 하단 띠에 사람별 배정 이력을 색으로 깔아, 색이 끊기는 곳이 ID 변경

박스와 골격의 색 기준이 다른 것이 핵심이다 — 박스는 정답, 골격은 모델.
둘이 같은 기준이면 화면만 보고는 아무 문제 없어 보인다.

두 모델을 같은 방식으로 그리면 '어느 쪽이 더 자주 바뀌는가' 가 색 띠의
조각 수로 바로 보인다.
"""
import argparse
import io
import os
import subprocess
from collections import defaultdict

import numpy as np

from smpl_eval.evaluate import load_gt
from smpl_eval.metrics.occlusion import associate
from smpl_eval.overlay import PALETTE, SMPL_LINKS, LEFT_JOINTS, _font
from smpl_eval.schema import load_tracks

STRIP_H = 74          # 하단 타임라인 띠 높이(px)


def build_assignment(tracks, gt):
    """{gt_id: {frame: pred_id}} 와 프레임 목록."""
    frames = [int(f) for f in np.unique(gt["frame_ids"])]
    assign = defaultdict(dict)
    for f in frames:
        for g, p in associate(tracks, gt, f).items():
            assign[int(g)][f] = int(p)
    return assign, frames


def draw_strip(d, assign, frames, gt_ids, x0, y0, w, h, cur_frame, font):
    """사람별 배정 이력을 가로 띠로 그린다. 색이 바뀌는 곳이 ID 변경."""
    n = max(len(gt_ids), 1)
    row_h = max(4, (h - 16) // n)
    label_w = 42
    bar_w = w - label_w - 8
    for r, g in enumerate(gt_ids):
        y = y0 + 8 + r * row_h
        d.text((x0 + 4, y - 2), "P%d" % g, fill=(210, 214, 222), font=font)
        prev = None
        for i, f in enumerate(frames):
            p = assign.get(g, {}).get(f)
            x = x0 + label_w + int(bar_w * i / max(len(frames) - 1, 1))
            xn = x0 + label_w + int(bar_w * (i + 1) / max(len(frames) - 1, 1))
            if p is None:
                col = (58, 62, 70)                      # 놓친 구간
            else:
                col = PALETTE[p % len(PALETTE)]
            d.rectangle([x, y, max(xn, x + 1), y + row_h - 3], fill=col)
            if prev is not None and p != prev:           # 변경 지점에 흰 눈금
                d.rectangle([x - 1, y - 2, x + 1, y + row_h - 1], fill=(255, 255, 255))
            prev = p
    # 현재 위치 커서
    idx = frames.index(cur_frame) if cur_frame in frames else 0
    cx = x0 + label_w + int(bar_w * idx / max(len(frames) - 1, 1))
    d.rectangle([cx - 1, y0 + 2, cx + 1, y0 + h - 4], fill=(255, 255, 255))


def render(video_path, tracks_path, rec, out_path, label, scale=0.6, max_frames=None):
    from PIL import Image, ImageDraw

    tracks, meta = load_tracks(tracks_path)
    gt, _ = load_gt(rec)
    assign, frames = build_assignment(tracks, gt)
    gt_ids = sorted(assign)
    if max_frames:
        frames = [f for f in frames if f < max_frames]

    W, H = rec["width"], rec["height"]
    OH = H + STRIP_H
    ow, oh = int(W * scale), int(OH * scale)
    ow -= ow % 2
    oh -= oh % 2
    fps = meta.get("fps") or rec["fps"]

    gt_rows = defaultdict(list)
    for i in range(len(gt["frame_ids"])):
        gt_rows[int(gt["frame_ids"][i])].append(i)

    total = (max(frames) + 1) if frames else 0
    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", video_path, "-frames:v", str(total),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{ow}x{oh}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out_path],
        stdin=subprocess.PIPE)

    fs = max(15, int(round(H / 40 / max(scale, .2))))
    font = _font(fs)
    small = _font(max(12, int(fs * .62)))
    nbytes = W * H * 3
    prev_assign, n_changes = {}, 0

    try:
        for f in range(total):
            buf = dec.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            frame = Image.frombytes("RGB", (W, H), buf)
            canvas = Image.new("RGB", (W, OH), (18, 20, 25))
            canvas.paste(frame, (0, 0))
            d = ImageDraw.Draw(canvas)

            changed_now = []
            for row in gt_rows.get(f, []):
                g = int(gt["track_ids"][row])
                p = assign.get(g, {}).get(f)
                col = PALETTE[g % len(PALETTE)]          # 박스 = 정답 사람 색
                tcol = (110, 116, 128) if p is None else PALETTE[p % len(PALETTE)]
                b = gt["bbox"][row]
                if b[2] > b[0]:
                    d.rectangle([float(v) for v in b], outline=col, width=3)
                # 골격은 **배정된 예측 트랙 색** — 색이 바뀌면 ID 변경
                if p is not None:
                    sel = np.where((tracks["frame_ids"] == f)
                                   & (tracks["track_ids"] == p))[0]
                    if len(sel):
                        j = tracks["joints2d"][sel[0]]
                        ok = np.isfinite(j).all(-1)
                        for a, c in SMPL_LINKS:
                            if ok[a] and ok[c]:
                                d.line([tuple(j[a]), tuple(j[c])], fill=tcol, width=4)
                        for k in range(len(j)):
                            if ok[k]:
                                x, y = float(j[k][0]), float(j[k][1])
                                r = 4 if k in LEFT_JOINTS else 3
                                d.ellipse([x - r, y - r, x + r, y + r],
                                          fill=(255, 255, 255), outline=tcol)
                txt = "P%d <- %s" % (g, "?" if p is None else "T%d" % p)
                d.rectangle([b[0], b[1] - fs - 10, b[0] + len(txt) * fs * .62 + 10, b[1] - 2],
                            fill=col)
                d.text((b[0] + 5, b[1] - fs - 7), txt, fill=(0, 0, 0), font=font)
                if g in prev_assign and p is not None and prev_assign[g] != p:
                    changed_now.append(g)
                if p is not None:
                    prev_assign[g] = p

            if changed_now:
                n_changes += len(changed_now)
                warn = "ID 변경  " + " ".join("P%d" % g for g in changed_now)
                d.rectangle([0, H - fs * 2.6, len(warn) * fs * .7 + 24, H], fill=(200, 30, 30))
                d.text((12, H - fs * 2.2), warn, fill=(255, 255, 255), font=font)

            draw_strip(d, assign, frames, gt_ids, 0, H, W, STRIP_H, f, small)
            cap = ("%s  |  frame %d  |  누적 ID 변경 %d   "
                   "[박스=정답사람 P, 골격=모델트랙 T]") % (label, f, n_changes)
            d.rectangle([0, 0, len(cap) * fs * .62 + 20, fs + 14], fill=(0, 0, 0))
            d.text((10, 7), cap, fill=(255, 255, 255), font=font)

            if scale != 1.0:
                canvas = canvas.resize((ow, oh), Image.BILINEAR)
            enc.stdin.write(canvas.tobytes())
    finally:
        enc.stdin.close(); enc.wait()
        dec.stdout.close(); dec.wait()
    return out_path, n_changes


def main(argv=None):
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--scale", type=float, default=0.6)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--manifest", default="smpl_eval/manifest.json")
    a = ap.parse_args(argv)

    rec = next(r for r in json.load(open(a.manifest)) if r["dataset"] == a.dataset)
    p, n = render(rec["video_path"], a.tracks, rec, a.out, a.label,
                  a.scale, a.max_frames)
    print("wrote %s  (ID 변경 총 %d회)" % (p, n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
