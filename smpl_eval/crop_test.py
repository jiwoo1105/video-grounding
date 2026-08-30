"""사람이 작게 찍힌 영상을 크롭해 다시 추론하고, 원본과 비교한다.

**왜 하는가** — golden(Data4) 은 1920x1080 에서 무용수 3명이 200x200 픽셀
영역만 차지한다. 사람 하나가 약 60x190px 인데, 두 모델 모두 입력을 정사각형
으로 줄이므로 (Multi-HMR 2 는 768) 사람이 76px 까지 작아진다. 메시가 뭉개진
것이 모델의 한계인지 입력 해상도의 한계인지 갈라야 한다.

    원본  1920x1080 -> 768 :  사람 190px -> 76px
    크롭   620x500  -> 768 :  사람 190px -> 235px   (약 3.1배)

**크롭 범위는 눈대중이 아니라 검출 결과로 정한다.** 원본 추론의 bbox 를
전 프레임에 대해 합집합하고 여백을 준다. 사람이 프레임 밖으로 잘리면
비교가 무의미해지므로 여백을 넉넉히 잡는다.

**GT 지표는 쓰지 않는다.** 크롭하면 카메라 내부 파라미터와 GT 좌표의 관계가
달라져 purity/PA-MPJPE 를 그대로 비교할 수 없다. 여기서는 GT 가 필요 없는
것만 본다 — 지터, 트랙 수, 검출 수, 그리고 눈으로 보는 메시 품질.
"""
import argparse
import json
import os
import subprocess

import numpy as np


def crop_from_tracks(tracks, W, H, margin=0.25, min_size=384):
    """검출 bbox 전체를 덮는 크롭 상자를 구한다.

    margin  사람 크기 대비 여백 비율. 팔다리를 뻗거나 머리카락이 날리는
            부분은 bbox 밖으로 나가므로 넉넉히 준다.
    min_size 너무 좁게 잘라 모델이 오히려 어색해지는 것을 막는 하한.
    """
    b = tracks["bbox"]
    x1, y1 = float(np.nanmin(b[:, 0])), float(np.nanmin(b[:, 1]))
    x2, y2 = float(np.nanmax(b[:, 2])), float(np.nanmax(b[:, 3]))
    bw, bh = x2 - x1, y2 - y1
    mx, my = bw * margin, bh * margin
    x1, y1, x2, y2 = x1 - mx, y1 - my, x2 + mx, y2 + my

    # 하한 적용 (중심 유지)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w = max(x2 - x1, min_size)
    h = max(y2 - y1, min_size)
    x1, y1 = cx - w / 2, cy - h / 2

    # 화면 안으로 밀어넣고 짝수로 맞춘다 (libx264 요구)
    w = min(int(round(w)), W)
    h = min(int(round(h)), H)
    w -= w % 2
    h -= h % 2
    x = int(round(min(max(x1, 0), W - w)))
    y = int(round(min(max(y1, 0), H - h)))
    return x, y, w, h


def make_crop(video, out, x, y, w, h):
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", video,
         "-vf", f"crop={w}:{h}:{x}:{y}",
         "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", out],
        check=True)
    return out


def person_pixels(tracks):
    """검출된 사람의 화면상 크기 — 크롭 효과를 재는 직접적인 값."""
    b = tracks["bbox"]
    hgt = b[:, 3] - b[:, 1]
    hgt = hgt[np.isfinite(hgt) & (hgt > 0)]
    return {"median_h": float(np.median(hgt)), "mean_h": float(hgt.mean()),
            "n": int(len(hgt))}


def summarize(tracks, meta, fps):
    from smpl_eval.metrics.plausibility import acceleration_jitter
    j = acceleration_jitter(tracks, fps)
    ids, cnt = np.unique(tracks["track_ids"], return_counts=True)
    nf = int(tracks["frame_ids"].max()) + 1
    px = person_pixels(tracks)
    return {
        "검출": int(len(tracks["frame_ids"])),
        "트랙": int(len(ids)),
        "프레임당 인원": round(len(tracks["frame_ids"]) / max(nf, 1), 2),
        "사람 높이(px, 중앙)": round(px["median_h"], 1),
        "가속 xy": round(j["mean_accel_xy"], 2),
        "가속 z": round(j["mean_accel_z"], 2),
        "깊이 비중": round(j["depth_share"], 4),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="크롭 전후 비교")
    ap.add_argument("--video", required=True)
    ap.add_argument("--orig-tracks", required=True,
                    help="원본 영상 추론 결과 — 여기서 크롭 범위를 계산한다")
    ap.add_argument("--out-video", required=True)
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--min-size", type=int, default=384)
    ap.add_argument("--print-crop-only", action="store_true")
    a = ap.parse_args(argv)

    from smpl_eval.schema import load_tracks
    tracks, meta = load_tracks(a.orig_tracks)
    W, H = meta.get("resolution", [1920, 1080])
    x, y, w, h = crop_from_tracks(tracks, W, H, a.margin, a.min_size)

    px = person_pixels(tracks)
    print(f"원본 {W}x{H}  사람 높이 중앙값 {px['median_h']:.0f}px")
    print(f"크롭 {w}x{h} @ ({x},{y})")
    print(f"  화면 대비 면적 {100*w*h/(W*H):.1f}%")
    print(f"  모델 입력 768 기준 사람 높이:")
    print(f"    원본 {px['median_h']*768/max(W,H):.0f}px"
          f"  ->  크롭 {px['median_h']*768/max(w,h):.0f}px"
          f"  ({max(W,H)/max(w,h):.1f}배)")
    if a.print_crop_only:
        print(json.dumps({"x": x, "y": y, "w": w, "h": h}))
        return
    make_crop(a.video, a.out_video, x, y, w, h)
    print(f"저장 {a.out_video}")
    with open(os.path.splitext(a.out_video)[0] + "_crop.json", "w") as f:
        json.dump({"x": x, "y": y, "w": w, "h": h, "src_w": W, "src_h": H}, f)


if __name__ == "__main__":
    main()
