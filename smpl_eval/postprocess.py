"""tracks.npz 후처리 — 모델을 건드리지 않고 알려진 실패를 보정한다.

현재 제공하는 것은 **2D 화면공간 NMS** 하나다.

왜 필요한가 (실측 근거, 2026-08-25):
Multi-HMR 2 의 NMS 는 3D 골반 거리 기준이다 (`dist_thresh_nms=0.25 m`).
단안 추정에서 깊이는 본질적으로 부정확해서, 같은 사람을 두 번 검출했을 때
화면상 위치는 거의 같아도(2D IoU 중앙값 0.55) 예측 깊이가 2m 벌어져
3D 거리가 임계값을 100% 통과한다. 그 결과 한 사람에 두 개의 트랙이 붙는다.

  농구 Data1: 2D 겹침>0.3 인 쌍 1,030개
    수평(xy) 거리 중앙값 0.187 m  /  깊이(z) 거리 중앙값 1.996 m
    깊이 차가 수평 차보다 큰 경우 99%

`dist_thresh_nms` 를 키우는 것은 답이 아니다 — 2m 떨어져 서 있는 실제
다른 사람까지 지워버린다. 화면공간에서 판정해야 한다.
"""
import argparse
import glob
import os

import numpy as np

from smpl_eval.metrics.geometry import iou_matrix
from smpl_eval.schema import load_tracks, save_tracks


def nms_2d(tracks, iou_thresh=0.55, keep="score"):
    """프레임마다 2D bbox 가 크게 겹치는 검출 중 하나만 남긴다.

    keep="score" 면 신뢰도가 높은 쪽, "longest" 면 더 긴 트랙을 남긴다.
    같은 사람에 붙은 중복 트랙을 지우는 것이 목적이므로 기본은 트랙 길이다
    (짧은 유령 트랙보다 오래 유지된 트랙이 실제 사람일 가능성이 높다).

    반환: (필터된 tracks, 제거된 행 수)
    """
    n = len(tracks["frame_ids"])
    track_len = {}
    for t in np.unique(tracks["track_ids"]):
        track_len[int(t)] = int((tracks["track_ids"] == t).sum())

    drop = np.zeros(n, bool)
    for f in np.unique(tracks["frame_ids"]):
        idx = np.where(tracks["frame_ids"] == f)[0]
        if len(idx) < 2:
            continue
        m = iou_matrix(tracks["bbox"][idx], tracks["bbox"][idx])
        np.fill_diagonal(m, 0.0)
        if keep == "score":
            rank = tracks["score"][idx]
        else:
            rank = np.array([track_len[int(tracks["track_ids"][i])] for i in idx],
                            dtype=float)
        order = np.argsort(-rank)                 # 좋은 것부터
        alive = np.ones(len(idx), bool)
        for a in order:
            if not alive[a]:
                continue
            for b in range(len(idx)):
                if b != a and alive[b] and m[a, b] >= iou_thresh:
                    alive[b] = False
        drop[idx[~alive]] = True

    if not drop.any():
        return tracks, 0
    keep_mask = ~drop
    return {k: v[keep_mask] for k, v in tracks.items()}, int(drop.sum())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True, help="입력 tracks.npz (glob 가능)")
    ap.add_argument("--suffix", default="_nms2d",
                    help="출력 파일명 접미사. 원본은 보존한다")
    ap.add_argument("--iou", type=float, default=0.55)
    ap.add_argument("--keep", default="longest", choices=["score", "longest"])
    a = ap.parse_args(argv)

    for p in sorted(glob.glob(a.tracks)):
        arrays, meta = load_tracks(p)
        out, n_drop = nms_2d(arrays, a.iou, a.keep)
        before = len(np.unique(arrays["track_ids"]))
        after = len(np.unique(out["track_ids"]))
        meta = dict(meta)
        meta.update({"postprocess": "nms_2d", "nms2d_iou": a.iou,
                     "nms2d_keep": a.keep, "nms2d_dropped_rows": n_drop})
        dst = p.replace(".npz", a.suffix + ".npz")
        save_tracks(dst, out, meta)
        print("  %s\n    행 %d -> %d (제거 %d)  트랙 %d -> %d"
              % (os.path.relpath(dst), len(arrays["frame_ids"]),
                 len(out["frame_ids"]), n_drop, before, after))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
