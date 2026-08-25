"""tracks.npz 후처리 — 모델을 건드리지 않고 알려진 실패를 보정한다.

현재 제공하는 것은 **2D 화면공간 중복 제거** 하나다.

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

### IoU 만으로는 부족하다

깊이 오차는 **투영 크기**도 바꾼다. 같은 사람이 5.8m 다른 깊이로 두 번
검출되면 한쪽 박스가 다른쪽의 56% 크기가 되고, IoU 는 0.385 로 떨어져
임계값을 통과한다 (테니스 프레임 1097 의 id45/id51 실측).

그래서 **IoS(작은 박스 대비 겹침)** 를 함께 본다. 위 사례의 IoS 는 0.771
이다. 다만 IoS 만 쓰면 가까운 사람 뒤에 서 있는 **진짜 다른 사람**도
지워질 수 있으므로, 두 박스가 **동심**인지(중심 offset) 함께 확인한다.
같은 사람의 중복은 동심이고, 뒤에 선 사람은 큰 박스 안 한쪽에 치우친다.
"""
import argparse
import glob
import os

import numpy as np

from smpl_eval.metrics.geometry import iou_matrix
from smpl_eval.schema import load_tracks, save_tracks


def _pair_stats(b):
    """(IoS 행렬, 중심 offset 행렬).

    IoS    = 교집합 / 작은 쪽 면적. 크기가 달라도 포함관계를 잡는다.
    offset = 두 중심 거리 / 작은 쪽 대각선. 0 에 가까우면 동심.
    """
    n = len(b)
    S = np.zeros((n, n))
    O = np.full((n, n), np.inf)
    if n < 2:
        return S, O
    x1 = np.maximum(b[:, None, 0], b[None, :, 0])
    y1 = np.maximum(b[:, None, 1], b[None, :, 1])
    x2 = np.minimum(b[:, None, 2], b[None, :, 2])
    y2 = np.minimum(b[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    small = np.maximum(np.minimum(area[:, None], area[None, :]), 1e-6)
    S = inter / small

    cx = (b[:, 0] + b[:, 2]) / 2.0
    cy = (b[:, 1] + b[:, 3]) / 2.0
    dist = np.hypot(cx[:, None] - cx[None, :], cy[:, None] - cy[None, :])
    dw = np.minimum(b[:, None, 2] - b[:, None, 0], b[None, :, 2] - b[None, :, 0])
    dh = np.minimum(b[:, None, 3] - b[:, None, 1], b[None, :, 3] - b[None, :, 1])
    O = dist / np.maximum(np.hypot(dw, dh), 1e-6)
    np.fill_diagonal(S, 0.0)
    np.fill_diagonal(O, np.inf)
    return S, O


def nms_2d(tracks, iou_thresh=0.55, keep="longest",
           ios_thresh=0.70, center_offset_thresh=0.25):
    """프레임마다 같은 사람에 붙은 중복 검출 중 하나만 남긴다.

    중복 판정은 둘 중 하나를 만족할 때다.
      (1) 2D IoU >= iou_thresh                      — 크기가 비슷한 중복
      (2) IoS >= ios_thresh **그리고** 중심 offset < center_offset_thresh
                                                     — 크기가 다른 동심 중복

    (2) 의 동심 조건이 없으면 가까운 사람 뒤에 서 있는 진짜 다른 사람도
    지워진다. ios_thresh=None 이면 (2) 를 끈다.

    keep="longest" 면 더 긴 트랙, "score" 면 신뢰도가 높은 쪽을 남긴다.
    기본이 트랙 길이인 이유는 짧은 유령 트랙보다 오래 유지된 트랙이
    실제 사람일 가능성이 높기 때문이다.

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
        b = np.asarray(tracks["bbox"][idx], float)
        m = iou_matrix(b, b)
        np.fill_diagonal(m, 0.0)
        dup = m >= iou_thresh
        if ios_thresh is not None:
            S, O = _pair_stats(b)
            dup = dup | ((S >= ios_thresh) & (O < center_offset_thresh))
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
            for c in range(len(idx)):
                if c != a and alive[c] and dup[a, c]:
                    alive[c] = False
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
    ap.add_argument("--ios", type=float, default=0.70,
                    help="작은박스 대비 겹침 임계. 0 이면 이 규칙을 끔")
    ap.add_argument("--center-offset", type=float, default=0.25,
                    help="동심 판정 임계 (작은박스 대각선 대비)")
    ap.add_argument("--keep", default="longest", choices=["score", "longest"])
    a = ap.parse_args(argv)

    for p in sorted(glob.glob(a.tracks)):
        arrays, meta = load_tracks(p)
        out, n_drop = nms_2d(arrays, a.iou, a.keep,
                             ios_thresh=(a.ios if a.ios > 0 else None),
                             center_offset_thresh=a.center_offset)
        before = len(np.unique(arrays["track_ids"]))
        after = len(np.unique(out["track_ids"]))
        meta = dict(meta)
        meta.update({"postprocess": "nms_2d", "nms2d_iou": a.iou,
                     "nms2d_ios": a.ios, "nms2d_center_offset": a.center_offset,
                     "nms2d_keep": a.keep, "nms2d_dropped_rows": n_drop})
        dst = p.replace(".npz", a.suffix + ".npz")
        save_tracks(dst, out, meta)
        print("  %s\n    행 %d -> %d (제거 %d)  트랙 %d -> %d"
              % (os.path.relpath(dst), len(arrays["frame_ids"]),
                 len(out["frame_ids"]), n_drop, before, after))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
