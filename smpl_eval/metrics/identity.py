"""ID 유지 지표.

py-motmetrics 로 MOT 지표를 집계한다. IDF1 은 "한 사람에게 하나의
ID 를 계속 유지했는가"를 재는 지표로, 본 평가의 축 ②에 직접 대응한다.

거리 행렬은 metrics.geometry 로 직접 계산한다 — motmetrics 1.4.0 의
iou_matrix 가 NumPy 2 에서 동작하지 않기 때문 (geometry.py 주석 참조).
"""
import numpy as np
import motmetrics as mm

from smpl_eval.metrics.geometry import iou_distance_matrix

_METRICS = ["idf1", "mota", "motp", "num_switches", "num_fragmentations",
            "mostly_tracked", "mostly_lost"]
_INT_METRICS = {"num_switches", "num_fragmentations",
                "mostly_tracked", "mostly_lost"}


def id_metrics(pred, gt, iou_thresh=0.5):
    """GT 트랙과 예측 트랙을 프레임별 IoU 로 짝지어 MOT 지표를 낸다."""
    acc = mm.MOTAccumulator(auto_id=False)
    frames = np.union1d(np.unique(gt["frame_ids"]), np.unique(pred["frame_ids"]))
    for f in frames:
        g = gt["frame_ids"] == f
        p = pred["frame_ids"] == f
        gids = gt["track_ids"][g]
        pids = pred["track_ids"][p]
        if len(gids) and len(pids):
            d = iou_distance_matrix(gt["bbox"][g], pred["bbox"][p], iou_thresh)
        else:
            d = np.empty((len(gids), len(pids)))
        acc.update(gids, pids, d, frameid=int(f))

    s = mm.metrics.create().compute(acc, metrics=_METRICS, name="v")
    out = {}
    for k in s.columns:
        v = s[k].iloc[0]
        out[k] = int(v) if k in _INT_METRICS else float(v)
    return out


def person_count_error(pred, expected):
    """프레임별 검출 인원 vs 기지값. GT 트랙 없이도 쓸 수 있다."""
    frames, counts = np.unique(pred["frame_ids"], return_counts=True)
    return {
        "mean_count": float(counts.mean()),
        "count_mae": float(np.abs(counts - expected).mean()),
        "frames_over": int((counts > expected).sum()),
        "frames_under": int((counts < expected).sum()),
        "n_frames": int(len(frames)),
    }
