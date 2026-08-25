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


def gt_track_purity(pred, gt):
    """GT 로 표시된 사람마다 하나의 예측 트랙이 일관되게 따라붙는가.

    **GT 가 불완전할 때 IDF1 을 대체하는 지표다.**
    본 데이터셋의 GT 는 화면의 모든 사람을 담지 않는다 (테니스는 선수
    2명만 표시하는데 화면에는 볼키즈·심판·관중 등 9~10명이 있다).
    IDF1 은 GT 에 없는 사람을 검출하면 오검출로 처벌하므로, 모델이
    **올바르게** 더 많은 사람을 찾을수록 점수가 낮아진다.

    purity 는 GT 트랙마다 "가장 많이 대응된 예측 트랙이 전체 프레임의
    몇 %를 차지하는가" 를 재므로 GT 에 없는 검출에 영향을 받지 않는다.

    반환:
      mean_purity   GT 트랙별 purity 의 평균 (1.0 = 완벽)
      min_purity    가장 나쁜 GT 트랙
      mean_coverage GT 트랙이 예측과 매칭된 프레임 비율
      n_gt_tracks   GT 트랙 수
      per_track     {gt_id: {purity, coverage, dominant_pred, n_pred_tracks}}
    """
    from collections import Counter

    from smpl_eval.metrics.occlusion import associate

    frames = np.unique(gt["frame_ids"])
    hits = {int(g): Counter() for g in np.unique(gt["track_ids"])}
    present = {int(g): 0 for g in hits}
    for f in frames:
        f = int(f)
        for g in np.unique(gt["track_ids"][gt["frame_ids"] == f]):
            present[int(g)] += 1
        for g, p in associate(pred, gt, f).items():
            hits[g][p] += 1

    per_track, purities, coverages = {}, [], []
    for g, c in hits.items():
        n_present = present[g] or 1
        matched = sum(c.values())
        dom, dom_n = (c.most_common(1)[0] if c else (None, 0))
        purity = dom_n / n_present
        coverage = matched / n_present
        per_track[g] = {"purity": float(purity), "coverage": float(coverage),
                        "dominant_pred": dom, "n_pred_tracks": len(c)}
        purities.append(purity)
        coverages.append(coverage)

    if not purities:
        return {"mean_purity": float("nan"), "min_purity": float("nan"),
                "mean_coverage": float("nan"), "n_gt_tracks": 0, "per_track": {}}
    return {"mean_purity": float(np.mean(purities)),
            "min_purity": float(np.min(purities)),
            "mean_coverage": float(np.mean(coverages)),
            "n_gt_tracks": len(purities), "per_track": per_track}


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
