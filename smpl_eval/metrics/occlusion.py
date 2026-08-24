"""가림 구간 분석.

사람이 겹치는 순간이 ID 스왑이 실제로 일어나는 지점이다. 전체 IDF1 만
보면 "어디서 무너졌는지"를 알 수 없으므로, bbox IoU 로 가림 이벤트를
자동 검출하고 그 전/후에서 ID 대응이 유지되는지를 따로 측정한다.
"""
import numpy as np
from collections import defaultdict

from smpl_eval.metrics.geometry import iou, iou_matrix


def find_occlusion_events(tracks, iou_thresh=0.3, min_len=3):
    """IoU 가 임계값을 min_len 프레임 이상 연속으로 넘는 트랙쌍 구간."""
    per_frame = defaultdict(lambda: ([], []))
    for i in range(len(tracks["frame_ids"])):
        ids, boxes = per_frame[int(tracks["frame_ids"][i])]
        ids.append(int(tracks["track_ids"][i]))
        boxes.append(tracks["bbox"][i])

    hot = defaultdict(list)                      # (a, b) → [(frame, iou), ...]
    for f in sorted(per_frame):
        ids, boxes = per_frame[f]
        if len(ids) < 2:
            continue
        m = iou_matrix(boxes, boxes)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if m[i, j] >= iou_thresh:
                    key = (min(ids[i], ids[j]), max(ids[i], ids[j]))
                    hot[key].append((f, float(m[i, j])))

    events = []
    for (ta, tb), hits in hot.items():
        hits.sort()
        run = [hits[0]]
        for cur in hits[1:]:
            if cur[0] == run[-1][0] + 1:
                run.append(cur)
            else:
                _emit(events, ta, tb, run, min_len)
                run = [cur]
        _emit(events, ta, tb, run, min_len)
    return sorted(events, key=lambda e: (e["start_frame"], e["track_a"]))


def _emit(events, ta, tb, run, min_len):
    if len(run) >= min_len:
        events.append({"start_frame": run[0][0], "end_frame": run[-1][0],
                       "track_a": ta, "track_b": tb,
                       "peak_iou": float(max(v for _f, v in run))})


def id_retention_around_events(pred, gt, events, margin=10):
    """이벤트 전/후 margin 프레임에서 GT→예측 ID 대응이 유지되는가."""
    retained = 0
    for ev in events:
        before = associate(pred, gt, ev["start_frame"] - margin)
        after = associate(pred, gt, ev["end_frame"] + margin)
        common = set(before) & set(after)
        if common and all(before[g] == after[g] for g in common):
            retained += 1
    n = len(events)
    return {"n_events": n, "retained": retained,
            "retention_rate": float(retained / n) if n else float("nan")}


def associate(pred, gt, frame):
    """해당 프레임에서 GT track → 예측 track 의 최근접 bbox 대응."""
    g = gt["frame_ids"] == frame
    p = pred["frame_ids"] == frame
    out = {}
    if not g.any() or not p.any():
        return out
    gb, gi = gt["bbox"][g], gt["track_ids"][g]
    pb, pi = pred["bbox"][p], pred["track_ids"][p]
    for k in range(len(gi)):
        ious = [iou(gb[k], pb[m]) for m in range(len(pi))]
        best = int(np.argmax(ious))
        if ious[best] > 0:
            out[int(gi[k])] = int(pi[best])
    return out


# metrics/pose.py 가 프레임 매칭에 쓰는 별칭
_assoc = associate
