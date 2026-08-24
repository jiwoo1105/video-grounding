"""bbox 기하 연산 공용 모듈.

identity.py 와 occlusion.py 가 같은 IoU 정의를 쓰도록 한 곳에 둔다.

py-motmetrics 1.4.0 의 `mm.distances.iou_matrix` 는 NumPy 2.0 에서 제거된
`np.asfarray` 를 사용해 동작하지 않는다 (최신 릴리스이며 수정본이 없음).
그래서 IoU 행렬을 직접 계산해 motmetrics 에 넘긴다 — motmetrics 의
MOT 집계 코어 자체는 NumPy 2 에서 정상 동작한다.
"""
import numpy as np


def iou(a, b):
    """단일 bbox 쌍의 IoU. 입력은 (x1, y1, x2, y2)."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return float(inter / union) if union > 0 else 0.0


def iou_matrix(boxes_a, boxes_b):
    """(Na, 4) x (Nb, 4) → (Na, Nb) IoU 행렬. 벡터화 구현."""
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)

    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    union = area_a + area_b - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(union > 0, inter / union, 0.0)
    return np.nan_to_num(out, nan=0.0)


def iou_distance_matrix(boxes_a, boxes_b, iou_thresh=0.5):
    """motmetrics 가 받는 거리 행렬. IoU 가 임계값 미만이면 NaN(매칭 불가)."""
    m = iou_matrix(boxes_a, boxes_b)
    d = 1.0 - m
    d[m < iou_thresh] = np.nan
    return d
