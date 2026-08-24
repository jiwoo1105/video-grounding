import numpy as np
from smpl_eval.metrics.geometry import iou, iou_matrix, iou_distance_matrix


def test_identical_boxes_iou_one():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_disjoint_boxes_iou_zero():
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_half_overlap():
    # 두 10x10 이 x 로 5 겹침 → inter=50, union=150
    assert abs(iou([0, 0, 10, 10], [5, 0, 15, 10]) - 50 / 150) < 1e-9


def test_matrix_matches_scalar():
    a = [[0, 0, 10, 10], [5, 5, 15, 15]]
    b = [[0, 0, 10, 10], [20, 20, 30, 30], [5, 0, 15, 10]]
    m = iou_matrix(a, b)
    assert m.shape == (2, 3)
    for i in range(2):
        for j in range(3):
            assert abs(m[i, j] - iou(a[i], b[j])) < 1e-9


def test_matrix_handles_empty():
    assert iou_matrix([], [[0, 0, 1, 1]]).shape == (0, 1)
    assert iou_matrix([[0, 0, 1, 1]], []).shape == (1, 0)


def test_zero_area_box_does_not_crash():
    assert iou_matrix([[0, 0, 0, 0]], [[0, 0, 10, 10]])[0, 0] == 0.0


def test_distance_matrix_masks_below_threshold():
    a = [[0, 0, 10, 10]]
    b = [[0, 0, 10, 10], [9, 9, 19, 19]]
    d = iou_distance_matrix(a, b, iou_thresh=0.5)
    assert d[0, 0] == 0.0          # 완전 일치 → 거리 0
    assert np.isnan(d[0, 1])       # 거의 안 겹침 → 매칭 불가
