import numpy as np
from smpl_eval.tests.synth import make_tracks
from smpl_eval.metrics.identity import (
    id_metrics, person_count_error, gt_track_purity)


def test_perfect_match_gives_idf1_one():
    t = make_tracks(n_frames=50, n_tracks=3)
    m = id_metrics(t, t)
    assert m["idf1"] > 0.99
    assert m["num_switches"] == 0


def test_id_swap_is_detected():
    """스펙 게이트 5 — 지표 코드 자체가 ID 스왑을 잡아내는지 검증."""
    gt = make_tracks(n_frames=50, n_tracks=3, seed=1)
    pred = make_tracks(n_frames=50, n_tracks=3, seed=1, id_swap_at=(25, 0, 1))
    m = id_metrics(pred, gt)
    assert m["num_switches"] >= 2, f"ID 스왑을 못 잡음: {m}"
    assert m["idf1"] < 0.9


def test_missing_track_lowers_mota():
    gt = make_tracks(n_frames=50, n_tracks=3, seed=2)
    keep = gt["track_ids"] != 2
    pred = {k: v[keep] for k, v in gt.items()}
    m = id_metrics(pred, gt)
    assert m["mota"] < 0.7


def test_person_count_error_on_exact():
    t = make_tracks(n_frames=30, n_tracks=4)
    r = person_count_error(t, expected=4)
    assert r["count_mae"] == 0.0 and r["mean_count"] == 4.0


def test_person_count_error_detects_over_detection():
    t = make_tracks(n_frames=30, n_tracks=6)
    r = person_count_error(t, expected=4)
    assert r["count_mae"] == 2.0 and r["frames_over"] == 30


def test_purity_is_one_for_perfect_tracking():
    t = make_tracks(n_frames=40, n_tracks=3)
    r = gt_track_purity(t, t)
    assert r["mean_purity"] > 0.99 and r["min_purity"] > 0.99
    assert r["n_gt_tracks"] == 3


def test_purity_drops_when_gt_person_is_split_across_tracks():
    """한 사람을 여러 예측 트랙이 이어받으면 purity 가 떨어져야 한다."""
    gt = make_tracks(n_frames=40, n_tracks=2, seed=5)
    pred = {k: v.copy() for k, v in gt.items()}
    ids = pred["track_ids"].copy()
    ids[(pred["frame_ids"] >= 20) & (pred["track_ids"] == 0)] = 99   # 중간에 트랙 교체
    pred["track_ids"] = ids
    r = gt_track_purity(pred, gt)
    assert r["min_purity"] < 0.6, r["per_track"]


def test_purity_ignores_extra_detections_that_gt_lacks():
    """GT 에 없는 사람을 추가로 검출해도 purity 는 떨어지지 않아야 한다.

    본 데이터셋 GT 는 불완전하다 (테니스는 화면 9~10명 중 2명만 표시).
    IDF1 은 이 경우 올바른 검출을 오검출로 처벌한다.
    """
    gt = make_tracks(n_frames=40, n_tracks=2, seed=6)
    extra = make_tracks(n_frames=40, n_tracks=2, seed=6)
    extra["track_ids"] = extra["track_ids"] + 50
    extra["bbox"] = extra["bbox"] + 800.0          # 전혀 다른 위치
    pred = {k: np.concatenate([gt[k], extra[k]]) for k in gt}

    pure = gt_track_purity(pred, gt)
    assert pure["mean_purity"] > 0.99, pure["per_track"]
    # 대조 — 같은 상황에서 IDF1 은 떨어진다
    assert id_metrics(pred, gt)["idf1"] < 0.9
