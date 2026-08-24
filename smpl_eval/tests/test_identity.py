import numpy as np
from smpl_eval.tests.synth import make_tracks
from smpl_eval.metrics.identity import id_metrics, person_count_error


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
