import numpy as np
from smpl_eval.tests.synth import make_tracks
from smpl_eval.metrics.occlusion import (
    find_occlusion_events, id_retention_around_events)


def _crossing(n_frames=60):
    """두 사람이 x축에서 교차해 중간에 bbox 가 크게 겹치는 시퀀스."""
    t = make_tracks(n_frames=n_frames, n_tracks=2, seed=7)
    f = t["frame_ids"]; k = t["track_ids"]
    for i in range(len(f)):
        shift = (f[i] - n_frames / 2) / (n_frames / 2) * 300.0
        s = shift if k[i] == 0 else -shift
        t["bbox"][i] = [500 + s, 100, 700 + s, 600]
    return t


def test_no_events_when_far_apart():
    t = make_tracks(n_frames=40, n_tracks=2)
    t["bbox"][t["track_ids"] == 0] = [0, 0, 100, 200]
    t["bbox"][t["track_ids"] == 1] = [900, 0, 1000, 200]
    assert find_occlusion_events(t) == []


def test_crossing_produces_one_event():
    ev = find_occlusion_events(_crossing())
    assert len(ev) == 1, ev
    assert ev[0]["peak_iou"] > 0.3
    assert ev[0]["track_a"] == 0 and ev[0]["track_b"] == 1


def test_event_is_centred_on_the_crossing():
    ev = find_occlusion_events(_crossing(60))[0]
    mid = (ev["start_frame"] + ev["end_frame"]) / 2
    assert abs(mid - 30) < 5, ev


def test_retention_is_one_when_ids_kept():
    t = _crossing()
    ev = find_occlusion_events(t)
    assert id_retention_around_events(t, t, ev)["retention_rate"] == 1.0


def test_retention_drops_when_ids_swap_at_event():
    gt = _crossing()
    ev = find_occlusion_events(gt)
    mid = (ev[0]["start_frame"] + ev[0]["end_frame"]) // 2
    pred = {k: v.copy() for k, v in gt.items()}
    sel = pred["frame_ids"] >= mid
    ids = pred["track_ids"].copy()
    ids[sel & (pred["track_ids"] == 0)] = 1
    ids[sel & (pred["track_ids"] == 1)] = 0
    pred["track_ids"] = ids
    assert id_retention_around_events(pred, gt, ev)["retention_rate"] < 1.0


def test_no_events_gives_nan_rate_not_crash():
    t = make_tracks(n_frames=10, n_tracks=1)
    r = id_retention_around_events(t, t, [])
    assert r["n_events"] == 0 and np.isnan(r["retention_rate"])
