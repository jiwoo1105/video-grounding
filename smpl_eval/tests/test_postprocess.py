import numpy as np
from smpl_eval.tests.synth import make_tracks
from smpl_eval.postprocess import nms_2d
from smpl_eval.schema import validate_tracks


def _add_duplicate(t, of_track=0, depth_offset=2.0):
    """같은 사람을 깊이만 다르게 한 번 더 검출한 상황을 만든다.

    Multi-HMR 2 의 실제 실패 양상 — 2D 로는 겹치는데 3D 골반 거리가
    커서 모델 내부 NMS 를 통과한다.
    """
    sel = t["track_ids"] == of_track
    dup = {k: v[sel].copy() for k, v in t.items()}
    dup["track_ids"] = np.full(sel.sum(), 99, np.int32)
    dup["transl"] = dup["transl"] + np.array([0, 0, depth_offset], np.float32)
    dup["score"] = dup["score"] * 0.5
    return {k: np.concatenate([t[k], dup[k]]) for k in t}


def test_duplicate_at_same_image_position_is_removed():
    t = make_tracks(n_frames=30, n_tracks=2, seed=3)
    dirty = _add_duplicate(t)
    assert len(np.unique(dirty["track_ids"])) == 3
    clean, dropped = nms_2d(dirty, iou_thresh=0.55, keep="longest")
    assert dropped == 30
    assert set(np.unique(clean["track_ids"]).tolist()) == {0, 1}
    validate_tracks(clean)


def test_distinct_people_are_not_merged():
    """멀리 떨어진 진짜 다른 사람은 지우면 안 된다."""
    t = make_tracks(n_frames=30, n_tracks=3, seed=4)
    clean, dropped = nms_2d(t, iou_thresh=0.55)
    assert dropped == 0
    assert len(np.unique(clean["track_ids"])) == 3


def test_keep_score_prefers_higher_confidence():
    t = make_tracks(n_frames=20, n_tracks=1, seed=5)
    dirty = _add_duplicate(t)          # 복제본 score 는 절반
    clean, _ = nms_2d(dirty, iou_thresh=0.55, keep="score")
    assert set(np.unique(clean["track_ids"]).tolist()) == {0}


def test_output_passes_schema():
    t = make_tracks(n_frames=20, n_tracks=2)
    clean, _ = nms_2d(_add_duplicate(t))
    validate_tracks(clean)
