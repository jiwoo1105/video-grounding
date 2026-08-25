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


def _scaled_duplicate(t, of_track=0, scale=0.75):
    """같은 사람을 더 작게 한 번 더 검출한 상황.

    깊이 오차가 투영 크기를 바꾸면 IoU 가 떨어져 IoU 기준 NMS 를 통과한다
    (테니스 실측: IoU 0.385, IoS 0.771, 면적비 0.56).
    """
    sel = t["track_ids"] == of_track
    dup = {k: v[sel].copy() for k, v in t.items()}
    dup["track_ids"] = np.full(int(sel.sum()), 98, np.int32)
    b = dup["bbox"]
    cx = (b[:, 0] + b[:, 2]) / 2
    cy = (b[:, 1] + b[:, 3]) / 2
    hw = (b[:, 2] - b[:, 0]) / 2 * scale
    hh = (b[:, 3] - b[:, 1]) / 2 * scale
    dup["bbox"] = np.stack([cx - hw, cy - hh, cx + hw, cy + hh], -1).astype(np.float32)
    return {k: np.concatenate([t[k], dup[k]]) for k in t}


def test_iou_alone_misses_scaled_duplicate():
    t = make_tracks(n_frames=20, n_tracks=2, seed=11)
    dirty = _scaled_duplicate(t, scale=0.7)
    _, dropped = nms_2d(dirty, iou_thresh=0.55, ios_thresh=None)
    assert dropped == 0, "IoU 만으로는 크기가 다른 중복을 못 잡는다"


def test_ios_rule_catches_scaled_duplicate():
    t = make_tracks(n_frames=20, n_tracks=2, seed=11)
    dirty = _scaled_duplicate(t, scale=0.7)
    clean, dropped = nms_2d(dirty, iou_thresh=0.55, ios_thresh=0.70)
    assert dropped == 20
    assert set(np.unique(clean["track_ids"]).tolist()) == {0, 1}


def test_offset_guard_keeps_small_person_inside_large_box():
    """큰 박스 안 한쪽에 치우친 작은 사람은 지우면 안 된다."""
    t = make_tracks(n_frames=20, n_tracks=1, seed=12)
    other = {k: v[:20].copy() for k, v in t.items()}
    other["track_ids"] = np.full(20, 97, np.int32)
    b = t["bbox"][:20]
    w = b[:, 2] - b[:, 0]
    h = b[:, 3] - b[:, 1]
    # 큰 박스의 좌상단 구석에 작은 박스를 놓는다 (동심이 아님)
    other["bbox"] = np.stack([b[:, 0] + 0.02 * w, b[:, 1] + 0.02 * h,
                              b[:, 0] + 0.30 * w, b[:, 1] + 0.30 * h],
                             -1).astype(np.float32)
    dirty = {k: np.concatenate([t[k], other[k]]) for k in t}
    _, dropped = nms_2d(dirty, iou_thresh=0.55, ios_thresh=0.70,
                        center_offset_thresh=0.25)
    assert dropped == 0, "동심이 아닌 포함관계는 다른 사람일 수 있다"
