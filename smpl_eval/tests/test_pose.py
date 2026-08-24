import numpy as np
from smpl_eval.conventions import (
    SMPL24, H36M17, COCO17, H36M17_TO_SMPL24, COCO17_TO_SMPL24, common_indices)
from smpl_eval.metrics.pose import pa_mpjpe, mpjpe, reprojection_error


def test_smpl24_has_24_named_joints():
    assert len(SMPL24) == 24
    assert SMPL24[0] == "pelvis" and SMPL24[23] == "right_hand"


def test_h36m17_has_17_joints_in_bvh_order():
    assert len(H36M17) == 17
    assert H36M17[0] == "Hip" and H36M17[16] == "RightWrist"
    assert H36M17[10] == "Head"


def test_coco17_has_17_joints():
    assert len(COCO17) == 17 and COCO17[0] == "nose"


def test_mapping_targets_are_valid_indices():
    for src, table, n_src in (("h36m", H36M17_TO_SMPL24, 17),
                              ("coco", COCO17_TO_SMPL24, 17)):
        for g, s in table.items():
            assert 0 <= g < n_src, (src, g)
            assert 0 <= s < 24, (src, s)


def test_mapping_targets_are_unique():
    for table in (H36M17_TO_SMPL24, COCO17_TO_SMPL24):
        v = list(table.values())
        assert len(v) == len(set(v)), "두 GT 관절이 같은 SMPL 슬롯에 매핑됨"


def test_common_indices_are_aligned_and_same_length():
    gi, si = common_indices(H36M17_TO_SMPL24)
    assert len(gi) == len(si) == 17
    gi2, si2 = common_indices(COCO17_TO_SMPL24)
    assert len(gi2) == len(si2) == 12   # 얼굴 5개는 SMPL 대응 없음


def test_pa_mpjpe_is_zero_for_identical():
    j = np.random.RandomState(0).randn(5, 14, 3)
    assert np.allclose(pa_mpjpe(j, j), 0, atol=1e-6)


def test_pa_mpjpe_invariant_to_rigid_transform_and_scale():
    rs = np.random.RandomState(1)
    j = rs.randn(4, 14, 3)
    q, _ = np.linalg.qr(rs.randn(3, 3))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    moved = (j * 2.5) @ q.T + np.array([10.0, -3.0, 7.0])
    assert np.allclose(pa_mpjpe(moved, j), 0, atol=1e-6)


def test_pa_mpjpe_nonzero_for_deformation():
    rs = np.random.RandomState(3)
    j = rs.randn(3, 14, 3)
    bad = j.copy(); bad[:, 5] += 0.5
    assert pa_mpjpe(bad, j).mean() > 1.0


def test_mpjpe_zero_after_root_align():
    rs = np.random.RandomState(2)
    j = rs.randn(3, 14, 3)
    assert np.allclose(mpjpe(j + 5.0, j), 0, atol=1e-6)


def test_mpjpe_is_in_millimetres():
    j = np.zeros((1, 5, 3)); bad = j.copy()
    bad[0, 1:] = 0.001              # 1 mm 이동 (루트 제외 4개)
    assert abs(mpjpe(bad, j)[0] - np.sqrt(3) * 4 / 5) < 1e-6


def test_reprojection_error_in_pixels():
    a = np.zeros((2, 10, 2)); b = np.zeros((2, 10, 2)); b[..., 0] = 3.0
    assert np.allclose(reprojection_error(a, b), 3.0)


# ── pose_metrics: 실제 매칭 경로 ────────────────────────────────────

def _pair(mapping, n_frames=10, offset=0.0, seed=11):
    """GT 와 예측을 같은 골격에서 만들되 예측만 살짝 어긋나게 한다."""
    from smpl_eval.tests.synth import make_tracks
    from smpl_eval.conventions import common_indices
    gi, si = common_indices(mapping)
    pred = make_tracks(n_frames=n_frames, n_tracks=2, seed=seed)
    gt = {k: v.copy() for k, v in pred.items()}
    # GT 는 매핑된 SMPL 슬롯만 값을 갖고 나머지는 NaN (실제 GT 와 같은 형태)
    j = np.full_like(gt["joints3d"], np.nan)
    j[:, si] = gt["joints3d"][:, si]
    gt["joints3d"] = j
    if offset:
        pred = {k: v.copy() for k, v in pred.items()}
        pred["joints3d"] = pred["joints3d"] + offset
    return pred, gt


def test_pose_metrics_matches_every_frame():
    from smpl_eval.metrics.pose import pose_metrics
    pred, gt = _pair(H36M17_TO_SMPL24, n_frames=10)
    r = pose_metrics(pred, gt, H36M17_TO_SMPL24)
    assert r["n_matched"] == 20          # 10프레임 x 2명
    assert r["pa_mpjpe"] < 1e-3


def test_pose_metrics_mpjpe_available_with_h36m_pelvis():
    from smpl_eval.metrics.pose import pose_metrics
    pred, gt = _pair(H36M17_TO_SMPL24)
    r = pose_metrics(pred, gt, H36M17_TO_SMPL24)
    assert r["mpjpe_available"] is True
    assert r["mpjpe"] < 1e-3


def test_pose_metrics_mpjpe_unavailable_for_coco_without_pelvis():
    """Data1 규약에는 골반이 없다 — MPJPE 를 조용히 틀리게 내지 말 것."""
    from smpl_eval.metrics.pose import pose_metrics
    pred, gt = _pair(COCO17_TO_SMPL24)
    r = pose_metrics(pred, gt, COCO17_TO_SMPL24)
    assert r["mpjpe_available"] is False
    assert np.isnan(r["mpjpe"])
    assert r["pa_mpjpe"] < 1e-3          # PA-MPJPE 는 골반 없어도 유효


def test_pose_metrics_pa_mpjpe_ignores_pure_translation():
    from smpl_eval.metrics.pose import pose_metrics
    pred, gt = _pair(H36M17_TO_SMPL24, offset=0.5)
    r = pose_metrics(pred, gt, H36M17_TO_SMPL24)
    assert r["pa_mpjpe"] < 1e-3          # 평행이동은 Procrustes 가 흡수


def test_pose_metrics_returns_nan_when_no_overlap():
    from smpl_eval.metrics.pose import pose_metrics
    pred, gt = _pair(H36M17_TO_SMPL24, n_frames=5)
    gt = {k: v.copy() for k, v in gt.items()}
    gt["frame_ids"] = gt["frame_ids"] + 1000
    r = pose_metrics(pred, gt, H36M17_TO_SMPL24)
    assert r["n_matched"] == 0 and np.isnan(r["pa_mpjpe"])
