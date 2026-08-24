import numpy as np
from smpl_eval.tests.synth import make_tracks
from smpl_eval.metrics.handsize import hand_pixel_stats


def _reproject(t, fx=1000.0, cx=960.0, cy=540.0):
    j = t["joints3d"]
    z = np.where(np.abs(j[..., 2]) < 1e-6, 1e-6, j[..., 2])
    t["joints2d"] = np.stack([j[..., 0] / z * fx + cx,
                              j[..., 1] / z * fx + cy], -1).astype(np.float32)
    t["bbox"] = np.stack([t["joints2d"][..., 0].min(1), t["joints2d"][..., 1].min(1),
                          t["joints2d"][..., 0].max(1), t["joints2d"][..., 1].max(1)],
                         -1).astype(np.float32)
    return t


def test_hand_stats_have_expected_keys():
    r = hand_pixel_stats(make_tracks(n_frames=20, n_tracks=2))
    for k in ("median_hand_px", "p10_hand_px", "median_person_px", "ratio"):
        assert k in r, k


def test_far_subject_has_smaller_hand_pixels():
    near = make_tracks(n_frames=20, n_tracks=1, seed=5)
    far = make_tracks(n_frames=20, n_tracks=1, seed=5)
    far["joints3d"] = far["joints3d"].copy()
    far["joints3d"][..., 2] *= 4.0
    far = _reproject(far)
    assert hand_pixel_stats(far)["median_hand_px"] < \
           hand_pixel_stats(near)["median_hand_px"]


def test_ratio_is_scale_invariant():
    """거리가 멀어져도 손/사람 비율은 유지돼야 한다."""
    near = make_tracks(n_frames=20, n_tracks=1, seed=6)
    far = make_tracks(n_frames=20, n_tracks=1, seed=6)
    far["joints3d"] = far["joints3d"].copy()
    far["joints3d"][..., 2] *= 3.0
    far = _reproject(far)
    a = hand_pixel_stats(near)["ratio"]
    b = hand_pixel_stats(far)["ratio"]
    assert abs(a - b) < 0.05 * max(a, b), (a, b)


def test_all_nan_joints_do_not_crash():
    t = make_tracks(n_frames=10, n_tracks=1)
    t["joints2d"] = np.full_like(t["joints2d"], np.nan)
    r = hand_pixel_stats(t)
    assert np.isnan(r["median_hand_px"])
