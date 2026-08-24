import numpy as np
from smpl_eval.tests.synth import make_tracks
from smpl_eval.metrics.plausibility import (
    limb_length_stats, acceleration_jitter, joint_angle_violations,
    beta_consistency, all_plausibility)


def test_clean_tracks_have_near_zero_limb_variation():
    t = make_tracks(n_frames=60, n_tracks=2)
    assert limb_length_stats(t)["max_cv"] < 1e-4


def test_limb_noise_raises_variation():
    clean = make_tracks(n_frames=60, n_tracks=2, seed=3)
    noisy = make_tracks(n_frames=60, n_tracks=2, seed=3, limb_noise=0.05)
    assert limb_length_stats(noisy)["max_cv"] > limb_length_stats(clean)["max_cv"] * 10


def test_jitter_raises_acceleration():
    clean = make_tracks(n_frames=60, n_tracks=2, seed=4)
    noisy = make_tracks(n_frames=60, n_tracks=2, seed=4, jitter=0.02)
    assert acceleration_jitter(noisy, 30.0)["mean_accel"] > \
           acceleration_jitter(clean, 30.0)["mean_accel"] * 5


def test_beta_stable_for_clean_tracks():
    t = make_tracks(n_frames=60, n_tracks=3)
    assert beta_consistency(t)["max_std"] < 1e-5


def test_beta_jump_is_detected():
    t = make_tracks(n_frames=60, n_tracks=3, beta_jump_at=(30, 1))
    r = beta_consistency(t)
    assert r["max_std"] > 1.0
    assert any(f in r["jump_frames"] for f in (29, 30, 31)), r["jump_frames"]


def test_beta_nan_is_skipped_not_crashed():
    t = make_tracks(n_frames=20, n_tracks=2)
    t["betas"] = np.full_like(t["betas"], np.nan)
    r = beta_consistency(t)
    assert r["available"] is False


def test_straight_tpose_limbs_are_not_violations():
    """T-포즈는 팔다리가 곧게 펴진 상태(180도) — 역굴곡 위반이 아니다."""
    t = make_tracks(n_frames=20, n_tracks=2)
    assert joint_angle_violations(t)["n_violations"] == 0


def test_hyperextension_is_detected():
    """무릎을 접어 부모-자식 벡터가 겹치게 만들면 위반으로 잡혀야 한다."""
    t = make_tracks(n_frames=20, n_tracks=1)
    # 왼발목(7)을 왼무릎(4) 기준으로 왼엉덩이(1) 쪽에 포개 놓는다
    t["joints3d"][:, 7] = t["joints3d"][:, 1]
    r = joint_angle_violations(t)
    assert r["n_violations"] > 0 and r["violation_rate"] > 0


def test_all_plausibility_returns_all_keys():
    r = all_plausibility(make_tracks(n_frames=30, n_tracks=2), 30.0)
    for k in ("limb_max_cv", "mean_accel", "violation_rate", "beta_max_std"):
        assert k in r, k
