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


def test_jitter_separates_depth_from_screen_plane():
    """깊이(z) 노이즈가 화면평면(xy) 지표를 오염시키면 안 된다.

    단안 추정은 깊이가 본질적으로 불안정해서, 3D 전체로 지터를 재면
    모델 비교가 깊이 추정 품질 비교로 변질된다 (CoMotion 실측: 98%가 z).
    """
    t = make_tracks(n_frames=40, n_tracks=2, seed=31)
    rng = np.random.default_rng(0)
    t["joints3d"] = t["joints3d"].copy()
    t["joints3d"][..., 2] += rng.normal(0, 0.3, t["joints3d"].shape[:2]).astype(np.float32)

    r = acceleration_jitter(t, 30.0)
    assert r["mean_accel_z"] > r["mean_accel_xy"] * 10, r
    assert r["depth_share"] > 0.9


def test_beta_constant_per_track_is_flagged():
    """β 가 트랙 내내 상수면 β 기반 ID 스왑 탐지가 무력함을 알려야 한다."""
    t = make_tracks(n_frames=30, n_tracks=3)
    assert beta_consistency(t)["constant_per_track"] is True
    t2 = make_tracks(n_frames=30, n_tracks=3, beta_jump_at=(15, 1))
    assert beta_consistency(t2)["constant_per_track"] is False


def test_all_plausibility_propagates_every_subkey():
    """all_plausibility 가 하위 지표의 키를 빠짐없이 옮기는지 확인한다.

    개별 함수만 테스트하면 병합 단계의 누락을 놓친다 (실제로
    beta_constant_per_track 이 여기서 빠져 실행 중 KeyError 가 났다).
    """
    t = make_tracks(n_frames=30, n_tracks=2)
    r = all_plausibility(t, 30.0)
    expected = {
        "limb_mean_cv", "limb_max_cv",
        "mean_accel", "p95_accel", "mean_accel_xy", "p95_accel_xy",
        "mean_accel_z", "depth_share",
        "n_violations", "violation_rate",
        "beta_available", "beta_constant_per_track",
        "beta_mean_std", "beta_max_std", "beta_jump_frames",
    }
    missing = expected - set(r)
    assert not missing, f"병합에서 누락된 키: {sorted(missing)}"
