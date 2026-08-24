"""테스트용 합성 tracks 생성기.

지표 코드를 GPU 없이 검증하기 위한 것. 정답을 아는 데이터를 만들어
"이 지표가 정말 그 실패를 잡아내는가"를 확인한다 (스펙 게이트 5).
"""
import numpy as np


def _tpose():
    """SMPL 24관절의 대략적인 T-포즈 좌표 (미터). 테스트용 근사값."""
    p = np.zeros((24, 3), np.float32)
    p[0] = (0.00, 0.00, 0.00)                                  # pelvis
    p[1] = (0.09, -0.08, 0.00); p[2] = (-0.09, -0.08, 0.00)    # hips
    p[3] = (0.00, 0.12, 0.00)                                  # spine1
    p[4] = (0.10, -0.48, 0.00); p[5] = (-0.10, -0.48, 0.00)    # knees
    p[6] = (0.00, 0.25, 0.00)                                  # spine2
    p[7] = (0.10, -0.88, 0.00); p[8] = (-0.10, -0.88, 0.00)    # ankles
    p[9] = (0.00, 0.35, 0.00)                                  # spine3
    p[10] = (0.10, -0.95, 0.10); p[11] = (-0.10, -0.95, 0.10)  # feet
    p[12] = (0.00, 0.52, 0.00)                                 # neck
    p[13] = (0.08, 0.45, 0.00); p[14] = (-0.08, 0.45, 0.00)    # collars
    p[15] = (0.00, 0.62, 0.00)                                 # head
    p[16] = (0.18, 0.45, 0.00); p[17] = (-0.18, 0.45, 0.00)    # shoulders
    p[18] = (0.44, 0.45, 0.00); p[19] = (-0.44, 0.45, 0.00)    # elbows
    p[20] = (0.68, 0.45, 0.00); p[21] = (-0.68, 0.45, 0.00)    # wrists
    p[22] = (0.76, 0.45, 0.00); p[23] = (-0.76, 0.45, 0.00)    # hands
    return p


def make_tracks(n_frames=100, n_tracks=3, seed=0, beta_jump_at=None,
                id_swap_at=None, jitter=0.0, limb_noise=0.0):
    """규칙적으로 움직이는 n_tracks 명을 만든다.

    beta_jump_at: (frame, track) 부터 betas 를 크게 튀게 한다
    id_swap_at:   (frame, tA, tB) 부터 두 트랙의 ID 를 맞바꾼다
    jitter:       joints3d 에 더할 가우시안 노이즈 표준편차 (m)
    limb_noise:   프레임마다 골격 전체 스케일을 흔드는 정도
    """
    rng = np.random.default_rng(seed)
    F, T = n_frames, n_tracks
    rows = F * T
    frame_ids = np.repeat(np.arange(F, dtype=np.int32), T)
    track_ids = np.tile(np.arange(T, dtype=np.int32), F)

    # 트랙별 고정 betas 를 frame-major 순서로 펼친다
    per_track_beta = rng.normal(0, 1, (T, 10)).astype(np.float32)
    betas = np.tile(per_track_beta, (F, 1))          # (F*T, 10), frame-major

    base = _tpose()
    j3d = np.zeros((rows, 24, 3), np.float32)
    for f in range(F):
        for t in range(T):
            off = np.array([t * 1.5, 0.0, 5.0 + 0.01 * f], np.float32)
            scale = 1.0 + limb_noise * rng.normal()
            j3d[f * T + t] = base * scale + off
    if jitter:
        j3d = j3d + rng.normal(0, jitter, j3d.shape).astype(np.float32)

    if beta_jump_at is not None:
        f0, t0 = beta_jump_at
        betas[(frame_ids >= f0) & (track_ids == t0)] += 5.0

    if id_swap_at is not None:
        f0, ta, tb = id_swap_at
        sel = frame_ids >= f0
        ids = track_ids.copy()
        ids[sel & (track_ids == ta)] = tb
        ids[sel & (track_ids == tb)] = ta
        track_ids = ids

    transl = j3d[:, 0, :].copy()
    z = np.where(np.abs(j3d[..., 2]) < 1e-6, 1e-6, j3d[..., 2])
    j2d = np.stack([j3d[..., 0] / z * 1000.0 + 960.0,
                    j3d[..., 1] / z * 1000.0 + 540.0], -1).astype(np.float32)
    bbox = np.stack([j2d[..., 0].min(1), j2d[..., 1].min(1),
                     j2d[..., 0].max(1), j2d[..., 1].max(1)], -1).astype(np.float32)

    return {
        "frame_ids": frame_ids,
        "track_ids": track_ids.astype(np.int32),
        "betas": betas.astype(np.float32),
        "global_orient": np.zeros((rows, 3), np.float32),
        "body_pose": np.zeros((rows, 23, 3), np.float32),
        "transl": transl.astype(np.float32),
        "joints3d": j3d,
        "joints2d": j2d,
        "bbox": bbox,
        "score": np.ones(rows, np.float32),
    }
