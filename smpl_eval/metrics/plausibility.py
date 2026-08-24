"""GT 불필요 지표 — 물리적 일관성으로 추정 오류를 탐지한다.

두 가지 역할을 한다.
 1. GT 조인트 매핑이 실패해도 결과를 낼 수 있는 폴백 경로 (스펙 §6)
 2. betas 가 NaN 인 경우(Anny→SMPL 피팅 실패) β 지표의 대체재
    — limb_length_stats 와 beta_consistency 는 둘 다 "신원 일관성"을
      재므로 대체가 성립한다.
"""
import numpy as np

# SMPL 24관절 기준 (부모, 자식) — 길이가 시간에 불변이어야 하는 뼈
LIMBS = [
    (1, 4), (4, 7), (2, 5), (5, 8),              # 다리
    (16, 18), (18, 20), (17, 19), (19, 21),      # 팔
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),   # 척추·목
]

# 굽힘만 허용되는 관절: (부모, 관절, 자식)
HINGES = [(1, 4, 7), (2, 5, 8), (16, 18, 20), (17, 19, 21)]


def _by_track(tracks):
    """트랙별로 (id, 선택마스크, 프레임 정렬순서) 를 넘긴다."""
    for t in np.unique(tracks["track_ids"]):
        sel = tracks["track_ids"] == t
        order = np.argsort(tracks["frame_ids"][sel], kind="stable")
        yield int(t), sel, order


def limb_length_stats(tracks):
    """트랙 내 뼈 길이의 변동계수(CV). 같은 사람이면 0 에 가까워야 한다.

    β 가 체형을 결정하고 체형이 뼈 길이를 결정하므로, 뼈 길이가 흔들린다는
    것은 추정이 불안정하거나 트랙이 다른 사람을 물었다는 뜻이다.
    """
    per_track, cvs = {}, []
    for tid, sel, order in _by_track(tracks):
        j = tracks["joints3d"][sel][order]
        lens = np.stack([np.linalg.norm(j[:, a] - j[:, b], axis=-1)
                         for a, b in LIMBS], 1)                    # (F, L)
        # GT 규약에 없는 뼈는 열 전체가 NaN 이다 (COCO 에는 척추가 없음).
        # nanstd/nanmean 이 경고를 뿜지 않도록 그런 열을 미리 버린다.
        usable = np.isfinite(lens).any(axis=0)
        lens = lens[:, usable]
        if lens.shape[1] == 0:
            per_track[tid] = float("nan")
            cvs.append(float("nan"))
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            cv = np.nanstd(lens, 0) / np.nanmean(lens, 0)
        cv = cv[np.isfinite(cv)]
        v = float(cv.max()) if cv.size else float("nan")
        per_track[tid] = v
        cvs.append(v)
    cvs = [c for c in cvs if np.isfinite(c)]
    return {"mean_cv": float(np.mean(cvs)) if cvs else float("nan"),
            "max_cv": float(np.max(cvs)) if cvs else float("nan"),
            "per_track": per_track}


def acceleration_jitter(tracks, fps):
    """관절 위치 2차 미분의 크기 (m/s^2). 떨림을 정량화한다."""
    accels = []
    for _tid, sel, order in _by_track(tracks):
        j = tracks["joints3d"][sel][order]
        if len(j) < 3:
            continue
        a = np.diff(j, n=2, axis=0) * (fps ** 2)
        accels.append(np.linalg.norm(a, axis=-1).ravel())
    if not accels:
        return {"mean_accel": float("nan"), "p95_accel": float("nan")}
    a = np.concatenate(accels)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean_accel": float("nan"), "p95_accel": float("nan")}
    return {"mean_accel": float(a.mean()),
            "p95_accel": float(np.percentile(a, 95))}


def joint_angle_violations(tracks, min_deg=5.0):
    """무릎·팔꿈치가 접히다 못해 겹쳤는지 센다.

    관절에서 부모 방향과 자식 방향이 이루는 각이 min_deg 미만이면
    해부학적으로 불가능한 자세다. 펴진 상태는 180도라 위반이 아니다.
    """
    n_bad = n_tot = 0
    for _tid, sel, order in _by_track(tracks):
        j = tracks["joints3d"][sel][order]
        for a, b, c in HINGES:
            v1, v2 = j[:, a] - j[:, b], j[:, c] - j[:, b]
            n1 = np.linalg.norm(v1, axis=-1)
            n2 = np.linalg.norm(v2, axis=-1)
            ok = (n1 > 1e-6) & (n2 > 1e-6) & np.isfinite(n1) & np.isfinite(n2)
            if not ok.any():
                continue
            cos = np.clip((v1 * v2).sum(-1)[ok] / (n1[ok] * n2[ok]), -1.0, 1.0)
            deg = np.degrees(np.arccos(cos))
            n_bad += int((deg < min_deg).sum())
            n_tot += int(ok.sum())
    return {"n_violations": n_bad,
            "violation_rate": float(n_bad / n_tot) if n_tot else float("nan")}


def beta_consistency(tracks, jump_sigma=3.0):
    """β 는 신원이므로 트랙 내에서 불변이어야 한다. 급변은 ID 스왑 의심.

    betas 가 전부 NaN 이면(Anny→SMPL 피팅 실패) available=False 를 돌려
    호출자가 limb_length_stats 로 대체하도록 한다.
    """
    b = tracks["betas"]
    if not np.isfinite(b).any():
        return {"available": False, "mean_std": float("nan"),
                "max_std": float("nan"), "jump_frames": []}

    stds, jumps = [], []
    for _tid, sel, order in _by_track(tracks):
        bb = b[sel][order]
        f = tracks["frame_ids"][sel][order]
        stds.append(float(np.nanstd(bb, 0).max()))
        if len(bb) > 2:
            d = np.linalg.norm(np.diff(bb, axis=0), axis=-1)
            if np.isfinite(d).any() and d.std() > 0:
                thr = d.mean() + jump_sigma * d.std()
                jumps += [int(f[i + 1]) for i in np.where(d > thr)[0]]
    return {"available": True,
            "mean_std": float(np.mean(stds)), "max_std": float(np.max(stds)),
            "jump_frames": sorted(set(jumps))}


def all_plausibility(tracks, fps):
    """네 지표를 평평한 dict 로 병합한다 (results/*.json 에 그대로 들어감)."""
    ll = limb_length_stats(tracks)
    aj = acceleration_jitter(tracks, fps)
    ja = joint_angle_violations(tracks)
    bc = beta_consistency(tracks)
    return {
        "limb_mean_cv": ll["mean_cv"], "limb_max_cv": ll["max_cv"],
        "mean_accel": aj["mean_accel"], "p95_accel": aj["p95_accel"],
        "n_violations": ja["n_violations"], "violation_rate": ja["violation_rate"],
        "beta_available": bc["available"], "beta_mean_std": bc["mean_std"],
        "beta_max_std": bc["max_std"], "beta_jump_frames": bc["jump_frames"],
    }
