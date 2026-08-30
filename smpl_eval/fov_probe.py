"""프레임별 초점거리를 역산하고, 그 흔들림이 장면 전체 움직임을 만드는지 본다.

**왜 역산이 가능한가** — `tracks.npz` 에 카메라 K 를 따로 담지는 않았지만,
3D 관절과 2D 관절이 둘 다 들어 있다. 러너가 쓴 투영식은

    u = x/z * fx + cx      v = y/z * fy + cy

이므로, 한 프레임의 관절들을 모아 (x/z, 1) 에 대한 u 의 최소제곱을 풀면
fx, cx 가 정확히 나온다. 관절이 여러 개라 과결정이고, 잔차가 0 에 가까우면
역산이 맞다는 증거가 된다.

**무엇을 보려는가** — Multi-HMR 2 는 CLS 토큰에서 시야각(FOV) 하나를 매
프레임 새로 예측한다(`mlp_fov_unique` 의 출력이 1개). 그 값이 흔들리면
초점거리가 흔들리고, 화면 위치가 그대로여도 3D 위치가 밀린다. 그것도
**모든 사람이 같은 방향으로.** 메시 영상이 매 프레임 끊겨 보이는 현상의
후보 원인이다.

이 모듈은 두 가지를 잰다.
  1. 초점거리가 실제로 프레임마다 얼마나 변하는가
  2. 그 변화가 사람들의 공통 움직임을 설명하는가 (상관)

CoMotion 은 f = 2*max(W,H) 로 고정하므로 대조군이 된다 — 역산값이 상수로
나와야 하고, 그렇지 않으면 역산 자체가 틀린 것이다.
"""
import numpy as np


def recover_intrinsics(tracks):
    """프레임별 (fx, fy, cx, cy, 잔차) 를 최소제곱으로 역산한다.

    반환: frames(오름차순), K(n,4), resid(n,) — resid 는 픽셀 단위 RMS
    """
    j3 = tracks["joints3d"]
    j2 = tracks["joints2d"]
    frames = np.unique(tracks["frame_ids"])
    out_f, out_K, out_r = [], [], []

    for f in frames:
        m = tracks["frame_ids"] == f
        a3, a2 = j3[m].reshape(-1, 3), j2[m].reshape(-1, 2)
        ok = np.all(np.isfinite(a3), 1) & np.all(np.isfinite(a2), 1) & (np.abs(a3[:, 2]) > 1e-6)
        a3, a2 = a3[ok], a2[ok]
        if len(a3) < 4:
            continue
        vals, res = [], []
        for ax in (0, 1):
            # u = (x/z) * f + c   →  [x/z, 1] @ [f, c]
            A = np.stack([a3[:, ax] / a3[:, 2], np.ones(len(a3))], 1)
            sol, *_ = np.linalg.lstsq(A, a2[:, ax], rcond=None)
            vals.append(sol)
            res.append(A @ sol - a2[:, ax])
        out_f.append(int(f))
        out_K.append([vals[0][0], vals[1][0], vals[0][1], vals[1][1]])
        out_r.append(float(np.sqrt(np.mean(np.concatenate(res) ** 2))))

    return np.array(out_f), np.array(out_K), np.array(out_r)


def common_mode_series(tracks):
    """프레임별 '사람들의 평균 가속도' — 장면 전체가 함께 움직인 양."""
    acc = {}
    for tid in np.unique(tracks["track_ids"]):
        m = tracks["track_ids"] == tid
        fr = tracks["frame_ids"][m]
        o = np.argsort(fr)
        fr = fr[o]
        p = np.nanmean(tracks["joints3d"][m][o], axis=1)
        ok = (fr[2:] - fr[:-2] == 2)
        a = (p[2:] - 2 * p[1:-1] + p[:-2])[ok]
        for i, f in enumerate(fr[1:-1][ok]):
            if np.all(np.isfinite(a[i])):
                acc.setdefault(int(f), []).append(a[i])
    fs = sorted(k for k, v in acc.items() if len(v) >= 2)
    return np.array(fs), np.array([np.mean(acc[f], axis=0) for f in fs])


def probe(tracks, label=""):
    f, K, resid = recover_intrinsics(tracks)
    if not len(f):
        print(f"{label}: 역산 불가 (관절 부족)")
        return None

    fx = K[:, 0]
    med = float(np.median(fx))
    rel = fx / med
    print(f"{label}")
    print(f"  역산 잔차 RMS  {resid.mean():.3f} px   "
          f"(0 에 가까울수록 역산이 정확)")
    print(f"  초점거리       중앙값 {med:8.1f}  "
          f"변동계수 {fx.std()/med:.4f}  "
          f"범위 {rel.min():.3f}~{rel.max():.3f} 배")

    # 프레임 간 초점거리 변화율과 공통 움직임의 상관
    cf, cm = common_mode_series(tracks)
    idx = {int(v): i for i, v in enumerate(f)}
    keep = [i for i, v in enumerate(cf) if v in idx and (v - 1) in idx and (v + 1) in idx]
    if len(keep) > 10:
        # 초점거리의 2차 차분 = 초점거리 '가속도'
        d2 = np.array([fx[idx[cf[i] + 1]] - 2 * fx[idx[cf[i]]] + fx[idx[cf[i] - 1]]
                       for i in keep]) / med
        cz = cm[keep][:, 2]        # 공통 움직임의 깊이 성분
        r = float(np.corrcoef(np.abs(d2), np.abs(cz))[0, 1])
        print(f"  |초점거리 가속| 과 |공통 깊이 움직임| 의 상관  r = {r:+.3f}"
              f"   (표본 {len(keep)})")
    return {"median_focal": med, "cv": float(fx.std() / med),
            "resid_px": float(resid.mean())}


def main(argv=None):
    import argparse
    from smpl_eval.schema import load_tracks
    ap = argparse.ArgumentParser(description="프레임별 초점거리 역산 및 진단")
    ap.add_argument("tracks", nargs="+")
    a = ap.parse_args(argv)
    for p in a.tracks:
        t, m = load_tracks(p)
        probe(t, f"{m.get('model', '?'):12s} {p}")
        print()


if __name__ == "__main__":
    main()
