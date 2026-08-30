"""초점거리를 고정해 장면 전체가 앞뒤로 밀리는 현상을 없앤다.

**문제** — Multi-HMR 2 는 CLS 토큰에서 시야각을 매 프레임 새로 예측한다
(`mlp_fov_unique` 의 출력이 1개). golden 영상 실측으로 초점거리가 프레임마다
±7.6% 흔들렸고, 그 흔들림이 사람들의 공통 깊이 움직임과 상관 r=+0.49 였다.
메시 영상이 매 프레임 끊겨 보이는 원인이다.

**무엇을 고치는가** — 사람의 3D 크기 H 는 체형에서 정해지고 화면상 크기 h
는 관측값이므로

    z = f · H / h        ->  뿌리 깊이는 f 에 정비례
    x = (u - cx) · z / f = (u - cx) · H / h    ->  가로 위치는 f 와 무관

실측으로도 log f 대 log z 의 기울기가 +1.016 로 정비례가 확인됐다.

**사람은 통째로 이동하지, 찌그러지지 않는다.** 관절마다 z 에 배율을 곱하면
몸이 깊이 방향으로 납작해진다 (CoMotion 을 실측 초점거리로 맞출 때 배율이
0.313 이라 눈에 띄게 뭉개졌다). 체형은 그대로 두고 **뿌리 깊이만 옮기는
강체 이동**이어야 한다.

    dz = transl_z * (f_fixed/f_t - 1)
    모든 관절의 z 에 dz 를 더한다 (곱하지 않는다)

**렌더할 때는 반드시 바뀐 초점거리를 써야 한다.** 사람을 가깝게 옮겨놓고
예전 초점거리로 투영하면 거대하게 그려진다. meta 의 fov_fix.focal_after
를 렌더러가 읽는다.

**어떤 값으로 고정하는가**
    median  모델 자신의 예측 중앙값. 외부 정보 없이 흔들림만 제거한다.
    colmap  COLMAP 이 측정한 실제 초점거리. 흔들림과 편향을 함께 없앤다.

CoMotion 에도 쓸 수 있다. CoMotion 은 f = 2*max(W,H) 로 고정하는데 이 값이
실측과 0.66~3.2배 어긋난다 (golden 은 3840 vs 실측 1201). colmap 모드로
돌리면 깊이 스케일이 바로잡힌다.
"""
import argparse
import os

import numpy as np

from smpl_eval.fov_probe import recover_intrinsics


def colmap_focal(colmap_dir, cam_name):
    """COLMAP 이 측정한 실제 초점거리 (fx, fy) 를 읽는다."""
    from smpl_eval.gt.colmap import load_camera
    cam = load_camera(colmap_dir, cam_name)
    return float(cam.prm["fx"]), float(cam.prm["fy"])


def fix_focal(tracks, mode="median", colmap=None, cam=None):
    """초점거리를 고정하고 깊이를 다시 계산한 새 tracks 를 돌려준다."""
    f_ids, K, resid = recover_intrinsics(tracks)
    if not len(f_ids):
        raise RuntimeError("초점거리를 역산할 수 없습니다 (관절 부족)")
    if resid.mean() > 1.0:
        raise RuntimeError(f"역산 잔차가 큽니다 ({resid.mean():.2f}px). "
                           "joints2d 가 joints3d 의 투영이 아닐 수 있습니다")

    fx_t = dict(zip(f_ids.tolist(), K[:, 0].tolist()))
    if mode == "median":
        f_fix = float(np.median(K[:, 0]))
    elif mode == "colmap":
        f_fix = colmap_focal(colmap, cam)[0]
    else:
        raise ValueError(mode)

    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in tracks.items()}
    scale = np.array([f_fix / fx_t.get(int(f), f_fix) for f in tracks["frame_ids"]],
                     dtype=np.float32)

    # 뿌리 깊이가 f 에 비례해 옮겨간 만큼을 강체 이동으로 적용한다.
    # 관절마다 곱하면 몸이 깊이 방향으로 납작해지므로 더해야 한다.
    dz = (out["transl"][:, 2] * (scale - 1.0)).astype(np.float32)
    out["joints3d"][..., 2] += dz[:, None]
    out["transl"][:, 2] += dz
    return out, {"focal_before_median": float(np.median(K[:, 0])),
                 "focal_after": f_fix,
                 "focal_cv_before": float(K[:, 0].std() / np.median(K[:, 0])),
                 "scale_min": float(scale.min()), "scale_max": float(scale.max())}


def main(argv=None):
    from smpl_eval.schema import load_tracks, save_tracks
    from smpl_eval.metrics.plausibility import acceleration_jitter

    ap = argparse.ArgumentParser(description="초점거리 고정 후처리")
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["median", "colmap"], default="median")
    ap.add_argument("--colmap-dir")
    ap.add_argument("--cam")
    a = ap.parse_args(argv)

    t, m = load_tracks(a.tracks)
    fps = m.get("fps", 30.0)
    j0 = acceleration_jitter(t, fps)
    t2, info = fix_focal(t, a.mode, a.colmap_dir, a.cam)
    j1 = acceleration_jitter(t2, fps)

    print(f"초점거리  {info['focal_before_median']:.1f} "
          f"(변동 {info['focal_cv_before']:.2%})  ->  {info['focal_after']:.1f} (고정)")
    print(f"깊이 배율  {info['scale_min']:.3f} ~ {info['scale_max']:.3f}")
    print()
    # 절대 지터(mm/frame^2)는 깊이 스케일에 비례하므로, 초점거리를 바꿔
    # 깊이가 통째로 줄면 지터도 같은 비율로 준다. 그것은 흔들림이 나아진
    # 것이 아니라 단위가 작아진 것이다. 중앙 깊이로 나눠 함께 본다.
    z0 = float(np.nanmedian(np.abs(t["joints3d"][..., 2])))
    z1 = float(np.nanmedian(np.abs(t2["joints3d"][..., 2])))
    r0 = j0["mean_accel_z"] / max(z0, 1e-9)
    r1 = j1["mean_accel_z"] / max(z1, 1e-9)

    print(f"{'':10s} {'가속 xy':>9s} {'가속 z':>10s} {'중앙깊이':>10s} {'상대 z지터':>11s}")
    print(f"{'보정 전':10s} {j0['mean_accel_xy']:9.2f} {j0['mean_accel_z']:10.2f}"
          f" {z0:10.1f} {r0:11.5f}")
    print(f"{'보정 후':10s} {j1['mean_accel_xy']:9.2f} {j1['mean_accel_z']:10.2f}"
          f" {z1:10.1f} {r1:11.5f}")
    print(f"{'변화':10s} {'':9s} {1 - j1['mean_accel_z']/max(j0['mean_accel_z'],1e-9):+9.1%}"
          f" {z1/max(z0,1e-9):9.3f}배 {1 - r1/max(r0,1e-9):+10.1%}")
    print()
    print("  상대 z지터 = 가속 z / 중앙 깊이.  깊이 스케일이 바뀌어도 공평하게")
    print("  비교된다. 흔들림이 실제로 줄었는지는 이 값을 봐야 한다.")

    m2 = dict(m)
    m2["postprocess"] = f"{m.get('postprocess', '')}+fov_fix_{a.mode}".lstrip("+")
    m2["fov_fix"] = info
    save_tracks(a.out, t2, m2)
    print(f"\n저장 {a.out}")


if __name__ == "__main__":
    main()
