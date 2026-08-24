"""GT 대비 포즈 오차. 단위는 3D=mm, 2D=픽셀.

PA-MPJPE 를 주 지표로 쓰는 이유: 관절 회전은 키네매틱 트리를 타고
아래로 오차가 누적되므로, 전역 위치·방향·스케일의 차이를 제거하고
'자세 자체가 맞았는가'만 보려면 Procrustes 정렬이 필요하다.
"""
import numpy as np

from smpl_eval.conventions import (
    common_indices, skeleton_scale, CANONICAL_SKELETON_SCALE_MM)


def _procrustes(pred, gt):
    """스케일·회전·평행이동을 gt 에 맞추도록 pred 를 정렬한다 (프레임별)."""
    pred = np.asarray(pred, float)
    gt = np.asarray(gt, float)
    mp = pred.mean(1, keepdims=True)
    mg = gt.mean(1, keepdims=True)
    p, g = pred - mp, gt - mg

    out = np.empty_like(pred)
    for i in range(len(pred)):
        u, s, vt = np.linalg.svd(p[i].T @ g[i])
        d = np.sign(np.linalg.det(u @ vt))
        r = u @ np.diag([1.0, 1.0, d]) @ vt
        var = (p[i] ** 2).sum()
        scale = (s[0] + s[1] + d * s[2]) / var if var > 0 else 1.0
        out[i] = scale * (p[i] @ r) + mg[i]
    return out


def pa_mpjpe(pred_j, gt_j, normalize=False):
    """Procrustes 정렬 후 관절 평균 오차.

    normalize=False → GT 좌표계 단위 × 1000 ("입력이 m 라면 mm").
    normalize=True  → GT 골격 크기로 나눈 뒤 표준 인체 크기를 곱한 값(mm).

    GT 가 SfM 재구성이라 데이터셋마다 스케일이 다르면 normalize=True 를
    써야 한다. Procrustes 는 스케일을 맞추므로 잔차가 GT 단위로 나오는데,
    그 단위 자체가 데이터셋마다 다르기 때문이다.
    """
    gt_j = np.asarray(gt_j, float)
    aligned = _procrustes(pred_j, gt_j)
    err = np.linalg.norm(aligned - gt_j, axis=-1).mean(-1)
    if normalize:
        sc = skeleton_scale(gt_j)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(sc > 0, err / sc, np.nan) * CANONICAL_SKELETON_SCALE_MM
    return err * 1000.0


def mpjpe(pred_j, gt_j, root_idx=0):
    """루트 정렬 후 관절 평균 오차 (mm)."""
    p = np.asarray(pred_j, float)
    g = np.asarray(gt_j, float)
    p = p - p[:, root_idx:root_idx + 1]
    g = g - g[:, root_idx:root_idx + 1]
    return np.linalg.norm(p - g, axis=-1).mean(-1) * 1000.0


def reprojection_error(pred_j2d, gt_j2d):
    """2D 관절 픽셀 거리."""
    return np.linalg.norm(np.asarray(pred_j2d, float)
                          - np.asarray(gt_j2d, float), axis=-1).mean(-1)


def pose_metrics(pred, gt, mapping, frame_offset=0):
    """예측·GT 를 프레임 + bbox IoU 로 짝지어 오차를 낸다.

    frame_offset: gt_frame = pred_frame + frame_offset
    GT 에 없는 SMPL 슬롯은 NaN 이므로 매핑된 관절만 골라서 비교한다.

    MPJPE 는 골반(SMPL index 0)을 기준으로 정렬해야 의미가 있는데,
    Data1 의 COCO 규약에는 골반 관절이 아예 없다. 골반이 매핑에 없으면
    MPJPE 는 NaN 을 돌려주고 `mpjpe_available: False` 로 표시한다.
    PA-MPJPE 는 전역 정렬을 스스로 하므로 골반 유무와 무관하다.
    """
    from smpl_eval.metrics.occlusion import associate

    # 주의: gt 는 to_gt_tracks 를 거쳐 **이미 SMPL24 레이아웃**이다.
    # 따라서 예측과 GT 를 같은 인덱스(si)로 뽑아야 한다. GT 규약 인덱스(gi)는
    # 원본 파일을 SMPL24 로 옮길 때(to_gt_tracks) 안에서만 쓰인다.
    _gi, si = common_indices(mapping)
    # 뽑아낸 배열에서 골반이 몇 번째인지 (매핑에 골반이 없으면 None)
    pelvis_pos = int(np.where(si == 0)[0][0]) if (si == 0).any() else None
    # associate 는 양쪽 프레임 번호가 같다고 가정하므로 pred 를 GT 축으로 옮긴다
    shifted = dict(pred)
    shifted["frame_ids"] = np.asarray(pred["frame_ids"]) + frame_offset

    pa, mp, scales = [], [], []
    gt_frames = set(np.unique(gt["frame_ids"]).tolist())
    for f in np.unique(pred["frame_ids"]):
        gf = int(f) + frame_offset
        if gf not in gt_frames:
            continue
        for gtid, ptid in associate(shifted, gt, gf).items():
            gsel = (gt["frame_ids"] == gf) & (gt["track_ids"] == gtid)
            psel = (pred["frame_ids"] == f) & (pred["track_ids"] == ptid)
            if not (gsel.any() and psel.any()):
                continue
            gj = gt["joints3d"][gsel][0][si]
            pj = pred["joints3d"][psel][0][si]
            ok = np.isfinite(gj).all(-1) & np.isfinite(pj).all(-1)
            if ok.sum() < 6:                 # 관절이 너무 적으면 정렬이 불안정
                continue
            pa.append(pa_mpjpe(pj[None, ok], gj[None, ok], normalize=True)[0])
            scales.append(float(skeleton_scale(gj[ok])))
            if pelvis_pos is not None and ok[pelvis_pos]:
                # 필터링으로 인덱스가 밀리므로 살아남은 관절 중 골반의 위치를 다시 센다
                root = int(ok[:pelvis_pos].sum())
                mp.append(mpjpe(pj[None, ok], gj[None, ok], root_idx=root)[0])

    if not pa:
        return {"pa_mpjpe": float("nan"), "pa_mpjpe_p95": float("nan"),
                "mpjpe": float("nan"), "mpjpe_available": False,
                "gt_scale": float("nan"), "n_matched": 0}
    pa = np.asarray(pa, float)
    pa = pa[np.isfinite(pa)]
    if pa.size == 0:
        return {"pa_mpjpe": float("nan"), "pa_mpjpe_p95": float("nan"),
                "mpjpe": float("nan"), "mpjpe_available": False,
                "gt_scale": float("nan"), "n_matched": 0}
    # mpjpe 는 GT 단위 그대로라 스케일 보정을 따로 적용한다
    scale_corr = (CANONICAL_SKELETON_SCALE_MM
                  / (float(np.mean(scales)) * 1000.0)) if scales else 1.0
    return {"pa_mpjpe": float(pa.mean()),
            "pa_mpjpe_p95": float(np.percentile(pa, 95)),
            "mpjpe": float(np.mean(mp) * scale_corr) if mp else float("nan"),
            "mpjpe_available": bool(mp),
            "gt_scale": float(np.mean(scales)) if scales else float("nan"),
            "n_matched": int(len(pa))}
