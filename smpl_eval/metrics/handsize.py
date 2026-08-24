"""손이 화면에서 몇 픽셀인지 실측한다.

SMPL-X 급(손가락 15관절×2) 확장이 의미 있는 데이터인지 판단하는 근거다.
손이 10px 수준이면 모델이 내놓는 손 파라미터는 관측이 아니라 prior 가
만들어낸 값이므로, 그걸 평가하면 노이즈를 평가하는 셈이 된다.

측정 방법: SMPL 의 손목(20/21)–손(22/23) 2D 거리를 손 크기 대용치로 쓰고,
사람 bbox 높이 대비 비율도 함께 낸다. 비율은 거리에 불변이라
데이터셋 간 비교에 쓸 수 있다.
"""
import numpy as np

WRIST_HAND = [(20, 22), (21, 23)]      # (손목, 손) SMPL 인덱스


def hand_pixel_stats(tracks):
    j2 = np.asarray(tracks["joints2d"], float)
    dists = []
    for w, h in WRIST_HAND:
        v = np.linalg.norm(j2[:, w] - j2[:, h], axis=-1)
        dists.append(v[np.isfinite(v)])
    d = np.concatenate(dists) if dists else np.array([])
    d = d[d > 0]

    bh = np.asarray(tracks["bbox"], float)[:, 3] - np.asarray(tracks["bbox"], float)[:, 1]
    bh = bh[np.isfinite(bh) & (bh > 0)]

    if d.size == 0 or bh.size == 0:
        return {"median_hand_px": float("nan"), "p10_hand_px": float("nan"),
                "median_person_px": float("nan"), "ratio": float("nan")}

    mh = float(np.median(d))
    mp = float(np.median(bh))
    return {"median_hand_px": mh,
            "p10_hand_px": float(np.percentile(d, 10)),
            "median_person_px": mp,
            "ratio": mh / mp if mp > 0 else float("nan")}
