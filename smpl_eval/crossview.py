"""뷰 간 일관성 분석 — 한 뷰에서 끊긴 추적을 다른 뷰가 메우는가.

파일럿에서 확인한 것: CoMotion 의 ID 변경은 **가림** 이 원인이다
(변경 순간 가림 IoU 가 유지 중의 4.2배, 변경의 46~62% 가 IoU>0.2).
단안으로는 가려진 사람을 계속 따라갈 정보가 없다.

그런데 본 데이터셋은 같은 장면을 4대의 카메라가 동시에 찍는다.
한 뷰에서 가려진 사람이 다른 뷰에서는 보이는 경우가 많다.
이 모듈은 **멀티뷰 융합이 실제로 답이 되는지를 융합 없이 미리 확인**한다.

방법: 세션의 모든 카메라에 대해 GT 사람별 추적 상태를 프레임 단위로 만들고,
      뷰마다 '이 프레임에서 이 사람을 대표 트랙이 잡고 있는가' 를 판정한다.
      그 다음
        - 뷰별 단독 purity
        - 최선의 뷰(oracle) purity — 프레임마다 가장 잘 잡은 뷰를 고름
        - 합집합 coverage — 한 뷰라도 잡았는가
      를 비교한다. oracle 이 단독보다 크게 높으면 멀티뷰의 여지가 크다.
"""
import argparse
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np

from smpl_eval.evaluate import load_gt
from smpl_eval.metrics.occlusion import associate
from smpl_eval.schema import load_tracks


def per_view_assignment(tracks, gt):
    """{gt_id: {frame: pred_id}} 를 만든다."""
    out = defaultdict(dict)
    for f in np.unique(gt["frame_ids"]):
        for g, p in associate(tracks, gt, int(f)).items():
            out[int(g)][int(f)] = int(p)
    return out


def dominant_map(assign):
    """gt_id 별 대표 예측 트랙 (가장 많이 대응된 것)."""
    return {g: (Counter(d.values()).most_common(1)[0][0] if d else None)
            for g, d in assign.items()}


def analyse_session(records, model, tracks_name="tracks.npz"):
    """한 세션(같은 장면, 여러 카메라)의 뷰 간 일관성을 분석한다."""
    views = {}
    gt = None
    for rec in records:
        p = os.path.join("smpl_eval/outputs", model, rec["dataset"],
                         rec["session"], rec["cam"], tracks_name)
        if not os.path.exists(p):
            continue
        g, _ = load_gt(rec)
        if g is None:
            continue
        gt = g
        a, _ = load_tracks(p)
        assign = per_view_assignment(a, g)
        views[rec["cam"]] = {"assign": assign, "dom": dominant_map(assign)}

    if gt is None or len(views) < 2:
        return None

    frames = [int(f) for f in np.unique(gt["frame_ids"])]
    gt_ids = [int(g) for g in np.unique(gt["track_ids"])]

    # 뷰 x 사람 x 프레임 -> 대표 트랙으로 잡혔는가
    ok = {v: {g: np.zeros(len(frames), bool) for g in gt_ids} for v in views}
    for v, info in views.items():
        for g in gt_ids:
            d = info["assign"].get(g, {})
            dom = info["dom"].get(g)
            for i, f in enumerate(frames):
                ok[v][g][i] = (dom is not None and d.get(f) == dom)

    per_view, oracle, union = {}, [], []
    for g in gt_ids:
        mats = np.stack([ok[v][g] for v in views])       # (V, F)
        oracle.append(mats.max(0).mean())                # 프레임마다 최선의 뷰
        union.append(mats.any(0).mean())
    for v in views:
        per_view[v] = float(np.mean([ok[v][g].mean() for g in gt_ids]))

    best = max(per_view.values())
    return {
        "n_views": len(views), "n_gt": len(gt_ids), "n_frames": len(frames),
        "per_view": per_view,
        "best_single": best,
        "mean_single": float(np.mean(list(per_view.values()))),
        "oracle": float(np.mean(oracle)),
        "union": float(np.mean(union)),
        "gain": float(np.mean(oracle)) - best,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="smpl_eval/manifest.json")
    ap.add_argument("--models", default="comotion,multihmr2")
    ap.add_argument("--tracks", default="tracks.npz")
    ap.add_argument("--out", default="smpl_eval/results/crossview.json")
    a = ap.parse_args(argv)

    man = json.load(open(a.manifest))
    sessions = defaultdict(list)
    for r in man:
        sessions[(r["dataset"], r["session"])].append(r)

    results = {}
    print("%-8s %-22s %-11s %6s %10s %10s %10s %9s"
          % ("데이터", "세션", "모델", "뷰수", "뷰평균", "최선단일", "oracle", "이득"))
    print("-" * 92)
    for (ds, sess), recs in sorted(sessions.items()):
        if len(recs) < 2:
            continue
        for model in a.models.split(","):
            r = analyse_session(recs, model, a.tracks)
            if r is None:
                continue
            results["%s/%s/%s" % (ds, sess, model)] = r
            print("%-8s %-22s %-11s %6d %10.3f %10.3f %10.3f %+8.3f"
                  % (ds, sess[:22], model, r["n_views"], r["mean_single"],
                     r["best_single"], r["oracle"], r["gain"]))

    if results:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        gains = [v["gain"] for v in results.values()]
        print("\n평균 oracle 이득: %+.3f  (최대 %+.3f)" % (np.mean(gains), np.max(gains)))
        print("→ 이 값이 크면 멀티뷰 융합으로 회복할 여지가 크다는 뜻")
        print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
