"""기존 tracks.npz 의 bbox 를 joints2d 기준으로 재계산한다.

CoMotion 어댑터가 초기에 모델의 MOT bbox 를 그대로 썼는데, GT 와
Multi-HMR 2 는 투영된 관절에서 bbox 를 유도한다. 생성 방식이 다르면
IoU 가 체계적으로 어긋나 ID 지표가 무의미해진다 (실측: IoU 0.41~0.48).

joints2d 는 이미 저장돼 있으므로 추론을 다시 돌릴 필요 없이 재계산만
하면 된다. 이 스크립트는 일회성 마이그레이션이다.
"""
import argparse
import glob
import os

import numpy as np

from smpl_eval.schema import load_tracks, save_tracks
from smpl_eval.runners.base import bbox_from_joints2d


def fix(path, dry_run=False):
    arrays, meta = load_tracks(path)
    if meta.get("bbox_source") == "joints2d":
        return None                       # 이미 처리됨
    old = arrays["bbox"].copy()
    new = bbox_from_joints2d(arrays["joints2d"])
    ow = np.median(old[:, 2] - old[:, 0])
    nw = np.median(new[:, 2] - new[:, 0])
    if not dry_run:
        arrays["bbox"] = new
        meta["bbox_source"] = "joints2d"
        meta["bbox_fixed_from"] = meta.pop("bbox_from_mot", "model_output")
        save_tracks(path, arrays, meta)
    return {"old_median_w": float(ow), "new_median_w": float(nw),
            "ratio": float(ow / nw) if nw > 0 else float("nan")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="smpl_eval/outputs")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    for p in sorted(glob.glob(os.path.join(a.root, "**", "tracks.npz"), recursive=True)):
        r = fix(p, a.dry_run)
        rel = os.path.relpath(p, a.root)
        if r is None:
            print("  skip (이미 joints2d): %s" % rel)
        else:
            print("  %s  bbox 폭 중앙값 %.0f -> %.0f (%.2f배)"
                  % (rel, r["old_median_w"], r["new_median_w"], r["ratio"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
