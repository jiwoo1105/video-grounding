"""tracks.npz 하나에 모든 지표를 적용해 results/*.json 을 만든다.

지표 코드는 모델을 모른다 — tracks.npz 스키마만 안다. 그래서 CoMotion 과
Multi-HMR 2 가 완전히 같은 코드로 평가되고 비교가 공정해진다.
"""
import argparse
import json
import os

import numpy as np

from smpl_eval.schema import load_tracks
from smpl_eval.conventions import DATASET_CONVENTION
from smpl_eval.gt.parse_pose3d import parse_pose3d, to_gt_tracks, detect_frame_offset
from smpl_eval.metrics.plausibility import all_plausibility
from smpl_eval.metrics.identity import (
    id_metrics, person_count_error, gt_track_purity)
from smpl_eval.metrics.occlusion import find_occlusion_events, id_retention_around_events
from smpl_eval.metrics.pose import pose_metrics
from smpl_eval.metrics.handsize import hand_pixel_stats


def load_gt(rec):
    """manifest 레코드로부터 GT 를 tracks 포맷으로 읽는다.

    GT 프레임 번호를 0-base 영상 인덱스로 옮겨서 돌려주므로,
    이후 비교에서 frame_offset 을 다시 적용할 필요가 없다.
    """
    path = rec.get("gt_pose3d")
    if not path or not os.path.isfile(path):
        return None, None
    n_joints, mapping = DATASET_CONVENTION[rec["dataset"]]
    parsed = parse_pose3d(path, n_joints)
    if len(parsed["frame_ids"]) == 0:
        return None, None
    off = detect_frame_offset(parsed["frame_ids"], rec["n_frames"])
    parsed["frame_ids"] = parsed["frame_ids"] - off

    # GT 를 이 카메라 화면으로 정확히 투영한다. 예측-GT 매칭이 bbox IoU 라
    # 투영이 틀리면 ID·포즈 지표 전체가 무의미해진다.
    camera, cam_err = None, None
    if rec.get("colmap_dir"):
        try:
            from smpl_eval.gt.colmap import load_camera
            camera = load_camera(rec["colmap_dir"], rec["cam"])
        except Exception as e:
            cam_err = f"{type(e).__name__}: {e}"

    gt = to_gt_tracks(parsed, mapping, (rec["width"], rec["height"]), camera)
    meta = {"gt_frame_offset": int(off),
            "gt_projection": (camera.model if camera else None),
            "gt_projection_error": cam_err,
            "gt_persons": int(len(np.unique(gt["track_ids"]))),
            "gt_malformed_lines": int(parsed["n_malformed"]),
            "gt_convention": f"{n_joints}관절",
            "gt_limb_cv": None}          # 아래에서 채움
    return gt, meta


def evaluate_one(tracks_path, rec):
    pred, pmeta = load_tracks(tracks_path)
    out = {
        "video": rec["video_path"], "dataset": rec["dataset"],
        "session": rec["session"], "cam": rec["cam"],
        "model": pmeta.get("model"), "body_model": pmeta.get("body_model"),
        "converted_from": pmeta.get("converted_from"),
        "runtime_sec": pmeta.get("runtime_sec"),
        "n_frames": rec["n_frames"], "fps": rec["fps"],
        "n_rows": int(len(pred["frame_ids"])),
        "n_tracks": int(len(np.unique(pred["track_ids"]))),
    }
    if out["runtime_sec"] and rec["n_frames"]:
        out["ms_per_frame"] = round(out["runtime_sec"] * 1000.0 / rec["n_frames"], 1)

    # ── GT 불필요 지표 ──────────────────────────────────────────
    out.update(all_plausibility(pred, rec["fps"]))
    out.update(hand_pixel_stats(pred))

    events = find_occlusion_events(pred)
    out["n_occlusion_events"] = len(events)

    if rec.get("expected_persons"):
        out.update(person_count_error(pred, rec["expected_persons"]))
        out["expected_persons"] = rec["expected_persons"]

    # ── GT 필요 지표 ────────────────────────────────────────────
    gt, gmeta = load_gt(rec)
    if gt is None:
        out["gt_available"] = False
        return out

    out["gt_available"] = True
    out.update(gmeta)
    # GT 자체의 노이즈 수준 — 모델 정확도 주장의 하한선이므로 함께 기록한다
    from smpl_eval.metrics.plausibility import limb_length_stats
    out["gt_limb_cv"] = limb_length_stats(gt)["mean_cv"]

    _, mapping = DATASET_CONVENTION[rec["dataset"]]
    try:
        out.update(pose_metrics(pred, gt, mapping))
        out.update({f"id_{k}": v for k, v in id_metrics(pred, gt).items()})
        pur = gt_track_purity(pred, gt)
        out.update({"gt_purity_mean": pur["mean_purity"],
                    "gt_purity_min": pur["min_purity"],
                    "gt_coverage_mean": pur["mean_coverage"],
                    "gt_n_tracks": pur["n_gt_tracks"]})
        out.update({f"occ_{k}": v for k, v in
                    id_retention_around_events(pred, gt, events).items()})
    except Exception as e:                       # GT 비교만 실패해도 나머지는 남긴다
        out["gt_error"] = f"{type(e).__name__}: {e}"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", default="smpl_eval/manifest.json")
    ap.add_argument("--out", default="smpl_eval/results")
    ap.add_argument("--tracks-root", default="smpl_eval/outputs")
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    n = 0
    for r in json.load(open(a.manifest)):
        p = os.path.join(a.tracks_root, a.model, r["dataset"], r["session"],
                         r["cam"], "tracks.npz")
        if not os.path.exists(p):
            continue
        res = evaluate_one(p, r)
        dst = os.path.join(a.out, f"{a.model}__{r['dataset']}__{r['cam']}.json")
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        n += 1
        print(f"{r['dataset']}/{r['cam']:<28} "
              f"PA-MPJPE {res.get('pa_mpjpe', float('nan')):>7.1f}  "
              f"IDF1 {res.get('id_idf1', float('nan')):>6.3f}")
    print(f"\n{n}개 → {a.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
