"""Multi-HMR 2 추적기 하이퍼파라미터 ablation.

동기: 기본 설정 `combine='wsum_1.0_8.0'` 은 정규화하면
      외모 특징(SAM2 증류) 11% / 골반 위치 예측 89% 다.
      즉 추적이 사실상 위치 예측 기반이고 외모는 보조 신호다.

가설: 농구처럼 움직임이 빠르고 예측이 어려운 장면에서는 위치 예측이
      무너지므로, 외모 비중을 올리면 개선될 수 있다.

주의: 이것은 **모델 기본값 그대로 비교** 라는 원칙을 벗어나는 실험이다.
      기본값 결과와 반드시 구분해 보고해야 한다.

효율: 추론(ViT-L forward)은 한 번만 하고 trackfeat/pelvis 를 캐시한 뒤,
      추적기 설정만 바꿔가며 재조립한다. GPU 를 반복해서 쓰지 않는다.
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch


def cache_detections(session, video_path, tmp_dir, cache_path, coco_w):
    """추론을 한 번만 돌려 추적·평가에 필요한 것만 저장한다.

    v3d(19,158 정점)를 그대로 담으면 캐시가 2GB 를 넘는다. 여기서
    COCO 키포인트로 미리 회귀해 담는다.
    """
    from multihmr2 import api

    os.makedirs(tmp_dir, exist_ok=True)
    preds = api.infer_video(session, os.path.abspath(video_path), tmp_dir)
    frames = []
    for _t, fp in enumerate(preds):
        p = fp.persons
        n = int(getattr(p, "num_person", 0) or 0)
        if n == 0:
            frames.append(None)
            continue
        v3d = p.v3d.detach().cpu().float()
        frames.append({
            "trackfeat": p.trackfeat.detach().cpu().float().numpy(),
            "transl_pelvis": p.transl_pelvis.detach().cpu().float().numpy(),
            # 추적기가 쓰는 값 — api.infer_video 와 동일하게 구성한다
            "j2d_root": p.j2d.detach().cpu().float()[:, 0].numpy(),
            "pelvis_ori": p.bone_poses.detach().cpu().float()[:, 0, :3, :3].numpy(),
            # 평가용
            "coco3d": torch.einsum("kv,nvc->nkc", coco_w, v3d).numpy(),
            "conf": p.conf.detach().cpu().float().numpy(),
            "K": fp.K.detach().cpu().float().numpy(),
        })
    with open(cache_path, "wb") as f:
        pickle.dump({"img_size": int(session.img_size), "frames": frames}, f)
    return frames


def retrack(frames, img_size, combine, min_sim=0.69, feat_aggsim="Knn35t15.0"):
    """캐시된 검출에 추적기만 다시 돌린다. GPU 불필요.

    입력 구성은 api.infer_video 와 **정확히 동일해야** 한다 —
      pelvis_ijn = cat([ j2d[:,0] / img_size,  log(1 / z) ])
      pelvis_ori = bone_poses[:, 0, :3, :3]
    이를 어기면(예: 정규화 생략, 방향을 단위행렬로) 기본 설정조차
    원래 결과를 재현하지 못해 ablation 전체가 무효가 된다.
    """
    from multihmr2.tracker import FeatPelvisTracker

    tr = FeatPelvisTracker(combine=combine, min_sim=min_sim,
                           feat_aggsim=feat_aggsim)
    tr.reset()
    out = []
    for t, fr in enumerate(frames):
        if fr is None:
            out.append(np.zeros((0,), np.int32))
            tr.track_next_frame(t, torch.zeros((0, 4096)), None, None, None)
            continue
        feat = torch.from_numpy(fr["trackfeat"])
        xyz = torch.from_numpy(fr["transl_pelvis"])
        ijn = torch.cat([torch.from_numpy(fr["j2d_root"]) / img_size,
                         torch.log(1.0 / xyz[:, 2:3].clamp_min(1e-5))], -1).float()
        ori = torch.from_numpy(fr["pelvis_ori"]).float()
        ids = tr.track_next_frame(t, feat, xyz, ijn, ori)
        out.append(np.asarray(ids).astype(np.int32))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Data1")
    ap.add_argument("--repo", default=os.path.expanduser("~/smpl_eval_env/multi-hmr2"))
    ap.add_argument("--cache", default=os.path.expanduser("~/mh_ablate_cache.pkl"))
    ap.add_argument("--tmp", default=os.path.expanduser("~/mh_ablate_tmp"))
    ap.add_argument("--manifest", default="smpl_eval/manifest.json")
    ap.add_argument("--combines", default="wsum_1.0_8.0,wsum_1.0_4.0,wsum_1.0_1.0,wsum_4.0_1.0,wsum_1.0_0.0")
    a = ap.parse_args(argv)

    rec = next(r for r in json.load(open(a.manifest)) if r["dataset"] == a.dataset)

    import anny, warnings
    from anny import KeypointsRegressor
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        bm = anny.create_fullbody_model(local_changes=True,
                                        pose_parameterization="root_relative",
                                        remove_unattached_vertices=False).to(dtype=torch.float32)
    reg = KeypointsRegressor.coco(bm)
    w = reg.regression_weights.float()
    labels = list(reg.labels)
    from smpl_eval.runners.multihmr2 import COCO_LABEL_TO_SMPL
    pairs = [(labels.index(k), v) for k, v in COCO_LABEL_TO_SMPL.items() if k in labels]

    if os.path.exists(a.cache):
        print("캐시 사용: %s" % a.cache)
        blob = pickle.load(open(a.cache, "rb"))
    else:
        from multihmr2 import api
        print("추론 1회 실행 중 (이후 설정 변경은 GPU 불필요)...")
        s = api.init_hmr_session(os.path.join(a.repo, "checkpoints", "multihmr2.pt"))
        cache_detections(s, rec["video_path"], a.tmp, a.cache, w)
        blob = pickle.load(open(a.cache, "rb"))
    frames, img_size = blob["frames"], blob["img_size"]
    n_det = sum(len(f["conf"]) for f in frames if f)
    print("프레임 %d, 검출 %d개, img_size %d\n" % (len(frames), n_det, img_size))

    from smpl_eval.evaluate import load_gt
    from smpl_eval.conventions import DATASET_CONVENTION
    from smpl_eval.runners.base import bbox_from_joints2d
    from smpl_eval.metrics.identity import gt_track_purity

    gt, _ = load_gt(rec)
    _, mapping = DATASET_CONVENTION[a.dataset]
    W, H = rec["width"], rec["height"]

    print("%-20s %9s %9s %9s %9s" % ("combine (외모:위치)", "purity", "최저", "트랙수", "검출행"))
    print("-" * 62)
    for combine in a.combines.split(","):
        ids_per_frame = retrack(frames, img_size, combine)
        rows = []
        for t, fr in enumerate(frames):
            if fr is None:
                continue
            kp3 = fr["coco3d"]
            K = fr["K"]
            sc = W / (2.0 * float(K[0, 2])) if K[0, 2] > 0 else 1.0
            fx, fy = float(K[0, 0]) * sc, float(K[1, 1]) * sc
            cx, cy = float(K[0, 2]) * sc, float(K[1, 2]) * sc
            for i in range(len(fr["conf"])):
                j3 = np.full((24, 3), np.nan, np.float32)
                for ci, si in pairs:
                    j3[si] = kp3[i, ci]
                z = np.where(np.abs(j3[:, 2]) < 1e-6, 1e-6, j3[:, 2])
                j2 = np.stack([j3[:, 0] / z * fx + cx, j3[:, 1] / z * fy + cy], -1)
                rows.append((t, int(ids_per_frame[t][i]), j2.astype(np.float32)))
        if not rows:
            print("%-20s  (검출 없음)" % combine)
            continue
        j2d = np.stack([r[2] for r in rows])
        tracks = {"frame_ids": np.array([r[0] for r in rows], np.int32),
                  "track_ids": np.array([r[1] for r in rows], np.int32),
                  "joints2d": j2d, "bbox": bbox_from_joints2d(j2d)}
        pu = gt_track_purity(tracks, gt)
        wf, wp = combine.split("_")[1:]
        tot = float(wf) + float(wp)
        print("%-20s %9.3f %9.3f %9d %9d"
              % ("%s (%.0f%%:%.0f%%)" % (combine.replace("wsum_", ""),
                                          100 * float(wf) / tot, 100 * float(wp) / tot),
                 pu["mean_purity"], pu["min_purity"],
                 len(np.unique(tracks["track_ids"])), len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
