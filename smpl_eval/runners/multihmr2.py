"""Multi-HMR 2 실행 + tracks.npz 정규화.

CLI(`multihmr2 --save_anny_params`)는 K/shape/pose_parameters 만 저장하고
**트랙 ID 를 남기지 않는다**. ID 유지가 본 평가의 축이므로 Python API 를
직접 쓴다. `api.infer_video()` 는 FeatPelvisTracker 로 track_id 를 채워
프레임별 DecoderOutput 을 돌려준다 (2026-08-25 실측 확인).

PersonOutput 필드 (실측):
    conf (N,1) / shape (N,11) / v3d (N,19158,3) / j3d (N,163,3)
    j2d (N,163,2) / transl_pelvis (N,3) / bone_poses (N,163,4,4)
    track_id (N,) / trackfeat (N,4096)

### 관절 규약 — Anny → COCO-17 (SMPL 변환 불필요)

Anny 의 163본은 SMPL 24관절과 대응이 없다. 그런데 anny 패키지가
COCO 키포인트 회귀기를 내장하고 있어 메쉬 정점에서 COCO-17(+발 6개)을
직접 뽑을 수 있고, **우리 GT 가 바로 그 COCO-17 이다.**
따라서 Anny→SMPL 리토폴로지·피팅 없이 곧바로 비교할 수 있다.

실측 검증: 허벅지 0.366m, 정강이 0.352m, 상완 0.243m, 전완 0.206m,
허벅지/상완 = 1.51 (실제 인체 약 1.5). 미터 스케일도 타당하다.

`betas` 는 SMPL β 가 아니므로 NaN 으로 두고 Anny 체형 11개를
`betas_native` 에 보존한다. β 기반 지표는 metrics/plausibility 가
자동으로 뼈길이 분산으로 대체한다.

반드시 multihmr2 venv 안에서 실행해야 한다.
"""
import os

import numpy as np

from smpl_eval.runners.base import Runner, bbox_from_joints2d
from smpl_eval.conventions import SMPL24

_S = {n: i for i, n in enumerate(SMPL24)}

# anny COCO 회귀기의 라벨 → SMPL24 슬롯.
# 얼굴 5개(nose/eyes/ears)는 SMPL 에 대응 관절이 없어 제외한다.
COCO_LABEL_TO_SMPL = {
    "left_shoulder": _S["left_shoulder"],   "right_shoulder": _S["right_shoulder"],
    "left_elbow":    _S["left_elbow"],      "right_elbow":    _S["right_elbow"],
    "left_wrist":    _S["left_wrist"],      "right_wrist":    _S["right_wrist"],
    "left_hip":      _S["left_hip"],        "right_hip":      _S["right_hip"],
    "left_knee":     _S["left_knee"],       "right_knee":     _S["right_knee"],
    "left_ankle":    _S["left_ankle"],      "right_ankle":    _S["right_ankle"],
    # SMPL 의 foot 관절은 발가락 쪽을 향하므로 big_toe 를 쓴다
    "left_big_toe":  _S["left_foot"],       "right_big_toe":  _S["right_foot"],
}


class MultiHMR2Runner(Runner):
    name = "multihmr2"
    body_model = "anny"

    def __init__(self, repo_dir, checkpoint=None, tmp_root=None):
        self.repo_dir = repo_dir
        self.checkpoint = checkpoint or os.path.join(
            repo_dir, "checkpoints", "multihmr2.pt")
        self.tmp_root = tmp_root or os.path.expanduser("~/mh_frames_tmp")
        self._session = None
        self._regressor = None

    # ── 내부 자원 ────────────────────────────────────────────────
    def _get_session(self):
        if self._session is None:
            from multihmr2 import api
            self._session = api.init_hmr_session(self.checkpoint)
        return self._session

    def _get_regressor(self):
        """세션이 실제로 쓴 body model 로부터 COCO 회귀기를 만든다.

        정점 수(19,158)가 일치해야 하므로 반드시 같은 모델이어야 한다.
        """
        if self._regressor is None:
            from anny import KeypointsRegressor
            bm = self._get_session().model.full_body_decoder.body_model
            self._regressor = KeypointsRegressor.coco(bm)
        return self._regressor

    # ── Runner 인터페이스 ────────────────────────────────────────
    def _invoke(self, video_path, raw_dir, max_frames):
        """추론 결과를 메모리에 담아둔다 (원본 파일을 남기지 않음).

        Multi-HMR 2 는 프레임을 PNG 로 모두 추출한 뒤 추론한다. 2K 영상
        기준 장당 약 2MB 이므로 전수 처리 시 임시파일이 수십 GB 에 달한다.
        영상마다 tmp 를 비우고 재사용한다.
        """
        import shutil
        from multihmr2 import api

        os.makedirs(raw_dir, exist_ok=True)
        src = video_path
        if max_frames:
            src = os.path.join(raw_dir, "_trim.mp4")
            from smpl_eval.runners.base import sh
            sh(["ffmpeg", "-v", "error", "-y", "-i", os.path.abspath(video_path),
                "-frames:v", str(max_frames), "-c", "copy", src])

        tmp = self.tmp_root
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp, exist_ok=True)
        try:
            self._preds = api.infer_video(self._get_session(),
                                          os.path.abspath(src), tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _normalize(self, raw_dir, video_meta):
        import torch

        preds = getattr(self, "_preds", None)
        if not preds:
            raise RuntimeError("추론 결과가 없습니다 (_invoke 가 실행되지 않음)")

        reg = self._get_regressor()
        w = reg.regression_weights.float()
        labels = list(reg.labels)
        # COCO 라벨 인덱스 → SMPL24 슬롯
        pairs = [(labels.index(k), v) for k, v in COCO_LABEL_TO_SMPL.items()
                 if k in labels]

        W, H = video_meta["width"], video_meta["height"]
        rows = []
        for t, fp in enumerate(preds):
            p = fp.persons
            n = int(getattr(p, "num_person", 0) or 0)
            if n == 0 or p.v3d is None:
                continue
            v3d = p.v3d.detach().cpu().float()
            kp3 = torch.einsum("kv,nvc->nkc", w, v3d).numpy()      # (n, 23, 3)

            K = fp.K.detach().cpu().float().numpy()
            # K 는 모델 입력 해상도(cx=W_in/2) 기준이다. 원본으로 되돌린다.
            scale = W / (2.0 * float(K[0, 2])) if K[0, 2] > 0 else 1.0
            fx, fy = float(K[0, 0]) * scale, float(K[1, 1]) * scale
            cx, cy = float(K[0, 2]) * scale, float(K[1, 2]) * scale

            tid = np.asarray(p.track_id).reshape(-1).astype(np.int32)
            conf = np.asarray(p.conf.detach().cpu().float()).reshape(-1)
            shape = np.asarray(p.shape.detach().cpu().float())
            transl = np.asarray(p.transl_pelvis.detach().cpu().float())

            for i in range(n):
                j3 = np.full((24, 3), np.nan, np.float32)
                for ci, si in pairs:
                    j3[si] = kp3[i, ci]
                z = np.where(np.abs(j3[:, 2]) < 1e-6, 1e-6, j3[:, 2])
                j2 = np.stack([j3[:, 0] / z * fx + cx,
                               j3[:, 1] / z * fy + cy], -1).astype(np.float32)
                rows.append({
                    "frame": t, "track": int(tid[i]),
                    "shape": shape[i], "conf": float(conf[i]),
                    "transl": transl[i], "j3d": j3, "j2d": j2,
                })

        if not rows:
            raise RuntimeError("검출된 사람이 없습니다")

        n = len(rows)
        j2d = np.stack([r["j2d"] for r in rows])
        arrays = {
            "frame_ids": np.array([r["frame"] for r in rows], np.int32),
            "track_ids": np.array([r["track"] for r in rows], np.int32),
            "betas": np.full((n, 10), np.nan, np.float32),
            "betas_native": np.stack([r["shape"] for r in rows]).astype(np.float32),
            "global_orient": np.full((n, 3), np.nan, np.float32),
            "body_pose": np.full((n, 23, 3), np.nan, np.float32),
            "transl": np.stack([r["transl"] for r in rows]).astype(np.float32),
            "joints3d": np.stack([r["j3d"] for r in rows]).astype(np.float32),
            "joints2d": j2d,
            "bbox": bbox_from_joints2d(j2d),
            "score": np.array([r["conf"] for r in rows], np.float32),
        }
        return arrays, {
            "converted_from": "anny",
            "joint_source": "anny COCO keypoint regressor (23 labels)",
            "n_mapped_joints": len(pairs),
            "camera": "multihmr2_predicted_K",
        }
