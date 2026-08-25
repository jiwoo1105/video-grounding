"""CoMotion 실행 + tracks.npz 정규화.

출력 구조 (2026-08-25 실측, ELICE.md 참조):
    id        (N,)     int64    트랙 ID (1-base)
    pose      (N, 72)  float32  SMPL θ, axis-angle 24관절
    trans     (N, 3)   float32  루트 위치
    betas     (N, 10)  float32  체형
    frame_idx (N,)     int64    프레임 (0-base)
  + <이름>.txt : MOT 포맷 (frame 1-base, id, x, y, w, h, conf, ...)

반드시 comotion venv 안에서 실행해야 한다 (comotion_demo 임포트 필요).
"""
import glob
import os

import numpy as np

from smpl_eval.runners.base import (
    Runner, sh, project_default_K, bbox_from_joints2d, parse_mot)

# 실측으로 확정한 키 이름
KEY_ID, KEY_POSE, KEY_TRANS, KEY_BETAS, KEY_FRAME = (
    "id", "pose", "trans", "betas", "frame_idx")


class CoMotionRunner(Runner):
    name = "comotion"
    body_model = "smpl"

    def __init__(self, repo_dir, python=None):
        self.repo_dir = repo_dir
        # 기본은 현재 인터프리터 — 러너는 자기 모델 venv 안에서 돌아야 한다
        import sys
        self.python = python or sys.executable

    def _invoke(self, video_path, raw_dir, max_frames):
        os.makedirs(raw_dir, exist_ok=True)
        cmd = [self.python, "demo.py",
               "-i", os.path.abspath(video_path),
               "-o", os.path.abspath(raw_dir),
               "--skip-visualization"]
        if max_frames:
            cmd += ["--num-frames", str(max_frames)]
        sh(cmd, cwd=self.repo_dir)

    def _normalize(self, raw_dir, video_meta):
        import torch

        pts = sorted(glob.glob(os.path.join(raw_dir, "**", "*.pt"), recursive=True))
        if not pts:
            raise FileNotFoundError(f"{raw_dir} 에 .pt 출력이 없습니다")
        d = torch.load(pts[0], map_location="cpu", weights_only=False)

        pose = np.asarray(d[KEY_POSE], np.float32).reshape(-1, 24, 3)
        n = len(pose)
        betas = np.asarray(d[KEY_BETAS], np.float32).reshape(n, 10)
        trans = np.asarray(d[KEY_TRANS], np.float32).reshape(n, 3)
        frame_ids = np.asarray(d[KEY_FRAME], np.int64).reshape(n).astype(np.int32)
        track_ids = np.asarray(d[KEY_ID], np.int64).reshape(n).astype(np.int32)

        joints3d = self._forward_smpl(betas, pose, trans)
        joints2d = project_default_K(joints3d, video_meta["width"], video_meta["height"])

        # ★ bbox 는 **관절에서 유도한다**. 모델이 낸 MOT bbox 를 쓰면 안 된다.
        #
        # CoMotion 의 MOT bbox 는 여백을 준 검출 박스라 GT 보다 각 변이 1.5배
        # 크다 (Data3 실측: GT 198x547 vs MOT 300x749). GT bbox 는 투영된
        # 관절에서 유도하고 Multi-HMR 2 도 같은 방식이므로, MOT bbox 를 쓰면
        # IoU 가 0.41~0.48 로 임계값 0.5 아래에 걸려 매칭이 전부 실패한다.
        # 그러면 추적 품질이 아니라 **bbox 생성 방식 차이**로 점수가 갈린다.
        #
        # 모델의 원본 bbox 는 참고용으로 meta 에만 남긴다.
        bbox = bbox_from_joints2d(joints2d)

        mot = parse_mot(os.path.splitext(pts[0])[0] + ".txt")
        score = np.ones(n, np.float32)
        n_from_mot = 0
        for i in range(n):
            hit = mot.get((int(frame_ids[i]), int(track_ids[i])))
            if hit:
                score[i] = hit[1]
                n_from_mot += 1

        arrays = {
            "frame_ids": frame_ids, "track_ids": track_ids,
            "betas": betas,
            "global_orient": pose[:, 0], "body_pose": pose[:, 1:],
            "transl": trans, "joints3d": joints3d, "joints2d": joints2d,
            "bbox": bbox, "score": score,
        }
        return arrays, {"raw_file": os.path.basename(pts[0]),
                        "bbox_source": "joints2d",
                        "score_from_mot": int(n_from_mot),
                        "camera": "comotion_default_K"}

    @staticmethod
    def _forward_smpl(betas, pose, trans):
        """SMPL 순전파로 24관절 3D 좌표를 얻는다 (CoMotion 내장 구현 사용)."""
        import torch
        from comotion_demo.utils.smpl_kinematics import SMPLKinematics

        m = SMPLKinematics()
        with torch.no_grad():
            j = m(torch.from_numpy(betas),
                  torch.from_numpy(pose.reshape(len(pose), 72)),
                  torch.from_numpy(trans),
                  output_format="joints")
        return np.asarray(j, np.float32)[:, :24]
