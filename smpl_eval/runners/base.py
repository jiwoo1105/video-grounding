"""러너 공통 인터페이스.

러너는 외부 모델을 호출하고 그 출력을 tracks.npz 로 정규화하는 것만
책임진다. 지표가 무엇인지 알지 못한다.

각 러너는 **자기 모델의 venv 안에서** 실행되어야 한다. 모델 패키지를
직접 임포트해 관절을 계산하기 때문이다.
    ~/venvs/comotion/bin/python  -m smpl_eval.run_all --model comotion
    ~/venvs/multihmr2/bin/python -m smpl_eval.run_all --model multihmr2
"""
import abc
import os
import subprocess
import sys
import time

import numpy as np


class Runner(abc.ABC):
    name = "base"
    body_model = "smpl"

    @abc.abstractmethod
    def _invoke(self, video_path, raw_dir, max_frames):
        """외부 모델을 실행해 raw_dir 에 원본 출력을 남긴다."""

    @abc.abstractmethod
    def _normalize(self, raw_dir, video_meta):
        """(arrays, extra_meta) 를 돌려준다. arrays 는 schema 를 만족해야 한다."""

    def run(self, video_path, out_dir, video_meta, max_frames=None, force=False):
        os.makedirs(out_dir, exist_ok=True)
        dst = os.path.join(out_dir, "tracks.npz")
        if os.path.exists(dst) and not force:
            return dst

        raw_dir = os.path.join(out_dir, "raw")
        t0 = time.time()
        self._invoke(video_path, raw_dir, max_frames)
        invoke_sec = time.time() - t0

        arrays, extra = self._normalize(raw_dir, video_meta)

        from smpl_eval.schema import save_tracks
        n_frames = max_frames or video_meta["n_frames"]
        meta = {
            "model": self.name,
            "body_model": self.body_model,
            "converted_from": None,
            "video": video_path,
            "fps": video_meta["fps"],
            "resolution": [video_meta["width"], video_meta["height"]],
            "n_frames": int(n_frames),
            "runtime_sec": round(invoke_sec, 1),
            "ms_per_frame": round(invoke_sec * 1000.0 / max(n_frames, 1), 1),
        }
        meta.update(extra)
        save_tracks(dst, arrays, meta)
        return dst


def sh(cmd, cwd=None):
    """서브프로세스 실행. 실패하면 stderr 를 포함해 예외를 던진다."""
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-25:])
        raise RuntimeError(f"명령 실패 ({r.returncode}): {' '.join(cmd)}\n{tail}")
    return r.stdout


def project_default_K(j3d, width, height):
    """CoMotion 의 get_default_K 와 동일한 근사 내부 파라미터로 투영한다.

    K = [[2*max(W,H), 0, W/2], [0, 2*max(W,H), H/2]]
    """
    f = 2.0 * max(width, height)
    z = np.where(np.abs(j3d[..., 2]) < 1e-6, 1e-6, j3d[..., 2])
    return np.stack([j3d[..., 0] / z * f + width / 2.0,
                     j3d[..., 1] / z * f + height / 2.0], -1).astype(np.float32)


def bbox_from_joints2d(j2d):
    """관절 2D 로부터 축정렬 bbox (x1,y1,x2,y2).

    ★ 모든 러너와 GT 가 **반드시 이 방식으로** bbox 를 만들어야 한다.
    예측-GT 매칭이 bbox IoU 이므로, 한쪽만 모델의 원본 검출 박스(보통
    여백이 있음)를 쓰면 IoU 가 체계적으로 낮아져 추적 품질과 무관하게
    점수가 갈린다. (실측: CoMotion 의 MOT bbox 는 GT 대비 각 변 1.5배,
    IoU 0.41~0.48 로 임계값 0.5 아래 → ID 지표 전체가 무의미해졌다)
    """
    ok = np.isfinite(j2d).all(-1)
    out = np.zeros((len(j2d), 4), np.float32)
    for i in range(len(j2d)):
        v = j2d[i][ok[i]]
        if len(v):
            out[i] = [v[:, 0].min(), v[:, 1].min(), v[:, 0].max(), v[:, 1].max()]
    return out


def parse_mot(path):
    """MOT 포맷 → {(frame0, track_id): (bbox_xyxy, conf)}

    MOT 의 프레임 번호는 1-base 이므로 0-base 로 낮춰서 키를 만든다.
    """
    out = {}
    if not os.path.isfile(path):
        return out
    for line in open(path):
        f = [x.strip() for x in line.strip().split(",") if x.strip() != ""]
        if len(f) < 7:
            continue
        fr = int(float(f[0])) - 1
        tid = int(float(f[1]))
        x, y, w, h = (float(v) for v in f[2:6])
        out[(fr, tid)] = ((x, y, x + w, y + h), float(f[6]))
    return out
