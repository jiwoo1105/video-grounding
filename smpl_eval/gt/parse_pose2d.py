"""Data1 전용 2D GT 파서.

형식: cam, frame, pid, bx, by, bw, bh, (x, y, conf) * 19  → 64 필드
필드가 7개뿐인 깨진 줄이 6개 존재한다 (2026-08-25 실측).
조용히 건너뛰지 않고 개수를 세어 돌려준다.

이 데이터가 게이트 2(GT 재투영 오버레이)의 입력이 된다 — Data1 의
19관절 규약을 영상 위에서 육안으로 확정하기 위한 유일한 수단이다.
"""
import numpy as np

N_JOINTS = 19
EXPECTED = 3 + 4 + N_JOINTS * 3      # 64


def parse_pose2d(path, cam_id):
    frames, pids, bboxes, joints, confs, n_bad = [], [], [], [], [], 0

    for line in open(path):
        f = [x.strip() for x in line.strip().split(",") if x.strip() != ""]
        if len(f) != EXPECTED:
            n_bad += 1
            continue
        if int(float(f[0])) != cam_id:
            continue
        bx, by, bw, bh = (float(x) for x in f[3:7])
        vals = np.array([float(x) for x in f[7:]], np.float32).reshape(N_JOINTS, 3)
        frames.append(int(float(f[1])))
        pids.append(int(float(f[2])))
        bboxes.append([bx, by, bx + bw, by + bh])
        joints.append(vals[:, :2])
        confs.append(vals[:, 2])

    n = len(frames)
    return {
        "frame_ids": np.array(frames, np.int32),
        "track_ids": np.array(pids, np.int32),
        "bbox": (np.array(bboxes, np.float32) if n
                 else np.zeros((0, 4), np.float32)),
        "joints2d": (np.array(joints, np.float32) if n
                     else np.zeros((0, N_JOINTS, 2), np.float32)),
        "conf": (np.array(confs, np.float32) if n
                 else np.zeros((0, N_JOINTS), np.float32)),
        "n_malformed": n_bad,
    }
