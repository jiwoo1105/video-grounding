"""GT 3D 포즈 파서.

Data1  PoseResults3d_*.txt : frame, pid, (x,y,z,conf) * 19   → 78 필드
Data2-4 3DPose.txt         : frame, pid, (x,y,z,conf) * 17   → 70 필드

파싱 결과는 원본 관절 규약 그대로이며, to_gt_tracks 가 SMPL24 레이아웃으로
옮긴다. 그 이후로는 모든 코드가 SMPL24 인덱스만 쓴다.
"""
import numpy as np


def parse_pose3d(path, n_joints):
    """필드 수가 안 맞는 줄은 세어서 n_malformed 로 돌려준다 (조용히 버리지 않음)."""
    expected = 2 + n_joints * 4
    frames, pids, joints, confs, n_bad = [], [], [], [], 0

    for line in open(path):
        f = [x.strip() for x in line.strip().split(",") if x.strip() != ""]
        if len(f) != expected:
            n_bad += 1
            continue
        vals = np.array([float(x) for x in f[2:]], np.float32).reshape(n_joints, 4)
        frames.append(int(float(f[0])))
        pids.append(int(float(f[1])))
        joints.append(vals[:, :3])
        confs.append(vals[:, 3])

    n = len(frames)
    return {
        "frame_ids": np.array(frames, np.int32),
        "track_ids": np.array(pids, np.int32),
        "joints3d": (np.array(joints, np.float32) if n
                     else np.zeros((0, n_joints, 3), np.float32)),
        "conf": (np.array(confs, np.float32) if n
                 else np.zeros((0, n_joints), np.float32)),
        "n_malformed": n_bad,
    }


def detect_frame_offset(gt_frames, n_video_frames):
    """GT 프레임 번호를 0-base 영상 인덱스로 맞추는 오프셋을 추정한다.

    반환값 k 에 대해  video_index = gt_frame - k.

    GT 의 최소 프레임 번호가 영상의 0번 프레임에 대응한다고 가정한 **추정치**다.
    데이터셋마다 1-base/0-base 가 다르고 앞부분 프레임이 빠진 경우도 있어
    파일 내용만으로는 확정할 수 없다. 파일럿 게이트 2(GT 재투영 오버레이)가
    이 값을 육안 검증한다.  → smpl_eval/GATES.md
    """
    gt_frames = np.asarray(gt_frames)
    if gt_frames.size == 0:
        return 0
    lo = int(gt_frames.min())
    span = int(gt_frames.max()) - lo + 1
    if span > n_video_frames:
        raise ValueError(
            f"GT 프레임 범위({span})가 영상 프레임수({n_video_frames})보다 큽니다")
    return lo


def to_gt_tracks(parsed, mapping, image_wh=None, camera=None):
    """파싱 결과를 tracks.npz 호환 dict 로 변환. 미대응 SMPL 슬롯은 NaN.

    GT 에는 β·포즈 파라미터가 없으므로 해당 필드는 전부 NaN 이다.
    스키마를 공유해야 지표 코드가 예측/GT 를 구분 없이 다룰 수 있다.

    camera: ColmapCamera. **반드시 넘겨야 한다.** GT 는 SfM 월드 좌표라
      화면 좌표를 모르는데, 예측과 GT 를 짝짓는 것은 bbox IoU 이기 때문이다.
      약원근 근사로 대신하면 z 가 0 을 지나며 발산해 bbox 가 수천만 픽셀까지
      튄다(Data4 실측: 화면 안 0%). 그러면 ID·포즈 지표가 전부 무의미해진다.
      camera 가 없으면 joints2d/bbox 는 NaN 으로 두고 3D 지표만 쓸 수 있다.
    """
    n = len(parsed["frame_ids"])
    j3 = np.full((n, 24, 3), np.nan, np.float32)
    for g, s in mapping.items():
        j3[:, s] = parsed["joints3d"][:, g]

    if camera is not None:
        j2 = camera.project(j3).astype(np.float32)
    else:
        j2 = np.full((n, 24, 2), np.nan, np.float32)

    bbox = np.zeros((n, 4), np.float32)
    finite = np.isfinite(j2).all(-1)
    for i in range(n):
        v = j2[i][finite[i]]
        if len(v):
            bbox[i] = [v[:, 0].min(), v[:, 1].min(), v[:, 0].max(), v[:, 1].max()]

    # 루트 위치. COCO 계열 GT 에는 골반 관절이 없으므로 좌우 엉덩이의
    # 중점을 루트로 삼는다 (SMPL 골반과 거의 같은 위치).
    root = j3[:, 0].copy()
    if n and not np.isfinite(root).any():
        with np.errstate(invalid="ignore"):
            root = np.nanmean(j3[:, [1, 2]], axis=1).astype(np.float32)

    with np.errstate(invalid="ignore"):
        score = np.nanmean(parsed["conf"], axis=-1) if n else np.zeros(0)
    score = np.nan_to_num(np.asarray(score, np.float32), nan=0.0)

    return {
        "frame_ids": parsed["frame_ids"].astype(np.int32),
        "track_ids": parsed["track_ids"].astype(np.int32),
        "betas": np.full((n, 10), np.nan, np.float32),
        "global_orient": np.full((n, 3), np.nan, np.float32),
        "body_pose": np.full((n, 23, 3), np.nan, np.float32),
        "transl": root,
        "joints3d": j3,
        "joints2d": j2,
        "bbox": bbox,
        "score": score,
    }
