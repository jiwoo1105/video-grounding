"""관절 규약 정의와 규약 간 매핑. 정본은 SMPL 24관절.

본 프로젝트에서 오류 가능성이 가장 높은 지점이라 별도 모듈로 격리했다.

## GT 규약 확정 경위 (2026-08-25)

처음에는 obj*.bvh 의 계층(Hip→RightHip→…→RightWrist)을 근거로
Data2/3/4 GT 를 Human3.6M 17관절로 가정했으나, **실측 결과 틀렸다.**
BVH 는 별도로 변환된 표현이고 3DPose.txt 는 COCO 순서를 쓴다.

근거 1 — 좌표 구조: 관절이 인덱스 순서대로 높이가 단조 증가하며
  idx 0~4 가 한 덩어리(얼굴), 이후 (5,6) (7,8) (9,10) (11,12) (13,14) (15,16)
  좌우 쌍이 차례로 내려온다. H36M 의 다리-먼저 순서와 맞지 않는다.

근거 2 — 뼈 길이 변동계수(낮을수록 강체 = 실제 뼈):

  | 데이터 | COCO-17 | H36M-17 |
  |--------|---------|---------|
  | Data1  | 0.045   | —       |
  | Data2  | 0.046   | 0.106   |
  | Data3  | 0.056   | 0.066   |
  | Data4  | 0.062   | 0.085   |

근거 3 — 인체 비율: COCO 로 해석하면 허벅지≈정강이, 상완>전완,
  허벅지/상완 ≈ 1.44~1.63 으로 실제 인체(≈1.5)와 일치한다.

## 스케일

GT 좌표는 COLMAP/SfM 재구성 결과라 **데이터셋마다 임의 스케일**이다.
같은 허벅지가 Data2 는 0.238, Data3 는 0.377, Data4 는 0.949 로 나온다.
따라서 GT 좌표를 mm 로 직역하면 안 되며, metrics/pose.py 가
CANONICAL_SKELETON_SCALE_MM 기준으로 정규화한다.
"""
import numpy as np

# 정본 — SMPL 표준 관절 순서
SMPL24 = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hand", "right_hand",
]

# Data2/3/4 GT (3DPose.txt, 70필드 = 2 + 17*4)
COCO17 = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Data1 GT (PoseResults3d, 78필드 = 2 + 19*4) = COCO-17 + 발 2개.
# 17·18 은 발목보다 아래에 있고 같은 쪽 발목과 0.58 거리로 붙어 있다
# (반대쪽 발목과는 2.3). 정강이 길이 1.21 대비 타당한 발 길이다.
COCO19 = COCO17 + ["left_foot", "right_foot"]

# 참고용 — BVH 파일의 계층 순서. 3DPose.txt 와는 무관하다.
BVH16 = [
    "Hip", "RightHip", "RightKnee", "RightAnkle", "LeftHip", "LeftKnee",
    "LeftAnkle", "Spine", "Thorax", "Neck", "LeftShoulder", "LeftElbow",
    "LeftWrist", "RightShoulder", "RightElbow", "RightWrist",
]

_S = {n: i for i, n in enumerate(SMPL24)}

# 얼굴 5개(nose/eyes/ears)는 SMPL 에 대응 관절이 없어 제외한다.
# SMPL 의 head(15) 는 두개골 중심이지 코가 아니므로 nose 를 붙이면 안 된다.
COCO17_TO_SMPL24 = {
    5:  _S["left_shoulder"],   6:  _S["right_shoulder"],
    7:  _S["left_elbow"],      8:  _S["right_elbow"],
    9:  _S["left_wrist"],      10: _S["right_wrist"],
    11: _S["left_hip"],        12: _S["right_hip"],
    13: _S["left_knee"],       14: _S["right_knee"],
    15: _S["left_ankle"],      16: _S["right_ankle"],
}

COCO19_TO_SMPL24 = dict(COCO17_TO_SMPL24)
COCO19_TO_SMPL24[17] = _S["left_foot"]
COCO19_TO_SMPL24[18] = _S["right_foot"]

# 데이터셋 → (GT 관절 수, 매핑)
DATASET_CONVENTION = {
    "Data1": (19, COCO19_TO_SMPL24),
    "Data2": (17, COCO17_TO_SMPL24),
    "Data3": (17, COCO17_TO_SMPL24),
    "Data4": (17, COCO17_TO_SMPL24),
}

# 표준 인체(키 1.5m T-포즈) 관절의 중심으로부터 평균거리.
# 임의 스케일 GT 의 오차를 해석 가능한 mm 로 환산하는 기준이다.
CANONICAL_SKELETON_SCALE_MM = 544.6


def common_indices(mapping):
    """정렬된 (gt_indices, smpl_indices) 쌍을 반환한다."""
    keys = sorted(mapping)
    return np.array(keys, int), np.array([mapping[k] for k in keys], int)


def skeleton_scale(joints):
    """관절 집합의 중심으로부터 평균거리. NaN 은 무시한다.

    joints: (..., J, 3)  →  (...,)
    """
    j = np.asarray(joints, float)
    c = np.nanmean(j, axis=-2, keepdims=True)
    return np.nanmean(np.linalg.norm(j - c, axis=-1), axis=-1)
