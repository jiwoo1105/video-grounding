"""관절 규약 정의와 규약 간 매핑. 정본은 SMPL 24관절.

본 프로젝트에서 오류 가능성이 가장 높은 지점이라 별도 모듈로 격리했다.
파일럿 게이트 2(GT 재투영 오버레이)가 이 파일의 매핑을 검증한다.
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

# Data2/3/4 GT (3DPose.txt, 70필드 = 2 + 17*4).
# obj*.bvh 의 계층이 Hip→RightHip→…→RightWrist 16관절이고 여기에
# Head 를 index 10 에 넣으면 Human3.6M 표준 17관절 순서와 일치한다.
H36M17 = [
    "Hip", "RightHip", "RightKnee", "RightAnkle", "LeftHip", "LeftKnee",
    "LeftAnkle", "Spine", "Thorax", "Neck", "Head", "LeftShoulder",
    "LeftElbow", "LeftWrist", "RightShoulder", "RightElbow", "RightWrist",
]

# Data1 GT 19관절 중 앞 17개에 대한 **가설**.
# 근거: PoseResults2d 에서 앞 5개 관절의 신뢰도가 유독 낮고(0.02~0.39)
#       위치가 머리 영역에 몰려 있다 → 얼굴 키포인트로 보인다.
# 나머지 2관절(index 17, 18)의 정체는 미상.
# 게이트 2 에서 확정할 것.  → smpl_eval/GATES.md
COCO17 = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
DATA1_N_JOINTS = 19

_S = {n: i for i, n in enumerate(SMPL24)}

H36M17_TO_SMPL24 = {
    0:  _S["pelvis"],          1:  _S["right_hip"],    2:  _S["right_knee"],
    3:  _S["right_ankle"],     4:  _S["left_hip"],     5:  _S["left_knee"],
    6:  _S["left_ankle"],      7:  _S["spine2"],       8:  _S["spine3"],
    9:  _S["neck"],            10: _S["head"],         11: _S["left_shoulder"],
    12: _S["left_elbow"],      13: _S["left_wrist"],   14: _S["right_shoulder"],
    15: _S["right_elbow"],     16: _S["right_wrist"],
}

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


def common_indices(mapping):
    """정렬된 (gt_indices, smpl_indices) 쌍을 반환한다."""
    keys = sorted(mapping)
    return np.array(keys, int), np.array([mapping[k] for k in keys], int)
