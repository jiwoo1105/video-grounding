"""tracks.npz 표준 포맷 — 모델별 출력을 여기로 정규화한다.

CoMotion 과 Multi-HMR 2 는 출력 형식·바디모델·관절 규약이 모두 다르다.
어댑터가 이 스키마로 정규화하면 이후 모든 평가 코드는 어느 모델인지
모른 채 동작하고, 비교의 공정성이 보장된다.

한 행 = "프레임 f 에 존재하는 사람 t". 모든 배열의 첫 축 길이 N 은 동일.
"""
import json
import numpy as np

# 필드명 → (shape suffix, dtype).  실제 shape 는 (N,) + suffix
TRACK_FIELDS = {
    "frame_ids":     ((),      "int32"),    # 0-based 프레임 번호
    "track_ids":     ((),      "int32"),    # 모델이 부여한 트랙 ID
    "betas":         ((10,),   "float32"),  # 체형. SMPL β 규약
    "global_orient": ((3,),    "float32"),  # 루트 회전 (axis-angle)
    "body_pose":     ((23, 3), "float32"),  # 관절 회전, 부모 대비 상대
    "transl":        ((3,),    "float32"),  # 카메라 좌표계 루트 위치 (m)
    "joints3d":      ((24, 3), "float32"),  # SMPL 24관절, 카메라 좌표계
    "joints2d":      ((24, 2), "float32"),  # 이미지 픽셀 좌표
    "bbox":          ((4,),    "float32"),  # x1, y1, x2, y2 픽셀
    "score":         ((),      "float32"),  # 검출 신뢰도
}

# shape (N, K) 이되 K 가 모델마다 다른 선택 필드.
# Anny 는 해석 가능한 체형 파라미터 6개를 여기 보존한다.
OPTIONAL_FIELDS = {"betas_native"}


def validate_tracks(arrays):
    """스키마 위반 시 ValueError. 저장 전에 반드시 통과해야 한다."""
    missing = [k for k in TRACK_FIELDS if k not in arrays]
    if missing:
        raise ValueError(f"필수 필드 누락: {', '.join(missing)}")

    n = len(arrays["frame_ids"])
    for k, (suffix, _dt) in TRACK_FIELDS.items():
        a = np.asarray(arrays[k])
        if len(a) != n:
            raise ValueError(f"{k}: length {len(a)} != {n}")
        if a.shape != (n,) + suffix:
            raise ValueError(f"{k}: shape {a.shape} != {(n,) + suffix}")

    for k in arrays:
        if k in OPTIONAL_FIELDS and len(np.asarray(arrays[k])) != n:
            raise ValueError(f"{k}: length mismatch (expected {n})")


def save_tracks(path, arrays, meta):
    """검증 후 압축 저장. meta 는 JSON 으로 직렬화해 같은 파일에 넣는다."""
    validate_tracks(arrays)
    payload = {}
    for k, v in arrays.items():
        dtype = TRACK_FIELDS[k][1] if k in TRACK_FIELDS else "float32"
        payload[k] = np.asarray(v, dtype=dtype)
    payload["__meta__"] = np.frombuffer(
        json.dumps(meta, ensure_ascii=False).encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(path, **payload)


def load_tracks(path):
    """(arrays, meta) 를 돌려준다."""
    d = np.load(path, allow_pickle=False)
    meta = json.loads(bytes(d["__meta__"]).decode("utf-8"))
    arrays = {k: d[k] for k in d.files if k != "__meta__"}
    return arrays, meta
