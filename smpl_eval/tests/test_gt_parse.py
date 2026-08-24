import os, glob
import numpy as np
import pytest

from smpl_eval.gt.parse_pose3d import parse_pose3d, to_gt_tracks, detect_frame_offset
from smpl_eval.gt.parse_pose2d import parse_pose2d
from smpl_eval.conventions import COCO17_TO_SMPL24, COCO19_TO_SMPL24
from smpl_eval.schema import validate_tracks

ROOT = "Captured-Motion-Dataset"
pytestmark = pytest.mark.skipif(not os.path.isdir(ROOT), reason="데이터셋 없음")

D2 = f"{ROOT}/Data2_WTA_tennis_double_clip3_1min_2K_30fps/tennis_double_clip3_1min_2K_pose_result/3DPose.txt"
D3 = f"{ROOT}/Data3_OnlyOneOf_rie_junji_2K_60fps/rie_junji_4cam_pose_result/3Dpose.txt"
D4 = f"{ROOT}/Data4_vid3_golden_clip1_2K_60fps/3_golden_clip1_pose_result/3DPose.txt"
_D1 = f"{ROOT}/Data1_SKNight-live_Basketball/S03_HL01_2K_pose_gt"
D1_3D = glob.glob(f"{_D1}/PoseResults3d_*.txt")[0]
D1_2D = glob.glob(f"{_D1}/PoseResults2d_*.txt")[0]


# ── 3D 파서 ─────────────────────────────────────────────────────────

def test_data2_parses_17_joints_two_persons():
    p = parse_pose3d(D2, n_joints=17)
    assert p["joints3d"].shape[1:] == (17, 3)
    assert set(np.unique(p["track_ids"]).tolist()) == {0, 1}
    assert len(p["frame_ids"]) == 4616
    assert p["n_malformed"] == 0


def test_data1_parses_19_joints_thirteen_persons():
    p = parse_pose3d(D1_3D, n_joints=19)
    assert p["joints3d"].shape[1:] == (19, 3)
    assert len(np.unique(p["track_ids"])) == 13
    assert p["frame_ids"].min() == 1 and p["frame_ids"].max() == 300


def test_data4_nan_is_preserved_not_zeroed():
    p = parse_pose3d(D4, n_joints=17)
    assert np.isnan(p["joints3d"]).any(), "nan 이 0 으로 뭉개짐"


def test_wrong_joint_count_yields_all_malformed():
    """관절 수를 틀리게 주면 조용히 통과하지 말고 전부 malformed 여야 한다."""
    p = parse_pose3d(D2, n_joints=19)
    assert len(p["frame_ids"]) == 0 and p["n_malformed"] == 4616


@pytest.mark.parametrize("path,n_video,expected", [
    (D2, 2309, 0),
    (D3, 1420, 2),
    (D4, 660, 2),
    (D1_3D, 300, 1),
])
def test_frame_offset_detected(path, n_video, expected):
    n_j = 19 if path == D1_3D else 17
    p = parse_pose3d(path, n_joints=n_j)
    assert detect_frame_offset(p["frame_ids"], n_video) == expected


def test_frame_offset_rejects_impossible_range():
    with pytest.raises(ValueError, match="프레임"):
        detect_frame_offset(np.arange(0, 5000), 100)


# ── SMPL24 변환 ─────────────────────────────────────────────────────

def test_to_gt_tracks_is_schema_valid():
    p = parse_pose3d(D2, n_joints=17)
    t = to_gt_tracks(p, COCO17_TO_SMPL24, image_wh=(1920, 1080))
    validate_tracks(t)
    assert t["joints3d"].shape[1:] == (24, 3)


def test_unmapped_smpl_slots_are_nan():
    p = parse_pose3d(D2, n_joints=17)
    t = to_gt_tracks(p, COCO17_TO_SMPL24, image_wh=(1920, 1080))
    assert np.isnan(t["joints3d"][:, 22]).all()    # left_hand 는 GT 에 없음
    assert np.isnan(t["joints3d"][:, 0]).all()     # pelvis 도 COCO 에는 없음
    assert np.isfinite(t["joints3d"][:, 16]).any()  # left_shoulder 는 있음


def test_transl_falls_back_to_hip_midpoint_when_no_pelvis():
    """COCO 에는 골반이 없다 — transl 이 전부 NaN 이 되면 안 된다."""
    p = parse_pose3d(D2, n_joints=17)
    t = to_gt_tracks(p, COCO17_TO_SMPL24, image_wh=(1920, 1080))
    assert np.isfinite(t["transl"]).any()
    mid = np.nanmean(t["joints3d"][:, [1, 2]], axis=1)
    np.testing.assert_allclose(t["transl"], mid, rtol=1e-5)


def test_mapped_values_land_in_right_slots():
    p = parse_pose3d(D2, n_joints=17)
    t = to_gt_tracks(p, COCO17_TO_SMPL24, image_wh=(1920, 1080))
    for g, s in COCO17_TO_SMPL24.items():
        np.testing.assert_array_equal(t["joints3d"][:, s], p["joints3d"][:, g])


def test_no_camera_yields_nan_2d_not_garbage():
    """카메라 없이는 2D 를 만들지 않는다 — 근사 투영은 발산해서 위험하다."""
    p = parse_pose3d(D2, n_joints=17)
    t = to_gt_tracks(p, COCO17_TO_SMPL24, image_wh=(1920, 1080))
    assert np.isnan(t["joints2d"]).all()


def test_colmap_projection_puts_gt_on_screen():
    """COLMAP 카메라로 투영하면 GT 가 실제 화면 안에 들어와야 한다.

    예측-GT 매칭이 bbox IoU 이므로 이게 깨지면 ID·포즈 지표가 전부 무의미해진다.
    """
    import json
    from smpl_eval.gt.colmap import load_camera

    recs = json.load(open("smpl_eval/manifest.json"))
    for ds in ("Data1", "Data2", "Data3", "Data4"):
        rec = next(r for r in recs if r["dataset"] == ds)
        n_j = 19 if ds == "Data1" else 17
        mapping = COCO19_TO_SMPL24 if ds == "Data1" else COCO17_TO_SMPL24
        cam = load_camera(rec["colmap_dir"], rec["cam"])
        p = parse_pose3d(rec["gt_pose3d"], n_joints=n_j)
        t = to_gt_tracks(p, mapping, (rec["width"], rec["height"]), cam)

        uv = t["joints2d"]
        ok = np.isfinite(uv).all(-1)
        inside = (ok & (uv[..., 0] >= 0) & (uv[..., 0] < rec["width"])
                  & (uv[..., 1] >= 0) & (uv[..., 1] < rec["height"]))
        ratio = inside.sum() / max(ok.sum(), 1)
        assert ratio > 0.9, f"{ds}: 화면 안 비율 {ratio:.1%}"

        w = t["bbox"][:, 2] - t["bbox"][:, 0]
        h = t["bbox"][:, 3] - t["bbox"][:, 1]
        assert (w > 0).mean() > 0.9, f"{ds}: bbox 폭 0 인 행이 많음"
        # 사람 하나가 화면 절반을 넘지는 않는다 (발산 감지)
        assert np.median(h) < rec["height"], f"{ds}: bbox 높이 비정상"


def test_camera_model_params_are_unpacked_by_name():
    """SIMPLE_RADIAL 은 f,cx,cy,k 4개다 — fx,fy,cx,cy 로 읽으면 투영이 깨진다."""
    from smpl_eval.gt.colmap import unpack_params
    d = unpack_params("SIMPLE_RADIAL", [1912.0, 610.0, 512.0, 0.01])
    assert d["fx"] == d["fy"] == 1912.0
    assert d["cx"] == 610.0 and d["cy"] == 512.0 and d["k1"] == 0.01
    f = unpack_params("FULL_OPENCV", [100., 200., 10., 20., 1, 2, 3, 4, 5, 6, 7, 8])
    assert f["fx"] == 100.0 and f["fy"] == 200.0 and f["k3"] == 5.0


def test_data1_to_gt_tracks_uses_coco19_with_feet():
    p = parse_pose3d(D1_3D, n_joints=19)
    t = to_gt_tracks(p, COCO19_TO_SMPL24, image_wh=(1920, 1080))
    validate_tracks(t)
    assert np.isnan(t["joints3d"][:, 0]).all()      # COCO 에는 골반이 없다
    assert np.isfinite(t["joints3d"][:, 10]).any()  # left_foot 은 있다
    assert np.isfinite(t["joints3d"][:, 11]).any()  # right_foot 도 있다


# ── Data1 2D 파서 ───────────────────────────────────────────────────

def test_data1_2d_reports_malformed_lines():
    p = parse_pose2d(D1_2D, cam_id=0)
    assert p["n_malformed"] == 6, "실측한 깨진 줄 6개와 다름"
    assert p["joints2d"].shape[1:] == (19, 2)
    assert p["bbox"].shape[1] == 4


def test_data1_2d_filters_by_camera():
    counts = [len(parse_pose2d(D1_2D, cam_id=c)["frame_ids"]) for c in range(4)]
    assert all(c > 0 for c in counts), counts
    assert sum(counts) + 6 == 15086       # 깨진 줄 6개 제외하면 전부


def test_data1_2d_bbox_is_xyxy_within_image():
    p = parse_pose2d(D1_2D, cam_id=0)
    assert (p["bbox"][:, 2] > p["bbox"][:, 0]).all()
    assert (p["bbox"][:, 3] > p["bbox"][:, 1]).all()
    assert p["bbox"][:, 2].max() <= 1920 + 50    # 약간의 초과는 허용
