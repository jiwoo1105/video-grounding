import os, json, tempfile
from collections import Counter
import pytest
from smpl_eval.dataset_index import scan, write_manifest

ROOT = "Captured-Motion-Dataset"
pytestmark = pytest.mark.skipif(not os.path.isdir(ROOT), reason="데이터셋 없음")


def test_scan_finds_28_videos():
    assert len(scan(ROOT)) == 28


def test_per_dataset_video_counts():
    c = Counter(r["dataset"] for r in scan(ROOT))
    assert c["Data1"] == 16 and c["Data2"] == 4 and c["Data3"] == 4 and c["Data4"] == 4


def test_probe_fills_video_metadata():
    r = next(x for x in scan(ROOT) if x["dataset"] == "Data2")
    assert r["width"] == 1920 and r["height"] == 1080
    assert abs(r["fps"] - 29.97) < 0.05
    assert r["n_frames"] == 2309


def test_gt_paths_resolved():
    for r in scan(ROOT):
        assert r["gt_pose3d"] and os.path.isfile(r["gt_pose3d"]), r["video_path"]


def test_data1_has_2d_gt_others_do_not():
    for r in scan(ROOT):
        if r["dataset"] == "Data1":
            assert r["gt_pose2d"] and os.path.isfile(r["gt_pose2d"])
        else:
            assert r["gt_pose2d"] is None


def test_gt_person_counts_match_measured():
    """GT 인원. Data1 은 세션마다 다르다 (2026-08-25 실측)."""
    exp = {
        ("Data1", "S03_HL01_2K"): 13, ("Data1", "S06_HL01_2K"): 15,
        ("Data1", "S08_HL04_2K"): 15, ("Data1", "S12_HL01_2K"): 13,
        ("Data2", "tennis_double_clip3_1min_2K"): 2,
        ("Data3", "rie_junji_4cam"): 2,
        ("Data4", "3_golden_clip1"): 3,
    }
    for r in scan(ROOT):
        key = (r["dataset"], r["session"])
        assert r["gt_persons"] == exp[key], f"{key}: {r['gt_persons']}"


def test_data2_gt_covers_fewer_people_than_scene():
    """테니스는 복식(4명)인데 GT 는 2명뿐 — ID 지표 해석 시 반드시 고려."""
    r = next(x for x in scan(ROOT) if x["dataset"] == "Data2")
    assert r["expected_persons"] == 4 and r["gt_persons"] == 2


def test_write_manifest_is_valid_json():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.json")
        write_manifest(ROOT, p)
        assert len(json.load(open(p))) == 28
