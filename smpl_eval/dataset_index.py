"""데이터셋을 스캔해 manifest.json 을 만든다.

영상 28개 각각에 대해 해상도·fps·프레임수와 대응하는 GT 파일 경로를 모은다.
이후 모든 단계가 이 manifest 를 입력으로 받는다.
"""
import os
import glob
import json
import subprocess

# 데이터셋 디렉터리 이름 → 짧은 이름
DATASETS = {
    "Data1_SKNight-live_Basketball": "Data1",
    "Data2_WTA_tennis_double_clip3_1min_2K_30fps": "Data2",
    "Data3_OnlyOneOf_rie_junji_2K_60fps": "Data3",
    "Data4_vid3_golden_clip1_2K_60fps": "Data4",
}

# 화면에 실제로 등장하는 인원. GT 인원과 다를 수 있다 —
# 테니스는 복식(4명)인데 GT 는 obj0/obj1 두 명뿐이다.
EXPECTED_PERSONS = {"Data1": None, "Data2": 4, "Data3": 2, "Data4": 3}

# 입력 영상이 아니라 기존 처리 결과가 들어 있는 디렉터리
RESULT_SUFFIXES = ("_pose_estimated", "_pose_gt", "_pose_result", "_mvg_result")


def _probe(path):
    """ffprobe 로 영상 메타데이터를 읽는다."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,nb_frames", "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {"width": int(s["width"]), "height": int(s["height"]),
            "fps": int(num) / int(den), "n_frames": int(s["nb_frames"])}


def _first(pattern):
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None


def _gt_person_count(path):
    """GT 파일의 2번째 열(person id)의 고유값 개수."""
    ids = set()
    for line in open(path):
        f = line.strip().split(",")
        if len(f) < 3:
            continue
        ids.add(f[1].strip())
    return len(ids)


def scan(dataset_root):
    records = []
    for dirname, short in DATASETS.items():
        base = os.path.join(dataset_root, dirname)
        if not os.path.isdir(base):
            continue
        for vid in sorted(glob.glob(os.path.join(base, "*", "*.mp4"))):
            session = os.path.basename(os.path.dirname(vid))
            if session.endswith(RESULT_SUFFIXES):
                continue
            cam = os.path.splitext(os.path.basename(vid))[0]

            if short == "Data1":
                gt_dir = os.path.join(base, session + "_pose_gt")
                gt3 = _first(os.path.join(gt_dir, "PoseResults3d_*.txt"))
                gt2 = _first(os.path.join(gt_dir, "PoseResults2d_*.txt"))
                bvh = []
            else:
                res = os.path.join(base, session + "_pose_result")
                # Data3 는 파일명이 소문자 p (3Dpose.txt)
                gt3 = (_first(os.path.join(res, "3DPose.txt"))
                       or _first(os.path.join(res, "3Dpose.txt")))
                gt2 = None
                bvh = sorted(glob.glob(os.path.join(res, "obj*.bvh")))

            rec = {
                "video_path": vid, "dataset": short, "session": session, "cam": cam,
                "gt_pose3d": gt3, "gt_pose2d": gt2, "gt_bvh": bvh,
                "colmap_dir": _first(os.path.join(base, session + "_mvg_result",
                                                  "colmap_text")),
                "expected_persons": EXPECTED_PERSONS[short],
            }
            rec.update(_probe(vid))
            rec["gt_persons"] = _gt_person_count(gt3) if gt3 else None
            records.append(rec)
    return records


def write_manifest(dataset_root, out_path):
    recs = scan(dataset_root)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False, indent=2)
    return recs


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "Captured-Motion-Dataset"
    recs = write_manifest(root, "smpl_eval/manifest.json")
    total = sum(r["n_frames"] for r in recs)
    print(f"{len(recs)} videos, {total} frames → smpl_eval/manifest.json")
