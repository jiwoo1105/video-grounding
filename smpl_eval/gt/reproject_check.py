"""GT 를 영상 위에 그려 관절 규약과 프레임 정렬을 육안 검증한다 (게이트 2).

관절 규약 자체는 뼈길이 강체성으로 수치 확정했으나(스펙 §12.1),
**프레임 오프셋은 파일 내용만으로 확정할 수 없다.** GT 의 최소 프레임
번호가 영상의 몇 번째 프레임인지는 눈으로 맞춰보는 수밖에 없다.

Data1 만 2D GT 가 있어 직접 겹쳐 그릴 수 있다. Data2/3/4 는
2D GT 가 없으므로 COLMAP 카메라로 투영해야 한다 (별도 단계).
"""
import io
import os
import subprocess

import numpy as np

from smpl_eval.conventions import COCO19

# COCO 골격 연결 (0-base). 얼굴은 점만 찍는다.
COCO_LINKS = [
    (5, 7), (7, 9), (6, 8), (8, 10),        # 팔
    (5, 6), (11, 12), (5, 11), (6, 12),     # 몸통
    (11, 13), (13, 15), (12, 14), (14, 16),  # 다리
    (15, 17), (16, 18),                     # 발 (Data1 전용)
]
LEFT = {5, 7, 9, 11, 13, 15, 17}


def read_frame(video_path, idx):
    """ffmpeg 로 특정 프레임 하나를 PNG 바이트로 뽑는다."""
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", video_path,
         "-vf", f"select=eq(n\\,{idx})", "-vframes", "1",
         "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True, check=True).stdout


def overlay_gt2d(video_path, gt2d, out_dir, frames=(0, 100, 200),
                 conf_thresh=0.3, label_joints=True):
    """gt2d 의 관절을 영상 프레임 위에 번호와 함께 그린다."""
    from PIL import Image, ImageDraw

    os.makedirs(out_dir, exist_ok=True)
    written = []
    for idx in frames:
        sel = gt2d["frame_ids"] == idx
        if not sel.any():
            continue
        img = Image.open(io.BytesIO(read_frame(video_path, idx))).convert("RGB")
        d = ImageDraw.Draw(img)

        for row in np.where(sel)[0]:
            j = gt2d["joints2d"][row]
            c = gt2d["conf"][row]
            d.rectangle([float(x) for x in gt2d["bbox"][row]],
                        outline=(0, 255, 0), width=2)
            for a, b in COCO_LINKS:
                if a < len(j) and b < len(j) and c[a] > conf_thresh and c[b] > conf_thresh:
                    color = (0, 160, 255) if a in LEFT else (255, 140, 0)
                    d.line([tuple(j[a]), tuple(j[b])], fill=color, width=3)
            for k in range(len(j)):
                if c[k] > conf_thresh:
                    x, y = float(j[k][0]), float(j[k][1])
                    d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 0, 0))
                    if label_joints:
                        d.text((x + 4, y - 6), str(k), fill=(255, 255, 255))

        p = os.path.join(out_dir, f"frame_{idx:05d}.png")
        img.save(p)
        written.append(p)
    return written


def main():
    import json
    from smpl_eval.gt.parse_pose2d import parse_pose2d

    manifest = json.load(open("smpl_eval/manifest.json"))
    rec = next(r for r in manifest
               if r["dataset"] == "Data1" and r["cam"].startswith("Cam1"))
    print(f"영상: {rec['video_path']}")

    gt2d = parse_pose2d(rec["gt_pose2d"], cam_id=0)
    print(f"cam0 행수 {len(gt2d['frame_ids'])}, 깨진 줄 {gt2d['n_malformed']}개")
    print(f"GT 프레임 범위: {gt2d['frame_ids'].min()} ~ {gt2d['frame_ids'].max()}")

    # 오프셋 후보를 나란히 그려 어느 쪽이 맞는지 눈으로 고른다
    for off in (0, 1):
        shifted = dict(gt2d)
        shifted["frame_ids"] = gt2d["frame_ids"] - off
        out = overlay_gt2d(rec["video_path"], shifted,
                           f"smpl_eval/gate_out/data1_offset{off}",
                           frames=(0, 100, 250))
        for p in out:
            print(f"  offset={off}  {p}")

    print("""
게이트 2 확인 사항
  1) 초록 bbox 가 실제 사람을 정확히 감싸는가  → 프레임 정렬 확인
     offset=0 과 offset=1 중 더 잘 맞는 쪽이 정답
  2) 파란(좌)/주황(우) 골격선이 팔다리를 따라가는가  → COCO 규약 확인
  3) 빨간 점 17·18 이 발에 찍히는가  → Data1 추가 2관절 확인
""")


if __name__ == "__main__":
    main()
