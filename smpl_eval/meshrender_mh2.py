"""Multi-HMR 2 의 Anny 메시를 영상 위에 렌더한다.

CoMotion 은 tracks.npz 에 SMPL 파라미터가 다 있어 재추론 없이 메시를
복원할 수 있다. Multi-HMR 2 는 다르다. 네트워크가 내놓는 것은 Anny 의
bone_poses 이고, 우리 러너는 그걸로 만든 정점(`persons.v3d`)에서 관절만
뽑아 쓰고 정점 자체는 버렸다. 19,158 x 3 float32 = 정점 한 벌에 230KB 라
프레임마다 저장하면 영상 하나에 수 GB 가 된다.

그래서 여기서는 추론을 다시 돌리고, 정점이 메모리에 있는 그 자리에서
바로 렌더한다.

**어떤 메시를 그릴지는 이미 평가한 tracks.npz 가 정한다.** 중복 제거
(postprocess.nms_2d) 를 거친 트랙 목록을 그대로 쓰므로, 화면에 보이는
것과 표에 있는 수치가 같은 대상을 가리킨다. 한 사람에 메시가 둘 겹치는
현상도 여기서 함께 사라진다.
"""
import argparse
import os
import shutil

import numpy as np

from smpl_eval.meshrender import render_stream


def kept_pairs(tracks_path):
    """(프레임, 트랙) 쌍의 집합. 이 조합만 렌더한다."""
    from smpl_eval.schema import load_tracks
    tracks, meta = load_tracks(tracks_path)
    pairs = set(zip(tracks["frame_ids"].astype(int).tolist(),
                    tracks["track_ids"].astype(int).tolist()))
    return pairs, meta


def run(video_path, tracks_path, out_path, checkpoint, label,
        max_frames=None, scale=0.5, tmp_root=None):
    """tmp_root 를 주지 않으면 출력 파일 이름으로 고유한 폴더를 만든다.

    두 렌더를 동시에 돌릴 때 임시 폴더를 공유하면 한쪽이 프레임을 추출하는
    사이 다른 쪽이 그 폴더를 지워 둘 다 실패한다 (실측). 출력마다 다른
    폴더를 쓰면 병렬 실행이 안전하다.
    """
    import torch
    from multihmr2 import api

    if tmp_root is None:
        tag = os.path.splitext(os.path.basename(out_path))[0]
        tmp_root = f"~/mh_mesh_tmp_{tag}"

    pairs, meta = kept_pairs(tracks_path)
    W, H = meta.get("resolution", [1920, 1080])
    fps = meta.get("fps", 30.0)

    src = os.path.abspath(video_path)
    tmp = os.path.expanduser(tmp_root)
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    if max_frames:
        src = os.path.join(tmp, "_trim.mp4")
        from smpl_eval.runners.base import sh
        sh(["ffmpeg", "-v", "error", "-y", "-i", os.path.abspath(video_path),
            "-frames:v", str(max_frames), "-c", "copy", src])

    # tracks 가 초점거리 보정본이면 그 값으로 렌더해야 한다.
    f_fix = meta.get("fov_fix", {}).get("focal_after")
    if f_fix:
        f_fix = float(f_fix)
        print(f"초점거리 고정 렌더: {f_fix:.1f}")

    session = api.init_hmr_session(checkpoint)
    faces = session.model.full_body_decoder.body_model.faces
    faces = np.asarray(faces.cpu() if torch.is_tensor(faces) else faces, np.int32)
    print(f"Anny 면 {len(faces):,}개")

    try:
        preds = api.infer_video(session, src, tmp)
    finally:
        if not max_frames:
            shutil.rmtree(tmp, ignore_errors=True)

    # 프레임별 (트랙, 정점, 면) 과 카메라를 미리 정리한다. 정점은 float16
    # 으로 눕혀 메모리를 절반으로 줄인다 (렌더 직전 float32 로 되돌린다).
    per_frame = {}
    for t, fp in enumerate(preds):
        p = fp.persons
        n = int(getattr(p, "num_person", 0) or 0)
        if n == 0 or p.v3d is None:
            continue
        K = fp.K.detach().cpu().float().numpy()
        # K 는 모델 입력 해상도 기준이다. 러너와 같은 식으로 원본에 맞춘다.
        s = W / (2.0 * float(K[0, 2])) if K[0, 2] > 0 else 1.0
        focal = (float(K[0, 0]) * s, float(K[1, 1]) * s)
        princpt = (float(K[0, 2]) * s, float(K[1, 2]) * s)
        v3d = p.v3d.detach().cpu().float().numpy()
        tid = np.asarray(p.track_id).reshape(-1).astype(int)

        if f_fix is not None:
            # 초점거리를 고정한 tracks 로 렌더하는 경우. 사람을 통째로
            # 깊이 방향으로 옮기고, 반드시 바뀐 초점거리로 투영한다.
            # (옮겨만 놓고 예전 초점거리로 그리면 거대하게 나온다)
            ratio = f_fix / max(focal[0], 1e-9)
            pz = p.transl_pelvis.detach().cpu().float().numpy()[:, 2]
            v3d = v3d.copy()
            v3d[..., 2] += (pz * (ratio - 1.0))[:, None]
            focal = (f_fix, f_fix)

        rows = [(int(tid[i]), v3d[i].astype(np.float16))
                for i in range(n) if (t, int(tid[i])) in pairs]
        if rows:
            per_frame[t] = (rows, focal, princpt)

    n_total = sum(len(v[0]) for v in per_frame.values())
    n_raw = sum(int(getattr(fp.persons, "num_person", 0) or 0) for fp in preds)
    print(f"렌더 대상 메시 {n_total:,}개 (원 검출 {n_raw:,}개, "
          f"중복제거로 {n_raw - n_total:,}개 제외)")

    default_cam = ((2.0 * max(W, H),) * 2, (W / 2.0, H / 2.0))

    def frame_meshes(f):
        e = per_frame.get(f)
        if e is None:
            return [], default_cam[0], default_cam[1]
        rows, focal, princpt = e
        return ([(tid, v.astype(np.float32), faces) for tid, v in rows],
                focal, princpt)

    total = max_frames or (max(per_frame) + 1 if per_frame else 0)
    out, n = render_stream(video_path, out_path, W, H, fps, int(total),
                           frame_meshes, label=label, scale=scale)
    print(f"{out}  ({n}/{total} 프레임에 메시)")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Multi-HMR 2 Anny 메시 렌더")
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks", required=True,
                    help="중복제거를 거친 tracks.npz — 이 목록만 렌더한다")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--label", default="Multi-HMR 2 (Anny 19,158 정점)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--scale", type=float, default=0.5)
    a = ap.parse_args(argv)
    run(a.video, a.tracks, a.out, a.checkpoint, a.label, a.max_frames, a.scale)


if __name__ == "__main__":
    main()
