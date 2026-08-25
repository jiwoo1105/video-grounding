"""모델이 낸 3D 메시를 원본 프레임 위에 실제로 래스터화한다.

`overlay.py` 는 관절 좌표를 받아 선으로 이어 **우리가 그린** 그림이다.
사람에 포즈가 붙었는지 보기에는 충분하지만, 모델이 실제로 내놓은 형상은
아니다. 이 모듈은 모델이 출력한 **정점(vertex)과 면(face)** 을 그대로
3D 래스터라이저에 넣는다. 화면에 보이는 몸 형상은 전부 모델 출력이고,
우리가 정하는 것은 조명과 트랙별 색뿐이다.

두 모델을 **같은 래스터라이저**로 렌더한다. 렌더러가 다르면 재질·조명
차이가 모델 차이처럼 보여 비교가 안 된다. 기하는 각 모델의 것이므로
    CoMotion      SMPL  6,890 정점 / 13,776 면
    Multi-HMR 2   Anny 19,158 정점
의 차이는 그대로 화면에 드러난다.

**백엔드** — MIG 인스턴스에는 하드웨어 EGL 이 없다. Multi-HMR 2 의
`utils/render.py` 가 이를 감지해 osmesa(CPU 소프트웨어 래스터라이저)로
자동 전환한다. 느리지만 동작하며, 결과는 GPU 렌더와 동일하다.
osmesa 를 쓰려면 PyOpenGL 이 3.1.7 이상이어야 한다 (pyrender 가 요구하는
`OSMesaCreateContextAttribs` 심볼이 3.1.0 에는 없다).
"""
import os
import subprocess

import numpy as np

# pyrender 를 import 하기 전에 백엔드가 정해져야 한다. Multi-HMR 2 의
# render 모듈이 하드웨어 EGL 을 탐지해 PYOPENGL_PLATFORM 을 설정해 주므로
# 있으면 그것을 쓴다. CoMotion 쪽 venv 에는 없으므로, 없으면 osmesa 로
# 떨어뜨린다 — MIG 에는 어차피 하드웨어 EGL 이 없다.
if "PYOPENGL_PLATFORM" not in os.environ:
    try:
        from multihmr2.utils import render as _mh2_render  # noqa: F401
    except ImportError:
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import pyrender
import trimesh

from smpl_eval.overlay import PALETTE, _font

# OpenCV(+Y 아래, +Z 앞) → OpenGL(+Y 위, +Z 뒤) 카메라 규약 변환.
# multihmr2.utils.render 의 동명 상수와 같다.
OPENCV_TO_OPENGL = np.array([[1, 0, 0, 0],
                             [0, -1, 0, 0],
                             [0, 0, -1, 0],
                             [0, 0, 0, 1]], dtype=np.float64)


def track_color(tid):
    """트랙 ID → 색. overlay.py 와 같은 팔레트라 두 영상을 나란히 볼 수 있다."""
    return PALETTE[int(tid) % len(PALETTE)]


class MeshRenderer:
    """메시를 이미지 위에 합성한다.

    Multi-HMR 2 의 `render_meshes()` 와 장면 구성(재질·조명·카메라 규약·
    경계 블렌딩)이 같다. 다른 점은 `OffscreenRenderer` 를 한 번만 만들어
    재사용한다는 것뿐이다. 원본은 호출마다 생성·파괴하는데, osmesa 에서는
    컨텍스트 생성이 수백 ms 라 프레임마다 하면 영상 렌더가 불가능하다.
    """

    def __init__(self, width, height):
        self.width, self.height = int(width), int(height)
        self._r = pyrender.OffscreenRenderer(
            viewport_width=self.width, viewport_height=self.height, point_size=1.0)

    def close(self):
        if self._r is not None:
            self._r.delete()
            self._r = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def render(self, img, meshes, faces, focal, princpt, colors):
        """img(H,W,3 uint8) 위에 메시들을 얹은 새 이미지를 돌려준다.

        meshes  각 (V,3) 정점 배열의 리스트 — 카메라 좌표계
        faces   각 (F,3) 면 인덱스. 모델마다 하나지만 리스트로 받는다
        colors  각 (r,g,b) 0-255
        """
        if not meshes:
            return img

        scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
        for v, f, c in zip(meshes, faces, colors):
            mat = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0, roughnessFactor=0.5, alphaMode="OPAQUE",
                baseColorFactor=(c[0] / 255, c[1] / 255, c[2] / 255, 1.0))
            tm = trimesh.Trimesh(v, f, process=False)
            scene.add(pyrender.Mesh.from_trimesh(tm, material=mat, smooth=True))

        cam_pose = np.linalg.inv(OPENCV_TO_OPENGL @ np.eye(4))
        scene.add(pyrender.IntrinsicsCamera(
            fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1]), pose=cam_pose)
        scene.add(pyrender.DirectionalLight(intensity=3.0), pose=cam_pose)

        rgb, depth = self._r.render(scene, flags=pyrender.RenderFlags.RGBA)
        rgb = rgb[:, :, :3].astype(np.float32)

        # 실루엣 경계를 1픽셀 부드럽게 한다 (원본 render_meshes 와 동일한 의도).
        fg = (depth > 0).astype(np.float32)
        k = np.ones((3, 3), np.float32) * 2.0 / 9.0
        pad = np.pad(fg, 1)
        blur = sum(pad[i:i + fg.shape[0], j:j + fg.shape[1]] * k[i, j]
                   for i in range(3) for j in range(3))
        fg = np.clip((blur - 1.0) * fg, 0.0, None)[:, :, None]

        return (fg * rgb + (1 - fg) * img.astype(np.float32)).astype(np.uint8)


def project(verts, focal, princpt):
    """정점을 2D 로 투영한다 — ID 라벨을 놓을 위치를 잡는 용도."""
    z = np.clip(verts[:, 2], 1e-6, None)
    return np.stack([verts[:, 0] / z * focal[0] + princpt[0],
                     verts[:, 1] / z * focal[1] + princpt[1]], -1)


def draw_labels(img, entries, focal, princpt, font_size):
    """메시 머리 위에 트랙 번호를 얹는다.

    형상은 모델 출력 그대로지만, 번호가 없으면 어느 메시가 어느 트랙인지
    알 수 없어 추적을 볼 수 없다. 색만으로는 팔레트가 8색이라 부족하다.
    """
    from PIL import Image, ImageDraw

    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    font = _font(font_size)
    for tid, verts in entries:
        uv = project(verts, focal, princpt)
        top = uv[np.argmin(verts[:, 1])] if len(uv) else None
        if top is None or not np.all(np.isfinite(top)):
            continue
        x, y = float(top[0]), float(top[1]) - font_size * 1.6
        col = track_color(tid)
        txt = f"ID {int(tid)}"
        try:
            x0, y0, x1, y1 = d.textbbox((0, 0), txt, font=font)
            tw, th = x1 - x0, y1 - y0
        except AttributeError:
            tw, th = d.textsize(txt, font=font)
        pad = max(3, font_size // 4)
        d.rectangle([x - tw / 2 - pad, y - pad, x + tw / 2 + pad, y + th + pad],
                    fill=col, outline=(0, 0, 0), width=2)
        lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
        d.text((x - tw / 2, y), txt, font=font,
               fill=(0, 0, 0) if lum > 140 else (255, 255, 255))
    return np.asarray(pil)


def render_stream(video_path, out_path, width, height, fps, total, frame_meshes,
                  label="", scale=1.0, draw_id=True):
    """영상을 디코딩 → 메시 합성 → 재인코딩.

    frame_meshes(f) 는 프레임 f 에 대해 (entries, focal, princpt) 를 준다.
    entries 는 (track_id, verts, faces) 의 리스트다.
    """
    from PIL import Image, ImageDraw

    W, H = int(width), int(height)
    ow, oh = int(W * scale), int(H * scale)
    ow -= ow % 2
    oh -= oh % 2
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", video_path, "-frames:v", str(total),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{ow}x{oh}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out_path],
        stdin=subprocess.PIPE)

    nbytes = W * H * 3
    n_drawn = 0
    cap_size = max(16, int(round(H / 34 / max(scale, 0.2))))
    id_size = max(14, int(round(H / 44 / max(scale, 0.2))))

    with MeshRenderer(W, H) as mr:
        try:
            for f in range(total):
                buf = dec.stdout.read(nbytes)
                if len(buf) < nbytes:
                    break
                img = np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
                entries, focal, princpt = frame_meshes(f)
                if entries:
                    img = mr.render(img,
                                    [e[1] for e in entries], [e[2] for e in entries],
                                    focal, princpt,
                                    [track_color(e[0]) for e in entries])
                    if draw_id:
                        img = draw_labels(img, [(e[0], e[1]) for e in entries],
                                          focal, princpt, id_size)
                    n_drawn += 1

                pil = Image.fromarray(img)
                d = ImageDraw.Draw(pil)
                cap = f"{label}  |  frame {f}  |  meshes {len(entries)}"
                cf = _font(cap_size)
                try:
                    x0, y0, x1, y1 = d.textbbox((0, 0), cap, font=cf)
                    cw, ch = x1 - x0, y1 - y0
                except AttributeError:
                    cw, ch = d.textsize(cap, font=cf)
                d.rectangle([0, 0, cw + 20, ch + 16], fill=(0, 0, 0))
                d.text((10, 8), cap, fill=(255, 255, 255), font=cf)
                if scale != 1.0:
                    pil = pil.resize((ow, oh), Image.BILINEAR)
                enc.stdin.write(pil.tobytes())
        finally:
            enc.stdin.close()
            enc.wait()
            dec.stdout.close()
            dec.wait()
    return out_path, n_drawn


# ── CoMotion: 저장된 SMPL 파라미터에서 메시를 복원한다 ──────────────────
#
# tracks.npz 에 betas/global_orient/body_pose/transl 이 모두 있으므로
# 재추론 없이 CoMotion 이 낸 것과 **같은** 메시를 얻는다. SMPL 은 결정적
# 함수라서 같은 파라미터는 같은 정점을 준다.

def comotion_source(tracks_path, device="cpu"):
    """(frame_meshes, meta) 를 돌려준다."""
    import torch
    from comotion_demo.utils import smpl_kinematics
    from smpl_eval.schema import load_tracks

    tracks, meta = load_tracks(tracks_path)
    kn = smpl_kinematics.SMPLKinematics().to(device).eval()
    faces = _smpl_faces()

    W, H = meta.get("resolution", [1920, 1080])
    f = 2.0 * max(W, H)                       # CoMotion 의 get_default_K
    focal, princpt = (f, f), (W / 2.0, H / 2.0)

    by_frame = {}
    for i in range(len(tracks["frame_ids"])):
        by_frame.setdefault(int(tracks["frame_ids"][i]), []).append(i)

    def frame_meshes(fr):
        rows = by_frame.get(fr, [])
        if not rows:
            return [], focal, princpt
        betas = torch.as_tensor(tracks["betas"][rows], dtype=torch.float32, device=device)
        pose = torch.as_tensor(
            np.concatenate([tracks["global_orient"][rows],
                            tracks["body_pose"][rows].reshape(len(rows), -1)], -1),
            dtype=torch.float32, device=device)
        trans = torch.as_tensor(tracks["transl"][rows], dtype=torch.float32, device=device)
        with torch.no_grad():
            v = kn(betas, pose, trans, output_format="mesh").cpu().numpy()
        return ([(int(tracks["track_ids"][r]), v[i].astype(np.float32), faces)
                 for i, r in enumerate(rows)], focal, princpt)

    return frame_meshes, meta


def _smpl_faces():
    """SMPL 면 인덱스 (13,776 x 3). 모델 pkl 에서 읽는다."""
    import pickle
    cands = [os.path.expanduser(p) for p in (
        "~/video-grounding/smpl/SMPL_NEUTRAL.pkl",
        "~/smpl_eval_env/ml-comotion/data/smpl/SMPL_NEUTRAL.pkl",
        "./smpl/SMPL_NEUTRAL.pkl")]
    for p in cands:
        if os.path.exists(p):
            with open(p, "rb") as fh:
                d = pickle.load(fh, encoding="latin1")
            return np.asarray(d["f"], np.int32)
    raise FileNotFoundError(f"SMPL_NEUTRAL.pkl 없음. 확인한 경로: {cands}")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="모델 메시를 영상 위에 렌더")
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="CoMotion (SMPL 6,890 정점)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)

    frame_meshes, meta = comotion_source(a.tracks, a.device)
    from smpl_eval.schema import load_tracks
    tracks, _ = load_tracks(a.tracks)
    total = int(a.max_frames or tracks["frame_ids"].max() + 1)
    W, H = meta.get("resolution", [1920, 1080])
    out, n = render_stream(a.video, a.out, W, H, meta.get("fps", 30.0), total,
                           frame_meshes, label=a.label, scale=a.scale)
    print(f"{out}  ({n}/{total} 프레임에 메시)")


if __name__ == "__main__":
    main()
