"""COLMAP 캘리브레이션으로 GT 3D 를 각 카메라 화면에 투영한다.

왜 필요한가: GT 는 SfM 월드 좌표라 화면 좌표를 모른다. 약원근 근사로
투영하면 z 가 0 을 지나며 발산해 bbox 가 수천만 픽셀까지 튄다(Data4 실측).
GT bbox 가 망가지면 예측-GT 를 IoU 로 짝지을 수 없어 ID·포즈 지표 전체가
무의미해진다. COLMAP 의 실제 내부·외부 파라미터를 쓰면 정확히 투영된다.

Data1 은 PoseResults2d 로 실제 2D GT 가 있으므로 그쪽을 우선 쓴다.
"""
import os
import numpy as np


def _parse_cameras(path):
    """CAMERA_ID → dict(model, width, height, params)"""
    cams = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        cams[int(f[0])] = {
            "model": f[1], "width": int(f[2]), "height": int(f[3]),
            "params": np.array([float(x) for x in f[4:]], float),
        }
    return cams


def _parse_images(path):
    """NAME(확장자 제외) → dict(qvec, tvec, camera_id)"""
    imgs = {}
    lines = [l.strip() for l in open(path)
             if l.strip() and not l.strip().startswith("#")]
    # 이미지마다 두 줄: 메타 줄 + POINTS2D 줄
    for i in range(0, len(lines), 2):
        f = lines[i].split()
        if len(f) < 10:
            continue
        # NAME 은 Windows 절대경로일 수 있다 (Data1 실측:
        # "I:/.../mvg_images_folder\\Cam1_....png"). 구분자 양쪽을 모두 처리한다.
        raw = f[9].replace("\\", "/")
        name = os.path.splitext(os.path.basename(raw))[0]
        imgs[name] = {
            "qvec": np.array([float(x) for x in f[1:5]], float),   # qw qx qy qz
            "tvec": np.array([float(x) for x in f[5:8]], float),
            "camera_id": int(f[8]),
        }
    return imgs


def qvec2rotmat(q):
    """COLMAP 쿼터니언(qw,qx,qy,qz) → 월드→카메라 회전행렬."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
    ], float)


# COLMAP 카메라 모델별 파라미터 순서.
# 모델마다 초점거리가 1개(f)인지 2개(fx,fy)인지가 다르다 —
# 이걸 틀리면 cx 를 fy 로 읽어 투영이 완전히 어긋난다 (Data3 실측).
CAMERA_MODELS = {
    "SIMPLE_PINHOLE": ("f", "cx", "cy"),
    "PINHOLE":        ("fx", "fy", "cx", "cy"),
    "SIMPLE_RADIAL":  ("f", "cx", "cy", "k1"),
    "RADIAL":         ("f", "cx", "cy", "k1", "k2"),
    "OPENCV":         ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"),
    "FULL_OPENCV":    ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2",
                       "k3", "k4", "k5", "k6"),
}


def unpack_params(model, params):
    """카메라 파라미터를 이름 붙은 dict 로 푼다. 없는 항목은 0."""
    if model not in CAMERA_MODELS:
        raise ValueError(f"지원하지 않는 카메라 모델: {model}")
    names = CAMERA_MODELS[model]
    if len(params) < len(names):
        raise ValueError(f"{model}: 파라미터 {len(names)}개 필요, "
                         f"{len(params)}개 있음")
    d = {n: float(v) for n, v in zip(names, params)}
    if "f" in d:                       # 초점거리가 하나뿐인 모델
        d["fx"] = d["fy"] = d["f"]
    for k in ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2"):
        d.setdefault(k, 0.0)
    return d


def _distort(x, y, prm):
    """정규화 좌표에 렌즈 왜곡을 적용한다."""
    k1, k2, k3 = prm["k1"], prm["k2"], prm["k3"]
    k4, k5, k6 = prm["k4"], prm["k5"], prm["k6"]
    p1, p2 = prm["p1"], prm["p2"]

    r2 = x*x + y*y
    r4, r6 = r2*r2, r2*r2*r2
    num = 1.0 + k1*r2 + k2*r4 + k3*r6
    den = 1.0 + k4*r2 + k5*r4 + k6*r6
    radial = np.where(np.abs(den) > 1e-12, num/den, num)
    xd = x*radial + 2.0*p1*x*y + p2*(r2 + 2.0*x*x)
    yd = y*radial + p1*(r2 + 2.0*y*y) + 2.0*p2*x*y
    return xd, yd


class ColmapCamera:
    """한 카메라의 투영기."""

    def __init__(self, cam, img):
        self.model = cam["model"]
        self.params = cam["params"]
        self.prm = unpack_params(self.model, self.params)
        self.width, self.height = cam["width"], cam["height"]
        self.R = qvec2rotmat(img["qvec"])
        self.t = img["tvec"]

    def project(self, pts_world):
        """(..., 3) 월드 좌표 → (..., 2) 픽셀. 카메라 뒤의 점은 NaN."""
        p = np.asarray(pts_world, float)
        shape = p.shape[:-1]
        flat = p.reshape(-1, 3)
        cam = flat @ self.R.T + self.t
        z = cam[:, 2]
        valid = np.isfinite(z) & (z > 1e-6)      # 카메라 앞에 있는 점만
        x = np.full(len(flat), np.nan)
        y = np.full(len(flat), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            x[valid] = cam[valid, 0] / z[valid]
            y[valid] = cam[valid, 1] / z[valid]
        xd, yd = _distort(x, y, self.prm)
        uv = np.stack([self.prm["fx"]*xd + self.prm["cx"],
                       self.prm["fy"]*yd + self.prm["cy"]], -1)
        return uv.reshape(shape + (2,))


def load_camera(colmap_dir, cam_name):
    """colmap_text 디렉터리와 카메라 이름(확장자 제외)으로 투영기를 만든다."""
    cams = _parse_cameras(os.path.join(colmap_dir, "cameras.txt"))
    imgs = _parse_images(os.path.join(colmap_dir, "images.txt"))
    if cam_name not in imgs:
        raise KeyError(f"{cam_name} 이(가) images.txt 에 없습니다. "
                       f"있는 이름: {sorted(imgs)}")
    img = imgs[cam_name]
    return ColmapCamera(cams[img["camera_id"]], img)
