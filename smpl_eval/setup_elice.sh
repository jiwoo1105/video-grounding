#!/usr/bin/env bash
# 엘리스클라우드 인스턴스 환경 구축. 재실행 안전(idempotent).
#
# 인스턴스가 소멸하면 이 스크립트 하나로 전체 환경을 복구한다.
#
# 실측 환경 (2026-08-25, G-NAHPM-40):
#   Ubuntu 22.04.5 / python 3.10.14 / sudo 무암호 / apt 사용가능
#   conda 없음  → venv 사용
#   ffmpeg 없음 → apt 로 설치
#   GPU: A100 80GB PCIe 의 MIG 3g.40gb / driver 535.183.06
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_ROOT="${SMPL_EVAL_ENV_ROOT:-${HOME}/smpl_eval_env}"
VENV_ROOT="${HOME}/venvs"
# driver 535 는 CUDA 12.2 까지 지원한다. cu121 휠이 가장 안전하다.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"

echo "저장소: $REPO_ROOT"
echo "환경  : $ENV_ROOT"
echo "venv  : $VENV_ROOT"
mkdir -p "$ENV_ROOT" "$VENV_ROOT"

# ── 1. 시스템 패키지 ──────────────────────────────────────────────
echo
echo "=== 1/6  시스템 패키지 ==="
if ! command -v ffmpeg >/dev/null; then
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
      ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 build-essential
fi
echo "  ffmpeg: $(ffmpeg -version 2>&1 | head -1)"

# ── 2. CoMotion ──────────────────────────────────────────────────
echo
echo "=== 2/6  CoMotion (Apple, ICLR 2025) ==="
cd "$ENV_ROOT"
[ -d ml-comotion ] || git clone --depth 1 https://github.com/apple/ml-comotion.git

V="$VENV_ROOT/comotion"
[ -d "$V" ] || python3 -m venv "$V"
"$V/bin/pip" install -q --upgrade pip wheel
"$V/bin/pip" install -q torch torchvision --index-url "$TORCH_INDEX"

# chumpy 는 setup.py 에서 pip 를 임포트하는데 빌드 격리 환경에는 pip 가
# 없어 실패한다 (실측 확인). 격리를 끄면 venv 의 pip 가 보여 설치된다.
# 먼저 넣어두면 아래 -e 설치가 이미 충족된 것으로 보고 넘어간다.
"$V/bin/pip" install -q --no-build-isolation \
  "chumpy @ git+https://github.com/mattloper/chumpy@9b045ff5d6588a24a0bab52c83f032e2ba433e17"

cd ml-comotion
# '.[all]' 은 aitviewer(3D 뷰어)를 끌어온다. 헤드리스 MIG 환경에서는
# 불필요하고 OpenGL 의존성만 늘어나므로 기본 의존성만 설치한다.
"$V/bin/pip" install -q -e .
if [ ! -f src/comotion_demo/data/comotion_detection_checkpoint.pt ]; then
  bash get_pretrained_models.sh
fi

# ── 3. Multi-HMR 2 ───────────────────────────────────────────────
echo
echo "=== 3/6  Multi-HMR 2 (NAVER, 2026) ==="
cd "$ENV_ROOT"
[ -d multi-hmr2 ] || git clone --depth 1 https://github.com/naver/multi-hmr2.git

V2="$VENV_ROOT/multihmr2"
[ -d "$V2" ] || python3 -m venv "$V2"
"$V2/bin/pip" install -q --upgrade pip wheel
"$V2/bin/pip" install -q torch torchvision --index-url "$TORCH_INDEX"
cd multi-hmr2
# MIG 는 OpenGL 을 지원하지 않으므로 '[render]' 는 넣지 않는다.
# 시각 검증은 저장소의 PIL 기반 오버레이 도구로 한다.
"$V2/bin/pip" install -q -e .

# ── 4. SMPL 바디모델 ─────────────────────────────────────────────
echo
echo "=== 4/6  SMPL 바디모델 확인 ==="
SMPL_DST="$ENV_ROOT/ml-comotion/src/comotion_demo/data/smpl/SMPL_NEUTRAL.pkl"
if [ ! -f "$SMPL_DST" ]; then
  mkdir -p "$(dirname "$SMPL_DST")"
  cat <<MSG

  !! 수동 작업이 필요합니다 (라이선스 동의가 필요해 자동화할 수 없음)

     1. https://smpl.is.tue.mpg.de/ 가입 후 로그인
     2. "SMPL version 1.1.0 for Python" 다운로드
     3. 압축 해제 후
          basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
        를 아래 경로로 이름 바꿔 복사:
          $SMPL_DST

     로컬에서 보내려면:
          scp <받은파일> elice:$SMPL_DST

     복사 후 이 스크립트를 다시 실행하세요.

MSG
  exit 1
fi
echo "  SMPL_NEUTRAL.pkl OK"

# ── 5. 평가 파이프라인 + 검증 ────────────────────────────────────
echo
echo "=== 5/6  평가 파이프라인 의존성 ==="
for V in "$VENV_ROOT/comotion" "$VENV_ROOT/multihmr2"; do
  "$V/bin/pip" install -q -r "$REPO_ROOT/smpl_eval/requirements.txt"
done

echo
echo "=== 검증 ==="
for name in comotion multihmr2; do
  printf "  %-10s " "$name"
  "$VENV_ROOT/$name/bin/python" -c "
import torch
ok = torch.cuda.is_available()
print(f\"torch {torch.__version__} cuda={ok}\", end='')
print(f\" {torch.cuda.get_device_name(0)}\" if ok else \"  !! GPU 미인식\")
"
done

# ── 6. 메시 렌더링 (osmesa) ──────────────────────────────────────
#
# MIG 에는 하드웨어 EGL 이 없어 GPU 오프스크린 OpenGL 을 쓸 수 없다.
# osmesa(CPU 소프트웨어 래스터라이저)로 우회한다. 느리지만(2K·10명
# 기준 프레임당 약 0.85초) 결과는 GPU 렌더와 같다.
#
# 버전이 셋 다 중요하다 (전부 실측으로 확인):
#   PyOpenGL 3.1.0  pyrender 가 요구하는 핀이지만 osmesa 컨텍스트 생성에
#                   필요한 OSMesaCreateContextAttribs 심볼이 없다 → 3.1.7
#   pyrender 0.1.18 pip 가 위 핀 충돌을 피하려 고른 구버전. IntrinsicsCamera
#                   가 없다 → --no-deps 로 0.1.45 강제
#   networkx 1.x    python 3.10 에서 `from collections import Mapping` 실패
echo
echo "=== 6/6  메시 렌더링 (osmesa) ==="
if ! ldconfig -p 2>/dev/null | grep -q OSMesa; then
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
      libosmesa6-dev freeglut3-dev
fi
for V in "$VENV_ROOT/comotion" "$VENV_ROOT/multihmr2"; do
  # --no-deps 로 넣으므로 pyrender 의 의존성을 여기서 직접 채운다.
  "$V/bin/pip" install -q "pyopengl==3.1.7" "networkx>=3" trimesh \
      freetype-py imageio pyglet 2>&1 | grep -v "dependency resolver\|incompatible" || true
  "$V/bin/pip" install -q --no-deps "pyrender==0.1.45"
  printf "  %-40s " "$(basename "$V") 렌더"
  PYOPENGL_PLATFORM=osmesa "$V/bin/python" - <<'PYCHK'
import numpy as np, trimesh, pyrender
r = pyrender.OffscreenRenderer(64, 64)
sc = pyrender.Scene()
sc.add(pyrender.Mesh.from_trimesh(trimesh.creation.icosphere(radius=0.5)))
p = np.eye(4); p[2, 3] = 3
sc.add(pyrender.IntrinsicsCamera(fx=60, fy=60, cx=32, cy=32), pose=p)
sc.add(pyrender.DirectionalLight(intensity=3.0), pose=p)
col, depth = r.render(sc)
print(f"OK  (메시 픽셀 {int((depth>0).sum())})")
PYCHK
done

echo
echo "완료. 다음:"
echo "  bash $SCRIPT_DIR/smoke_elice.sh <영상경로>"
