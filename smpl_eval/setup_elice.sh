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
echo "=== 1/5  시스템 패키지 ==="
if ! command -v ffmpeg >/dev/null; then
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
      ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 build-essential
fi
echo "  ffmpeg: $(ffmpeg -version 2>&1 | head -1)"

# ── 2. CoMotion ──────────────────────────────────────────────────
echo
echo "=== 2/5  CoMotion (Apple, ICLR 2025) ==="
cd "$ENV_ROOT"
[ -d ml-comotion ] || git clone --depth 1 https://github.com/apple/ml-comotion.git

V="$VENV_ROOT/comotion"
[ -d "$V" ] || python3 -m venv "$V"
"$V/bin/pip" install -q --upgrade pip wheel
"$V/bin/pip" install -q torch torchvision --index-url "$TORCH_INDEX"
cd ml-comotion
"$V/bin/pip" install -e '.[all]'
if [ ! -f src/comotion_demo/data/comotion_detection_checkpoint.pt ]; then
  bash get_pretrained_models.sh
fi

# ── 3. Multi-HMR 2 ───────────────────────────────────────────────
echo
echo "=== 3/5  Multi-HMR 2 (NAVER, 2026) ==="
cd "$ENV_ROOT"
[ -d multi-hmr2 ] || git clone --depth 1 https://github.com/naver/multi-hmr2.git

V2="$VENV_ROOT/multihmr2"
[ -d "$V2" ] || python3 -m venv "$V2"
"$V2/bin/pip" install -q --upgrade pip wheel
"$V2/bin/pip" install -q torch torchvision --index-url "$TORCH_INDEX"
cd multi-hmr2
# MIG 는 OpenGL 을 지원하지 않으므로 렌더 의존성은 넣지 않는다.
# 시각 검증은 저장소의 PIL 기반 오버레이 도구로 한다.
"$V2/bin/pip" install -e .

# ── 4. SMPL 바디모델 ─────────────────────────────────────────────
echo
echo "=== 4/5  SMPL 바디모델 확인 ==="
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
echo "=== 5/5  평가 파이프라인 의존성 ==="
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

echo
echo "완료. 다음:"
echo "  bash $SCRIPT_DIR/smoke_elice.sh <영상경로>"
