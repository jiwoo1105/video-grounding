#!/usr/bin/env bash
# 엘리스클라우드 인스턴스에서 1회 실행. 재실행 안전(idempotent).
#
# 인스턴스가 소멸하면 이 스크립트 하나로 전체 환경을 복구한다.
set -euo pipefail

# 이 스크립트가 있는 위치에서 저장소 루트를 역산 (경로 하드코딩 회피)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_ROOT="${SMPL_EVAL_ENV_ROOT:-${HOME}/smpl_eval_env}"

echo "저장소: $REPO_ROOT"
echo "환경  : $ENV_ROOT"
mkdir -p "$ENV_ROOT"

have_env() { conda env list | awk '{print $1}' | grep -qx "$1"; }

echo
echo "=== 1/4  CoMotion (Apple, ICLR 2025) ==="
cd "$ENV_ROOT"
[ -d ml-comotion ] || git clone --depth 1 https://github.com/apple/ml-comotion.git
cd ml-comotion
have_env comotion || conda create -n comotion -y python=3.10
conda run --no-capture-output -n comotion pip install -e '.[all]'
if [ ! -f src/comotion_demo/data/comotion_detection_checkpoint.pt ]; then
  bash get_pretrained_models.sh
fi

echo
echo "=== 2/4  Multi-HMR 2 (NAVER, 2026) ==="
cd "$ENV_ROOT"
[ -d multi-hmr2 ] || git clone --depth 1 https://github.com/naver/multi-hmr2.git
cd multi-hmr2
have_env multihmr2 || conda create -n multihmr2 -y python=3.10
conda run --no-capture-output -n multihmr2 pip install -e '.[render]'

echo
echo "=== 3/4  SMPL 바디모델 확인 ==="
SMPL_DST="$ENV_ROOT/ml-comotion/src/comotion_demo/data/smpl/SMPL_NEUTRAL.pkl"
if [ ! -f "$SMPL_DST" ]; then
  cat <<MSG

  !! 수동 작업이 필요합니다 (라이선스 동의가 필요해 자동화할 수 없음)

     1. https://smpl.is.tue.mpg.de/ 가입 후 로그인
     2. SMPL version 1.1.0 for Python (neutral) 다운로드
     3. 압축 해제 후 아래 파일을
          basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
        다음 경로로 이름 바꿔 복사:
          $SMPL_DST

     복사 후 이 스크립트를 다시 실행하세요.

MSG
  exit 1
fi
echo "SMPL_NEUTRAL.pkl OK"

echo
echo "=== 4/4  평가 파이프라인 의존성 ==="
for e in comotion multihmr2; do
  conda run --no-capture-output -n "$e" pip install -r "$REPO_ROOT/smpl_eval/requirements.txt"
done

echo
echo "완료. 다음: bash $SCRIPT_DIR/smoke_elice.sh <영상경로>"
