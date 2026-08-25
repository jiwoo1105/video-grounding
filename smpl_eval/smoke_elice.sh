#!/usr/bin/env bash
# 게이트 0 — 두 모델이 실제로 돌아가고 출력을 내는지 확인한다.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_ROOT="${SMPL_EVAL_ENV_ROOT:-${HOME}/smpl_eval_env}"
VENV_ROOT="${HOME}/venvs"
V_CO="$VENV_ROOT/comotion/bin/python"
V_MH="$VENV_ROOT/multihmr2/bin"

V_IN="${1:?사용법: smoke_elice.sh <영상경로>}"
V_ABS="$(cd "$(dirname "$V_IN")" && pwd)/$(basename "$V_IN")"
OUT="${HOME}/smoke_out"
rm -rf "$OUT" && mkdir -p "$OUT"

echo "=== CoMotion (30프레임) ==="
cd "$ENV_ROOT/ml-comotion"
"$V_CO" demo.py -i "$V_ABS" -o "$OUT/comotion" --num-frames 30
find "$OUT/comotion" -type f -exec ls -lh {} \;

echo
echo "=== Multi-HMR 2 (30프레임) ==="
# --num-frames 옵션이 없으므로 ffmpeg 로 잘라 넣는다.
# MIG 는 OpenGL 을 지원하지 않으므로 --render 는 쓰지 않는다.
ffmpeg -v error -y -i "$V_ABS" -frames:v 30 -c copy "$OUT/_trim30.mp4"
cd "$ENV_ROOT/multi-hmr2"
"$V_MH/multihmr2" --checkpoint checkpoints/multihmr2.pt \
  --video "$OUT/_trim30.mp4" --out "$OUT/multihmr2" --save_anny_params
find "$OUT/multihmr2" -type f -exec ls -lh {} \;

echo
echo "=== 출력 구조 덤프 → ELICE.md ==="
bash "$SCRIPT_DIR/dump_outputs.sh" "$OUT" | tee -a "$REPO_ROOT/smpl_eval/ELICE.md"

echo
echo "게이트 0 확인 사항:"
echo "  1) 두 모델 모두 출력 파일이 생성되었는가"
echo "  2) ELICE.md 에 키·shape 가 기록되었는가"
echo "  (메쉬 렌더는 MIG 에서 불가 — 포즈 육안 검증은 파이프라인의"
echo "   PIL 오버레이 도구로 한다)"
