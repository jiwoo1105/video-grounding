#!/usr/bin/env bash
# 게이트 0 — 두 모델이 실제로 돌아가고 출력을 내는지 확인한다.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_ROOT="${SMPL_EVAL_ENV_ROOT:-${HOME}/smpl_eval_env}"
V="$(cd "$(dirname "${1:?사용법: smoke_elice.sh <영상경로>}")" && pwd)/$(basename "$1")"
OUT="${HOME}/smoke_out"
rm -rf "$OUT" && mkdir -p "$OUT"

echo "=== CoMotion (30프레임) ==="
cd "$ENV_ROOT/ml-comotion"
conda run --no-capture-output -n comotion python demo.py \
  -i "$V" -o "$OUT/comotion" --num-frames 30
find "$OUT/comotion" -type f -exec ls -lh {} \;

echo
echo "=== Multi-HMR 2 (30프레임) ==="
# --num-frames 옵션이 없으므로 ffmpeg 로 잘라 넣는다
ffmpeg -v error -y -i "$V" -frames:v 30 -c copy "$OUT/_trim30.mp4"
cd "$ENV_ROOT/multi-hmr2"
conda run --no-capture-output -n multihmr2 multihmr2 \
  --checkpoint checkpoints/multihmr2.pt \
  --video "$OUT/_trim30.mp4" --out "$OUT/multihmr2" \
  --save_anny_params --render
find "$OUT/multihmr2" -type f -exec ls -lh {} \;

echo
echo "=== 출력 구조 덤프 → ELICE.md ==="
bash "$SCRIPT_DIR/dump_outputs.sh" "$OUT" | tee -a "$REPO_ROOT/smpl_eval/ELICE.md"

echo
echo "게이트 0 확인 사항:"
echo "  1) 두 모델 모두 출력 파일이 생성되었는가"
echo "  2) 렌더 mp4 에서 사람 위에 메쉬가 보이는가 (직접 열어볼 것)"
