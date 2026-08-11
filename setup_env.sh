#!/usr/bin/env bash
# =============================================================================
#  Video Temporal Grounding 환경 셋업
#  대상 모델 : Time-R1-7B (Qwen2.5-VL 계열) + TimeLens2-8B (Qwen3-VL 계열)
#  검증 GPU  : A100 (sm_80) / H100 (sm_90) / RTX 5090 (sm_120)
#
#  드라이버 버전에 맞춰 cu126 / cu128 휠을 자동으로 고릅니다.
#
#  실행:  bash setup_env.sh
# =============================================================================
set -euo pipefail

echo "=============================================="
echo " [0/5] 하드웨어 / 드라이버 확인"
echo "=============================================="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || {
    echo "!! nvidia-smi 실패. GPU 인스턴스가 맞는지 확인하세요."; exit 1;
}

# 드라이버 요구사항은 GPU 세대별로 다릅니다.
#   A100 / H100 : 드라이버 525+ 면 충분 (엘리스 기본 이미지는 대부분 만족)
#   RTX 5090    : Blackwell(sm_120)이라 CUDA 12.8 + 드라이버 570+ 필수
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)

if echo "${GPU_NAME}" | grep -qiE "50[89]0|blackwell"; then
    if [ "${DRIVER}" -lt 570 ]; then
        echo ""
        echo "!! 중단: Blackwell GPU인데 드라이버가 ${DRIVER}.x 입니다 (570+ 필요)."
        echo "   CUDA 12.8 이상 이미지로 인스턴스를 다시 만드세요."
        exit 1
    fi
elif [ "${DRIVER}" -lt 525 ]; then
    echo ""
    echo "!! 경고: 드라이버 ${DRIVER}.x 는 다소 낮습니다. 문제가 생기면 이미지를 바꾸세요."
    echo ""
fi

echo ""
echo "=============================================="
echo " [1/5] 가상환경 생성"
echo "=============================================="
python3 -m venv ~/vtg-env
# shellcheck disable=SC1090
source ~/vtg-env/bin/activate
pip install -q --upgrade pip wheel setuptools

echo ""
echo "=============================================="
echo " [2/5] PyTorch 설치 (드라이버에 맞춰 자동 선택)"
echo "=============================================="
# ★ 가장 중요한 단계 ★
# 드라이버가 지원하는 CUDA 버전에 맞춰 휠을 고릅니다.
#   드라이버 570+ (CUDA 12.8)  -> cu128   (5090 Blackwell 은 이것만 가능)
#   드라이버 525+ (CUDA 12.x)  -> cu126   (A100/H100 에 안전한 선택)
# CUDA 는 12.x 안에서 마이너 버전 호환이 되지만, 굳이 위험을 감수할 이유가 없습니다.
#
# Time-R1 공식 레포가 지정한 torch 2.6 + cu124 를 그대로 쓰면
# 최신 GPU(sm_120)에서 "no kernel image is available" 로 죽습니다.
# 공식 requirements.txt 를 따라가지 마세요.
if [ "${DRIVER}" -ge 570 ]; then
    CU=cu128
else
    CU=cu126
fi
echo "드라이버 ${DRIVER}.x -> ${CU} 휠 설치"
pip install --index-url "https://download.pytorch.org/whl/${CU}" \
    torch torchvision

echo ""
echo "=============================================="
echo " [3/5] 라이브러리 설치"
echo "=============================================="
# transformers : 하한 4.57 = Qwen3-VL(TimeLens2) 지원 시작 버전.
#                상한 5.0  = transformers 5.x는 from_pretrained 인자명이 바뀌는 등
#                            breaking change가 있어 구형 Qwen2.5-VL 경로가 깨질 수 있음.
#                이 구간(4.57.x~4.9x)이 Qwen2.5-VL과 Qwen3-VL을 동시에 지원하는 유일한 창입니다.
# qwen-vl-utils>=0.0.14 : TimeLens2가 쓰는 return_video_metadata 인자에 필요.
#                         구버전 호출 규약과 하위호환되므로 Time-R1도 문제 없음.
pip install -q \
    "transformers>=4.57.0,<5.0" \
    "accelerate>=1.0" \
    "qwen-vl-utils[decord]>=0.0.14" \
    av matplotlib gradio \
    "huggingface_hub[hf_transfer]"

# 다운로드 가속 (시간당 과금이므로 체감 차이가 큽니다)
export HF_HUB_ENABLE_HF_TRANSFER=1
echo 'export HF_HUB_ENABLE_HF_TRANSFER=1' >> ~/.bashrc

# flash-attn은 sm_120용 사전빌드 휠이 없는 경우가 많고
# 소스 빌드에 30분 이상 걸립니다. 기본은 sdpa로 갑니다.
# 굳이 쓰겠다면:  pip install flash-attn --no-build-isolation

echo ""
echo "=============================================="
echo " [4/5] GPU 커널 동작 검증"
echo "=============================================="
python3 - <<'PY'
import sys, torch
print("torch        :", torch.__version__)
print("built w/ cuda:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA 사용 불가"
cc = torch.cuda.get_device_capability(0)
print("GPU          :", torch.cuda.get_device_name(0))
print("compute cap  :", f"{cc[0]}.{cc[1]}")
print("arch list    :", torch.cuda.get_arch_list())

if f"sm_{cc[0]}{cc[1]}" not in torch.cuda.get_arch_list():
    print("\n!! 이 torch 빌드에 sm_%d%d 커널이 없습니다." % cc)
    print("   다른 CUDA 휠로 재시도:  pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision")
    sys.exit(1)

# 실제 커널 실행까지 확인 (arch_list에 있어도 실행이 깨지는 경우가 있음)
a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
print("bf16 matmul  : OK", float((a @ a).float().sum()))
print("\n>>> 환경 정상 <<<")
PY

echo ""
echo "=============================================="
echo " [5/5] 모델 미리 받기 (선택)"
echo "=============================================="
echo "지금 받아두면 첫 추론이 바로 시작됩니다. 총 ~26GB."
echo "건너뛰려면 Ctrl+C — 나중에 자동으로 받습니다."
echo ""
read -r -p "지금 다운로드할까요? [y/N] " ans
if [[ "${ans}" =~ ^[Yy]$ ]]; then
    hf download TencentARC/TimeLens-8B      # UI 기본값 (CVPR'26)
    hf download MCG-NJU/TimeLens2-4B        # 스모크 테스트용 (가벼움)
    echo "다운로드 완료."
fi

echo ""
echo "=============================================="
echo " 완료"
echo "=============================================="
cat <<'EOF'

다음부터 접속할 때마다:

    source ~/vtg-env/bin/activate

웹 UI 실행 (권장):

    python3 app.py                 # http://<인스턴스주소>:7860
    python3 app.py --share         # 외부 접속용 임시 공개 링크

CLI 실행 예시:

    python3 vtg_run.py --model timelens-8b --video videos/2Y8XQ.mp4 \
        --query "A man drinks water with a glass"
    python3 run_experiments.py --model safe

EOF
