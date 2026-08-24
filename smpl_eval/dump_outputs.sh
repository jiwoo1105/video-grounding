#!/usr/bin/env bash
# 모델 출력의 실제 키 이름과 shape 를 덤프한다.
# Task 10/11 어댑터의 KEY_* 상수와 ANNY_TO_SMPL 매핑은 이 출력으로 확정한다.
set -euo pipefail
OUT="${1:?사용법: dump_outputs.sh <스모크출력디렉터리>}"

echo
echo "## 모델 출력 구조  ($(date +%Y-%m-%d))"

echo
echo "### CoMotion .pt"
echo '```'
conda run --no-capture-output -n comotion python - "$OUT" <<'PY'
import sys, glob, torch
hits = glob.glob(f"{sys.argv[1]}/comotion/**/*.pt", recursive=True)
if not hits:
    print("(.pt 없음)"); raise SystemExit
print("file:", hits[0])
def walk(o, p="", depth=0):
    if depth > 4: return
    if isinstance(o, dict):
        for k, v in o.items(): walk(v, f"{p}/{k}", depth+1)
    elif isinstance(o, (list, tuple)):
        print(f"{p}: {type(o).__name__}[{len(o)}]")
        if o: walk(o[0], f"{p}[0]", depth+1)
    elif hasattr(o, "shape"):
        print(f"{p}: {tuple(o.shape)} {o.dtype}")
    else:
        print(f"{p}: {type(o).__name__} = {repr(o)[:80]}")
walk(torch.load(hits[0], map_location="cpu"))
PY
echo '```'

echo
echo "### CoMotion MOT .txt (앞 5줄)"
echo '```'
head -5 "$OUT"/comotion/**/*.txt 2>/dev/null || head -5 "$OUT"/comotion/*.txt 2>/dev/null || echo "(.txt 없음)"
echo '```'

echo
echo "### Multi-HMR 2 .pkl"
echo '```'
conda run --no-capture-output -n multihmr2 python - "$OUT" <<'PY'
import sys, glob, pickle
hits = sorted(glob.glob(f"{sys.argv[1]}/multihmr2/**/*.pkl", recursive=True))
if not hits:
    print("(.pkl 없음)"); raise SystemExit
print(f"files: {len(hits)}, first: {hits[0]}")
def walk(o, p="", depth=0):
    if depth > 4: return
    if isinstance(o, dict):
        for k, v in o.items(): walk(v, f"{p}/{k}", depth+1)
    elif isinstance(o, (list, tuple)):
        print(f"{p}: {type(o).__name__}[{len(o)}]")
        if o: walk(o[0], f"{p}[0]", depth+1)
    elif hasattr(o, "shape"):
        print(f"{p}: {tuple(o.shape)} {getattr(o,'dtype','')}")
    else:
        print(f"{p}: {type(o).__name__} = {repr(o)[:80]}")
walk(pickle.load(open(hits[0], "rb")))
PY
echo '```'

echo
echo "### Anny 관절명 / 토폴로지"
echo '```'
conda run --no-capture-output -n multihmr2 python - <<'PY'
try:
    import anny, inspect
    print("anny", getattr(anny, "__version__", "?"))
    print("topo 관련 심볼:", [t for t in dir(anny) if "topo" in t.lower()])
    cls = getattr(anny, "Anny", None)
    if cls:
        print("Anny.__init__:", inspect.signature(cls.__init__))
        for attr in ("joint_names", "JOINT_NAMES", "bone_names"):
            if hasattr(cls, attr):
                print(attr, "=", getattr(cls, attr))
except Exception as e:
    print("anny 조회 실패:", e)
PY
echo '```'
