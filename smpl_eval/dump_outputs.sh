#!/usr/bin/env bash
# 모델 출력의 실제 키 이름과 shape 를 덤프한다.
# Task 10/11 어댑터의 KEY_* 상수와 ANNY_TO_SMPL 매핑은 이 출력으로 확정한다.
set -euo pipefail
OUT="${1:?사용법: dump_outputs.sh <스모크출력디렉터리>}"
VENV_ROOT="${HOME}/venvs"

echo
echo "## 모델 출력 구조  ($(date +%Y-%m-%d))"

WALK='
def walk(o, p="", depth=0):
    if depth > 5: return
    if isinstance(o, dict):
        print(f"{p}/ : dict[{len(o)}] keys={list(o)[:12]}")
        for k, v in o.items(): walk(v, f"{p}/{k}", depth+1)
    elif isinstance(o, (list, tuple)):
        print(f"{p} : {type(o).__name__}[{len(o)}]")
        if o: walk(o[0], f"{p}[0]", depth+1)
    elif hasattr(o, "shape"):
        print(f"{p} : {tuple(o.shape)} {getattr(o,'"'"'dtype'"'"','"'"''"'"')}")
    else:
        print(f"{p} : {type(o).__name__} = {repr(o)[:100]}")
'

echo
echo "### CoMotion .pt"
echo '```'
"$VENV_ROOT/comotion/bin/python" - "$OUT" <<PY
import sys, glob, torch
$WALK
hits = sorted(glob.glob(f"{sys.argv[1]}/comotion/**/*.pt", recursive=True))
if not hits:
    print("(.pt 없음)"); raise SystemExit
print("file:", hits[0])
walk(torch.load(hits[0], map_location="cpu", weights_only=False))
PY
echo '```'

echo
echo "### CoMotion 기타 출력 파일"
echo '```'
find "$OUT/comotion" -type f -printf '%-60p %10s bytes\n' 2>/dev/null || ls -l "$OUT/comotion"
echo "--- .txt 앞 5줄 ---"
find "$OUT/comotion" -name '*.txt' -exec head -5 {} \; 2>/dev/null || echo "(.txt 없음)"
echo '```'

echo
echo "### Multi-HMR 2 .pkl"
echo '```'
"$VENV_ROOT/multihmr2/bin/python" - "$OUT" <<PY
import sys, glob, pickle
$WALK
hits = sorted(glob.glob(f"{sys.argv[1]}/multihmr2/**/*.pkl", recursive=True))
if not hits:
    print("(.pkl 없음)"); raise SystemExit
print(f"files: {len(hits)}")
for h in hits[:3]: print("  ", h)
walk(pickle.load(open(hits[0], "rb")))
PY
echo '```'

echo
echo "### Multi-HMR 2 기타 출력 파일"
echo '```'
find "$OUT/multihmr2" -type f -printf '%-60p %10s bytes\n' 2>/dev/null | head -20
echo '```'

echo
echo "### Anny 바디모델 정보"
echo '```'
"$VENV_ROOT/multihmr2/bin/python" - <<'PY'
try:
    import anny, inspect
    print("anny", getattr(anny, "__version__", "?"))
    print("모듈 심볼:", [t for t in dir(anny) if not t.startswith("_")][:30])
    cls = getattr(anny, "Anny", None)
    if cls:
        print("Anny.__init__:", inspect.signature(cls.__init__))
        for attr in ("joint_names", "JOINT_NAMES", "bone_names", "BONE_NAMES",
                     "topologies", "TOPOLOGIES"):
            if hasattr(cls, attr):
                v = getattr(cls, attr)
                print(f"{attr} ({len(v) if hasattr(v,'__len__') else '?'}):", v)
except Exception as e:
    print("anny 조회 실패:", type(e).__name__, e)
PY
echo '```'
