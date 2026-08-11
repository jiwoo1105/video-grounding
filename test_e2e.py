#!/usr/bin/env python3
"""
run_experiments.py E2E 검증 — GPU / 네트워크 없이 실행.

임시 디렉터리에 가짜 영상과 **TimeLens-Bench 실제 형식**의 주석 파일을 만들고,
전체 실험 루프를 돌려서

  · 벤치마크 주석 파싱 (질의별 GT 정렬)
  · 모델 로딩 횟수 (롱비디오 재로드가 모델당 1회인지)
  · tIoU 계산
  · report.md / results.jsonl / PNG 생성

을 확인합니다.  실행:  python3 test_e2e.py
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
fails = []

# matplotlib 은 타임라인 그림에만 쓰입니다. 로컬 검증에는 없어도 무방하고,
# 엘리스 인스턴스에서는 setup_env.sh 가 설치합니다.
# find_spec 만으로는 "파일은 있는데 import 가 깨진" 경우를 못 걸러서 실제로 import 해봅니다.
try:
    import matplotlib  # noqa: F401
    HAS_MPL = True
except Exception:      # noqa: BLE001
    HAS_MPL = False


def check(cond, msg):
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        fails.append(msg)


def skip(msg):
    print(f"  SKIP  {msg}")


# ---- 가짜 모듈 주입 (test_vtg_run.py 앞부분 재사용) -----------------------
src = (HERE / "test_vtg_run.py").read_text("utf-8")
G = {"__name__": "inject"}
exec(compile(src[:src.index("sys.path.insert")], "inject", "exec"), G)

OUTPUTS = [
    "The event happens in 24.00 - 30.00 seconds.",   # 3MSZA q1  GT [25,30]
    "The event happens in 2.00 - 22.00 seconds.",    # 3MSZA q2  GT [1,24]
    "The event happens in 0.00 - 1.00 seconds.",     # 3MSZA q3  GT [0,1]   완벽
    "The event happens in 40.00 - 50.00 seconds.",   # 3MSZA q4  GT [23,24] 완전 빗나감
    "The event happens in 6.00 - 28.00 seconds.",    # AMT7R q1  GT [5,30]
    "I cannot find it.",                             # 파싱 실패 케이스
]
_it = iter(OUTPUTS * 50)
G["FakeProcessor"].batch_decode = lambda self, ids, **kw: [next(_it)]

# ---- 실제 TimeLens-Bench 형식 그대로의 주석 ------------------------------
REAL_FORMAT_ANN = {
    "3MSZA": {
        "duration": 31.0,
        "spans": [[25, 30], [1, 24], [0, 1], [23, 24]],
        "queries": [
            "A woman is repeatedly flipping the switch on the wall.",
            "A woman is eating chips.",
            "A woman holding chips leans against the door frame.",
            "The woman stands up straight from leaning on the door frame.",
        ],
    },
    "AMT7R": {
        "duration": 30.11241970021413,
        "spans": [[5, 30]],
        "queries": ["A person is putting clothes into a washing machine."],
    },
    "ZZZZZ": {           # 영상 파일이 없는 케이스 -> 건너뛰어야 함
        "duration": 20.0,
        "spans": [[1, 5]],
        "queries": ["should be skipped"],
    },
}

tmp = Path(tempfile.mkdtemp(prefix="vtg_e2e_"))
for f in ("vtg_run.py", "run_experiments.py"):
    shutil.copy(HERE / f, tmp / f)
os.chdir(tmp)
sys.path.insert(0, str(tmp))

vd = tmp / "videos"
vd.mkdir()
(vd / "2Y8XQ.mp4").touch()
bench_vd = vd / "_bench" / "videos"
bench_vd.mkdir(parents=True)
(bench_vd / "3MSZA.mp4").touch()
(bench_vd / "AMT7R.mp4").touch()

import run_experiments as R  # noqa: E402

print("=" * 74)
print(" 1. 벤치마크 주석 파싱 (TimeLens-Bench 실제 형식)")
print("=" * 74)

# download_bench 의 선택 로직만 떼어내 검증
avail = {p.stem: p for p in bench_vd.rglob("*.mp4")}
picked = []
for vid, rec in REAL_FORMAT_ANN.items():
    if vid not in avail:
        continue
    qs, sp = rec.get("queries") or [], rec.get("spans") or []
    pairs = [(q, s) for q, s in zip(qs, sp)
             if isinstance(s, (list, tuple)) and len(s) == 2 and s[1] > s[0]]
    if not pairs:
        continue
    picked.append(dict(
        name=vid, group="벤치마크(GT있음)", path=str(avail[vid]),
        desc=f"Charades-TimeLens / {vid} ({rec.get('duration','?')}s)",
        queries=[q for q, _ in pairs],
        gts=[[float(s[0]), float(s[1])] for _, s in pairs],
    ))

check(len(picked) == 2, f"영상 파일 있는 것만 선택됨 ({len(picked)}개, ZZZZZ 제외)")
check(picked[0]["name"] == "3MSZA" and len(picked[0]["queries"]) == 4,
      "3MSZA 질의 4개 로드")
check(picked[0]["gts"] == [[25.0, 30.0], [1.0, 24.0], [0.0, 1.0], [23.0, 24.0]],
      "GT 가 queries 순서와 정렬됨")
check(picked[0]["queries"][1] == "A woman is eating chips.",
      "queries[1] <-> gts[1] 짝이 맞음")
check(picked[1]["name"] == "AMT7R" and picked[1]["gts"] == [[5.0, 30.0]],
      "AMT7R 질의 1개 로드")

(vd / "bench.json").write_text(json.dumps(picked, ensure_ascii=False, indent=2), "utf-8")

# 내 영상(5분) — external.json 에 long:true 로 등록된 형태
(vd / "mylong.mp4").touch()
(vd / "external.json").write_text(json.dumps([
    dict(name="2Y8XQ", group="외부(공식 데모)", desc="TimeLens 공식 데모",
         queries=["A man drinks water with a glass", "a dog runs across the room"]),
    dict(name="mylong", group="내 영상", long=True, path=str(vd / "mylong.mp4"),
         desc="직접 추가 / 5분", queries=["a woman opens the door",
                                          "two people shake hands"]),
], ensure_ascii=False, indent=2), "utf-8")

print()
print("=" * 74)
print(" 1-b. 모듈 무결성 — 함수 누락 / 미정의 이름 검사")
print("=" * 74)
# main() 이 호출하는 함수가 실제로 정의돼 있는지. (편집 중 함수가 통째로
# 사라져도 --download-bench 를 안 돌리면 모르고 지나가는 사고를 막습니다.)
import ast as _ast


_src = (HERE / "run_experiments.py").read_text("utf-8")
_tree = _ast.parse(_src)
_defined = {n.name for n in _ast.walk(_tree) if isinstance(n, _ast.FunctionDef)}
for fn in ("sh", "hf_cli", "download_external", "download_bench",
           "add_video", "load_videos", "tiou", "main"):
    check(fn in _defined and hasattr(R, fn), f"{fn}() 정의됨")

# main() 본문에서 호출하는 모듈 내부 함수가 전부 존재하는지
_main = next(n for n in _tree.body if isinstance(n, _ast.FunctionDef) and n.name == "main")
_called = {n.func.id for n in _ast.walk(_main)
           if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name)}
_missing = {c for c in _called if c in _defined or c in ("download_bench",)} - _defined
check(not _missing, f"main() 이 호출하는 내부 함수 전부 존재 (누락: {_missing or '없음'})")

# CLI 조합이 argparse 를 통과하는지 (오타난 flag / 잘못된 choices 를 잡음)
_ok = True
for argv in (["--download"], ["--download-bench"],
             ["--download-bench", "--bench-dataset", "activitynet",
              "--bench-longest", "--bench-n", "1"],
             ["--download-bench", "--bench-dataset", "qvhighlights", "--all-shards"],
             ["--download-bench", "--bench-pick", "E6DLK,KOVTR", "--cleanup-shard"],
             ["--add-video", "x.mp4", "--queries", "a|b", "--long"],
             ["--list-videos"], ["--model", "safe"],
             ["--model", "main", "--skip-long", "--attn", "flash_attention_2"],
             ["--long-tokens", "8192", "--long-fps", "1.0"]):
    try:
        R.build_parser().parse_args(argv)
    except SystemExit:
        _ok = False
        print(f"    argparse 실패: {argv}")
check(_ok, "모든 CLI 조합이 argparse 를 통과")
check(set(R.BENCH_SETS) == {"charades", "activitynet", "qvhighlights"},
      "벤치마크 데이터셋 3종 등록")
check(all(len(v) == 5 for v in R.BENCH_SETS.values()),
      "BENCH_SETS 항목마다 (주석, 폴더, 평균길이, 샤드1개GB, 전체GB) 5개")
# 샤드 다운로드 패턴이 실제 파일명 규칙과 맞는지 (charades_shard_01.tar.gz)
for k, (js, folder, avg, gb1, gball) in R.BENCH_SETS.items():
    check(f"{folder}_shard_01.tar.gz".startswith(folder), f"{k}: 샤드 파일명 규칙 일치")
check(R.BENCH_SETS["activitynet"][4] == 50.2, "ActivityNet 전체 용량 50.2GB 기록됨")

print()
print("=" * 74)
print(" 2. tIoU 계산")
print("=" * 74)
for (a, b, want) in [([10, 20], [10, 20], 1.0), ([11, 19], [10, 20], 0.8),
                     ([15, 25], [10, 20], 1 / 3), ([30, 40], [10, 20], 0.0)]:
    got = R.tiou(a, b)
    check(abs(got - want) < 1e-9, f"tiou({a},{b}) = {got:.4f}")

print()
print("=" * 74)
print(" 3. 실험 계획 로딩 (--list-videos 경로)")
print("=" * 74)
vids = R.load_videos()
check(len(vids) == 4, f"영상 4개 (벤치 2 + 외부 1 + 내 영상 1), 실제 {len(vids)}개")
check(vids[-1]["name"] == "mylong", "긴 영상이 맨 뒤로 정렬됨 (재로드 1회로 축소)")
check(vids[0].get("gts") is not None, "벤치 = GT 있음")
check([v for v in vids if v["name"] == "2Y8XQ"][0].get("gts") is None, "외부 = GT 없음")
check([v for v in vids if v["name"] == "mylong"][0].get("long") is True,
      "내 영상 = long 플래그로 예산 축소 대상")

print()
print("=" * 74)
print(" 4. 전체 실험 실행 (모델 2개)")
print("=" * 74)
G["CALLS"].clear()
sys.argv = ["run_experiments.py", "--model", "timelens-8b,timelens2-4b"]
R.main()

loads = G["CALLS"]["model_load"]
check(len(loads) == 4, f"모델 로딩 {len(loads)}회 == 4 (모델2 x [일반+롱비디오])")
check([l[0] for l in loads] == [
    "TencentARC/TimeLens-8B", "TencentARC/TimeLens-8B",
    "MCG-NJU/TimeLens2-4B", "MCG-NJU/TimeLens2-4B"], "로딩 순서 정상")

print()
print("=" * 74)
print(" 5. 산출물 검증")
print("=" * 74)
res = tmp / "results"
check((res / "results.jsonl").exists(), "results.jsonl 생성")
check((res / "report.md").exists(), "report.md 생성")
for n in ("3MSZA", "AMT7R", "2Y8XQ", "mylong"):
    if HAS_MPL:
        check((res / f"{n}.png").exists(), f"{n}.png 타임라인 생성")
    else:
        skip(f"{n}.png — matplotlib 미설치 (그림만 영향, 수치는 정상)")

rows = [json.loads(l) for l in (res / "results.jsonl").read_text("utf-8").splitlines()]
check(len(rows) == 2 * (4 + 1 + 2 + 2), f"결과 {len(rows)}건 == 모델2 x 질의9")

with_gt = [r for r in rows if "tiou" in r]
check(len(with_gt) > 0, f"tIoU 계산된 건수 {len(with_gt)}")
check(all(r["group"] == "벤치마크(GT있음)" for r in with_gt),
      "tIoU 는 GT 있는 영상에만 붙음")

# 완벽 일치 케이스가 1.0 인지
perfect = [r for r in rows if r.get("gt") == [0.0, 1.0] and "tiou" in r]
check(perfect and perfect[0]["tiou"] == 1.0,
      f"GT [0,1] 완전 일치 -> tIoU 1.0 (실제 {perfect[0]['tiou'] if perfect else 'N/A'})")

miss = [r for r in rows if r.get("gt") == [23.0, 24.0] and "tiou" in r]
check(miss and miss[0]["tiou"] == 0.0,
      f"완전 빗나감 -> tIoU 0.0 (실제 {miss[0]['tiou'] if miss else 'N/A'})")

rep = (res / "report.md").read_text("utf-8")
check("평균 tIoU" in rep, "리포트에 정량 섹션 포함")
check("R1@0.5" in rep and "R1@0.7" in rep, "리포트에 R1 지표 포함")
check("A woman is eating chips" in rep, "리포트에 질의 문장 포함")
check("내 영상" in rep, "리포트에 내 영상 결과 포함")

print()
print("─" * 74)
print(" report.md 미리보기")
print("─" * 74)
print("\n".join(rep.splitlines()[:22]))

print()
print("=" * 74)
print(f" 결과: {'전부 통과' if not fails else str(len(fails)) + '건 실패'}")
print("=" * 74)
for f in fails:
    print("  FAIL:", f)
if not HAS_MPL:
    print("\n  참고: matplotlib 이 없어 타임라인 그림 검증은 건너뛰었습니다.")
    print("        로컬에서도 보고 싶으면:  pip3 install matplotlib")
    print("        엘리스에서는 setup_env.sh 가 자동으로 설치하므로 문제 없습니다.")
shutil.rmtree(tmp, ignore_errors=True)
sys.exit(1 if fails else 0)
