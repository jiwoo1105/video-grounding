#!/usr/bin/env python3
"""
Video Temporal Grounding — 영상 여러 개 x 모델 여러 개 일괄 실험

영상 구성
  [벤치마크]  TimeLens-Bench(Charades-TimeLens)에서 추출. **질의마다 진짜 GT** -> tIoU 계산
  [외부]      TimeLens 공식 데모 영상. GT 없음 -> 정성 평가
  [내 영상]   직접 넣은 영상. --add-video 로 등록

vtg_run.py 를 import 해서 **모델을 한 번만 로드**하고 모든 영상을 처리합니다.
(vtg_run.py 를 영상마다 호출하면 8B 로딩 1~2분을 매번 반복합니다.)

사용:
    python3 run_experiments.py --download           # 공식 데모 영상 (2.3MB)
    python3 run_experiments.py --download-bench     # 벤치마크 영상 + GT (수 GB)
    python3 run_experiments.py --add-video my.mp4 \
        --queries "a woman opens the door|two people shake hands" --long
    python3 run_experiments.py --list-videos        # 실험 계획 확인
    python3 run_experiments.py --model safe         # 실행

산출물:  results/results.jsonl, results/report.md, results/*.png
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

VIDEO_DIR = Path("videos")
OUT_DIR = Path("results")
SPEC_EXT = VIDEO_DIR / "external.json"       # 사용자가 편집 가능
SPEC_BENCH = VIDEO_DIR / "bench.json"        # --download-bench 가 생성

BENCH_REPO = "TencentARC/TimeLens-Bench"

# TimeLens-Bench 는 세 데이터셋의 정제본을 담고 있습니다.
# 전부 사람이 직접 검수한 질의 + 정답 구간이라, 제가 임의로 질의를 지어낼 필요가 없습니다.
# 샤드 용량은 2026-08 기준 실측값입니다. 다운로드 전에 경고를 띄우는 데 씁니다.
BENCH_SETS = {
    #  키            주석 파일                   영상 폴더      평균 길이          샤드1개  전체
    "charades":    ("charades-timelens.json",    "charades",    "29.6초",          6.5,   6.5),
    "activitynet": ("activitynet-timelens.json", "activitynet", "134.9초 / 최대 4분대", 8.4,  50.2),
    "qvhighlights":("qvhighlights-timelens.json","qvhighlights","149.6초",         8.0,   None),
}

# ==========================================================================
#  외부 영상 기본값
#
#  구글 샘플 버킷(commondatastorage.googleapis.com)은 2026년 8월 기준 접근이
#  막혀서 쓸 수 없습니다. 대신 TimeLens 공식 데모 영상(HuggingFace)을 씁니다.
#
#  ★ 본인 영상을 쓰시려면 videos/ 에 mp4 를 넣고 videos/external.json 에
#    항목을 추가하세요 (url 없이 name/queries 만 있으면 됩니다).
# ==========================================================================
EXTERNAL_DEFAULT = [
    dict(
        name="2Y8XQ", group="외부(공식 데모)",
        url="https://huggingface.co/datasets/JungleGym/TimeLens-Assets/resolve/main/2Y8XQ.mp4",
        desc="TimeLens 공식 데모 영상 (2.3MB). 모델 카드에 쓰인 질의를 그대로 사용",
        queries=[
            "A man drinks water with a glass",      # 공식 모델 카드의 질의
            "a person walks through a doorway",
            "a dog runs across the room",           # 없는 사건 (no-target 테스트)
        ],
    ),
]


def sh(cmd):
    print(f"  $ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def hf_cli():
    """huggingface_hub 버전에 따라 명령어 이름이 다릅니다."""
    if shutil.which("hf"):
        return "hf download"
    if shutil.which("huggingface-cli"):
        return "huggingface-cli download"
    sys.exit("!! hf CLI 를 찾을 수 없습니다.  pip install -U huggingface_hub")


def download_external():
    VIDEO_DIR.mkdir(exist_ok=True)
    for v in EXTERNAL_DEFAULT:
        p = VIDEO_DIR / f"{v['name']}.mp4"
        if p.exists():
            print(f"  [skip] {p} 이미 있음")
            continue
        # -f: HTTP 오류 시 빈 파일을 남기지 않고 실패시킴
        try:
            sh(f'curl -fL --progress-bar -o "{p}" "{v["url"]}"')
        except subprocess.CalledProcessError:
            p.unlink(missing_ok=True)
            print(f"  !! 다운로드 실패: {v['url']}")
            print("     URL 이 막혔을 수 있습니다. videos/ 에 직접 mp4 를 넣고")
            print("     videos/external.json 에 항목을 추가해도 됩니다.")
    if not SPEC_EXT.exists():
        SPEC_EXT.write_text(json.dumps(EXTERNAL_DEFAULT, ensure_ascii=False, indent=2), "utf-8")
        print(f"\n  질의 문장은 {SPEC_EXT} 에서 편집하세요 (코드 수정 불필요).")


def download_bench(dataset="charades", n_videos=2, max_q=4, longest=False,
                   all_shards=False, pick=None, cleanup=False):
    """
    TimeLens-Bench 주석 + 영상 샤드를 받아 GT 있는 영상 n개를 고릅니다.

    주석 실제 형식 (2026-08 확인):
        {
          "3MSZA": {
            "duration": 31.0,
            "spans":   [[25,30], [1,24], ...],     # queries 와 같은 순서
            "queries": ["A woman is ...", ...]
          }, ...
        }
    키는 영상 ID(확장자 없음). spans[i] 가 queries[i] 의 정답 구간입니다.

    longest=True 면 **받은 샤드 안에서** 긴 영상부터 고릅니다. 긴 영상용 질의를
    임의로 지어낼 필요 없이 **사람이 검수한 질의 + 정답 구간**을 그대로 씁니다.

    ★ 샤드 1개가 6~8GB 입니다. 기본은 1개만 받습니다.
      ActivityNet 은 샤드가 6개(총 50GB)라 전부 받으면 디스크와 시간이 크게 듭니다.
      샤드 1개에도 영상이 수백 개 들어 있어 긴 영상 후보는 충분합니다.
    """
    if dataset not in BENCH_SETS:
        sys.exit(f"!! 알 수 없는 데이터셋: {dataset}  (선택: {', '.join(BENCH_SETS)})")
    js, folder, avg, gb1, gball = BENCH_SETS[dataset]

    est = (gball or gb1 * 6) if all_shards else gb1
    print(f"\n예상 다운로드: 약 {est:.1f} GB"
          f"{' (샤드 전체)' if all_shards else ' (샤드 1개)'}"
          f" + 압축 해제에 비슷한 용량이 더 필요합니다.")
    if all_shards and (gball or 0) > 20:
        print(f"!! {dataset} 전체는 {gball} GB 입니다. 정말 필요한지 확인하세요.")

    VIDEO_DIR.mkdir(exist_ok=True)
    root = VIDEO_DIR / "_bench"
    root.mkdir(exist_ok=True)
    hf = hf_cli()

    print(f"\n[1/3] 주석 파일 다운로드 — {dataset} (평균 {avg})")
    sh(f'{hf} {BENCH_REPO} {js} --repo-type=dataset --local-dir "{root}"')
    ann = json.loads((root / js).read_text("utf-8"))
    if not isinstance(ann, dict):
        sys.exit("!! 주석 형식이 예상과 다릅니다 (dict 가 아님). 레포가 갱신됐을 수 있습니다.")
    print(f"      영상 {len(ann)}개 / 주석 "
          f"{sum(len(v.get('queries', [])) for v in ann.values())}건")
    durs = sorted((v.get("duration", 0) for v in ann.values()), reverse=True)
    if durs:
        print(f"      길이: 최장 {durs[0]:.0f}초 / 중앙값 {durs[len(durs)//2]:.0f}초")

    print(f"\n[2/3] 영상 샤드 다운로드 (~{est:.1f} GB)")
    # ★ 기본은 샤드 1개만. 전부 받으려면 명시적으로 --all-shards.
    pattern = (f"video_shards/{folder}/*" if all_shards
               else f"video_shards/{folder}/{folder}_shard_01.tar.gz")
    sh(f'{hf} {BENCH_REPO} --repo-type=dataset '
       f'--include "{pattern}" --local-dir "{root}"')
    shards = sorted((root / "video_shards").rglob("*.tar.gz"))
    if not shards:
        sys.exit(f"!! 샤드를 찾지 못했습니다. 예상 경로: {pattern}\n"
                 f"   레포 구조가 바뀌었을 수 있습니다.")
    ex = root / "videos"
    ex.mkdir(exist_ok=True)
    # --bench-pick 을 주면 그 영상만 디스크에 씁니다 (압축 해제는 전체를 훑지만
    # 저장되는 건 고른 것뿐이라 용량을 크게 아낍니다).
    wild = " ".join(f"'*{v}*'" for v in pick) if pick else ""
    for s in shards:
        sh(f'tar -xzf "{s}" -C "{ex}"' + (f" --wildcards {wild}" if wild else ""))

    if cleanup:
        for s in shards:
            print(f"  샤드 삭제: {s.name}")
            s.unlink()

    avail = {p.stem: p for p in ex.rglob("*.mp4")}      # 3MSZA -> 경로
    print(f"      영상 {len(avail)}개 추출됨")

    cands = [(vid, rec) for vid, rec in ann.items() if vid in avail]
    if pick:                        # 지정한 순서를 유지
        order = {v: i for i, v in enumerate(pick)}
        cands.sort(key=lambda kv: order.get(kv[0], 999))
        n_videos = max(n_videos, len(pick))
    elif longest:
        cands.sort(key=lambda kv: -kv[1].get("duration", 0))

    picked = []
    for vid, rec in cands:
        qs, sp = rec.get("queries") or [], rec.get("spans") or []
        pairs = [(q, s) for q, s in zip(qs, sp)
                 if isinstance(s, (list, tuple)) and len(s) == 2 and s[1] > s[0]]
        if not pairs:
            continue
        pairs = pairs[:max_q]
        d = float(rec.get("duration", 0))
        picked.append(dict(
            name=vid, group="벤치마크(GT있음)", path=str(avail[vid]),
            desc=f"{dataset}-TimeLens / {vid} ({d:.0f}초)",
            queries=[q for q, _ in pairs],
            gts=[[float(s[0]), float(s[1])] for _, s in pairs],
            **({"long": True} if d >= 180 else {}),     # 3분 이상이면 예산 축소 대상
        ))
        if len(picked) >= n_videos:
            break

    if not picked:
        sys.exit("!! 주석과 영상이 매칭되지 않았습니다. --all-shards 로 전체를 풀어보세요.")

    # 기존 목록에 병합 (charades + activitynet 을 함께 쓸 수 있게)
    prev = json.loads(SPEC_BENCH.read_text("utf-8")) if SPEC_BENCH.exists() else []
    names = {p["name"] for p in picked}
    merged = [p for p in prev if p["name"] not in names] + picked
    SPEC_BENCH.write_text(json.dumps(merged, ensure_ascii=False, indent=2), "utf-8")

    # 웹 UI 에서 바로 올릴 수 있도록 videos/ 바로 아래에도 복사해 둡니다.
    for p in picked:
        dst = VIDEO_DIR / f"{p['name']}.mp4"
        if not dst.exists():
            shutil.copy(p["path"], dst)

    print(f"\n[3/3] 선택된 영상 {len(picked)}개 (목록 전체 {len(merged)}개)")
    print("=" * 74)
    for p in picked:
        tag = "   [긴 영상]" if p.get("long") else ""
        print(f"\n  ■ videos/{p['name']}.mp4    {p['desc']}{tag}")
        print("    ┌ 웹 UI 에 올린 뒤 아래 문장을 그대로 넣어보세요 ─────────────")
        for q, g in zip(p["queries"], p["gts"]):
            print(f"    │  {q}")
            print(f"    │      -> 정답 {g[0]:.0f}초 ~ {g[1]:.0f}초")
        print("    └────────────────────────────────────────────────────────")
    print("=" * 74)
    print("\n  웹 UI 실행:  python3 app.py")


def add_video(path, queries, long=False, name=None):
    """내 영상을 실험 목록(videos/external.json)에 등록합니다."""
    src = Path(path)
    if not src.exists():
        sys.exit(f"!! 파일이 없습니다: {src}")
    VIDEO_DIR.mkdir(exist_ok=True)
    name = name or src.stem
    dst = VIDEO_DIR / f"{name}{src.suffix}"
    if src.resolve() != dst.resolve():
        shutil.copy(src, dst)

    specs = json.loads(SPEC_EXT.read_text("utf-8")) if SPEC_EXT.exists() else list(EXTERNAL_DEFAULT)
    specs = [s for s in specs if s.get("name") != name]        # 같은 이름은 교체
    entry = dict(name=name, group="내 영상", path=str(dst),
                 desc=f"직접 추가 / {dst.name}", queries=queries)
    if long:
        entry["long"] = True          # 토큰 예산을 자동으로 낮춰 처리
    specs.append(entry)
    SPEC_EXT.write_text(json.dumps(specs, ensure_ascii=False, indent=2), "utf-8")

    print(f"등록 완료: {dst}")
    for q in queries:
        print(f"  - {q}")
    if long:
        print("  (롱비디오로 표시됨 — 토큰 예산을 낮춰 처리합니다)")
    print(f"\n수정하려면 {SPEC_EXT} 를 직접 편집하세요.")


def load_videos(skip_long=False, bench_only=False, ext_only=False):
    vids = []
    if not ext_only and SPEC_BENCH.exists():
        vids += json.loads(SPEC_BENCH.read_text("utf-8"))
    if not bench_only:
        specs = (json.loads(SPEC_EXT.read_text("utf-8"))
                 if SPEC_EXT.exists() else EXTERNAL_DEFAULT)
        for v in specs:
            p = Path(v.get("path") or (VIDEO_DIR / f"{v['name']}.mp4"))
            if p.exists():
                vids.append({**v, "path": str(p), "gts": v.get("gts")})
    if skip_long:
        vids = [v for v in vids if not v.get("long")]
    # 롱비디오는 토큰 예산이 달라 재로드가 필요하므로 맨 뒤로
    return sorted(vids, key=lambda v: bool(v.get("long")))


def tiou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def build_parser():
    """CLI 정의. 테스트에서 그대로 재사용합니다."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="외부 영상 다운로드")
    ap.add_argument("--download-bench", action="store_true",
                    help="TimeLens-Bench에서 GT 있는 영상 추출 (수 GB)")
    ap.add_argument("--bench-dataset", default="charades", choices=list(BENCH_SETS),
                    help="charades(~30초) / activitynet(최대 4분대) / qvhighlights(~150초)")
    ap.add_argument("--bench-n", type=int, default=2, help="가져올 영상 수")
    ap.add_argument("--bench-longest", action="store_true",
                    help="받은 샤드 안에서 긴 영상부터 선택 (긴 영상 테스트용)")
    ap.add_argument("--all-shards", action="store_true",
                    help="샤드를 전부 받음. ActivityNet 은 50GB 이므로 보통 불필요")
    ap.add_argument("--bench-pick", metavar="IDS",
                    help="원하는 영상 ID만 추출. 콤마 구분 (예: E6DLK,KOVTR)")
    ap.add_argument("--cleanup-shard", action="store_true",
                    help="추출 후 tar.gz 삭제 (수 GB 회수)")
    ap.add_argument("--add-video", metavar="PATH", help="내 영상을 실험 목록에 등록")
    ap.add_argument("--queries", help="--add-video 와 함께. '|' 로 구분")
    ap.add_argument("--long", action="store_true",
                    help="--add-video 와 함께. 3분 이상이면 지정하세요 (토큰 예산 자동 축소)")
    ap.add_argument("--list-videos", action="store_true", help="실험 계획만 출력")
    ap.add_argument("--model", default="safe", help="vtg_run 의 모델 키 또는 프리셋")
    ap.add_argument("--skip-long", action="store_true", help="긴 영상 제외")
    ap.add_argument("--bench-only", action="store_true")
    ap.add_argument("--ext-only", action="store_true")
    ap.add_argument("--long-tokens", type=int, default=16384,
                    help="긴 영상에 쓸 토큰 예산 (5분 기준 16384, OOM 이면 8192)")
    ap.add_argument("--long-fps", type=float, default=1.0)
    ap.add_argument("--attn", default="sdpa",
                    choices=["sdpa", "eager", "flash_attention_2"])
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()

    if args.add_video:
        if not args.queries:
            sys.exit("!! --queries 가 필요합니다.  예: --queries \"질의1|질의2|질의3\"")
        qs = [q.strip() for q in args.queries.split("|") if q.strip()]
        add_video(args.add_video, qs, long=args.long)
        return
    if args.download:
        download_external()
    if args.download_bench:
        download_bench(dataset=args.bench_dataset, n_videos=args.bench_n,
                       longest=args.bench_longest, all_shards=args.all_shards,
                       pick=[v.strip() for v in args.bench_pick.split(",")]
                            if args.bench_pick else None,
                       cleanup=args.cleanup_shard)
    if args.download or args.download_bench:
        print("\n준비 완료. --list-videos 로 계획을 확인한 뒤 --model 로 실행하세요.")
        return

    videos = load_videos(args.skip_long, args.bench_only, args.ext_only)
    if not videos:
        sys.exit("영상이 없습니다. 먼저 --download 를 실행하세요.")

    n_runs = sum(len(v["queries"]) for v in videos)
    print("=" * 74)
    for v in videos:
        print(f"[{v['group']}] {v['name']}  ({v.get('desc','')})")
        gts = v.get("gts") or [None] * len(v["queries"])
        for q, g in zip(v["queries"], gts):
            print(f"    {('GT=' + str(g)) if g else 'GT 없음':<20} {q}")
    print("=" * 74)
    print(f"영상 {len(videos)}개 / 질의 {n_runs}건")

    if args.list_videos:
        return

    import vtg_run as V          # 여기서 torch/transformers 로드
    V.preflight()                # GPU / 라이브러리 확인 후 진행
    keys = V.PRESETS.get(args.model) or [k.strip() for k in args.model.split(",")]
    for k in keys:
        if k not in V.MODELS:
            sys.exit(f"알 수 없는 모델: {k}  (python vtg_run.py --list 로 확인)")
    print(f"모델 {len(keys)}개: {', '.join(keys)}  ->  총 {n_runs * len(keys)}회 추론\n")

    OUT_DIR.mkdir(exist_ok=True)
    results = []
    f = open(OUT_DIR / "results.jsonl", "w", encoding="utf-8")

    for key in keys:                     # 모델은 바깥 루프 — 한 번만 로드
        g = None
        try:
            g = V.Grounder(key, attn=args.attn)
            long_mode = False
            for v in videos:
                if v.get("long") and not long_mode:
                    g.free()             # 롱비디오 구간 진입: 예산 축소해 재구성
                    g = V.Grounder(key, total_tokens=args.long_tokens,
                                   attn=args.attn, fps=args.long_fps)
                    long_mode = True
                    print(f"\n  (롱비디오 모드: {args.long_tokens} 토큰 / {args.long_fps} fps)")
                dur = V.video_duration(v["path"])
                print(f"\n[{key}] {v['name']} ({dur:.1f}s)")
                gts = v.get("gts") or [None] * len(v["queries"])
                for q, gt in zip(v["queries"], gts):
                    try:
                        r = g(v["path"], q)
                    except Exception as e:          # noqa: BLE001
                        print(f"  !! 실패: {type(e).__name__}: {e}")
                        continue
                    r.update(video=v["name"], group=v["group"],
                             duration=round(dur, 2), gt=gt)
                    if gt and r["spans"]:
                        r["tiou"] = round(tiou(r["spans"][0], gt), 3)
                    results.append(r)
                    f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
                    mark = f"  tIoU={r['tiou']:<5}" if "tiou" in r else " " * 13
                    print(f"  {str(r['spans']):<26} {r['latency_s']:>6.1f}s"
                          f" {r['peak_vram_gb']:>5.1f}GB{mark} | {q[:52]}")
        finally:
            if g is not None:
                g.free()
    f.close()

    # ---- 리포트 --------------------------------------------------------
    lines = ["# Video Temporal Grounding 실험 결과", "",
             f"- 생성: {time.strftime('%Y-%m-%d %H:%M')}",
             f"- 모델: {', '.join(keys)}",
             f"- 영상 {len(videos)}개 / 결과 {len(results)}건", ""]

    def stats(ts):
        return (f"**{sum(ts)/len(ts):.3f}** | {sum(t>=0.5 for t in ts)/len(ts)*100:.1f}% "
                f"| {sum(t>=0.7 for t in ts)/len(ts)*100:.1f}% | {len(ts)}")

    by_model, by_mg = {}, {}
    for r in results:
        if "tiou" in r:
            by_model.setdefault(r["model"], []).append(r["tiou"])
            by_mg.setdefault((r["model"], r["group"]), []).append(r["tiou"])
    if by_model:
        lines += ["## 정량 결과 — 평균 tIoU (GT 있는 영상)", "",
                  "| 모델 | 평균 tIoU | R1@0.5 | R1@0.7 | n |", "|---|---|---|---|---|"]
        for m, ts in by_model.items():
            lines.append(f"| {m} | {stats(ts)} |")

        groups = sorted({g for _, g in by_mg})
        if len(groups) > 1:
            lines += ["", "### 짧은 영상 vs 긴 영상", "",
                      "| 모델 | 구분 | 평균 tIoU | R1@0.5 | R1@0.7 | n |",
                      "|---|---|---|---|---|---|"]
            for m in by_model:
                for g in groups:
                    ts = by_mg.get((m, g))
                    if ts:
                        lines.append(f"| {m} | {g} | {stats(ts)} |")
            lines += ["", "> 영상 길이에 따른 성능 저하를 직접 비교할 수 있습니다.",
                      "> 합성 롱비디오는 같은 클립을 이어붙인 것이라 난이도 자체는 동일합니다."]

        lines += ["", "> 표본이 작아 논문 수치와 직접 비교할 수는 없습니다. 동작 확인용입니다.", "",
                  "### 질의별", "",
                  "| 모델 | 영상 | 질의 | GT | 예측 | tIoU |", "|---|---|---|---|---|---|"]
        for r in results:
            if "tiou" in r:
                lines.append(f"| {r['model']} | {r['video']} | {r['query'][:40]} "
                             f"| {r['gt']} | {r['spans'][0]} | **{r['tiou']}** |")
        lines.append("")

    lines += ["## 전체 결과", "",
              "| 모델 | 구분 | 영상 | 길이 | 질의 | 예측 구간 | 지연 | VRAM |",
              "|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['model']} | {r['group']} | {r['video']} | {r['duration']}s "
                     f"| {r['query'][:44]} | {r['spans']} | {r['latency_s']}s "
                     f"| {r['peak_vram_gb']}GB |")

    lines += ["", "## 원본 출력 (모델별 1건)", ""]
    seen = set()
    for r in results:
        if r["model"] in seen:
            continue
        seen.add(r["model"])
        lines += [f"**{r['model']}** — `{r['query']}`", "", "```", r["raw"][:800], "```", ""]

    (OUT_DIR / "report.md").write_text("\n".join(lines), "utf-8")
    print(f"\n저장: {OUT_DIR}/results.jsonl, {OUT_DIR}/report.md")

    try:
        for v in videos:
            rs = [r for r in results if r["video"] == v["name"]]
            if rs:
                V.plot(rs, rs[0]["duration"], str(OUT_DIR / f"{v['name']}.png"))
    except ImportError:
        print("[plot] matplotlib 이 없어 타임라인 그림을 건너뜁니다. "
              "(수치 결과는 report.md 에 정상 저장됨)\n"
              "       설치:  pip install matplotlib")
    except Exception as e:      # noqa: BLE001
        print(f"[plot] 건너뜀: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
