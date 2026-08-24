"""오케스트레이션 — 영상 단위 격리 + 재개 가능.

엘리스 인스턴스가 중간에 죽어도 이미 만든 tracks.npz 는 건너뛰고
이어서 진행한다. 한 영상이 실패해도 나머지는 계속 돌린다.
"""
import argparse
import json
import os
import traceback

# 파일럿 4영상 — 데이터셋별로 다른 실패 모드를 노린다 (스펙 §7.1)
PILOT = [
    ("Data1", "Cam1_Deck0009_HL01_2K"),   # 300f  최다인원 밀집 → ID 스왑
    ("Data2", "cam1_2K"),                 # 2309f 최장 시퀀스 → 드리프트
    ("Data3", "cam-001"),                 # 1420f 60fps 2인 근접
    ("Data4", "CAM_M01"),                 # 660f  유사 무대의상 3인
]


def build_runner(model, repo_root):
    if model == "comotion":
        from smpl_eval.runners.comotion import CoMotionRunner
        return CoMotionRunner(os.path.join(repo_root, "ml-comotion"))
    if model == "multihmr2":
        from smpl_eval.runners.multihmr2 import MultiHMR2Runner
        return MultiHMR2Runner(os.path.join(repo_root, "multi-hmr2"))
    raise ValueError(f"알 수 없는 모델: {model}")


def select(records, pilot=False, dataset=None, cam=None):
    if pilot:
        want = set(PILOT)
        records = [r for r in records if (r["dataset"], r["cam"]) in want]
    if dataset:
        records = [r for r in records if r["dataset"] == dataset]
    if cam:
        records = [r for r in records if r["cam"] == cam]
    return records


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["comotion", "multihmr2"])
    ap.add_argument("--manifest", default="smpl_eval/manifest.json")
    ap.add_argument("--out", default="smpl_eval/outputs")
    ap.add_argument("--repo-root", default=os.path.expanduser("~/smpl_eval_env"))
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--cam", default=None)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="실행 대상만 출력하고 끝낸다 (모델 불필요)")
    a = ap.parse_args(argv)

    recs = select(json.load(open(a.manifest)), a.pilot, a.dataset, a.cam)
    if a.dry_run:
        total = sum(r["n_frames"] for r in recs)
        for r in recs:
            print(f"  {r['dataset']}/{r['session']}/{r['cam']}  {r['n_frames']}f")
        print(f"대상 {len(recs)}개 영상, {total} 프레임")
        return 0

    os.makedirs(a.out, exist_ok=True)
    runner = build_runner(a.model, a.repo_root)
    log = os.path.join(a.out, "failures.log")
    ok = fail = skip = 0

    for i, r in enumerate(recs, 1):
        od = os.path.join(a.out, a.model, r["dataset"], r["session"], r["cam"])
        dst = os.path.join(od, "tracks.npz")
        tag = f"{r['dataset']}/{r['session']}/{r['cam']}"
        if os.path.exists(dst) and not a.force:
            print(f"[{i}/{len(recs)}] {tag}  (이미 있음, 건너뜀)", flush=True)
            skip += 1
            continue
        print(f"[{i}/{len(recs)}] {tag}  {r['n_frames']}f", flush=True)
        try:
            p = runner.run(r["video_path"], od, r,
                           max_frames=a.max_frames, force=a.force)
            print(f"    → {p}", flush=True)
            ok += 1
        except Exception:
            fail += 1
            with open(log, "a", encoding="utf-8") as f:
                f.write(f"\n### {r['video_path']}\n{traceback.format_exc()}\n")
            print(f"    !! 실패 — {log} 에 기록", flush=True)

    print(f"\n완료: 성공 {ok}, 건너뜀 {skip}, 실패 {fail}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
