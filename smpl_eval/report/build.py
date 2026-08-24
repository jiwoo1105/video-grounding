"""results/*.json 을 모아 모델 대결 리포트를 만든다.

두 가지를 한 화면에 놓는 것이 목적이다.
 1. 데이터셋 x 모델 격자 — 어느 모델이 어느 데이터에서 무너지는가
 2. GT 노이즈 수준 — 모델 오차를 해석할 기준선
"""
import argparse
import glob
import html as _html
import json
import os
from collections import defaultdict

# (키, 표시명, 낮을수록 좋은가)
POSE_KEYS = [
    ("pa_mpjpe", "PA-MPJPE", True),
    ("pa_mpjpe_p95", "PA-MPJPE p95", True),
    ("limb_max_cv", "뼈길이 CV", True),
    ("mean_accel", "가속 지터", True),
    ("violation_rate", "관절각 위반율", True),
    ("beta_max_std", "β 최대 std", True),
]
ID_KEYS = [
    ("id_idf1", "IDF1", False),
    ("id_mota", "MOTA", False),
    ("id_num_switches", "ID 스왑", True),
    ("id_num_fragmentations", "트랙 단편화", True),
    ("count_mae", "인원수 MAE", True),
    ("occ_retention_rate", "가림 후 ID 유지", False),
    ("n_occlusion_events", "가림 이벤트", None),
]
INFO_KEYS = [
    ("n_tracks", "검출 트랙", None),
    ("ms_per_frame", "ms/frame", True),
    ("gt_limb_cv", "GT 뼈길이 CV", None),
    ("median_hand_px", "손 픽셀", None),
    ("ratio", "손/사람 비", None),
]


def aggregate(results_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(p, encoding="utf-8") as f:
            rows.append(json.load(f))
    return rows


def worst_k(rows, key, k=10, largest=True):
    have = [r for r in rows if _num(r.get(key)) is not None]
    return sorted(have, key=lambda r: r[key], reverse=largest)[:k]


def _num(v):
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return None if v != v else float(v)     # NaN 제외
    return None


def _mean(rows, key):
    vals = [_num(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _fmt(v):
    if v is None:
        return "—"
    return f"{v:.0f}" if abs(v) >= 100 else f"{v:.3f}"


def _table(by_ds, keys, best_marks=True):
    models = sorted({m for d in by_ds.values() for m in d})
    out = ["<table><thead><tr><th>데이터</th><th>모델</th>"
           + "".join(f"<th>{_html.escape(n)}</th>" for _k, n, _l in keys)
           + "</tr></thead><tbody>"]
    for ds in sorted(by_ds):
        # 데이터셋 안에서 각 지표의 승자를 표시한다
        best = {}
        if best_marks and len(by_ds[ds]) > 1:
            for k, _n, lower in keys:
                if lower is None:
                    continue
                vals = {m: _mean(by_ds[ds][m], k) for m in by_ds[ds]}
                vals = {m: v for m, v in vals.items() if v is not None}
                if len(vals) > 1:
                    best[k] = (min if lower else max)(vals, key=vals.get)
        for i, model in enumerate(sorted(by_ds[ds])):
            rs = by_ds[ds][model]
            cells = []
            for k, _n, _l in keys:
                v = _mean(rs, k)
                win = " class='win'" if best.get(k) == model else ""
                cells.append(f"<td{win}>{_fmt(v)}</td>")
            first = (f"<td rowspan='{len(by_ds[ds])}'>{_html.escape(ds)}"
                     f"<br><small>{len(rs)}개 영상</small></td>" if i == 0 else "")
            out.append(f"<tr>{first}<td>{_html.escape(model)}</td>"
                       + "".join(cells) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--muted:#666;--line:#ddd;--win:#e8f5e9;--warn:#fff4e5}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
  --bg:#141414;--fg:#e8e8e8;--muted:#999;--line:#333;--win:#1b3a1f;--warn:#3a2f1b}}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);margin:0;padding:2rem 1.5rem;
  font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  max-width:1200px;margin-inline:auto}
h1{font-size:1.6rem;margin:0 0 .3rem} h2{font-size:1.15rem;margin:2.2rem 0 .6rem}
.sub{color:var(--muted);margin:0 0 1.5rem}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:.5rem 0}
th,td{border:1px solid var(--line);padding:.4rem .6rem;text-align:right;white-space:nowrap}
th{background:color-mix(in srgb,var(--fg) 6%,transparent);font-weight:600}
th:first-child,td:first-child,td:nth-child(2){text-align:left}
.win{background:var(--win);font-weight:600}
.note{background:var(--warn);border-left:3px solid #e8a33d;padding:.8rem 1rem;
  border-radius:4px;margin:1rem 0;font-size:14px}
small{color:var(--muted)}
"""


def build_html(rows, out_path, title="CoMotion vs Multi-HMR 2"):
    by_ds = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_ds[r.get("dataset", "?")][r.get("model", "?")].append(r)

    gt_cvs = [_num(r.get("gt_limb_cv")) for r in rows]
    gt_cvs = [v for v in gt_cvs if v is not None]
    gt_note = (f"GT 자체의 뼈길이 변동계수는 {min(gt_cvs):.3f}~{max(gt_cvs):.3f} 입니다. "
               "삼각측량 GT 가 이미 이만큼의 골격 길이 변동을 갖고 있으므로, "
               "모델 오차를 해석할 때 이 값을 하한선으로 함께 보아야 합니다."
               if gt_cvs else "GT 노이즈 정보 없음.")

    p = [f"<title>{_html.escape(title)}</title>", f"<style>{CSS}</style>",
         f"<h1>{_html.escape(title)}</h1>",
         f"<p class='sub'>{len(rows)}개 결과 · "
         f"{len(by_ds)}개 데이터셋 · 초록 배경 = 해당 지표 우세</p>",
         f"<div class='note'>{_html.escape(gt_note)}<br>"
         "PA-MPJPE 는 표준 인체(544.6mm) 기준으로 정규화한 값입니다 — "
         "GT 가 SfM 재구성이라 데이터셋마다 스케일이 다르기 때문입니다. "
         "MPJPE 는 GT 규약(COCO)에 골반이 없어 산출하지 않습니다.</div>",
         "<h2>① 포즈 정확도</h2>", "<div class='scroll'>",
         _table(by_ds, POSE_KEYS), "</div>",
         "<h2>② ID 유지</h2>", "<div class='scroll'>",
         _table(by_ds, ID_KEYS), "</div>",
         "<h2>실행 정보</h2>", "<div class='scroll'>",
         _table(by_ds, INFO_KEYS), "</div>"]

    for key, label in (("pa_mpjpe", "PA-MPJPE 최악 10"),
                       ("id_num_switches", "ID 스왑 최다 10")):
        w = worst_k(rows, key, 10)
        if not w:
            continue
        p += [f"<h2>{label}</h2>", "<div class='scroll'><table><thead><tr>"
              "<th>데이터</th><th>카메라</th><th>모델</th>"
              f"<th>{_html.escape(key)}</th></tr></thead><tbody>"]
        for r in w:
            p.append(f"<tr><td>{_html.escape(str(r.get('dataset')))}</td>"
                     f"<td>{_html.escape(str(r.get('cam')))}</td>"
                     f"<td>{_html.escape(str(r.get('model')))}</td>"
                     f"<td>{_fmt(_num(r.get(key)))}</td></tr>")
        p.append("</tbody></table></div>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(p))
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="smpl_eval/results")
    ap.add_argument("--out", default="smpl_eval/report.html")
    a = ap.parse_args(argv)
    rows = aggregate(a.results)
    if not rows:
        print(f"{a.results}/ 에 결과가 없습니다")
        return 1
    print(f"wrote {build_html(rows, a.out)}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
