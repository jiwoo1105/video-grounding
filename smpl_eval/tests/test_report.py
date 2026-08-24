import json
import os
import tempfile

from smpl_eval.report.build import aggregate, worst_k, build_html


def _row(model, dataset, cam, **kw):
    r = {"model": model, "dataset": dataset, "cam": cam}
    r.update(kw)
    return r


def _write(d, rows):
    for i, r in enumerate(rows):
        with open(os.path.join(d, f"{r['model']}__{r['dataset']}__{i}.json"),
                  "w", encoding="utf-8") as f:
            json.dump(r, f)


def test_aggregate_reads_all_json():
    with tempfile.TemporaryDirectory() as d:
        _write(d, [_row("comotion", "Data2", "c1", pa_mpjpe=80.0),
                   _row("multihmr2", "Data2", "c1", pa_mpjpe=70.0)])
        rows = aggregate(d)
        assert len(rows) == 2
        assert {r["model"] for r in rows} == {"comotion", "multihmr2"}


def test_worst_k_sorts_descending_and_skips_missing():
    rows = [{"pa_mpjpe": 10.0}, {"pa_mpjpe": 99.0}, {"other": 1},
            {"pa_mpjpe": 50.0}, {"pa_mpjpe": float("nan")}]
    w = worst_k(rows, "pa_mpjpe", k=2)
    assert [x["pa_mpjpe"] for x in w] == [99.0, 50.0]


def test_worst_k_handles_none_and_bool():
    rows = [{"x": None}, {"x": True}, {"x": 5.0}]
    assert [r["x"] for r in worst_k(rows, "x", k=5)] == [5.0]


def test_build_html_contains_both_models_and_datasets():
    rows = [_row("comotion", "Data2", "c", pa_mpjpe=80.0, id_idf1=0.9),
            _row("multihmr2", "Data2", "c", pa_mpjpe=70.0, id_idf1=0.8),
            _row("comotion", "Data4", "c", pa_mpjpe=60.0, id_idf1=0.7)]
    with tempfile.TemporaryDirectory() as d:
        p = build_html(rows, os.path.join(d, "r.html"))
        html = open(p, encoding="utf-8").read()
    for token in ("comotion", "multihmr2", "Data2", "Data4",
                  "PA-MPJPE", "IDF1", "prefers-color-scheme"):
        assert token in html, token


def test_winner_is_marked_on_the_better_model():
    """PA-MPJPE 는 낮을수록, IDF1 은 높을수록 승자여야 한다."""
    rows = [_row("comotion", "Data2", "c", pa_mpjpe=80.0, id_idf1=0.95),
            _row("multihmr2", "Data2", "c", pa_mpjpe=70.0, id_idf1=0.60)]
    with tempfile.TemporaryDirectory() as d:
        html = open(build_html(rows, os.path.join(d, "r.html")),
                    encoding="utf-8").read()
    # 각 모델 행에서 win 클래스가 붙은 셀 수를 센다
    import re
    body = html.split("① 포즈 정확도")[1].split("② ID 유지")[0]
    multi_row = [ln for ln in body.split("<tr>") if "multihmr2" in ln][0]
    assert "class='win'" in multi_row      # PA-MPJPE 70 < 80 이므로 승
    id_body = html.split("② ID 유지")[1].split("실행 정보")[0]
    como_row = [ln for ln in id_body.split("<tr>") if "comotion" in ln][0]
    assert "class='win'" in como_row       # IDF1 0.95 > 0.60 이므로 승


def test_gt_noise_floor_appears_in_report():
    rows = [_row("comotion", "Data2", "c", pa_mpjpe=80.0, gt_limb_cv=0.13),
            _row("comotion", "Data4", "c", pa_mpjpe=90.0, gt_limb_cv=0.22)]
    with tempfile.TemporaryDirectory() as d:
        html = open(build_html(rows, os.path.join(d, "r.html")),
                    encoding="utf-8").read()
    assert "0.130~0.220" in html or "0.130" in html


def test_html_has_no_horizontal_body_scroll():
    """넓은 표는 자체 컨테이너 안에서 스크롤돼야 한다."""
    rows = [_row("comotion", "Data2", "c", pa_mpjpe=80.0)]
    with tempfile.TemporaryDirectory() as d:
        html = open(build_html(rows, os.path.join(d, "r.html")),
                    encoding="utf-8").read()
    assert "class='scroll'" in html and "overflow-x:auto" in html
