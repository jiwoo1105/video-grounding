import json, os
import pytest
from smpl_eval.run_all import select, PILOT, main

ROOT = "smpl_eval/manifest.json"
pytestmark = pytest.mark.skipif(not os.path.isfile(ROOT), reason="manifest 없음")


def _recs():
    return json.load(open(ROOT))


def test_pilot_selects_exactly_four_videos():
    r = select(_recs(), pilot=True)
    assert len(r) == 4
    assert {x["dataset"] for x in r} == {"Data1", "Data2", "Data3", "Data4"}


def test_pilot_entries_all_exist_in_manifest():
    """PILOT 상수의 오타를 잡는다 — 없는 cam 이름이면 조용히 빠진다."""
    pairs = {(x["dataset"], x["cam"]) for x in _recs()}
    for p in PILOT:
        assert p in pairs, f"manifest 에 없는 파일럿 대상: {p}"


def test_dataset_filter():
    assert len(select(_recs(), dataset="Data1")) == 16


def test_cam_filter_narrows_further():
    r = select(_recs(), dataset="Data2", cam="cam1_2K")
    assert len(r) == 1


def test_dry_run_needs_no_model(capsys):
    assert main(["--model", "comotion", "--pilot", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "대상 4개 영상" in out
