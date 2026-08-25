"""evaluate.py 통합 테스트.

GT 를 예측인 것처럼 넣으면 PA-MPJPE 가 0, IDF1 이 1 이어야 한다.
GT 파싱 → 규약 매핑 → 프레임 정렬 → 지표까지 전 경로를 한 번에 검증한다.
"""
import json
import os
import tempfile

import numpy as np
import pytest

from smpl_eval.evaluate import evaluate_one, load_gt
from smpl_eval.schema import save_tracks

MANIFEST = "smpl_eval/manifest.json"
pytestmark = pytest.mark.skipif(not os.path.isfile(MANIFEST), reason="manifest 없음")


def _recs():
    return json.load(open(MANIFEST))


def _rec(dataset, cam=None):
    for r in _recs():
        if r["dataset"] == dataset and (cam is None or r["cam"] == cam):
            return r
    raise AssertionError(f"{dataset}/{cam} 없음")


def _gt_as_prediction(rec, tmpdir, model="fake"):
    """GT 를 tracks.npz 로 저장해 '완벽한 예측' 을 만든다."""
    gt, _ = load_gt(rec)
    assert gt is not None
    p = os.path.join(tmpdir, "tracks.npz")
    save_tracks(p, gt, {"model": model, "body_model": "smpl",
                        "fps": rec["fps"], "runtime_sec": 1.0})
    return p


@pytest.mark.parametrize("dataset", ["Data1", "Data2", "Data3", "Data4"])
def test_gt_against_itself_is_perfect(dataset):
    rec = _rec(dataset)
    with tempfile.TemporaryDirectory() as d:
        res = evaluate_one(_gt_as_prediction(rec, d), rec)
    assert res["gt_available"] is True
    assert res["n_matched"] > 0, res
    assert res["pa_mpjpe"] < 1.0, f"{dataset}: {res['pa_mpjpe']}"
    assert res["id_idf1"] > 0.99, f"{dataset}: {res['id_idf1']}"
    assert res["id_num_switches"] == 0


@pytest.mark.parametrize("dataset", ["Data1", "Data2", "Data3", "Data4"])
def test_mpjpe_unavailable_for_all_coco_datasets(dataset):
    """모든 GT 가 COCO 계열이라 골반이 없다 — MPJPE 를 내면 안 된다."""
    rec = _rec(dataset)
    with tempfile.TemporaryDirectory() as d:
        res = evaluate_one(_gt_as_prediction(rec, d), rec)
    assert res["mpjpe_available"] is False
    assert np.isnan(res["mpjpe"])


def test_gt_noise_floor_is_reported():
    """GT 자체 뼈길이 CV 를 기록해야 모델 정확도를 해석할 수 있다."""
    rec = _rec("Data2")
    with tempfile.TemporaryDirectory() as d:
        res = evaluate_one(_gt_as_prediction(rec, d), rec)
    assert 0.0 < res["gt_limb_cv"] < 1.0


def test_perturbed_prediction_raises_error():
    """예측을 흔들면 PA-MPJPE 가 실제로 올라가야 한다."""
    rec = _rec("Data3")
    gt, _ = load_gt(rec)
    noisy = {k: v.copy() for k, v in gt.items()}
    rng = np.random.default_rng(0)
    scale = float(np.nanstd(gt["joints3d"]))
    noisy["joints3d"] = (noisy["joints3d"]
                         + rng.normal(0, 0.05 * scale, noisy["joints3d"].shape)
                         ).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        clean_p = _gt_as_prediction(rec, d)
        clean = evaluate_one(clean_p, rec)
        noisy_p = os.path.join(d, "noisy.npz")
        save_tracks(noisy_p, noisy, {"model": "fake", "fps": rec["fps"]})
        dirty = evaluate_one(noisy_p, rec)
    assert dirty["pa_mpjpe"] > clean["pa_mpjpe"] + 5.0, (clean["pa_mpjpe"],
                                                        dirty["pa_mpjpe"])


def test_id_swap_in_prediction_is_caught_end_to_end():
    """실 GT 위에서 ID 를 뒤바꾸면 IDF1 이 떨어져야 한다."""
    rec = _rec("Data4")
    gt, _ = load_gt(rec)
    swapped = {k: v.copy() for k, v in gt.items()}
    mid = int(np.median(gt["frame_ids"]))
    ids = swapped["track_ids"].copy()
    a, b = sorted(np.unique(ids))[:2]
    sel = swapped["frame_ids"] >= mid
    ids[sel & (swapped["track_ids"] == a)] = b
    ids[sel & (swapped["track_ids"] == b)] = a
    swapped["track_ids"] = ids
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "swap.npz")
        save_tracks(p, swapped, {"model": "fake", "fps": rec["fps"]})
        res = evaluate_one(p, rec)
    assert res["id_idf1"] < 0.9, res["id_idf1"]
    assert res["id_num_switches"] >= 2


def test_gt_and_prediction_bbox_use_same_construction():
    """예측과 GT 의 bbox 생성 방식이 같아야 IoU 매칭이 공정하다.

    CoMotion 이 모델의 MOT bbox(여백 포함)를 쓰고 GT 는 관절에서 유도할 때
    IoU 가 0.41~0.48 로 임계값 0.5 아래에 걸려 ID 지표가 통째로 무너졌다.
    추적 품질이 아니라 bbox 관례 차이로 점수가 갈리면 안 된다.
    """
    from smpl_eval.runners.base import bbox_from_joints2d

    rec = _rec("Data3")
    gt, _ = load_gt(rec)
    derived = bbox_from_joints2d(gt["joints2d"])
    np.testing.assert_allclose(gt["bbox"], derived, rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize("dataset", ["Data1", "Data2", "Data3", "Data4"])
def test_gt_self_match_iou_is_high(dataset):
    """GT 를 예측으로 넣으면 IoU 가 1.0 이어야 한다 (매칭 경로 정상성)."""
    from smpl_eval.metrics.geometry import iou_matrix

    rec = _rec(dataset)
    gt, _ = load_gt(rec)
    f = int(np.median(np.unique(gt["frame_ids"])))
    sel = gt["frame_ids"] == f
    if sel.sum() == 0:
        pytest.skip("프레임 없음")
    m = iou_matrix(gt["bbox"][sel], gt["bbox"][sel])
    assert np.allclose(np.diag(m), 1.0, atol=1e-6)
