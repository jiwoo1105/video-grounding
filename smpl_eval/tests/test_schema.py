import numpy as np, pytest, tempfile, os
from smpl_eval.schema import save_tracks, load_tracks, validate_tracks, TRACK_FIELDS


def _minimal(n=5):
    return {
        "frame_ids": np.arange(n, dtype=np.int32),
        "track_ids": np.zeros(n, dtype=np.int32),
        "betas": np.zeros((n, 10), np.float32),
        "global_orient": np.zeros((n, 3), np.float32),
        "body_pose": np.zeros((n, 23, 3), np.float32),
        "transl": np.zeros((n, 3), np.float32),
        "joints3d": np.zeros((n, 24, 3), np.float32),
        "joints2d": np.zeros((n, 24, 2), np.float32),
        "bbox": np.zeros((n, 4), np.float32),
        "score": np.ones(n, np.float32),
    }


def test_validate_accepts_minimal():
    validate_tracks(_minimal())


def test_validate_rejects_length_mismatch():
    a = _minimal(); a["score"] = np.ones(4, np.float32)
    with pytest.raises(ValueError, match="length"):
        validate_tracks(a)


def test_validate_rejects_missing_field():
    a = _minimal(); del a["bbox"]
    with pytest.raises(ValueError, match="bbox"):
        validate_tracks(a)


def test_validate_rejects_wrong_shape():
    a = _minimal(); a["betas"] = np.zeros((5, 7), np.float32)
    with pytest.raises(ValueError, match="betas"):
        validate_tracks(a)


def test_roundtrip_preserves_arrays_and_meta():
    a = _minimal(); a["frame_ids"] = np.array([0, 1, 2, 3, 4], np.int32)
    meta = {"model": "comotion", "fps": 29.97, "body_model": "smpl"}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.npz")
        save_tracks(p, a, meta)
        b, m = load_tracks(p)
        assert m["model"] == "comotion" and m["fps"] == 29.97
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])


def test_optional_betas_native_roundtrips():
    a = _minimal(); a["betas_native"] = np.zeros((5, 6), np.float32)
    validate_tracks(a)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.npz")
        save_tracks(p, a, {"model": "multihmr2"})
        b, _ = load_tracks(p)
        assert b["betas_native"].shape == (5, 6)


def test_synth_satisfies_schema():
    from smpl_eval.tests.synth import make_tracks
    validate_tracks(make_tracks(n_frames=10, n_tracks=3))
