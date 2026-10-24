"""Tests for src/clean_splits.py, using a small fake Roboflow-style dataset."""
import pytest

from clean_splits import frame_id, list_images, overlaps, remove_leakage


def make(root, split, frame, copies, tag):
    """Create `copies` augmented images (+ labels + a .npy cache file) for a frame."""
    (root / split / "images").mkdir(parents=True, exist_ok=True)
    (root / split / "labels").mkdir(parents=True, exist_ok=True)
    for i in range(copies):
        stem = f"{frame}_jpg.rf.{tag}{i}"
        (root / split / "images" / f"{stem}.jpg").write_bytes(b"x")
        (root / split / "images" / f"{stem}.npy").write_bytes(b"x")
        (root / split / "labels" / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1")


@pytest.fixture
def dataset(tmp_path):
    make(tmp_path, "train", "f1", 3, "a")   # shared with valid
    make(tmp_path, "train", "f2", 3, "a")   # shared with test
    make(tmp_path, "train", "f4", 3, "a")   # train only
    make(tmp_path, "valid", "f1", 1, "v")
    make(tmp_path, "valid", "f9", 1, "v")   # shared with test
    make(tmp_path, "test", "f2", 1, "t")
    make(tmp_path, "test", "f9", 1, "t")
    make(tmp_path, "test", "f8", 1, "t")    # clean
    return tmp_path


def test_frame_id_strips_roboflow_suffix():
    assert frame_id("frame_with_red_95_jpg.rf.abc123.jpg") == "frame_with_red_95_jpg"


def test_list_images_ignores_cache_files(dataset):
    assert len(list_images(dataset, "train")) == 9   # 9 jpgs, .npy ignored


def test_overlaps_finds_shared_frames(dataset):
    ov = overlaps(dataset)
    assert ov["train&valid"] == {"f1_jpg"}
    assert ov["train&test"] == {"f2_jpg"}
    assert ov["valid&test"] == {"f9_jpg"}


def test_remove_leakage_removes_expected_files(dataset):
    removed = remove_leakage(dataset)
    assert removed == {"train": 6, "test": 1}      # 2 frames x 3 copies, 1 test frame
    assert all(len(v) == 0 for v in overlaps(dataset).values())


def test_remove_leakage_deletes_matching_labels(dataset):
    remove_leakage(dataset)
    left = {p.stem for p in (dataset / "train" / "labels").glob("*.txt")}
    assert all(not s.startswith(("f1", "f2")) for s in left)


def test_remove_leakage_keeps_valid_untouched(dataset):
    remove_leakage(dataset)
    assert len(list_images(dataset, "valid")) == 2


def test_remove_leakage_is_idempotent(dataset):
    remove_leakage(dataset)
    assert remove_leakage(dataset) == {"train": 0, "test": 0}
