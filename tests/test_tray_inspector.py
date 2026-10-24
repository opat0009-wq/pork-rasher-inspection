"""Tests for src/tray_inspector.py. A fake model stands in for YOLO."""
import math

import cv2
import numpy as np
import pytest

from tray_inspector import check_image, inspect_tray, tray_verdict, wilson_interval


class FakeBox:
    def __init__(self, cls, conf):
        self.cls, self.conf, self.xyxy = cls, conf, [[1.0, 2.0, 3.0, 4.0]]


class FakeResult:
    names = {0: "unsealed", 1: "wrinkle"}

    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    """Returns the given boxes, but only those at or above the requested conf."""

    def __init__(self, boxes):
        self._boxes = boxes

    def predict(self, path, conf, imgsz, verbose):
        return [FakeResult([b for b in self._boxes if b.conf >= conf])]


@pytest.fixture
def good_image(tmp_path):
    p = tmp_path / "tray.jpg"
    cv2.imwrite(str(p), np.zeros((20, 20, 3), np.uint8))
    return p


# ---- tray_verdict
def test_verdict_fail_when_any_detection_above_threshold():
    assert tray_verdict([0.1, 0.6], 0.5) == "FAIL"


def test_verdict_pass_when_all_below_threshold():
    assert tray_verdict([0.1, 0.4], 0.5) == "PASS"


def test_verdict_pass_when_no_detections():
    assert tray_verdict([], 0.5) == "PASS"


def test_verdict_boundary_counts_as_fail():
    assert tray_verdict([0.5], 0.5) == "FAIL"


# ---- wilson_interval
def test_wilson_all_successes_upper_bound_is_one():
    low, high = wilson_interval(12, 12)
    assert high == 1.0 and 0.7 < low < 0.8


def test_wilson_zero_successes_lower_bound_is_zero():
    low, high = wilson_interval(0, 12)
    assert low == 0.0 and 0.2 < high < 0.3


def test_wilson_is_symmetric_around_half():
    low, high = wilson_interval(5, 10)
    assert low + high == pytest.approx(1.0)


def test_wilson_zero_trials_returns_nan():
    low, high = wilson_interval(0, 0)
    assert math.isnan(low) and math.isnan(high)


# ---- check_image (invalid / missing / unsupported input handling)
def test_check_image_accepts_valid_image(good_image):
    assert check_image(good_image) == good_image


def test_check_image_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        check_image(tmp_path / "nope.jpg")


def test_check_image_unsupported_extension_raises(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("hello")
    with pytest.raises(ValueError):
        check_image(p)


def test_check_image_corrupt_file_raises(tmp_path):
    p = tmp_path / "broken.jpg"
    p.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        check_image(p)


# ---- inspect_tray
def test_inspect_tray_fails_tray_with_confident_defect(good_image):
    out = inspect_tray(good_image, FakeModel([FakeBox(0, 0.9)]), threshold=0.35)
    assert out["verdict"] == "FAIL"
    assert out["detections"][0]["class"] == "unsealed"


def test_inspect_tray_passes_when_only_weak_detections(good_image):
    out = inspect_tray(good_image, FakeModel([FakeBox(1, 0.2)]), threshold=0.35)
    assert out["verdict"] == "PASS" and out["detections"] == []


@pytest.mark.parametrize("bad", [0, -0.1, 1.5])
def test_inspect_tray_rejects_bad_threshold(good_image, bad):
    with pytest.raises(ValueError):
        inspect_tray(good_image, FakeModel([]), threshold=bad)


def test_inspect_tray_missing_image_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        inspect_tray(tmp_path / "missing.jpg", FakeModel([]))
