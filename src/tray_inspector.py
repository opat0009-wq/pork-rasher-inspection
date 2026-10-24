"""Tray-level PASS/FAIL logic on top of a YOLO detector.

A tray FAILS if the detector finds at least one defect with confidence at or
above the threshold; otherwise it PASSES.
"""
import math
from pathlib import Path

import cv2

IMG_EXT = {".jpg", ".jpeg", ".png"}


def tray_verdict(confidences, threshold):
    """Return "FAIL" if any confidence >= threshold, else "PASS".

    Args:
        confidences: Iterable of detection confidences (0 to 1).
        threshold: Confidence threshold (0 to 1).

    Returns:
        "PASS" or "FAIL".
    """
    return "FAIL" if any(c >= threshold for c in confidences) else "PASS"


def wilson_interval(successes, n, z=1.96):
    """Wilson score confidence interval for a proportion (95% by default).

    Args:
        successes: Number of successes.
        n: Number of trials.
        z: Normal quantile (1.96 gives 95%).

    Returns:
        (low, high), or (nan, nan) if n is 0.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def check_image(image_path):
    """Validate an input image path.

    Args:
        image_path: Path to a JPG/PNG image.

    Returns:
        The path as a Path object.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the extension is unsupported or the file is unreadable.
    """
    p = Path(image_path)
    if not p.is_file():
        raise FileNotFoundError(f"No such file: {p}")
    if p.suffix.lower() not in IMG_EXT:
        raise ValueError(f"Unsupported image type: {p.suffix}")
    if cv2.imread(str(p)) is None:
        raise ValueError(f"Unreadable image: {p}")
    return p


def inspect_tray(image_path, model, threshold=0.25, imgsz=640):
    """Detect defects in one tray image and return a PASS/FAIL verdict.

    Args:
        image_path: Path to a JPG/PNG tray image.
        model: A loaded ultralytics YOLO model.
        threshold: Confidence threshold in (0, 1].
        imgsz: Inference image size.

    Returns:
        Dict with "verdict" and "detections" (list of dicts with "class",
        "confidence", "box_xyxy").

    Raises:
        FileNotFoundError, ValueError: see check_image; also if threshold
            is outside (0, 1].
    """
    if not 0 < threshold <= 1:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    p = check_image(image_path)
    res = model.predict(str(p), conf=threshold, imgsz=imgsz, verbose=False)[0]
    dets = [{"class": res.names[int(b.cls)],
             "confidence": float(b.conf),
             "box_xyxy": [float(v) for v in b.xyxy[0]]} for b in res.boxes]
    return {"verdict": tray_verdict([d["confidence"] for d in dets], threshold),
            "detections": dets}
