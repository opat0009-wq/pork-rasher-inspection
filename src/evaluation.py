"""Evaluation helpers for the pork rasher tray-inspection project.

Pure functions (no GPU and no model) that turn detections and label files into
the numbers reported in the project: box overlap (IoU), tray-level confusion
counts, confidence-threshold selection and simple label statistics.

Labels use the YOLO text format: one box per line, ``class x_centre y_centre
width height``, with coordinates normalised to the range 0 to 1. A missing or
empty label file means the image has no labelled defect.
"""
from pathlib import Path

import numpy as np


def iou(a, b):
    """Intersection over union of two axis-aligned boxes.

    Args:
        a: Box as ``[x1, y1, x2, y2]`` with ``x2 >= x1`` and ``y2 >= y1``.
        b: Box in the same format and the same units as ``a``.

    Returns:
        IoU between 0 and 1. Returns 0.0 if the boxes do not overlap or if
        both have zero area.
    """
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def yolo_to_xyxy(x, y, bw, bh, width, height):
    """Convert a normalised YOLO box to pixel corner coordinates.

    Args:
        x: Box centre x, as a fraction of the image width.
        y: Box centre y, as a fraction of the image height.
        bw: Box width, as a fraction of the image width.
        bh: Box height, as a fraction of the image height.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        ``[x1, y1, x2, y2]`` in pixels.
    """
    return [(x - bw / 2) * width, (y - bh / 2) * height,
            (x + bw / 2) * width, (y + bh / 2) * height]


def gt_boxes(label_path, width, height):
    """Read ground-truth boxes from a YOLO label file.

    Args:
        label_path: Path to the ``.txt`` label file. A missing file is treated
            as an image with no labelled boxes.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        List of ``(class_id, [x1, y1, x2, y2])`` with pixel coordinates.
        Blank lines are ignored.

    Raises:
        ValueError: if a non-blank line has fewer than five fields or a field
            is not a number.
    """
    path = Path(label_path)
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text().splitlines():
        if line.strip():
            c, x, y, bw, bh = map(float, line.split()[:5])
            boxes.append((int(c), yolo_to_xyxy(x, y, bw, bh, width, height)))
    return boxes


def has_defect(label_path):
    """Return True if a label file contains at least one box.

    Args:
        label_path: Path to a YOLO label file.

    Returns:
        False for a missing, empty or whitespace-only file, otherwise True.
    """
    path = Path(label_path)
    return path.exists() and path.read_text().strip() != ''


def count_boxes(label_path):
    """Count the boxes in a YOLO label file.

    Args:
        label_path: Path to a YOLO label file. A missing file counts as 0.

    Returns:
        Number of non-blank lines.
    """
    path = Path(label_path)
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def _ratio(numerator, denominator):
    """Divide two counts, returning NaN instead of raising when the denominator is 0."""
    return numerator / denominator if denominator else float('nan')


def tray_metrics(max_conf, has_defect, threshold):
    """Tray-level confusion counts and rates at one confidence threshold.

    A tray is rejected (FAIL) if its highest detection confidence is at or
    above the threshold. A tray is truly defective if its label file has a box.

    Args:
        max_conf: Highest detection confidence per tray (0 if none), as a list,
            array or pandas Series.
        has_defect: Whether each tray is truly defective, same length as
            ``max_conf``.
        threshold: Confidence threshold between 0 and 1.

    Returns:
        Dict with ``threshold`` (rounded to 2 decimals), the counts ``TP``
        (defective and rejected), ``FN`` (defective and passed), ``FP`` (good
        and rejected), ``TN`` (good and passed), ``recall`` = TP / (TP + FN)
        and ``false_alarm`` = FP / (FP + TN). A rate is NaN if its denominator
        is 0, for example when the set has no good trays.

    Raises:
        ValueError: if ``max_conf`` and ``has_defect`` differ in length.
    """
    conf = np.asarray(max_conf, dtype=float)
    truth = np.asarray(has_defect, dtype=bool)
    if conf.shape != truth.shape:
        raise ValueError(f'max_conf and has_defect differ in shape: {conf.shape} vs {truth.shape}')
    fail = conf >= threshold
    tp = int(np.sum(fail & truth))
    fn = int(np.sum(~fail & truth))
    fp = int(np.sum(fail & ~truth))
    tn = int(np.sum(~fail & ~truth))
    return {'threshold': round(float(threshold), 2), 'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn,
            'recall': _ratio(tp, tp + fn), 'false_alarm': _ratio(fp, fp + tn)}


def threshold_sweep(max_conf, has_defect, thresholds=None):
    """Evaluate tray-level metrics over a range of confidence thresholds.

    Args:
        max_conf: Highest detection confidence per tray.
        has_defect: Whether each tray is truly defective.
        thresholds: Thresholds to try. Defaults to 0.05, 0.10, ..., 0.95.

    Returns:
        List of dicts as returned by :func:`tray_metrics`, each with an extra
        key ``J`` = recall - false_alarm (Youden's J).
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)
    rows = []
    for t in thresholds:
        row = tray_metrics(max_conf, has_defect, t)
        row['J'] = row['recall'] - row['false_alarm']
        rows.append(row)
    return rows


def choose_threshold(sweep):
    """Pick the threshold with the highest J from a sweep.

    Ties go to the lowest threshold, which favours catching defects. Rows
    whose J is NaN are ignored.

    Args:
        sweep: List of dicts from :func:`threshold_sweep`.

    Returns:
        The chosen threshold as a float.

    Raises:
        ValueError: if no row has a valid J.
    """
    best = None
    for row in sweep:
        j = row['J']
        if np.isnan(j):
            continue
        if (best is None or j > best['J']
                or (j == best['J'] and row['threshold'] < best['threshold'])):
            best = row
    if best is None:
        raise ValueError('No threshold has a valid J (is one class missing?)')
    return float(best['threshold'])


def is_located(pred_boxes, gts, iou_threshold):
    """Check whether any predicted box overlaps any ground-truth box.

    Class is ignored. Used to test that a tray was rejected because of a
    detection on a real defect, not a detection elsewhere.

    Args:
        pred_boxes: Predicted boxes as ``[x1, y1, x2, y2]`` lists.
        gts: Ground-truth boxes as returned by :func:`gt_boxes`.
        iou_threshold: Minimum IoU for an overlap to count.

    Returns:
        True if at least one prediction has IoU at or above the threshold
        with at least one ground-truth box, otherwise False.
    """
    return any(iou(p, g[1]) >= iou_threshold for p in pred_boxes for g in gts)


def near_edge(lines, margin=0.10):
    """Flag which boxes lie within a margin of an image edge.

    Args:
        lines: YOLO label lines (blank lines are skipped).
        margin: Fraction of the image size treated as the edge band.

    Returns:
        List of booleans, one per non-blank line. True if the box extends
        into the band on any side (strictly less than ``margin`` from an edge).

    Raises:
        ValueError: if a non-blank line has fewer than five fields.
    """
    flags = []
    for line in lines:
        if line.strip():
            _, x, y, bw, bh = map(float, line.split()[:5])
            flags.append(x - bw / 2 < margin or x + bw / 2 > 1 - margin
                         or y - bh / 2 < margin or y + bh / 2 > 1 - margin)
    return flags


def edge_share(labels_dir, stems=None, margin=0.10):
    """Count boxes near an image edge across a folder of label files.

    Args:
        labels_dir: Folder containing YOLO ``.txt`` label files.
        stems: Optional set of file stems to include. All files if None.
        margin: Edge band as a fraction of the image size.

    Returns:
        ``(near_edge_count, total_boxes)``.
    """
    flags = []
    for path in Path(labels_dir).glob('*.txt'):
        if stems is None or path.stem in stems:
            flags += near_edge(path.read_text().splitlines(), margin)
    return sum(flags), len(flags)
