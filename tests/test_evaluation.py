"""Tests for src/evaluation.py, using hand-computed values."""
import math

import pytest

from evaluation import (choose_threshold, count_boxes, edge_share, gt_boxes, has_defect, iou,
                        is_located, near_edge, threshold_sweep, tray_metrics, yolo_to_xyxy)


# ---- iou
def test_iou_identical_boxes_is_one():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_iou_half_overlap_is_one_third():
    # intersection 5 x 10 = 50, union 100 + 100 - 50 = 150
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)


def test_iou_disjoint_boxes_is_zero():
    assert iou([0, 0, 1, 1], [2, 2, 3, 3]) == 0.0


def test_iou_touching_edges_is_zero():
    assert iou([0, 0, 10, 10], [10, 0, 20, 10]) == 0.0


def test_iou_contained_box_is_area_ratio():
    assert iou([0, 0, 10, 10], [2, 2, 4, 4]) == pytest.approx(4 / 100)


def test_iou_is_symmetric():
    a, b = [0, 0, 8, 6], [3, 2, 12, 9]
    assert iou(a, b) == pytest.approx(iou(b, a))


def test_iou_zero_area_boxes_return_zero_not_error():
    assert iou([1, 1, 1, 1], [1, 1, 1, 1]) == 0.0


# ---- yolo_to_xyxy and gt_boxes
def test_yolo_to_xyxy_known_values():
    # centre (0.5, 0.5), 0.2 x 0.4 of a 100 x 200 image
    assert yolo_to_xyxy(0.5, 0.5, 0.2, 0.4, 100, 200) == pytest.approx([40, 60, 60, 140])


def test_gt_boxes_reads_class_and_pixels_and_skips_blank_lines(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("3 0.5 0.5 0.2 0.4\n\n   \n1 0.25 0.25 0.5 0.5\n")
    boxes = gt_boxes(p, 100, 200)
    assert [c for c, _ in boxes] == [3, 1]
    assert boxes[0][1] == pytest.approx([40, 60, 60, 140])
    assert boxes[1][1] == pytest.approx([0, 0, 50, 100])


def test_gt_boxes_missing_file_means_no_boxes(tmp_path):
    assert gt_boxes(tmp_path / "missing.txt", 100, 100) == []


def test_gt_boxes_malformed_line_raises(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("1 0.5 0.5\n")
    with pytest.raises(ValueError):
        gt_boxes(p, 100, 100)


# ---- has_defect and count_boxes
def test_has_defect_true_when_file_has_a_box(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0 0.5 0.5 0.1 0.1\n")
    assert has_defect(p) is True


@pytest.mark.parametrize("content", ["", "\n", "   \n  "])
def test_has_defect_false_for_empty_or_blank_file(tmp_path, content):
    p = tmp_path / "a.txt"
    p.write_text(content)
    assert has_defect(p) is False


def test_has_defect_false_for_missing_file(tmp_path):
    assert has_defect(tmp_path / "missing.txt") is False


def test_count_boxes_counts_only_non_blank_lines(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0 .5 .5 .1 .1\n\n1 .2 .2 .1 .1\n   \n2 .8 .8 .1 .1")
    assert count_boxes(p) == 3
    assert count_boxes(tmp_path / "missing.txt") == 0


# ---- tray_metrics (5 trays, 3 defective)
CONF = [0.9, 0.6, 0.2, 0.8, 0.1]
DEFECT = [True, True, True, False, False]


def test_tray_metrics_counts_and_rates_at_half():
    # rejected: trays 0, 1, 3. TP=2, FN=1, FP=1, TN=1
    m = tray_metrics(CONF, DEFECT, 0.5)
    assert (m["TP"], m["FN"], m["FP"], m["TN"]) == (2, 1, 1, 1)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["false_alarm"] == pytest.approx(1 / 2)


def test_tray_metrics_confidence_equal_to_threshold_is_rejected():
    assert tray_metrics([0.5], [True], 0.5)["TP"] == 1


def test_tray_metrics_threshold_is_rounded_to_two_decimals():
    assert tray_metrics(CONF, DEFECT, 0.15000000000000002)["threshold"] == 0.15


def test_tray_metrics_counts_always_sum_to_number_of_trays():
    m = tray_metrics(CONF, DEFECT, 0.3)
    assert m["TP"] + m["FN"] + m["FP"] + m["TN"] == len(CONF)


def test_tray_metrics_no_good_trays_gives_nan_false_alarm():
    m = tray_metrics([0.9, 0.2], [True, True], 0.5)
    assert m["recall"] == pytest.approx(0.5) and math.isnan(m["false_alarm"])


def test_tray_metrics_accepts_numpy_and_pandas_inputs():
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    a = tray_metrics(np.array(CONF), np.array(DEFECT), 0.5)
    b = tray_metrics(pd.Series(CONF), pd.Series(DEFECT), 0.5)
    assert a == b == tray_metrics(CONF, DEFECT, 0.5)


def test_tray_metrics_length_mismatch_raises():
    with pytest.raises(ValueError):
        tray_metrics([0.5, 0.6], [True], 0.5)


# ---- threshold_sweep and choose_threshold
def test_threshold_sweep_default_grid_and_j():
    sweep = threshold_sweep(CONF, DEFECT)
    assert len(sweep) == 19
    assert [r["threshold"] for r in sweep][:3] == [0.05, 0.1, 0.15]
    row = next(r for r in sweep if r["threshold"] == 0.5)
    assert row["J"] == pytest.approx(2 / 3 - 1 / 2)


def test_threshold_sweep_uses_given_thresholds():
    sweep = threshold_sweep(CONF, DEFECT, thresholds=[0.3, 0.7])
    assert [r["threshold"] for r in sweep] == [0.3, 0.7]


def test_choose_threshold_picks_highest_j():
    sweep = [{"threshold": 0.2, "J": 0.1}, {"threshold": 0.4, "J": 0.6}, {"threshold": 0.6, "J": 0.3}]
    assert choose_threshold(sweep) == 0.4


def test_choose_threshold_tie_goes_to_lowest_threshold():
    sweep = [{"threshold": 0.10, "J": 0.5}, {"threshold": 0.35, "J": 0.5}, {"threshold": 0.5, "J": 0.2}]
    assert choose_threshold(sweep) == 0.10
    assert choose_threshold(list(reversed(sweep))) == 0.10


def test_choose_threshold_ignores_nan_rows():
    sweep = [{"threshold": 0.2, "J": float("nan")}, {"threshold": 0.4, "J": 0.1}]
    assert choose_threshold(sweep) == 0.4


def test_choose_threshold_all_nan_raises():
    with pytest.raises(ValueError):
        choose_threshold([{"threshold": 0.2, "J": float("nan")}])


def test_choose_threshold_end_to_end_on_separable_data():
    # defective trays all score >= 0.7, good trays all <= 0.3: any threshold in (0.3, 0.7] is perfect
    conf = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    truth = [True, True, True, False, False, False]
    assert choose_threshold(threshold_sweep(conf, truth)) == 0.35


# ---- is_located
def test_is_located_depends_on_iou_threshold():
    gts = [(0, [0, 0, 10, 10])]
    pred = [[0, 0, 10, 6]]        # IoU = 60 / 100 = 0.6
    assert is_located(pred, gts, 0.5) is True
    assert is_located(pred, gts, 0.7) is False


def test_is_located_false_without_predictions_or_ground_truth():
    assert is_located([], [(0, [0, 0, 1, 1])], 0.3) is False
    assert is_located([[0, 0, 1, 1]], [], 0.3) is False


def test_is_located_any_pair_is_enough():
    gts = [(0, [0, 0, 10, 10]), (1, [50, 50, 60, 60])]
    assert is_located([[100, 100, 110, 110], [50, 50, 60, 60]], gts, 0.5) is True


# ---- near_edge and edge_share (exact binary fractions avoid rounding at the boundary)
def test_near_edge_centred_box_is_not_near_edge():
    assert near_edge(["0 0.5 0.5 0.1 0.1"]) == [False]


def test_near_edge_box_reaching_the_border_is_near_edge():
    assert near_edge(["0 0.95 0.5 0.2 0.1"]) == [True]       # right side at 1.05
    assert near_edge(["0 0.5 0.03 0.1 0.04"]) == [True]      # top side at 0.01


def test_near_edge_boundary_is_strict():
    assert near_edge(["0 0.5 0.5 0.5 0.5"], margin=0.25) == [False]   # sides exactly at 0.25 and 0.75
    assert near_edge(["0 0.5 0.5 0.6 0.5"], margin=0.25) == [True]    # left side at 0.2


def test_near_edge_skips_blank_lines_and_raises_on_bad_lines():
    assert near_edge(["", "  ", "0 0.5 0.5 0.1 0.1"]) == [False]
    with pytest.raises(ValueError):
        near_edge(["0 0.5 0.5"])


def test_edge_share_counts_boxes_and_respects_stems(tmp_path):
    (tmp_path / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n0 0.95 0.5 0.2 0.1\n")   # 1 of 2 near edge
    (tmp_path / "b.txt").write_text("0 0.95 0.5 0.2 0.1\n")                       # 1 of 1
    (tmp_path / "empty.txt").write_text("")
    assert edge_share(tmp_path) == (2, 3)
    assert edge_share(tmp_path, stems={"a"}) == (1, 2)
    assert edge_share(tmp_path, stems={"nope"}) == (0, 0)
