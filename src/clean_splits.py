"""Detect and remove data leakage between dataset splits.

Roboflow names augmented copies ``<original>.rf.<hash>.jpg``, so the text
before ``.rf.`` identifies the original frame. If the same original frame
appears in two splits, test scores are inflated.
"""
from pathlib import Path

SPLITS = ("train", "valid", "test")
IMG_EXT = {".jpg", ".jpeg", ".png"}


def list_images(root, split):
    """List image files in one split (ignores cache files such as .npy).

    Args:
        root: Dataset root containing train/, valid/, test/ folders.
        split: One of "train", "valid", "test".

    Returns:
        List of image paths.
    """
    folder = Path(root) / split / "images"
    return [p for p in folder.glob("*") if p.suffix.lower() in IMG_EXT]


def frame_id(path):
    """Return the original-frame id of an augmented image path."""
    return Path(path).name.split(".rf.")[0]


def overlaps(root="/content/data"):
    """Find original frames shared between splits.

    Args:
        root: Dataset root.

    Returns:
        Dict mapping "train&valid", "train&test", "valid&test" to sets of
        shared frame ids.
    """
    ids = {s: {frame_id(p) for p in list_images(root, s)} for s in SPLITS}
    return {
        "train&valid": ids["train"] & ids["valid"],
        "train&test": ids["train"] & ids["test"],
        "valid&test": ids["valid"] & ids["test"],
    }


def remove_frames(root, split, frames):
    """Delete every image (and its label file) in a split whose frame is in `frames`.

    Args:
        root: Dataset root.
        split: Split to delete from.
        frames: Iterable of original-frame ids.

    Returns:
        Number of images removed.
    """
    frames = set(frames)
    removed = 0
    for img in list_images(root, split):
        if frame_id(img) in frames:
            lab = Path(root) / split / "labels" / (img.stem + ".txt")
            img.unlink()
            if lab.exists():
                lab.unlink()
            removed += 1
    return removed


def remove_leakage(root="/content/data"):
    """Remove leaked frames so the three splits share no original frame.

    Train copies of any frame also in valid/test are deleted (train is the
    split that can be shrunk cheaply). A frame shared by valid and test is
    deleted from test, which is only used for final evaluation. Safe to
    rerun: a second call removes nothing.

    Args:
        root: Dataset root.

    Returns:
        Dict with the number of images removed from "train" and "test".
    """
    ov = overlaps(root)
    leak_train = ov["train&valid"] | ov["train&test"]
    leak_test = ov["valid&test"]
    return {
        "train": remove_frames(root, "train", leak_train),
        "test": remove_frames(root, "test", leak_test),
    }
