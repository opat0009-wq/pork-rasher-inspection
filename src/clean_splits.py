"""Remove train frames that also appear in valid/test (data leakage)."""
from pathlib import Path

def clean_splits(root="/content/data"):
    """Delete augmented train copies of any frame present in valid or test.

    Args:
        root: Dataset root containing train/, valid/, test/ folders.

    Returns:
        Number of train images removed.
    """
    root = Path(root)
    base = lambda p: p.name.split(".rf.")[0]
    sets = {s: {base(p) for p in (root/s/"images").glob("*")}
            for s in ["train", "valid", "test"]}
    leak = (sets["train"] & sets["test"]) | (sets["train"] & sets["valid"])
    removed = 0
    for b in leak:
        for img in (root/"train"/"images").glob(b + ".rf.*"):
            lab = root/"train"/"labels"/(img.stem + ".txt")
            img.unlink()
            if lab.exists():
                lab.unlink()
            removed += 1
    return removed
