"""Explicit asset paths; all callers may select their own data and outputs."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def require_within(path, root):
    path, root = Path(path).expanduser().resolve(), Path(root).expanduser().resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"asset path escapes its declared root: {path}")
    return path
