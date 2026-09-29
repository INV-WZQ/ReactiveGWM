"""Small JSON and path helpers."""

import json
import os
from pathlib import Path
import tempfile


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=".write-", delete=False, encoding="utf-8"
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def resolve_path(value, base):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(base) / path).resolve()


def relative_path(path, base):
    return os.path.relpath(Path(path).resolve(), Path(base).resolve())
