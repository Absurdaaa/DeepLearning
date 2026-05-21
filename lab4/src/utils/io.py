"""输出文件写出工具占位。"""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """确保目录存在。"""

    path.mkdir(parents=True, exist_ok=True)
    return path
