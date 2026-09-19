"""Đọc/ghi JSON và YAML.

Toàn bộ thực nghiệm được tham số hóa qua `configs/*.yaml` (mục 2.4 báo cáo
01/09) — đổi backbone hay đổi giải đấu chỉ sửa cấu hình, không sửa mã nguồn.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIGS_DIR, ensure_dir


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Nạp một tệp YAML. Chấp nhận tên ngắn, tự tìm trong configs/."""
    path = Path(path)
    if not path.exists() and not path.is_absolute():
        candidate = CONFIGS_DIR / path
        if candidate.exists():
            path = candidate
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
