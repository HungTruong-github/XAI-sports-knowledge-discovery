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


def save_atomic_json(data: Any, path: str | Path, indent: int = 2) -> Path:
    """Ghi tệp JSON an toàn / nguyên tử (atomic write).

    Ghi vào tệp tạm trước rồi đổi tên nguyên tử để tránh tệp dở dang nếu xảy ra lỗi.
    """
    import os
    import uuid

    dest = Path(path)
    ensure_dir(dest.parent)
    tmp_path = dest.with_name(f".{dest.name}.tmp_{uuid.uuid4().hex[:8]}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, dest)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
    return dest


def save_atomic_csv(
    df: Any, path: str | Path, index: bool = False, encoding: str = "utf-8-sig"
) -> Path:
    """Ghi tệp CSV an toàn / nguyên tử (atomic write) cho pandas DataFrame.

    Tránh trường hợp bảng official bị ghi đè dở dang khi gặp ngoại lệ.
    """
    import os
    import uuid

    dest = Path(path)
    ensure_dir(dest.parent)
    tmp_path = dest.with_name(f".{dest.name}.tmp_{uuid.uuid4().hex[:8]}")
    try:
        df.to_csv(tmp_path, index=index, encoding=encoding)
        with open(tmp_path, "a", encoding=encoding) as f:
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, dest)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
    return dest


def save_atomic_yaml(data: dict[str, Any], path: str | Path) -> Path:
    """Ghi tệp YAML an toàn / nguyên tử (atomic write)."""
    import os
    import uuid

    dest = Path(path)
    ensure_dir(dest.parent)
    tmp_path = dest.with_name(f".{dest.name}.tmp_{uuid.uuid4().hex[:8]}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, dest)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
    return dest
