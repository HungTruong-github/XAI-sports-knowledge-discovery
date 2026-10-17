"""Tiện ích thu thập thông tin Provenance (Git revision, cấu hình, môi trường).

Đảm bảo mọi kết quả thực nghiệm có thể truy ngược về commit, branch, config SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def get_git_info(cwd: Path | str | None = None) -> dict[str, Any]:
    """Lấy thông tin git commit, branch và trạng thái working tree thực tế."""
    git_info: dict[str, Any] = {
        "branch": "unknown",
        "commit_sha": "unknown",
        "clean_working_tree": False,
        "is_git_repo": False,
    }
    repo_dir = Path(cwd) if cwd else Path(__file__).resolve().parents[3]
    try:
        # Kiểm tra commit sha
        res_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        git_info["commit_sha"] = res_sha.stdout.strip()
        git_info["is_git_repo"] = True

        # Kiểm tra branch
        res_branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        git_info["branch"] = res_branch.stdout.strip()

        # Kiểm tra status
        res_status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        git_info["clean_working_tree"] = len(res_status.stdout.strip()) == 0
    except Exception:
        pass
    return git_info


def compute_file_sha256(file_path: Path | str) -> str:
    """Tính mã SHA-256 của một tệp tin."""
    path = Path(file_path)
    if not path.is_file():
        return ""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dict_sha256(data: dict[str, Any]) -> str:
    """Tính mã SHA-256 từ nội dung dict cấu hình tất định."""
    encoded = json.dumps(data, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
