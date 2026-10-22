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


# Phân loại và trạng thái run thực nghiệm
RUN_TYPES = ("official", "pilot", "debug", "smoke", "historical")
RUN_STATUSES = ("complete", "partial", "failed", "historical", "unverified")


def validate_run_manifest(
    manifest: dict[str, Any],
    expected_run_name: str | None = None,
    require_official: bool = False,
    min_samples: int | None = None,
) -> tuple[bool, str]:
    """Kiểm tra tính hợp lệ và toàn vẹn của một run manifest.

    Args:
        manifest: Nội dung JSON manifest của run.
        expected_run_name: Tên run kỳ vọng ('official' hoặc 'pilot').
        require_official: Bắt buộc run phải đủ điều kiện công nhận official.
        min_samples: Số lượng mẫu tối thiểu kỳ vọng (nếu có).

    Returns:
        (is_valid, reason): True nếu hợp lệ; False kèm lý do từ chối nếu không hợp lệ.
    """
    if not isinstance(manifest, dict):
        return False, "Manifest không phải là dictionary JSON hợp lệ"

    # 1. Các trường nhận diện cơ bản
    run_id = manifest.get("run_id")
    if not run_id:
        return False, "Manifest thiếu trường 'run_id'"

    run_name = manifest.get("run_name", "unknown")
    if expected_run_name and run_name != expected_run_name:
        return (
            False,
            f"run_name trong manifest ('{run_name}') không khớp với kỳ vọng ('{expected_run_name}')",
        )

    # 2. Kiểm tra Git provenance
    git = manifest.get("git")
    if not git or not isinstance(git, dict) or not git.get("commit_sha"):
        return False, "Manifest thiếu thông tin Git commit SHA hợp lệ"

    # 3. Kiểm tra điều kiện Official
    if require_official:
        if run_name != "official":
            return False, f"Run được gắn nhãn '{run_name}', không phải 'official'"

        status = manifest.get("status", "").lower()
        if status in (
            "pilot",
            "debug",
            "smoke",
            "partial",
            "failed",
            "historical",
            "historical_provisional",
        ):
            return (
                False,
                f"Trạng thái run là '{status}', không thể công nhận là kết quả official mới",
            )

        if manifest.get("is_debug_run") is True:
            return False, "Run được đánh dấu là debug run (is_debug_run=True)"
        if manifest.get("smoke_test") is True:
            return False, "Run được đánh dấu là smoke test (smoke_test=True)"
        if manifest.get("allow_partial") is True:
            return False, "Run được thực thi với allow_partial=True"

        # Nếu có trường is_complete_official thì phải True
        if "is_complete_official" in manifest and not manifest["is_complete_official"]:
            return False, "is_complete_official = False trong manifest"

        # Kiểm tra số lượng mẫu nếu có quy định
        n_samples = manifest.get("n_graphs_explained", manifest.get("n_samples"))
        if min_samples is not None and n_samples is not None:
            if int(n_samples) < min_samples:
                return False, f"Số mẫu ({n_samples}) nhỏ hơn ngưỡng yêu cầu ({min_samples})"

    return True, ""


def scan_available_xai_runs(base_dir: Path | str) -> list[dict[str, Any]]:
    """Quét và lập danh mục toàn bộ các run XAI có sẵn trong thư mục.

    Không tự động chọn run mới nhất; cung cấp đầy đủ metadata để người dùng/caller
    lựa chọn chính xác.
    """
    root = Path(base_dir)
    if not root.exists():
        return []

    # Tìm tất cả thư mục có xai_metrics_summary.csv hoặc xai_run_manifest.json
    candidate_dirs = sorted(
        {
            p.parent
            for p in root.glob("**/*")
            if p.name in ("xai_metrics_summary.csv", "xai_run_manifest.json")
        }
    )

    runs = []
    for d in candidate_dirs:
        manifest_path = d / "xai_run_manifest.json"
        summary_path = d / "xai_metrics_summary.csv"
        manifest_data = None
        has_manifest = manifest_path.exists()

        if has_manifest:
            try:
                with open(manifest_path, encoding="utf-8") as f:
                    manifest_data = json.load(f)
            except Exception:
                manifest_data = None

        run_id = manifest_data.get("run_id", d.name) if manifest_data else d.name
        run_name = (
            manifest_data.get("run_name", "official" if "official" in d.parts else "unknown")
            if manifest_data
            else "unknown"
        )
        model_name = (
            manifest_data.get("model_name", d.parent.name if d.parent else "unknown")
            if manifest_data
            else "unknown"
        )
        split = manifest_data.get("split", "test") if manifest_data else "unknown"
        n_graphs = manifest_data.get("n_graphs_explained") if manifest_data else None

        git = manifest_data.get("git", {}) if manifest_data else {}
        commit_sha = git.get("commit_sha", "unknown")[:8] if isinstance(git, dict) else "unknown"
        branch = git.get("branch", "unknown") if isinstance(git, dict) else "unknown"
        timestamp = manifest_data.get("timestamp", "unknown") if manifest_data else "unknown"

        # Đánh giá status
        if not has_manifest:
            status = "UNVERIFIED (missing manifest)"
            is_official = False
        elif run_id == "20261017_091540" or (n_graphs is not None and n_graphs < 50):
            status = f"HISTORICAL_PROVISIONAL (n={n_graphs or '?'})"
            is_official = False
        elif run_name == "pilot":
            status = "PILOT"
            is_official = False
        else:
            status = manifest_data.get("status", "COMPLETE")
            is_official = True

        runs.append(
            {
                "run_id": run_id,
                "run_name": run_name,
                "model_name": model_name,
                "split": split,
                "n_graphs": n_graphs,
                "status": status,
                "commit_sha": commit_sha,
                "branch": branch,
                "timestamp": timestamp,
                "has_manifest": has_manifest,
                "is_official": is_official,
                "dir": str(d),
                "summary_exists": summary_path.exists(),
            }
        )

    return runs


def format_run_table(runs: list[dict[str, Any]]) -> str:
    """Định dạng danh sách các run thành bảng hiển thị CLI rõ ràng."""
    if not runs:
        return "Không có run nào được tìm thấy."

    lines = [
        f"{'RUN_ID':<18} | {'MODEL':<8} | {'SPLIT':<6} | {'N_GRAPHS':<8} | {'STATUS':<26} | {'COMMIT':<10} | {'PATH'}",
        "-" * 110,
    ]
    for r in runs:
        n_str = str(r["n_graphs"]) if r["n_graphs"] is not None else "N/A"
        lines.append(
            f"{r['run_id']:<18} | {r['model_name']:<8} | {r['split']:<6} | {n_str:<8} | "
            f"{r['status']:<26} | {r['commit_sha']:<10} | {r['dir']}"
        )
    return "\n".join(lines)


def compute_dataset_fingerprint(paths: list[Path | str]) -> str:
    """Tính mã băm kết hợp (fingerprint) của tập các file dữ liệu/split."""
    hasher = hashlib.sha256()
    for p in sorted(str(p) for p in paths):
        fpath = Path(p)
        if fpath.is_file():
            hasher.update(fpath.name.encode("utf-8"))
            hasher.update(compute_file_sha256(fpath).encode("utf-8"))
    return hasher.hexdigest()
