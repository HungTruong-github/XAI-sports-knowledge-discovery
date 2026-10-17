"""Tiện ích thu thập thông tin Môi trường thực thi (Environment Metadata).

Đảm bảo ghi lại chính xác:
    - OS, kiến trúc, CPU, Python version
    - Phiên bản các thư viện chính (torch, torch_geometric, sklearn, pandas, numpy, shap...)
    - CUDA / GPU availability
    - Git branch, commit SHA, clean tree status
    - SHA-256 hashes của tất cả các file cấu hình configs/
"""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

from xai_football.utils.provenance import compute_file_sha256, get_git_info


def collect_environment_metadata(repo_root: Path | str | None = None) -> dict[str, Any]:
    """Thu thập thông tin môi trường chi tiết từ hệ thống thực tế."""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]

    packages = {}
    package_names = [
        "torch",
        "torch_geometric",
        "pandas",
        "numpy",
        "scipy",
        "sklearn",
        "matplotlib",
        "seaborn",
        "yaml",
        "nbformat",
        "shap",
        "xgboost",
        "lightgbm",
    ]
    for pkg in package_names:
        try:
            if pkg == "sklearn":
                import sklearn

                packages[pkg] = getattr(sklearn, "__version__", "unknown")
            elif pkg == "yaml":
                import yaml

                packages[pkg] = getattr(yaml, "__version__", "unknown")
            else:
                mod = __import__(pkg)
                packages[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            packages[pkg] = "not installed"

    # CUDA info
    cuda_available = False
    cuda_device_count = 0
    cuda_device_name = "CPU only"
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        cuda_device_count = torch.cuda.device_count() if cuda_available else 0
        cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU only"
    except Exception:
        pass

    # Git info
    git_info = get_git_info(root)

    # Config hashes
    config_hashes = {}
    config_dir = root / "configs"
    if config_dir.exists():
        for cfg_path in sorted(config_dir.glob("**/*.yaml")):
            rel_path = str(cfg_path.relative_to(root))
            config_hashes[rel_path] = compute_file_sha256(cfg_path)

    metadata = {
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_device_name": cuda_device_name,
        "git": {
            "branch": git_info["branch"],
            "commit_sha": git_info["commit_sha"],
            "clean_working_tree": git_info["clean_working_tree"],
        },
        "config_hashes": config_hashes,
    }
    return metadata


def save_environment_metadata(
    output_file: Path | str, repo_root: Path | str | None = None
) -> dict[str, Any]:
    """Thu thập và lưu metadata môi trường ra file JSON."""
    meta = collect_environment_metadata(repo_root)
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta
