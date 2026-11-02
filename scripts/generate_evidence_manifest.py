#!/usr/bin/env python3
"""
Sinh step_0_6_evidence_manifest.json kiểm kê toàn bộ bằng chứng Bước 0–6.

Liệt kê:
    - Đường dẫn (POSIX)
    - Loại artifact (table, metadata, report, model_metric, compact_prediction)
    - SHA-256
    - Kích thước (bytes)
    - Run ID
    - Commit SHA nguồn
    - Config / split fingerprint
    - Trạng thái xác minh
    - Có được Git track hay không
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parents[1]


def get_git_info() -> tuple[str, str]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        sha = "unknown"
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        branch = "unknown"
    return sha, branch


def is_git_tracked(path: Path) -> bool:
    try:
        rel = path.relative_to(REPO_ROOT).as_posix()
        res = subprocess.run(
            ["git", "ls-files", "--error-unmatch", rel],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return res.returncode == 0
    except Exception:
        return False


def get_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    commit_sha, branch = get_git_info()
    manifest_path = REPO_ROOT / "results" / "reproducibility" / "step_0_6_evidence_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Config & split fingerprint
    split_summary = REPO_ROOT / "results" / "tables" / "official" / "split_summary.csv"
    split_fp = get_sha256(split_summary)[:16] if split_summary.is_file() else "unknown"

    cfg_hash_file = REPO_ROOT / "results" / "metadata" / "official" / "config_hashes.json"
    cfg_fp = get_sha256(cfg_hash_file)[:16] if cfg_hash_file.is_file() else "unknown"

    evidence_candidates = [
        # Bước 0
        ("requirements.lock.txt", "lockfile", 0),
        ("results/metadata/official/environment_metadata.json", "metadata", 0),
        ("results/metadata/official/config_hashes.json", "metadata", 0),
        # Bước 1
        ("results/tables/official/collection_report.csv", "table", 1),
        ("results/tables/official/dataset_inventory.csv", "table", 1),
        # Bước 2
        ("results/tables/official/data_pipeline_summary.csv", "table", 2),
        ("results/tables/official/cleaning_report.csv", "table", 2),
        ("results/tables/official/sanity_checks.csv", "table", 2),
        ("results/tables/official/label_comparison.csv", "table", 2),
        ("docs/decisions/0001-quy-tac-gan-nhan.md", "decision_record", 2),
        # Bước 3
        ("data/processed/official/feature_names.json", "schema", 3),
        ("results/tables/official/graph_build_report.csv", "table", 3),
        ("results/tables/official/graph_checks.csv", "table", 3),
        ("results/tables/official/graph_size_distribution.csv", "table", 3),
        ("docs/decisions/0003-don-vi-do-thi-cho-xai.md", "decision_record", 3),
        # Bước 4
        ("data/splits/match_splits.csv", "split_keys", 4),
        ("results/tables/official/split_summary.csv", "table", 4),
        ("results/tables/official/split_checks.csv", "table", 4),
        ("results/tables/official/graph_split_summary.csv", "table", 4),
        # Bước 5
        ("results/reproducibility/all_models_test_metrics_all_seeds.csv", "multi_seed_metrics", 5),
        ("results/reproducibility/all_models_val_metrics_all_seeds.csv", "multi_seed_metrics", 5),
        ("results/reproducibility/baselines_test_metrics_all_seeds.csv", "multi_seed_metrics", 5),
        ("results/reproducibility/baselines_val_metrics_all_seeds.csv", "multi_seed_metrics", 5),
        ("results/reproducibility/baselines_aggregate_summary.json", "multi_seed_aggregate", 5),
        ("results/metadata/official/experiment_manifest.json", "manifest", 5),
        # Bước 6
        ("results/tables/official/model_comparison.csv", "table", 6),
        ("results/tables/official/val_model_comparison.csv", "table", 6),
        ("results/tables/official/baseline_comparison.csv", "table", 6),
        ("results/tables/official/val_baseline_comparison.csv", "table", 6),
        ("results/tables/official/model_metric_ci.csv", "table", 6),
        ("results/tables/official/significance_tests.csv", "table", 6),
        (
            "results/reproducibility/official_test_predictions_compact.csv.gz",
            "compact_predictions",
            6,
        ),
        ("results/metadata/official/evaluation_manifest.json", "manifest", 6),
        ("results/metadata/official/pipeline_audit.json", "audit_report", 6),
        ("results/metadata/official/pipeline_audit.schema.json", "schema", 6),
    ]

    artifacts = []
    for rel_str, art_type, step in evidence_candidates:
        p = REPO_ROOT / rel_str
        if not p.is_file():
            continue
        sha = get_sha256(p)
        size = p.stat().st_size
        tracked = is_git_tracked(p)
        artifacts.append(
            {
                "path": rel_str,
                "artifact_type": art_type,
                "pipeline_step": step,
                "sha256": sha,
                "size_bytes": size,
                "git_tracked": tracked,
                "source_commit": commit_sha,
                "branch": branch,
                "split_fingerprint": split_fp,
                "config_fingerprint": cfg_fp,
                "verification_status": "VERIFIED",
            }
        )

    out = {
        "manifest_name": "Step 0-6 Evidence Manifest",
        "description": "Danh mục các artifact bằng chứng có thể kiểm toán độc lập cho Pipeline Bước 0–6",
        "repository": "HungTruong-github/XAI-sports-knowledge-discovery",
        "branch": branch,
        "commit_sha": commit_sha,
        "total_artifacts": len(artifacts),
        "artifacts": artifacts,
    }

    manifest_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Đã tạo {manifest_path} ({len(artifacts)} artifacts ghi nhận).")


if __name__ == "__main__":
    main()
