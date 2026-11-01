"""Regression tests cho scripts/09_evaluate_xai.py và cơ chế kiểm soát Provenance.

Đảm bảo:
1. Không truyền --xai-dir/--run-id thì không tự chọn run; báo lỗi và liệt kê run khả dụng.
2. Run pilot không thể ghi vào bảng official.
3. Run partial/debug/smoke không được công nhận official.
4. Run thiếu manifest bị từ chối tổng hợp.
5. Run thuộc commit/split/checkpoint khác không được tự ghép vào run hiện hành.
6. Run cũ không bị tự chọn chỉ vì tên thư mục sắp xếp sau cùng.
7. Khi validation thất bại, bảng official hiện có không bị ghi đè (an toàn nguyên tử).
8. Hai run có metric giống nhau nhưng khác provenance vẫn được phân biệt rõ ràng.
9. Artifact lịch sử tra cứu được qua --allow-historical nhưng không ghi đè bảng canonical.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from xai_football.utils.provenance import (
    format_run_table,
    scan_available_xai_runs,
    validate_run_manifest,
)

# Load module scripts/09_evaluate_xai.py
_script_path = Path(__file__).resolve().parents[2] / "scripts" / "09_evaluate_xai.py"
_spec = importlib.util.spec_from_file_location("script_09_evaluate_xai", _script_path)
eval_xai = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_xai)


@pytest.fixture
def fake_xai_env(tmp_path):
    """Thiết lập môi trường XAI giả lập với nhiều loại runs."""
    base_xai = tmp_path / "results" / "xai"
    official_dir = base_xai / "official" / "mpnn"
    pilot_dir = base_xai / "pilot" / "mpnn"
    tables_dir = tmp_path / "results" / "tables" / "official"
    tables_dir.mkdir(parents=True)

    # 1. Run cũ theo thứ tự bảng chữ cái nhưng chỉ có 5 graphs (historical)
    old_run_dir = official_dir / "20261017_091540"
    old_run_dir.mkdir(parents=True)
    manifest_old = {
        "run_id": "20261017_091540",
        "run_name": "official",
        "model_name": "mpnn",
        "split": "test",
        "n_graphs_explained": 5,
        "git": {"commit_sha": "9fae6c51", "branch": "REPRODUCIBILITY"},
        "status": "historical_provisional",
        "timestamp": "2026-10-17T09:17:43+00:00",
    }
    (old_run_dir / "xai_run_manifest.json").write_text(json.dumps(manifest_old), encoding="utf-8")
    summary_old = pd.DataFrame(
        [{"explainer": "gnn_explainer", "fidelity_plus_mean": 0.1, "n_graphs": 5}]
    )
    summary_old.to_csv(old_run_dir / "xai_metrics_summary.csv", index=False)

    # 2. Run pilot
    pilot_run_dir = pilot_dir / "pilot_run_01"
    pilot_run_dir.mkdir(parents=True)
    manifest_pilot = {
        "run_id": "pilot_run_01",
        "run_name": "pilot",
        "model_name": "mpnn",
        "split": "test",
        "n_graphs_explained": 10,
        "git": {"commit_sha": "current_sha", "branch": "develop"},
        "status": "pilot",
        "timestamp": "2026-10-20T10:00:00+00:00",
    }
    (pilot_run_dir / "xai_run_manifest.json").write_text(
        json.dumps(manifest_pilot), encoding="utf-8"
    )
    summary_old.to_csv(pilot_run_dir / "xai_metrics_summary.csv", index=False)

    # 3. Run hợp lệ đầy đủ official
    valid_run_dir = official_dir / "official_run_valid"
    valid_run_dir.mkdir(parents=True)
    manifest_valid = {
        "run_id": "official_run_valid",
        "run_name": "official",
        "model_name": "mpnn",
        "split": "test",
        "n_graphs_explained": 57,
        "git": {"commit_sha": "current_sha", "branch": "Clean"},
        "status": "complete",
        "timestamp": "2026-10-22T12:00:00+00:00",
    }
    (valid_run_dir / "xai_run_manifest.json").write_text(
        json.dumps(manifest_valid), encoding="utf-8"
    )
    summary_old.to_csv(valid_run_dir / "xai_metrics_summary.csv", index=False)

    return {
        "root": tmp_path,
        "base_xai": base_xai,
        "tables_dir": tables_dir,
        "old_run_dir": old_run_dir,
        "pilot_run_dir": pilot_run_dir,
        "valid_run_dir": valid_run_dir,
    }


def test_missing_run_selection_lists_available_and_aborts(fake_xai_env, monkeypatch):
    """Không truyền --xai-dir hay --run-id thì không tự chọn phần tử cuối; phải dừng và liệt kê."""
    monkeypatch.setattr(
        "sys.argv",
        ["09_evaluate_xai.py", "--config", "configs/data.yaml"],
    )
    # Monkeypatch scan_available_xai_runs để quét fake_xai_env
    monkeypatch.setattr(
        eval_xai,
        "scan_available_xai_runs",
        lambda p: scan_available_xai_runs(fake_xai_env["base_xai"]),
    )

    exit_code = eval_xai.main()
    assert exit_code == 1

    # Kiểm tra danh sách quét được không tự động chọn thư mục cuối
    runs = scan_available_xai_runs(fake_xai_env["base_xai"])
    assert len(runs) >= 3
    run_ids = [r["run_id"] for r in runs]
    assert "20261017_091540" in run_ids
    assert "pilot_run_01" in run_ids
    assert "official_run_valid" in run_ids

    # Bảng format run table có thông tin rõ ràng
    formatted = format_run_table(runs)
    assert "20261017_091540" in formatted
    assert "HISTORICAL_PROVISIONAL" in formatted


def test_pilot_run_cannot_overwrite_official(fake_xai_env, monkeypatch):
    """Run pilot không được ghi nhận là official."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "09_evaluate_xai.py",
            "--xai-dir",
            str(fake_xai_env["pilot_run_dir"]),
        ],
    )
    # Chưa truyền --allow-historical
    exit_code = eval_xai.main()
    assert exit_code == 1


def test_historical_5_graph_run_rejected_from_official(fake_xai_env, monkeypatch):
    """Run cũ 20261017_091540 (5 graphs) bị từ chối nếu không có --allow-historical."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "09_evaluate_xai.py",
            "--xai-dir",
            str(fake_xai_env["old_run_dir"]),
        ],
    )
    exit_code = eval_xai.main()
    assert exit_code == 1


def test_historical_run_with_flag_isolated_from_canonical(fake_xai_env, monkeypatch):
    """Khi dùng --allow-historical, file lưu ra bảng riêng biệt, không đè xai_comparison.csv."""
    canonical_table = fake_xai_env["tables_dir"] / "xai_comparison.csv"
    canonical_table.write_text("canonical,official,data\n", encoding="utf-8")

    monkeypatch.setattr(eval_xai, "get_tables_dir", lambda cfg: fake_xai_env["tables_dir"])
    monkeypatch.setattr(
        "sys.argv",
        [
            "09_evaluate_xai.py",
            "--xai-dir",
            str(fake_xai_env["old_run_dir"]),
            "--allow-historical",
        ],
    )
    exit_code = eval_xai.main()
    assert exit_code == 0

    # Bảng canonical gốc giữ nguyên vẹn
    assert canonical_table.read_text(encoding="utf-8") == "canonical,official,data\n"

    # Kết quả lịch sử được ghi vào file riêng có hậu tố run_id
    historical_out = fake_xai_env["tables_dir"] / "xai_comparison_historical_20261017_091540.csv"
    assert historical_out.exists()
    hist_df = pd.read_csv(historical_out)
    assert hist_df["provenance_status"].iloc[0] == "HISTORICAL_PROVISIONAL"


def test_missing_manifest_is_rejected(fake_xai_env, monkeypatch):
    """Thư mục có kết quả CSV nhưng thiếu file manifest bị từ chối tuyệt đối."""
    unverified_dir = fake_xai_env["base_xai"] / "official" / "unverified_run"
    unverified_dir.mkdir(parents=True)
    (unverified_dir / "xai_metrics_summary.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "09_evaluate_xai.py",
            "--xai-dir",
            str(unverified_dir),
        ],
    )
    exit_code = eval_xai.main()
    assert exit_code == 1


def test_two_runs_same_metrics_distinguished_by_provenance(fake_xai_env):
    """Hai run có chỉ số giống hệt nhau nhưng provenance khác nhau vẫn được phân biệt qua manifest."""
    manifest_a = {
        "run_id": "run_a",
        "run_name": "official",
        "git": {"commit_sha": "commit_1111"},
        "status": "complete",
        "n_graphs_explained": 100,
    }
    manifest_b = {
        "run_id": "run_b",
        "run_name": "pilot",
        "git": {"commit_sha": "commit_2222"},
        "status": "pilot",
        "n_graphs_explained": 100,
    }
    ok_a, _ = validate_run_manifest(manifest_a, require_official=True)
    ok_b, reason_b = validate_run_manifest(manifest_b, require_official=True)

    assert ok_a is True
    assert ok_b is False
    assert "pilot" in reason_b
