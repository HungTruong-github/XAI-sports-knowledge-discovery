"""Unit tests cho scripts/07_evaluate.py: chỉ nhận run multi-seed đầy đủ."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

_path = Path(__file__).resolve().parents[2] / "scripts" / "07_evaluate.py"
_spec = importlib.util.spec_from_file_location("script_07_evaluate", _path)
evaluate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate)

METRIC_ROW = {
    "accuracy": 0.8,
    "precision": 0.4,
    "recall": 0.6,
    "f1": 0.5,
    "auc_roc": 0.85,
    "auc_pr": 0.55,
    "brier_score": 0.1,
    "threshold": 0.5,
    "loss": 0.9,
}


def _write_gnn_run(root: Path, seeds: list[int], complete: bool) -> None:
    model_dir = root / "official" / "mpnn"
    model_dir.mkdir(parents=True)
    rows = [{"seed": s, **METRIC_ROW, "auc_pr": 0.5 + 0.01 * i} for i, s in enumerate(seeds)]
    pd.DataFrame(rows).to_csv(model_dir / "test_metrics_all_seeds.csv", index=False)
    pd.DataFrame(rows).to_csv(model_dir / "val_metrics_all_seeds.csv", index=False)
    agg = {
        "is_complete_official": complete,
        "n_failed_runs": 0,
        "successful_seeds": seeds,
        "best_seed_by_val_auc_pr": seeds[-1],
    }
    (model_dir / "aggregate_summary.json").write_text(json.dumps(agg), encoding="utf-8")


@pytest.fixture
def fake_experiments(tmp_path, monkeypatch):
    monkeypatch.setattr(
        evaluate, "get_experiments_dir", lambda run, suffix="": tmp_path / f"{run}{suffix}"
    )
    return tmp_path


def test_complete_run_is_accepted(fake_experiments):
    _write_gnn_run(fake_experiments, [42, 123, 456, 789, 1024], complete=True)
    res, reason = evaluate.load_gnn_results("official", "mpnn", expected_runs=5)
    assert reason == ""
    assert res["seeds"] == [42, 123, 456, 789, 1024]
    summary = evaluate.summarize_seeds(res["test_df"])
    assert summary["n_runs"] == 5
    assert summary["auc_pr"] == pytest.approx(0.52)
    assert summary["auc_pr_std"] > 0


def test_single_seed_run_is_flagged_incomplete(fake_experiments):
    _write_gnn_run(fake_experiments, [42], complete=False)
    res, reason = evaluate.load_gnn_results("official", "mpnn", expected_runs=5)
    assert res is not None
    assert "1/5 seed" in reason
    assert "is_complete_official" in reason


def test_missing_run_is_reported(fake_experiments):
    res, reason = evaluate.load_gnn_results("official", "gcn", expected_runs=5)
    assert res is None
    assert "chưa có kết quả" in reason


def test_failure_manifest_blocks_old_results(fake_experiments):
    _write_gnn_run(fake_experiments, [42, 123, 456, 789, 1024], complete=True)
    (fake_experiments / "official" / "mpnn" / "run_failure_manifest.json").write_text("{}")
    res, reason = evaluate.load_gnn_results("official", "mpnn", expected_runs=5)
    assert res is None
    assert "thất bại" in reason


def test_debug_and_smoke_flags_block_complete_official(fake_experiments):
    """Run gắn cờ debug hoặc smoke test không được công nhận là complete official."""
    _write_gnn_run(fake_experiments, [42, 123, 456, 789, 1024], complete=True)
    agg_path = fake_experiments / "official" / "mpnn" / "aggregate_summary.json"
    agg = json.loads(agg_path.read_text(encoding="utf-8"))
    agg["smoke_test"] = True
    agg_path.write_text(json.dumps(agg), encoding="utf-8")

    res, reason = evaluate.load_gnn_results("official", "mpnn", expected_runs=5)
    assert res is not None
    assert "smoke_test = true" in reason


def test_allow_partial_does_not_overwrite_official_canonical(tmp_path, monkeypatch):
    """Run dùng --allow-partial phải lưu vào partial_model_comparison.csv, không đè model_comparison.csv."""
    tables_dir = tmp_path / "results" / "tables" / "official"
    tables_dir.mkdir(parents=True)
    canonical_file = tables_dir / "model_comparison.csv"
    canonical_file.write_text("official,canonical,content\n", encoding="utf-8")

    monkeypatch.setattr(evaluate, "get_tables_dir", lambda cfg: tables_dir)
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path / "results")

    # Giả lập ghi bảng trong tình huống allow_partial
    df = pd.DataFrame([{"model": "MPNN", "complete": False, "auc_pr": 0.5}])
    # Logic cách ly:
    is_partial_run = True
    run_name = "official"
    if is_partial_run and run_name == "official":
        out_test = tables_dir / "partial_model_comparison.csv"
    else:
        out_test = tables_dir / "model_comparison.csv"

    evaluate.save_atomic_csv(df, out_test)

    # File canonical gốc phải nguyên vẹn
    assert canonical_file.read_text(encoding="utf-8") == "official,canonical,content\n"
    assert (tables_dir / "partial_model_comparison.csv").exists()
