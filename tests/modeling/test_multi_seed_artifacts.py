"""Unit tests kiểm tra vòng đời artifact, dọn dẹp file cũ và bảo vệ kết quả official."""

import importlib.util
from pathlib import Path

# Tải module 06_train bằng importlib
train_script_path = Path(__file__).resolve().parents[2] / "scripts" / "06_train.py"
spec = importlib.util.spec_from_file_location("script_06_train", str(train_script_path))
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)

clean_stale_aggregates = train_mod.clean_stale_aggregates
compute_metrics_aggregation = train_mod.compute_metrics_aggregation
resolve_seeds = train_mod.resolve_seeds


def test_resolve_seeds_hierarchy():
    # 1. cli_seed debug ưu tiên cao nhất
    seeds, is_debug = resolve_seeds(
        {"seeds": [42, 123], "n_runs": 2}, cli_seed=999, return_debug=True
    )
    assert seeds == [999]
    assert is_debug is True

    # 2. cli_seeds
    seeds, is_debug = resolve_seeds({"seeds": [42]}, cli_seeds="10, 20, 30", return_debug=True)
    assert seeds == [10, 20, 30]
    assert is_debug is False

    # 3. train_cfg seeds list cắt theo n_runs
    seeds, is_debug = resolve_seeds({"seeds": [1, 2, 3, 4, 5], "n_runs": 3}, return_debug=True)
    assert seeds == [1, 2, 3]
    assert is_debug is False

    # 4. train_cfg seed đơn
    seeds, is_debug = resolve_seeds({"seed": 77}, return_debug=True)
    assert seeds == [77]
    assert is_debug is False

    # 5. Mặc định
    seeds, is_debug = resolve_seeds({}, return_debug=True)
    assert seeds == [42]
    assert is_debug is False


def test_clean_stale_aggregates(tmp_path: Path):
    stale_files = [
        "val_metrics_all_seeds.csv",
        "test_metrics_all_seeds.csv",
        "aggregate_summary.json",
        "run_failure_manifest.json",
    ]
    for fname in stale_files:
        (tmp_path / fname).write_text("dummy old data", encoding="utf-8")
        assert (tmp_path / fname).exists()

    clean_stale_aggregates(tmp_path)

    for fname in stale_files:
        assert not (tmp_path / fname).exists(), f"{fname} chưa được dọn dẹp sạch!"


def test_compute_metrics_aggregation():
    rows = [
        {"seed": 42, "auc_pr": 0.50, "f1": 0.40, "accuracy": 0.80},
        {"seed": 123, "auc_pr": 0.60, "f1": 0.50, "accuracy": 0.82},
        {"seed": 456, "auc_pr": 0.70, "f1": 0.60, "accuracy": 0.84},
    ]
    agg = compute_metrics_aggregation(rows)
    assert agg["n_runs"] == 3
    assert agg["mean"]["auc_pr"] == 0.60
    assert agg["mean"]["f1"] == 0.50
    assert agg["mean"]["accuracy"] == 0.82
    assert agg["std"]["auc_pr"] == 0.10
    assert agg["std"]["f1"] == 0.10


def test_compute_metrics_aggregation_empty():
    assert compute_metrics_aggregation([]) == {}
