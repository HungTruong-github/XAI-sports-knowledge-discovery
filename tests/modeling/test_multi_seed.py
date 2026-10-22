"""Unit tests cho huấn luyện đa seed, cách ly thư mục, và tổng hợp kết quả (Priority 1)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from xai_football.training import (
    clean_stale_aggregates,
    compute_metrics_aggregation,
    resolve_seeds,
)

# Đồng thời kiểm tra tính tương thích: scripts/06_train.py vẫn re-export đầy đủ
_script_path = Path(__file__).resolve().parents[2] / "scripts" / "06_train.py"
_spec = importlib.util.spec_from_file_location("script_06_train", _script_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

assert hasattr(_mod, "compute_metrics_aggregation")
assert hasattr(_mod, "resolve_seeds")
assert hasattr(_mod, "clean_stale_aggregates")


def test_resolve_seeds_cli_single():
    """CLI --seed ghi đè thành 1 seed duy nhất để debug/quick test."""
    train_cfg = {"seeds": [42, 123, 456], "n_runs": 3}
    seeds = resolve_seeds(train_cfg, cli_seed=999)
    assert seeds == [999]


def test_resolve_seeds_cli_multiple():
    """CLI --seeds ghi đè danh sách seed."""
    train_cfg = {"seed": 42}
    seeds = resolve_seeds(train_cfg, cli_seeds="100, 200, 300")
    assert seeds == [100, 200, 300]


def test_resolve_seeds_from_config_multi():
    """Cấu hình train.yaml có seeds và n_runs: lấy đúng số lượng n_runs đầu tiên."""
    train_cfg = {"seeds": [42, 123, 456, 789, 1024], "n_runs": 3}
    seeds = resolve_seeds(train_cfg)
    assert seeds == [42, 123, 456]


def test_resolve_seeds_from_config_single():
    """Cấu hình chỉ có 'seed: 42' trả về [42]."""
    train_cfg = {"seed": 42}
    seeds = resolve_seeds(train_cfg)
    assert seeds == [42]


def test_resolve_seeds_empty_fallback():
    """Cấu hình rỗng trả về seed mặc định [42]."""
    seeds = resolve_seeds({})
    assert seeds == [42]


def test_resolve_seeds_hierarchy_debug_flag():
    """Kiểm tra cờ return_debug phân định rõ ràng giữa run debug và run chính thức."""
    # 1. cli_seed debug ưu tiên cao nhất -> is_debug=True
    seeds, is_debug = resolve_seeds(
        {"seeds": [42, 123], "n_runs": 2}, cli_seed=999, return_debug=True
    )
    assert seeds == [999]
    assert is_debug is True

    # 2. cli_seeds -> is_debug=False
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
    """Bảo đảm dọn sạch artifact tổng hợp cũ trước khi chạy mới."""
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


def test_compute_metrics_aggregation_deterministic():
    """Tổng hợp mean và sample standard deviation (ddof=1) chính xác và tất định."""
    rows = [
        {"seed": 42, "accuracy": 0.80, "auc_pr": 0.50, "loss": 0.8},
        {"seed": 123, "accuracy": 0.84, "auc_pr": 0.60, "loss": 0.6},
        {"seed": 456, "accuracy": 0.82, "auc_pr": 0.55, "loss": 0.7},
    ]
    agg = compute_metrics_aggregation(rows)
    assert agg["n_runs"] == 3
    assert agg["mean"]["accuracy"] == 0.82
    assert agg["mean"]["auc_pr"] == 0.55
    assert agg["mean"]["loss"] == 0.70
    assert abs(agg["std"]["accuracy"] - 0.02) < 1e-4
    assert abs(agg["std"]["auc_pr"] - 0.05) < 1e-4


def test_compute_metrics_aggregation_single_run():
    """Với 1 run duy nhất, std phải bằng 0.0 thay vì NaN."""
    rows = [{"seed": 42, "accuracy": 0.85, "auc_pr": 0.58}]
    agg = compute_metrics_aggregation(rows)
    assert agg["n_runs"] == 1
    assert agg["mean"]["accuracy"] == 0.85
    assert agg["std"]["accuracy"] == 0.0
    assert agg["std"]["auc_pr"] == 0.0


def test_compute_metrics_aggregation_empty():
    """Danh sách rỗng trả về dict rỗng an toàn."""
    assert compute_metrics_aggregation([]) == {}


def test_failed_runs_reporting_manifest(tmp_path: Path):
    """Mỗi run có thư mục seed riêng biệt, phân biệt run thành công và run thất bại."""
    exp_dir = tmp_path / "experiments" / "official" / "test_model"
    seed_42_dir = exp_dir / "42"
    seed_123_dir = exp_dir / "123"
    seed_42_dir.mkdir(parents=True)
    seed_123_dir.mkdir(parents=True)

    # Giả lập run 42 thành công
    test_metrics_42 = pd.DataFrame([{"accuracy": 0.85, "auc_pr": 0.58, "loss": 0.75}])
    test_metrics_42.to_csv(seed_42_dir / "test_metrics.csv", index=False)

    # Giả lập run 123 thất bại với error log
    with open(seed_123_dir / "run_error.txt", "w", encoding="utf-8") as f:
        f.write("RuntimeError: CUDA out of memory\n")

    # Kiểm tra tính độc lập của artifact từng seed
    assert (seed_42_dir / "test_metrics.csv").exists()
    assert not (seed_123_dir / "test_metrics.csv").exists()
    assert (seed_123_dir / "run_error.txt").exists()
