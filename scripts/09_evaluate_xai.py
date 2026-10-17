"""Bước 8 — Đối sánh chất lượng các phương pháp giải thích (XAI Evaluation & Comparison).

Tham chiếu: docs/Quy_trinh_XAI_Football.md, mục 8 và configs/xai.yaml.
So sánh:
    - GNNExplainer vs PGExplainer vs Random Baseline
    - Các độ đo: Fidelity+, Fidelity-, Sparsity, Stability, Deletion AUC, Insertion AUC.

Cách dùng:
    python scripts/09_evaluate_xai.py --config configs/data.yaml
    python scripts/09_evaluate_xai.py --config configs/data.yaml --xai-dir results/xai/official/mpnn/<run_id>

Đầu ra:
    results/tables/<run_name>/xai_comparison.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from xai_football.utils.io import load_yaml  # noqa: E402
from xai_football.utils.logging import get_logger  # noqa: E402
from xai_football.utils.paths import get_tables_dir  # noqa: E402

logger = get_logger("evaluate_xai")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--xai-dir", default=None, help="Thư mục kết quả XAI cụ thể cần đối sánh")
    args = parser.parse_args()

    data_cfg = load_yaml(args.config)
    run_name = data_cfg.get("run_name", "official")
    tables_dir = get_tables_dir(data_cfg)

    # Tìm thư mục XAI mới nhất nếu không chỉ định
    if args.xai_dir:
        target_dir = Path(args.xai_dir)
    else:
        base_xai = Path("results/xai") / run_name / "mpnn"
        if not base_xai.exists():
            base_xai = Path("results/xai") / run_name
        candidate_dirs = sorted(
            [d for d in base_xai.glob("**/*") if (d / "xai_metrics_summary.csv").exists()]
        )
        if not candidate_dirs:
            logger.error(
                "Không tìm thấy kết quả XAI trong %s. Hãy chạy scripts/08_explain.py trước.",
                base_xai,
            )
            return 1
        target_dir = candidate_dirs[-1]

    summary_file = target_dir / "xai_metrics_summary.csv"
    if not summary_file.exists():
        logger.error("Không tìm thấy %s", summary_file)
        return 1

    summary_df = pd.read_csv(summary_file)
    logger.info("Đã tải tệp tổng hợp XAI từ: %s", summary_file)

    tables_dir.mkdir(parents=True, exist_ok=True)
    out_table = tables_dir / "xai_comparison.csv"
    summary_df.to_csv(out_table, index=False, encoding="utf-8-sig")

    print("\n=== BẢNG ĐỐI SÁNH CHẤT LƯỢNG GIẢI THÍCH XAI (BƯỚC 8) ===")
    print(summary_df.to_string(index=False))
    print(f"\nĐã lưu bảng đối sánh XAI vào: {out_table}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
