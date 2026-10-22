"""Bước 5–6 (bổ sung) — Ablation đặc trưng của baseline phi đồ thị.

Trả lời câu hỏi GV (19/10/2026): "Nhóm phi đồ thị lại cho kết quả tốt hơn?"
Giả thuyết: lợi thế của RF/ANN đến chủ yếu từ final_x/final_y — điểm rơi của
đường chuyền cuối cùng trước khi possession kết thúc (với possession có sút:
ngay trước cú sút). GNN gộp đỉnh bằng mean pooling nên không biết cạnh nào là
cạnh cuối.

Thí nghiệm: huấn luyện lại LR/RF/ANN (cùng split, cùng 5 seed, cùng cấu hình)
trên các tập đặc trưng khác nhau, và đo độ quan trọng hoán vị (permutation
importance) của RF trên tập validation.

Cách dùng:
    python scripts/06c_baseline_feature_ablation.py --config configs/data.yaml

Đầu ra:
    results/tables/<run>/baseline_feature_ablation.csv          (mean/std 5 seed, test)
    results/tables/<run>/baseline_feature_ablation_all_seeds.csv
    results/tables/<run>/rf_permutation_importance.csv          (validation)
    results/tables/<run>/final_x_by_label.csv                   (mô tả final_x theo nhãn)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

from xai_football.data.splits import load_match_splits  # noqa: E402
from xai_football.graphs.dataset import load_graphs  # noqa: E402
from xai_football.modeling.baselines import (  # noqa: E402
    graphs_to_tabular_dataset,
    train_and_eval_tabular_baselines,
)
from xai_football.utils.io import load_yaml  # noqa: E402
from xai_football.utils.logging import get_logger  # noqa: E402
from xai_football.utils.paths import get_processed_dir, get_split_dir, get_tables_dir  # noqa: E402
from xai_football.utils.seed import seed_everything  # noqa: E402

logger = get_logger("baseline_ablation")

FULL = [
    "n_passes",
    "n_players_involved",
    "sequence_length",
    "mean_pass_distance",
    "mean_pass_angle",
    "duration",
    "start_x",
    "start_y",
    "final_x",
    "final_y",
]
FEATURE_SETS: dict[str, list[str]] = {
    "A_full_10": FULL,
    "B_no_final_xy": [f for f in FULL if f not in ("final_x", "final_y")],
    "C_no_start_final_xy": [
        f for f in FULL if f not in ("final_x", "final_y", "start_x", "start_y")
    ],
    "D_only_final_xy": ["final_x", "final_y"],
}
METRICS = ["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr", "brier_score"]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--model-config", default="configs/model/baseline_tabular.yaml")
    ap.add_argument("--train-config", default="configs/train.yaml")
    ap.add_argument("--split-config", default="configs/split.yaml")
    args = ap.parse_args()

    data_cfg = load_yaml(args.config)
    baseline_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.train_config)
    split_cfg = load_yaml(args.split_config)
    seeds = [int(s) for s in train_cfg["seeds"][: int(train_cfg.get("n_runs", 5))]]
    run_name = data_cfg.get("run_name", "official")
    tables_dir = get_tables_dir(data_cfg)

    graphs, _, _ = load_graphs(get_processed_dir(data_cfg))
    splits = load_match_splits(get_split_dir(split_cfg, run_name))
    X_all, y_all, mids = graphs_to_tabular_dataset(graphs, feature_names=FULL)
    mids = np.asarray(mids)
    tr = np.isin(mids, list(splits["train"]))
    va = np.isin(mids, list(splits["val"]))
    te = np.isin(mids, list(splits["test"]))
    logger.info("Train=%d Val=%d Test=%d", tr.sum(), va.sum(), te.sum())

    # Mô tả: final_x theo nhãn (chỉ tập train để không nhìn test).
    # Tọa độ cạnh đã chuẩn hóa theo kích thước sân 120x80 -> quy đổi vòng cấm
    # (x >= 102, 18 <= y <= 62) về thang 0-1.
    L = float(data_cfg.get("pitch", {}).get("length", 120))
    W = float(data_cfg.get("pitch", {}).get("width", 80))
    desc = []
    for lab in (0, 1):
        fx = X_all.loc[tr & (y_all == lab), "final_x"]
        fy = X_all.loc[tr & (y_all == lab), "final_y"]
        in_box = ((fx >= 102 / L) & (fy >= 18 / W) & (fy <= 62 / W)).mean()
        desc.append(
            {
                "split": "train",
                "label": lab,
                "n": int(len(fx)),
                "final_x_mean_m": round(float(fx.mean()) * L, 1),
                "final_x_median_m": round(float(fx.median()) * L, 1),
                "share_final_in_penalty_box": round(float(in_box), 4),
            }
        )
    pd.DataFrame(desc).to_csv(
        tables_dir / "final_x_by_label.csv", index=False, encoding="utf-8-sig"
    )

    rows = []
    for set_name, feats in FEATURE_SETS.items():
        for seed in seeds:
            seed_everything(seed)
            res = train_and_eval_tabular_baselines(
                X_all.loc[tr, feats],
                y_all[tr],
                X_all.loc[va, feats],
                y_all[va],
                X_all.loc[te, feats],
                y_all[te],
                config=baseline_cfg,
                seed=seed,
            )
            for model, m in res["test_metrics"].items():
                rows.append(
                    {
                        "feature_set": set_name,
                        "n_features": len(feats),
                        "model": model,
                        "seed": seed,
                        **{k: m[k] for k in METRICS},
                    }
                )
            logger.info("%s seed=%d xong", set_name, seed)

            if set_name == "A_full_10" and seed == seeds[0]:
                rf = res["models"]["random_forest"]
                pi = permutation_importance(
                    rf,
                    X_all.loc[va, feats],
                    y_all[va],
                    scoring="average_precision",
                    n_repeats=10,
                    random_state=seed,
                    n_jobs=-1,
                )
                pd.DataFrame(
                    {
                        "feature": feats,
                        "auc_pr_drop_mean": np.round(pi.importances_mean, 4),
                        "auc_pr_drop_std": np.round(pi.importances_std, 4),
                        "gini_importance": np.round(rf.feature_importances_, 4),
                    }
                ).sort_values("auc_pr_drop_mean", ascending=False).to_csv(
                    tables_dir / "rf_permutation_importance.csv", index=False, encoding="utf-8-sig"
                )

    df = pd.DataFrame(rows)
    df.to_csv(
        tables_dir / "baseline_feature_ablation_all_seeds.csv", index=False, encoding="utf-8-sig"
    )
    agg = df.groupby(["feature_set", "n_features", "model"], sort=False)[METRICS].agg(
        ["mean", "std"]
    )
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.round(4).reset_index()
    agg.to_csv(tables_dir / "baseline_feature_ablation.csv", index=False, encoding="utf-8-sig")
    print(
        agg[
            ["feature_set", "model", "auc_roc_mean", "auc_pr_mean", "f1_mean", "brier_score_mean"]
        ].to_string(index=False)
    )
    print(pd.read_csv(tables_dir / "rf_permutation_importance.csv").to_string(index=False))
    print(pd.DataFrame(desc).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
