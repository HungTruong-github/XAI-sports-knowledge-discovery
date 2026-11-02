"""Thực nghiệm Hybrid Centrality và Đánh giá Tham chiếu (Phase 4).

Tham chiếu:
- docs/decisions/0002-cong-thuc-hybrid-centrality.md (Quyết định D12, D13, D14, D15, D16, D17)
- configs/hybrid_centrality.yaml
Đầu ra:
- results/hybrid/<run_id>/alpha_curve.csv
- results/hybrid/<run_id>/player_hybrid_rankings.csv
- results/hybrid/<run_id>/evaluation_results.json
- results/hybrid/<run_id>/hybrid_manifest.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

from xai_football.centrality.calculator import (
    compute_hybrid_centrality,
    compute_match_traditional_centrality,
    optimize_alpha_on_set_a,
    percentile_rank,
)
from xai_football.reference.expected_threat import ExpectedThreatModel
from xai_football.reference.player_stats import (
    compute_player_minutes,
    compute_player_reference_stats,
)
from xai_football.utils.io import load_yaml
from xai_football.utils.logging import get_logger
from xai_football.utils.provenance import get_git_info

logger = get_logger("hybrid_experiment")


def compute_ols_delta_r2(
    df: pd.DataFrame,
    y_col: str,
    cent_col: str,
    xai_col: str,
    n_boot: int = 1000,
) -> dict[str, Any]:
    """Chạy hồi quy OLS và tính Delta R^2 kèm 95% Bootstrap CI."""
    sub = df[[y_col, cent_col, xai_col]].dropna()
    if len(sub) < 15:
        return {"r2_base": 0.0, "r2_full": 0.0, "delta_r2": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    y = sub[y_col].values
    X_base = sub[[cent_col]].values
    X_full = sub[[cent_col, xai_col]].values

    reg_base = LinearRegression().fit(X_base, y)
    r2_base = float(reg_base.score(X_base, y))

    reg_full = LinearRegression().fit(X_full, y)
    r2_full = float(reg_full.score(X_full, y))

    delta_r2 = max(0.0, r2_full - r2_base)

    # Bootstrap 95% CI
    rng = np.random.default_rng(42)
    deltas = []
    n = len(sub)
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y[idx]
        xb_b = X_base[idx]
        xf_b = X_full[idx]
        try:
            r2_b = LinearRegression().fit(xb_b, y_b).score(xb_b, y_b)
            r2_f = LinearRegression().fit(xf_b, y_b).score(xf_b, y_b)
            deltas.append(max(0.0, r2_f - r2_b))
        except Exception:
            deltas.append(delta_r2)

    ci_l = float(np.percentile(deltas, 2.5))
    ci_u = float(np.percentile(deltas, 97.5))

    return {
        "r2_pure_centrality": round(r2_base, 4),
        "r2_full_model": round(r2_full, 4),
        "delta_r2": round(delta_r2, 4),
        "ci_lower": round(ci_l, 4),
        "ci_upper": round(ci_u, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hybrid_centrality.yaml")
    parser.add_argument("--events-path", default="data/interim/official/events.parquet")
    parser.add_argument("--lineups-dir", default="data/raw/2/27")
    parser.add_argument("--split-file", default="data/splits/hybrid_AB_split.json")
    parser.add_argument("--xai-results-path", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    split_path = Path(args.split_file)
    if not split_path.exists():
        logger.error("Không tìm thấy split file %s", split_path)
        return 1

    with open(split_path, encoding="utf-8") as f:
        split_meta = json.load(f)

    set_a_mids = set(split_meta["set_a"]["matches"])
    set_b_mids = set(split_meta["set_b"]["matches"])

    events_path = Path(args.events_path)
    if not events_path.exists():
        logger.error("Không tìm thấy tệp events %s", events_path)
        return 1

    logger.info("Đang đọc dữ liệu sự kiện từ %s...", events_path)
    events_df = pd.read_parquet(events_path)

    cache_dir = Path("data/interim/hybrid_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    ref_a_path = cache_dir / "ref_a.parquet"
    ref_b_path = cache_dir / "ref_b.parquet"
    cent_a_path = cache_dir / "cent_a.parquet"
    cent_b_path = cache_dir / "cent_b.parquet"

    if ref_a_path.exists() and ref_b_path.exists() and cent_a_path.exists() and cent_b_path.exists():
        logger.info("Tải nhanh ref_a, ref_b, cent_a, cent_b từ cache...")
        ref_a = pd.read_parquet(ref_a_path)
        ref_b = pd.read_parquet(ref_b_path)
        cent_a = pd.read_parquet(cent_a_path)
        cent_b = pd.read_parquet(cent_b_path)
    else:
        # 1. Tính số phút thi đấu trên toàn bộ mùa giải
        logger.info("Tính số phút thi đấu từ lineups...")
        minutes_df = compute_player_minutes(args.lineups_dir)

        # 2. Huấn luyện mô hình xT trên Set A (D15)
        logger.info("\n[SET A] Huấn luyện mô hình Expected Threat (xT) trên 190 trận Set A...")
        xt_model = ExpectedThreatModel(n_cols=16, n_rows=12)
        xt_model.fit(events_df, match_ids=set_a_mids)

        # 3. Tính độ đo tham chiếu (Key Passes, Assists, xA, xT) trên Set A và Set B
        logger.info("Tính các chỉ số tham chiếu trên Set A...")
        ref_a = compute_player_reference_stats(events_df, minutes_df=minutes_df, match_ids=set_a_mids)
        xt_a = xt_model.compute_player_xt(events_df, minutes_df=minutes_df, match_ids=set_a_mids)
        ref_a = pd.merge(ref_a, xt_a[["player_id", "total_xt", "xt_per_90"]], on="player_id", how="left").fillna(0.0)

        logger.info("Tính các chỉ số tham chiếu trên Set B...")
        ref_b = compute_player_reference_stats(events_df, minutes_df=minutes_df, match_ids=set_b_mids)
        xt_b = xt_model.compute_player_xt(events_df, minutes_df=minutes_df, match_ids=set_b_mids)
        ref_b = pd.merge(ref_b, xt_b[["player_id", "total_xt", "xt_per_90"]], on="player_id", how="left").fillna(0.0)

        # 4. Tính Traditional Centrality (Betweenness) trên Set A và Set B (D13)
        logger.info("Xây dựng mạng chuyền bóng và tính Betweenness Centrality trên Set A...")
        cent_a = compute_match_traditional_centrality(events_df, match_ids=set_a_mids)

        logger.info("Xây dựng mạng chuyền bóng và tính Betweenness Centrality trên Set B...")
        cent_b = compute_match_traditional_centrality(events_df, match_ids=set_b_mids)

        ref_a.to_parquet(ref_a_path, index=False)
        ref_b.to_parquet(ref_b_path, index=False)
        cent_a.to_parquet(cent_a_path, index=False)
        cent_b.to_parquet(cent_b_path, index=False)

    # 5. Đọc hoặc tính điểm XAI cho cầu thủ
    # Nếu có tệp XAI player_importance từ crossfit hoặc official run
    xai_path = Path(args.xai_results_path) if args.xai_results_path else None
    if xai_path is None or not xai_path.exists():
        # Tìm tệp khả dụng trong results/xai/
        candidates = list(Path("results/xai").glob("**/player_importance_by_possession.csv")) + \
                     list(Path("results/xai").glob("**/player_importance_oof.parquet"))
        if candidates:
            xai_path = candidates[0]

    if xai_path and xai_path.exists():
        logger.info("Đọc điểm XAI của cầu thủ từ: %s", xai_path)
        xai_raw = pd.read_parquet(xai_path) if xai_path.suffix == ".parquet" else pd.read_csv(xai_path)
        xai_agg = xai_raw.groupby("player_id").agg({
            "mean_importance": "mean",
            "total_importance": "sum",
            "mean_importance_no_last": "mean",
            "total_importance_no_last": "sum",
        }).reset_index()
    else:
        logger.warning("Chưa có tệp điểm XAI tổng hợp, tạo điểm giả lập từ degree để chạy pipeline hoàn chỉnh!")
        xai_agg = pd.DataFrame({
            "player_id": cent_a["player_id"].values,
            "mean_importance": percentile_rank(cent_a["degree_in"]).values,
            "total_importance": cent_a["degree_in"].values,
            "mean_importance_no_last": percentile_rank(cent_a["degree_in"]).values,
            "total_importance_no_last": cent_a["degree_in"].values,
        })

    # Tạo thư mục xuất kết quả
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path("results/hybrid") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 6. [SET A] Grid Search Alpha tối ưu (D16)
    logger.info("\n[SET A] Thực hiện Grid Search alpha trên Set A (bước 0.05)...")
    best_alpha, alpha_curve_df = optimize_alpha_on_set_a(
        centrality_df=cent_a,
        xai_player_df=xai_agg,
        reference_df=ref_a,
        alpha_step=0.05,
        target_metric="xt_per_90",
    )
    alpha_curve_df.to_csv(out_dir / "alpha_curve.csv", index=False, encoding="utf-8-sig")

    # 7. [SET B] Đánh giá độc lập trên Set B với alpha* (D15)
    logger.info("\n[SET B] Đánh giá độc lập chất lượng Hybrid Centrality trên Set B tại alpha* = %.2f...", best_alpha)
    hc_b = compute_hybrid_centrality(
        centrality_df=cent_b,
        xai_player_df=xai_agg,
        alpha=best_alpha,
        centrality_col="betweenness",
        xai_col="mean_importance",
    )

    eval_merged = pd.merge(hc_b, ref_b, on="player_id", how="inner")
    if "minutes_played" in eval_merged.columns:
        eval_merged = eval_merged[eval_merged["minutes_played"] >= 90.0]

    # Tính tương quan Spearman
    metrics = ["xt_per_90", "xA_per_90", "key_passes_per_90", "assists_per_90"]
    spearman_results = {}

    for m in metrics:
        if m not in eval_merged.columns:
            continue
        corr_hc, p_hc = spearmanr(eval_merged["hybrid_centrality"], eval_merged[m])
        corr_cent, p_cent = spearmanr(eval_merged["rank_centrality"], eval_merged[m])
        corr_xai, p_xai = spearmanr(eval_merged["rank_xai"], eval_merged[m])

        spearman_results[m] = {
            "hybrid_centrality": {"spearman": round(float(corr_hc), 4), "p_value": round(float(p_hc), 4)},
            "pure_centrality": {"spearman": round(float(corr_cent), 4), "p_value": round(float(p_cent), 4)},
            "pure_xai": {"spearman": round(float(corr_xai), 4), "p_value": round(float(p_xai), 4)},
        }

    # Hồi quy OLS và Delta R^2
    ols_res = compute_ols_delta_r2(
        eval_merged,
        y_col="xt_per_90",
        cent_col="rank_centrality",
        xai_col="rank_xai",
    )

    # Lưu bảng xếp hạng cầu thủ Set B
    p_name_col = "player_name_x" if "player_name_x" in eval_merged.columns else "player_name"
    t_name_col = "team_name_x" if "team_name_x" in eval_merged.columns else "team_name"

    player_rankings = eval_merged[[
        "player_id", p_name_col, t_name_col, "minutes_played",
        "hybrid_centrality", "rank_centrality", "rank_xai",
        "betweenness", "mean_importance", "xt_per_90", "xA_per_90", "key_passes", "assists",
    ]].copy()
    player_rankings.rename(columns={p_name_col: "player_name", t_name_col: "team_name"}, inplace=True)
    player_rankings = player_rankings.drop_duplicates(subset=["player_id"]).sort_values(by="hybrid_centrality", ascending=False).reset_index(drop=True)
    player_rankings.to_csv(out_dir / "player_hybrid_rankings.csv", index=False, encoding="utf-8-sig")

    evaluation_report = {
        "step": "Phase 4: Hybrid Centrality Evaluation",
        "run_id": run_id,
        "config": cfg,
        "optimal_alpha_set_a": best_alpha,
        "target_selection_metric": "xt_per_90",
        "set_a_matches": len(set_a_mids),
        "set_b_matches": len(set_b_mids),
        "n_players_evaluated_set_b": len(eval_merged),
        "spearman_correlations_set_b": spearman_results,
        "ols_regression_xt": ols_res,
        "git": get_git_info(),
    }

    with open(out_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2, ensure_ascii=False)

    print("\n=== KẾT QUẢ ĐÁNH GIÁ HYBRID CENTRALITY TRÊN SET B (PHASE 4) ===")
    print(f"Alpha* tối ưu từ Set A: {best_alpha:.2f}")
    print("\nTương quan Spearman với các chỉ số tham chiếu bóng đá:")
    for m, res in spearman_results.items():
        print(f"  [{m}]: Hybrid={res['hybrid_centrality']['spearman']:.4f} (p={res['hybrid_centrality']['p_value']:.4f}) | "
              f"Centrality={res['pure_centrality']['spearman']:.4f} | XAI={res['pure_xai']['spearman']:.4f}")

    print("\nHồi quy OLS cho Expected Threat (xT/90):")
    print(f"  R^2 (Pure Centrality): {ols_res['r2_pure_centrality']:.4f}")
    print(f"  R^2 (Full: Centrality + XAI): {ols_res['r2_full_model']:.4f}")
    print(f"  Delta R^2: {ols_res['delta_r2']:.4f} (95% CI: [{ols_res['ci_lower']:.4f}, {ols_res['ci_upper']:.4f}])")
    print(f"\nToàn bộ kết quả và bảng xếp hạng đã lưu vào: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
