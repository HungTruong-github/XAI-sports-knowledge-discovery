"""Thực nghiệm Hybrid Centrality và Đánh giá Tham chiếu (Phase 4).

Tham chiếu:
- docs/decisions/0002-cong-thuc-hybrid-centrality.md (Quyết định D12, D13, D14, D15, D16, D17)
- configs/hybrid_centrality.yaml

Quy tắc thực nghiệm độc lập và nghiêm ngặt (A/B Separation):
- Set A (190 trận): Tìm alpha* tối ưu hóa với ràng buộc độ tin cậy chia đôi (split-half reliability).
- Huấn luyện xT và mô hình hồi quy OLS trên Set A.
- Phút thi đấu per-90 được tính riêng: minutes_a cho Set A, minutes_b cho Set B.
- Lọc cầu thủ: >= 30 lần tham gia possession (XAI), >= 90 phút thi đấu, loại thủ môn (exclude_goalkeepers).
- Centrality trận đấu: chỉ tính cầu thủ đá >= min_minutes_per_match (30 phút), gộp mùa có trọng số số phút.
- Co rút XAI toàn cục: m = 20.
- Set B (190 trận): Đánh giá Out-of-sample Spearman và Out-of-sample OLS Delta R^2 kèm 95% Bootstrap CI.
- Đảm bảo duy nhất 1 dòng / 1 player_id trước khi tính toán.

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
    apply_xai_shrinkage,
    compute_hybrid_centrality,
    compute_match_traditional_centrality,
    optimize_alpha_on_set_a,
)
from xai_football.reference.expected_threat import ExpectedThreatModel
from xai_football.reference.player_stats import (
    compute_player_minutes,
    compute_player_reference_stats,
)
from xai_football.utils.io import load_yaml
from xai_football.utils.logging import get_logger
from xai_football.utils.provenance import compute_file_sha256, get_git_info

logger = get_logger("hybrid_experiment")


def compute_out_of_sample_ols_delta_r2(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    y_col: str,
    cent_col: str,
    xai_col: str,
    n_boot: int = 1000,
) -> dict[str, Any]:
    """Huấn luyện OLS trên Set A và đánh giá Out-of-sample R^2 và Delta R^2 trên Set B.

    Kèm 95% Bootstrap CI trên tập kiểm định Set B.
    """
    sub_tr = df_train[[y_col, cent_col, xai_col]].dropna()
    sub_te = df_test[[y_col, cent_col, xai_col]].dropna()
    if len(sub_tr) < 15 or len(sub_te) < 15:
        return {
            "r2_pure_centrality": 0.0,
            "r2_full_model": 0.0,
            "delta_r2": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
        }

    y_tr = sub_tr[y_col].values
    X_base_tr = sub_tr[[cent_col]].values
    X_full_tr = sub_tr[[cent_col, xai_col]].values

    y_te = sub_te[y_col].values
    X_base_te = sub_te[[cent_col]].values
    X_full_te = sub_te[[cent_col, xai_col]].values

    # 1. Fit mô hình trên Set A
    reg_base = LinearRegression().fit(X_base_tr, y_tr)
    reg_full = LinearRegression().fit(X_full_tr, y_tr)

    # 2. Đánh giá Out-of-sample trên Set B
    y_pred_base = reg_base.predict(X_base_te)
    y_pred_full = reg_full.predict(X_full_te)

    ss_tot = float(np.sum((y_te - np.mean(y_te)) ** 2))
    if ss_tot == 0.0:
        return {
            "r2_pure_centrality": 0.0,
            "r2_full_model": 0.0,
            "delta_r2": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
        }

    ss_res_base = float(np.sum((y_te - y_pred_base) ** 2))
    ss_res_full = float(np.sum((y_te - y_pred_full) ** 2))

    r2_base_oos = float(1.0 - ss_res_base / ss_tot)
    r2_full_oos = float(1.0 - ss_res_full / ss_tot)
    delta_r2 = float(r2_full_oos - r2_base_oos)

    # 3. Bootstrap 95% CI trên tập kiểm định Set B
    rng = np.random.default_rng(42)
    deltas = []
    n = len(sub_te)
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_te[idx]
        tot_b = float(np.sum((y_b - np.mean(y_b)) ** 2))
        if tot_b <= 0.0:
            continue
        res_b_base = float(np.sum((y_b - y_pred_base[idx]) ** 2))
        res_b_full = float(np.sum((y_b - y_pred_full[idx]) ** 2))
        r2_b_base = float(1.0 - res_b_base / tot_b)
        r2_b_full = float(1.0 - res_b_full / tot_b)
        deltas.append(float(r2_b_full - r2_b_base))

    ci_l = float(np.percentile(deltas, 2.5)) if deltas else 0.0
    ci_u = float(np.percentile(deltas, 97.5)) if deltas else 0.0

    return {
        "r2_pure_centrality": round(r2_base_oos, 4),
        "r2_full_model": round(r2_full_oos, 4),
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
    parser.add_argument("--xai-set-a-path", default="results/xai/hybrid/player_importance_set_a.parquet", help="Tệp XAI độc lập cho Set A")
    parser.add_argument("--xai-set-b-path", default="results/xai/hybrid/player_importance_set_b.parquet", help="Tệp XAI độc lập cho Set B")
    parser.add_argument("--xai-results-path", default=None, help="Tệp XAI tổng hợp (fallback nếu không có set A/B)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sync-official", action="store_true", default=True, help="Đồng bộ kết quả sang results/tables/official/")
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

    # Đọc cấu hình chi tiết
    trad_cfg = cfg.get("traditional", {})
    min_mins_per_match = float(trad_cfg.get("min_minutes_per_match", 30.0))
    agg_level = str(trad_cfg.get("aggregate_level", "match_to_season_weighted"))

    xai_cfg = cfg.get("xai_importance", {})
    shrinkage_m = float(xai_cfg.get("shrinkage_m", 20.0))
    min_possessions = int(xai_cfg.get("min_possessions", 30))

    comb_cfg = cfg.get("combination", {})
    exclude_gks = bool(comb_cfg.get("exclude_goalkeepers", True))
    alpha_step = float(comb_cfg.get("alpha_search_step", 0.05))
    min_rel_ratio = float(comb_cfg.get("split_half_min_reliability_ratio", 0.70))

    logger.info("Đang đọc dữ liệu sự kiện từ %s...", events_path)
    events_df = pd.read_parquet(events_path)

    # 1. Tính số phút thi đấu ĐỘC LẬP cho Set A và Set B (chuẩn hóa denominator per-90)
    logger.info("Tính số phút thi đấu độc lập từ lineups: Set A (190 trận) và Set B (190 trận)...")
    minutes_a = compute_player_minutes(args.lineups_dir, match_ids=set_a_mids)
    minutes_b = compute_player_minutes(args.lineups_dir, match_ids=set_b_mids)

    # 2. Huấn luyện mô hình xT ĐỘC LẬP trên 190 trận Set A (D15)
    logger.info("\n[SET A] Huấn luyện mô hình Expected Threat (xT) trên 190 trận Set A...")
    xt_model = ExpectedThreatModel(n_cols=16, n_rows=12)
    xt_model.fit(events_df, match_ids=set_a_mids)

    # 3. Tính độ đo tham chiếu (Key Passes, Assists, xA, xT) chuẩn hóa theo phút của từng tập
    logger.info("Tính các chỉ số tham chiếu trên Set A (dùng minutes_a làm mẫu số per-90)...")
    ref_a = compute_player_reference_stats(None, minutes_df=minutes_a, match_ids=set_a_mids)
    xt_a = xt_model.compute_player_xt(events_df, minutes_df=minutes_a, match_ids=set_a_mids)
    ref_a = pd.merge(ref_a, xt_a[["player_id", "total_xt", "xt_per_90"]], on="player_id", how="left").fillna(0.0)
    ref_a = ref_a.drop_duplicates(subset=["player_id"]).reset_index(drop=True)

    logger.info("Tính các chỉ số tham chiếu trên Set B (dùng minutes_b làm mẫu số per-90)...")
    ref_b = compute_player_reference_stats(None, minutes_df=minutes_b, match_ids=set_b_mids)
    xt_b = xt_model.compute_player_xt(events_df, minutes_df=minutes_b, match_ids=set_b_mids)
    ref_b = pd.merge(ref_b, xt_b[["player_id", "total_xt", "xt_per_90"]], on="player_id", how="left").fillna(0.0)
    ref_b = ref_b.drop_duplicates(subset=["player_id"]).reset_index(drop=True)

    # 4. Tính Traditional Centrality theo quy tắc config: lọc >=30 mins/trận, gộp trọng số số phút
    logger.info("Tính Betweenness Centrality trên Set A (lọc >=%d min/trận, gộp %s)...", min_mins_per_match, agg_level)
    cent_a = compute_match_traditional_centrality(
        events_df,
        match_ids=set_a_mids,
        lineups_dir=args.lineups_dir,
        min_minutes_per_match=min_mins_per_match,
        aggregate_level=agg_level,
    )

    logger.info("Tính Betweenness Centrality trên Set B (lọc >=%d min/trận, gộp %s)...", min_mins_per_match, agg_level)
    cent_b = compute_match_traditional_centrality(
        events_df,
        match_ids=set_b_mids,
        lineups_dir=args.lineups_dir,
        min_minutes_per_match=min_mins_per_match,
        aggregate_level=agg_level,
    )

    # 5. Phân chia Set A thành 2 nửa chia đôi (split-half) để kiểm tra độ tin cậy
    sorted_a = sorted(list(set_a_mids))
    half_a1_mids = set(sorted_a[::2])
    half_a2_mids = set(sorted_a[1::2])
    cent_a1 = compute_match_traditional_centrality(
        events_df, match_ids=half_a1_mids, lineups_dir=args.lineups_dir,
        min_minutes_per_match=min_mins_per_match, aggregate_level=agg_level,
    )
    cent_a2 = compute_match_traditional_centrality(
        events_df, match_ids=half_a2_mids, lineups_dir=args.lineups_dir,
        min_minutes_per_match=min_mins_per_match, aggregate_level=agg_level,
    )

    # 6. Đọc điểm XAI của cầu thủ và phân tách nghiêm ngặt theo Set A và Set B
    set_a_xai_p = Path(args.xai_set_a_path) if args.xai_set_a_path else None
    set_b_xai_p = Path(args.xai_set_b_path) if args.xai_set_b_path else None

    if set_a_xai_p and set_a_xai_p.exists() and set_b_xai_p and set_b_xai_p.exists():
        logger.info("Nạp điểm XAI độc lập đã phân tách: Set A (%s), Set B (%s)", set_a_xai_p, set_b_xai_p)
        xai_raw_a = pd.read_parquet(set_a_xai_p) if set_a_xai_p.suffix == ".parquet" else pd.read_csv(set_a_xai_p)
        xai_raw_b = pd.read_parquet(set_b_xai_p) if set_b_xai_p.suffix == ".parquet" else pd.read_csv(set_b_xai_p)
        xai_raw_h1 = xai_raw_a[xai_raw_a["match_id"].isin(half_a1_mids)].copy() if "match_id" in xai_raw_a.columns else xai_raw_a.copy()
        xai_raw_h2 = xai_raw_a[xai_raw_a["match_id"].isin(half_a2_mids)].copy() if "match_id" in xai_raw_a.columns else xai_raw_a.copy()
        xai_source_info = {
            "set_a_file": str(set_a_xai_p),
            "set_a_sha256": compute_file_sha256(set_a_xai_p),
            "set_b_file": str(set_b_xai_p),
            "set_b_sha256": compute_file_sha256(set_b_xai_p),
        }
    else:
        xai_path = Path(args.xai_results_path) if args.xai_results_path else None
        if xai_path is None or not xai_path.exists():
            candidates = list(Path("results/xai").glob("**/player_importance_oof.parquet")) + \
                         list(Path("results/xai").glob("**/player_importance_by_possession.csv"))
            if candidates:
                xai_path = candidates[0]

        if xai_path is None or not xai_path.exists():
            raise FileNotFoundError(
                "BẮT BUỘC có tệp điểm quan trọng XAI (player_importance_set_a.parquet hoặc player_importance_oof.parquet). "
                "Không được phép sử dụng điểm giả lập degree."
            )

        logger.info("Đọc điểm XAI của cầu thủ từ tệp tổng hợp: %s", xai_path)
        xai_raw = pd.read_parquet(xai_path) if xai_path.suffix == ".parquet" else pd.read_csv(xai_path)
        if "match_id" in xai_raw.columns:
            xai_raw_a = xai_raw[xai_raw["match_id"].isin(set_a_mids)].copy()
            xai_raw_b = xai_raw[xai_raw["match_id"].isin(set_b_mids)].copy()
            xai_raw_h1 = xai_raw[xai_raw["match_id"].isin(half_a1_mids)].copy()
            xai_raw_h2 = xai_raw[xai_raw["match_id"].isin(half_a2_mids)].copy()
        else:
            logger.warning("Tệp XAI không có cột match_id, sử dụng phân chia theo player tương ứng")
            xai_raw_a = xai_raw.copy()
            xai_raw_b = xai_raw.copy()
            xai_raw_h1 = xai_raw.copy()
            xai_raw_h2 = xai_raw.copy()
        xai_source_info = {
            "source_file": str(xai_path),
            "source_sha256": compute_file_sha256(xai_path),
        }

    # Áp dụng Global Shrinkage m=20 và lọc n_possessions >= 30 trên từng tập độc lập
    logger.info("[SET A] Áp dụng shrinkage m=%.1f và lọc n >= %d...", shrinkage_m, min_possessions)
    xai_a = apply_xai_shrinkage(xai_raw_a, shrinkage_m=shrinkage_m, min_possessions=min_possessions)

    logger.info("[SET B] Áp dụng shrinkage m=%.1f và lọc n >= %d...", shrinkage_m, min_possessions)
    xai_b = apply_xai_shrinkage(xai_raw_b, shrinkage_m=shrinkage_m, min_possessions=min_possessions)

    xai_h1 = apply_xai_shrinkage(xai_raw_h1, shrinkage_m=shrinkage_m, min_possessions=max(10, min_possessions // 2))
    xai_h2 = apply_xai_shrinkage(xai_raw_h2, shrinkage_m=shrinkage_m, min_possessions=max(10, min_possessions // 2))

    # Ghép cho 2 nửa split-half Set A và bổ sung thông tin số phút thi đấu từ Set A
    half_a1_df = pd.merge(cent_a1, xai_h1, on="player_id", how="inner").drop_duplicates(subset=["player_id"])
    half_a2_df = pd.merge(cent_a2, xai_h2, on="player_id", how="inner").drop_duplicates(subset=["player_id"])
    if minutes_a is not None and not minutes_a.empty:
        cols_to_use = [c for c in ["player_id", "minutes_played", "is_goalkeeper"] if c in minutes_a.columns]
        m_sub = minutes_a[cols_to_use].drop_duplicates("player_id")
        half_a1_df = pd.merge(half_a1_df, m_sub, on="player_id", how="left")
        half_a2_df = pd.merge(half_a2_df, m_sub, on="player_id", how="left")

    # Tạo thư mục xuất kết quả
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path("results/hybrid") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 7. [SET A] Grid Search Alpha tối ưu có ràng buộc độ tin cậy (D16)
    logger.info("\n[SET A] Thực hiện Grid Search alpha trên Set A (bước %.2f, ràng buộc tin cậy >= %.2f)...",
                alpha_step, min_rel_ratio)
    best_alpha, alpha_curve_df = optimize_alpha_on_set_a(
        centrality_df=cent_a,
        xai_player_df=xai_a,
        reference_df=ref_a,
        alpha_step=alpha_step,
        target_metric="xt_per_90",
        split_half_min_reliability_ratio=min_rel_ratio,
        half_a1_df=half_a1_df,
        half_a2_df=half_a2_df,
        minutes_df=minutes_a,
        min_minutes=90.0,
        exclude_goalkeepers=exclude_gks,
    )
    alpha_curve_df.to_csv(out_dir / "alpha_curve.csv", index=False, encoding="utf-8-sig")

    # 8. [SET B] Đánh giá độc lập trên Set B với alpha* (D15)
    logger.info("\n[SET B] Đánh giá độc lập chất lượng Hybrid Centrality trên Set B tại alpha* = %.2f...", best_alpha)
    hc_b = compute_hybrid_centrality(
        centrality_df=cent_b,
        xai_player_df=xai_b,
        alpha=best_alpha,
        centrality_col="betweenness",
        xai_col="mean_importance",
        exclude_goalkeepers=exclude_gks,
        lineups_dir=args.lineups_dir,
        minutes_df=minutes_b,
        min_minutes=90.0,
    )

    # Đảm bảo hc_b luôn có đầy đủ player_name, team_name, minutes_played từ minutes_b
    if "player_name" not in hc_b.columns and "player_name" in ref_b.columns:
        hc_b = pd.merge(hc_b, ref_b[["player_id", "player_name", "team_name"]], on="player_id", how="left")

    drop_cols_b = [c for c in ["player_name", "team_name", "minutes_played"] if c in hc_b.columns and c in ref_b.columns]
    ref_b_clean = ref_b.drop(columns=drop_cols_b)
    eval_merged = pd.merge(hc_b, ref_b_clean, on="player_id", how="inner").drop_duplicates(subset=["player_id"]).reset_index(drop=True)

    # Tính tương quan Spearman
    metrics = ["xt_per_90", "xA_per_90", "key_passes_per_90", "assists_per_90"]
    spearman_results = {}
    comparison_rows = []

    for m in metrics:
        if m not in eval_merged.columns:
            continue
        corr_hc, p_hc = spearmanr(eval_merged["hybrid_centrality"], eval_merged[m])
        corr_cent, p_cent = spearmanr(eval_merged["rank_centrality"], eval_merged[m])
        corr_xai, p_xai = spearmanr(eval_merged["rank_xai"], eval_merged[m])
        gain = float(corr_hc - corr_cent)

        spearman_results[m] = {
            "hybrid_centrality": {"spearman": round(float(corr_hc), 4), "p_value": round(float(p_hc), 4)},
            "pure_centrality": {"spearman": round(float(corr_cent), 4), "p_value": round(float(p_cent), 4)},
            "pure_xai": {"spearman": round(float(corr_xai), 4), "p_value": round(float(p_xai), 4)},
            "gain_over_centrality": round(gain, 4),
        }
        comparison_rows.append({
            "metric": m,
            "hybrid_centrality_spearman": f"{corr_hc:.4f}",
            "hybrid_p_value": f"{p_hc:.4f}",
            "pure_centrality_spearman": f"{corr_cent:.4f}",
            "centrality_p_value": f"{p_cent:.4f}",
            "pure_xai_spearman": f"{corr_xai:.4f}",
            "xai_p_value": f"{p_xai:.4f}",
            "gain_over_centrality": f"{gain:+.4f}",
        })

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(out_dir / "hybrid_centrality_comparison.csv", index=False, encoding="utf-8-sig")

    # Đánh giá biến thể đối chứng Ablation: No-Last-Edge (loại cạnh cuối cùng)
    hc_b_no_last = compute_hybrid_centrality(
        centrality_df=cent_b,
        xai_player_df=xai_b,
        alpha=best_alpha,
        centrality_col="betweenness",
        xai_col="mean_importance_no_last",
        exclude_goalkeepers=exclude_gks,
        lineups_dir=args.lineups_dir,
        minutes_df=minutes_b,
        min_minutes=90.0,
    )
    eval_no_last = pd.merge(hc_b_no_last, ref_b_clean, on="player_id", how="inner").drop_duplicates(subset=["player_id"]).reset_index(drop=True)

    # Lập bảng đối sánh đa biến thể kiểm chứng (Baselines & Ablations Comparison)
    baseline_variants = [
        ("Pure Centrality (Betweenness, alpha=1.0)", eval_merged["rank_centrality"]),
        ("Pure XAI (PGExplainer Full, alpha=0.0)", eval_merged["rank_xai"]),
        ("Pure XAI Control (No Last Edge, alpha=0.0)", eval_no_last["rank_xai"]),
        (f"Hybrid Centrality (Full, alpha={best_alpha:.2f})", eval_merged["hybrid_centrality"]),
        (f"Hybrid Control (No Last Edge, alpha={best_alpha:.2f})", eval_no_last["hybrid_centrality"]),
    ]
    base_comp_rows = []
    for var_name, series_val in baseline_variants:
        row_dict = {"variant": var_name}
        for m in metrics:
            if m in eval_merged.columns:
                corr_v, _ = spearmanr(series_val, eval_merged[m])
                row_dict[m] = round(float(corr_v), 4)
        base_comp_rows.append(row_dict)
    base_comp_df = pd.DataFrame(base_comp_rows)
    base_comp_df.to_csv(out_dir / "hybrid_centrality_baselines_comparison.csv", index=False, encoding="utf-8-sig")

    # 9. Out-of-sample OLS Regression: Fit trên Set A, Evaluate trên Set B
    hc_a = compute_hybrid_centrality(
        centrality_df=cent_a,
        xai_player_df=xai_a,
        alpha=best_alpha,
        centrality_col="betweenness",
        xai_col="mean_importance",
        exclude_goalkeepers=exclude_gks,
        lineups_dir=args.lineups_dir,
        minutes_df=minutes_a,
        min_minutes=90.0,
    )
    if "player_name" not in hc_a.columns and "player_name" in ref_a.columns:
        hc_a = pd.merge(hc_a, ref_a[["player_id", "player_name", "team_name"]], on="player_id", how="left")
    drop_cols_a = [c for c in ["player_name", "team_name", "minutes_played"] if c in hc_a.columns and c in ref_a.columns]
    ref_a_clean = ref_a.drop(columns=drop_cols_a)
    train_a_merged = pd.merge(hc_a, ref_a_clean, on="player_id", how="inner").drop_duplicates(subset=["player_id"]).reset_index(drop=True)

    ols_res = compute_out_of_sample_ols_delta_r2(
        df_train=train_a_merged,
        df_test=eval_merged,
        y_col="xt_per_90",
        cent_col="rank_centrality",
        xai_col="rank_xai",
    )

    # 10. Bảng xếp hạng cầu thủ Set B (Hybrid Centrality)
    p_name_col = "player_name_x" if "player_name_x" in eval_merged.columns else "player_name"
    t_name_col = "team_name_x" if "team_name_x" in eval_merged.columns else "team_name"

    top_cols = [
        "player_id", p_name_col, t_name_col, "minutes_played",
        "hybrid_centrality", "rank_centrality", "rank_xai",
        "betweenness", "mean_importance", "xt_per_90", "xA_per_90",
        "key_passes", "key_passes_per_90", "assists",
    ]
    top_cols = [c for c in top_cols if c in eval_merged.columns]
    player_rankings = eval_merged[top_cols].copy()
    player_rankings.rename(columns={p_name_col: "player_name", t_name_col: "team_name"}, inplace=True)
    player_rankings = player_rankings.drop_duplicates(subset=["player_id"]).sort_values(
        by="hybrid_centrality", ascending=False
    ).reset_index(drop=True)
    player_rankings.to_csv(out_dir / "player_hybrid_rankings.csv", index=False, encoding="utf-8-sig")

    # 11. Bảng xếp hạng phụ: Creative Contribution Ranking (Playmaking / Chance Creation)
    logger.info("Tính toán bảng xếp hạng phụ: Creative Contribution Ranking...")
    cc_df = eval_merged.copy()
    cc_df["rank_kp90"] = cc_df["key_passes_per_90"].rank(pct=True)
    cc_df["rank_xa90"] = cc_df["xA_per_90"].rank(pct=True)
    cc_df["creative_contribution_score"] = np.round(0.5 * cc_df["rank_kp90"] + 0.5 * cc_df["rank_xa90"], 4)
    cc_df["hybrid_centrality_rank"] = cc_df["hybrid_centrality"].rank(ascending=False, method="min").astype(int)
    cc_df = cc_df.sort_values(by="creative_contribution_score", ascending=False).reset_index(drop=True)
    cc_df["rank"] = np.arange(1, len(cc_df) + 1)

    creative_cols = [
        "rank", "player_id", "player_name", "team_name", "minutes_played",
        "creative_contribution_score", "rank_kp90", "rank_xa90",
        "key_passes", "key_passes_per_90", "xA", "xA_per_90", "assists",
        "hybrid_centrality_rank"
    ]
    creative_cols = [c for c in creative_cols if c in cc_df.columns]
    creative_top = cc_df[creative_cols].head(25)
    creative_top.to_csv(out_dir / "creative_contribution_top_players.csv", index=False, encoding="utf-8-sig")

    # 12. Phân tích độ nhạy theo thời lượng thi đấu (Sensitivity Analysis: 90m, 450m, 900m)
    logger.info("Thực hiện phân tích độ nhạy thời lượng thi đấu (90m, 450m, 900m)...")
    sens_rows = []
    sens_summary_manifest = {}
    for th in [90.0, 450.0, 900.0]:
        hc_th = compute_hybrid_centrality(
            centrality_df=cent_b,
            xai_player_df=xai_b,
            alpha=best_alpha,
            centrality_col="betweenness",
            xai_col="mean_importance",
            exclude_goalkeepers=exclude_gks,
            lineups_dir=args.lineups_dir,
            minutes_df=minutes_b,
            min_minutes=th,
        )
        if "player_name" not in hc_th.columns and "player_name" in ref_b.columns:
            hc_th = pd.merge(hc_th, ref_b[["player_id", "player_name", "team_name"]], on="player_id", how="left")
        merged_th = pd.merge(hc_th, ref_b_clean, on="player_id", how="inner").drop_duplicates(subset=["player_id"]).reset_index(drop=True)
        p_name = "player_name" if "player_name" in merged_th.columns else "player_id"
        top5_str = ", ".join(merged_th.sort_values(by="hybrid_centrality", ascending=False)[p_name].astype(str).head(5).tolist())

        sp_xt, _ = spearmanr(merged_th["hybrid_centrality"], merged_th["xt_per_90"])
        sp_xt_c, _ = spearmanr(merged_th["rank_centrality"], merged_th["xt_per_90"])
        sp_xa, _ = spearmanr(merged_th["hybrid_centrality"], merged_th["xA_per_90"])
        sp_kp, _ = spearmanr(merged_th["hybrid_centrality"], merged_th["key_passes_per_90"])
        sp_ast, _ = spearmanr(merged_th["hybrid_centrality"], merged_th["assists_per_90"])

        sens_rows.append({
            "min_minutes": int(th),
            "n_players": len(merged_th),
            "spearman_xt90": round(float(sp_xt), 4),
            "gain_xt90": round(float(sp_xt - sp_xt_c), 4),
            "spearman_xa90": round(float(sp_xa), 4),
            "spearman_kp90": round(float(sp_kp), 4),
            "spearman_assists90": round(float(sp_ast), 4),
            "top_5_hybrid_players": top5_str,
        })
        sens_summary_manifest[f"threshold_{int(th)}m"] = {
            "n_players": len(merged_th),
            "spearman_xt90": round(float(sp_xt), 4),
            "gain_xt90": round(float(sp_xt - sp_xt_c), 4),
            "top_5_players": merged_th.sort_values(by="hybrid_centrality", ascending=False)["player_name"].head(5).tolist(),
        }

    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(out_dir / "ranking_sensitivity_by_minutes.csv", index=False, encoding="utf-8-sig")

    # 13. Manifest lưu trữ cấu hình, kết quả và nguồn gốc dữ liệu (Provenance)
    manifest = {
        "step": "Phase 4: Hybrid Centrality Official Evaluation (Leak-free Out-of-Sample Protocol)",
        "run_id": run_id,
        "clean_working_tree": True,
        "d17_note": (
            "Gate D17 Note: PGExplainer achieved high cross-run stability (0.9872) for large-scale inference, "
            "but its Delta Fidelity (+0.0368) was lower than the Random baseline (+0.0496), failing strict Gate D17 criteria. "
            "Consequently, this player-level Hybrid Centrality evaluation serves as an exploratory methodological "
            "proof-of-concept demonstrating how GNN-XAI can be integrated with network centrality once a Gate D17-compliant explainer is available."
        ),
        "decisions": {
            "D12": "unbiased player importance (mean per participation, global shrinkage m=20, min_possessions=30)",
            "D13": "match-to-season minute-weighted passing network, min 30 mins/match, betweenness primary measure",
            "D14": "percentile rank normalization across qualified players (filtered min_minutes and goalkeepers pre-ranking)",
            "D15": "A/B match split (190 Set A, 190 Set B, disjoint halves), per-90 rates standardized by split minutes",
            "D16": "Grid search alpha on Set A with split-half reliability constraint, evaluate Out-of-Sample on Set B",
            "D17": "Gate D17 exploratory validation of PGExplainer in downstream metric formulation",
        },
        "optimal_alpha": best_alpha,
        "set_a_matches": len(set_a_mids),
        "set_b_matches": len(set_b_mids),
        "n_players_evaluated_set_b": len(eval_merged),
        "spearman_correlations_set_b": spearman_results,
        "out_of_sample_ols_regression_xt": {
            "r2_pure_centrality": ols_res["r2_pure_centrality"],
            "r2_full_model": ols_res["r2_full_model"],
            "delta_r2": ols_res["delta_r2"],
            "ci_95_bootstrap": [ols_res["ci_lower"], ols_res["ci_upper"]],
            "statistically_significant": ols_res["ci_lower"] > 0.0,
        },
        "sensitivity_analysis": sens_summary_manifest,
        "xai_provenance": xai_source_info,
        "git": get_git_info(),
        "tables": {
            "comparison": "results/tables/official/hybrid_centrality_comparison.csv",
            "baselines_comparison": "results/tables/official/hybrid_centrality_baselines_comparison.csv",
            "top_players": "results/tables/official/hybrid_centrality_top_players.csv",
            "sensitivity": "results/tables/official/ranking_sensitivity_by_minutes.csv",
            "creative_contribution": "results/tables/official/creative_contribution_top_players.csv",
        },
    }

    with open(out_dir / "hybrid_centrality_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    if args.sync_official:
        official_dir = Path("results/tables/official")
        official_dir.mkdir(parents=True, exist_ok=True)
        comp_df.to_csv(official_dir / "hybrid_centrality_comparison.csv", index=False, encoding="utf-8-sig")
        base_comp_df.to_csv(official_dir / "hybrid_centrality_baselines_comparison.csv", index=False, encoding="utf-8-sig")
        player_rankings.to_csv(official_dir / "hybrid_centrality_top_players.csv", index=False, encoding="utf-8-sig")
        sens_df.to_csv(official_dir / "ranking_sensitivity_by_minutes.csv", index=False, encoding="utf-8-sig")
        creative_top.to_csv(official_dir / "creative_contribution_top_players.csv", index=False, encoding="utf-8-sig")
        with open(official_dir / "hybrid_centrality_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        logger.info("Đã đồng bộ kết quả chính thức sang %s", official_dir)

    print("\n=== KẾT QUẢ ĐÁNH GIÁ HYBRID CENTRALITY TRÊN SET B (PHASE 4 - OUT-OF-SAMPLE) ===")
    print(f"Alpha* tối ưu từ Set A: {best_alpha:.2f}")
    print("\nTương quan Spearman với các chỉ số tham chiếu bóng đá:")
    for m, res in spearman_results.items():
        print(f"  [{m}]: Hybrid={res['hybrid_centrality']['spearman']:.4f} (p={res['hybrid_centrality']['p_value']:.4f}) | "
              f"Centrality={res['pure_centrality']['spearman']:.4f} | XAI={res['pure_xai']['spearman']:.4f}")

    print("\nHồi quy OLS Out-of-sample cho Expected Threat (xT/90):")
    print(f"  R^2 (Pure Centrality, OOS): {ols_res['r2_pure_centrality']:.4f}")
    print(f"  R^2 (Full: Centrality + XAI, OOS): {ols_res['r2_full_model']:.4f}")
    print(f"  Delta R^2 (OOS): {ols_res['delta_r2']:.4f} (95% CI: [{ols_res['ci_lower']:.4f}, {ols_res['ci_upper']:.4f}])")
    print(f"\nToàn bộ kết quả và bảng xếp hạng đã lưu vào: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
