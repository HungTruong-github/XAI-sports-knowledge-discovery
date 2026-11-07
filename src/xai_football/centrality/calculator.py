"""Module tính toán và tối ưu hóa chỉ số Hybrid Centrality (Phase 4).

Tham chiếu:
- docs/decisions/0002-cong-thuc-hybrid-centrality.md (Quyết định D12, D13, D14, D15, D16, D17)
- configs/hybrid_centrality.yaml

Công thức:
    HC(p; alpha) = alpha * PercentileRank(C(p)) + (1 - alpha) * PercentileRank(X(p))
Trong đó:
    - C(p): Độ đo Betweenness centrality truyền thống trên đồ thị chuyền bóng cấp trận đấu (D13),
            lọc cầu thủ đá >= min_minutes_per_match (30 phút) và gộp theo mùa có trọng số số phút.
    - X(p): Điểm XAI có co rút toàn cục (Global Shrinkage m=20) và lọc n_possessions >= 30 (D12).
    - PercentileRank: Chuẩn hoá thứ hạng phân vị trên đoạn [0, 1] (D14).
    - exclude_goalkeepers: Loại thủ môn khỏi bảng xếp hạng.
    - alpha*: Tối ưu hoá trên tập Set A với ràng buộc độ tin cậy chia đôi (D16) và kiểm định độc lập trên Set B (D15).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from xai_football.reference.player_stats import compute_player_minutes
from xai_football.utils.logging import get_logger

logger = get_logger("hybrid_centrality")


def compute_match_traditional_centrality(
    events_df: pd.DataFrame,
    match_ids: set[int] | None = None,
    lineups_dir: str | Path | None = None,
    min_minutes_per_match: float = 30.0,
    aggregate_level: str = "match_to_season_weighted",
) -> pd.DataFrame:
    """Xây dựng mạng lưới chuyền bóng cấp trận đấu và tính các độ đo Centrality (D13).

    Quy tắc cấu hình:
        - min_minutes_per_match: Chỉ tính trận cầu thủ đá từ min_minutes_per_match phút (mặc định 30).
        - aggregate_level: Gộp theo mùa bằng trung bình có trọng số số phút ("match_to_season_weighted")
          hoặc trung bình đơn giản ("mean").
    """
    df = events_df
    if match_ids is not None:
        df = df[df["match_id"].isin(match_ids)]

    # Lấy thông tin số phút từng trận nếu có lineups_dir
    match_minutes_map: dict[tuple[int, int], float] = {}
    valid_players_by_match: dict[int, set[int]] | None = None

    if lineups_dir is not None and Path(lineups_dir).exists() and min_minutes_per_match > 0:
        match_mins_df = compute_player_minutes(
            lineups_dir=lineups_dir, match_ids=match_ids, return_match_level=True
        )
        if not match_mins_df.empty:
            for _, r in match_mins_df.iterrows():
                mid = int(r["match_id"])
                pid = int(r["player_id"])
                mins = float(r["minutes"])
                match_minutes_map[(mid, pid)] = mins

            valid_sub = match_mins_df[match_mins_df["minutes"] >= min_minutes_per_match]
            valid_players_by_match = valid_sub.groupby("match_id")["player_id"].apply(set).to_dict()

    # Lọc các đường chuyền thành công
    passes = df[
        (df["event_type"] == "Pass")
        & (df["pass_outcome"].isna())
        & (df["player_id"].notna())
        & (df["pass_recipient_id"].notna())
    ].copy()

    if passes.empty:
        return pd.DataFrame(columns=[
            "player_id", "player_name", "betweenness", "degree_in", "degree_out",
            "closeness", "eigenvector", "matches_played"
        ])

    has_team = "team_id" in passes.columns
    group_cols = ["match_id", "team_id"] if has_team else ["match_id"]
    match_grouped = passes.groupby(group_cols)
    match_records = []

    for grp_key, m_passes in match_grouped:
        if has_team:
            mid, tid = grp_key
            mid = int(mid)
            tid = int(tid) if pd.notna(tid) else None
        else:
            mid = int(grp_key[0]) if isinstance(grp_key, (tuple, list)) else int(grp_key)
            tid = None

        # Nếu có thông tin lọc số phút, chỉ giữ đường chuyền giữa các cầu thủ đủ điều kiện
        if valid_players_by_match is not None and mid in valid_players_by_match:
            valid_set = valid_players_by_match[mid]
            m_passes = m_passes[
                m_passes["player_id"].isin(valid_set) & m_passes["pass_recipient_id"].isin(valid_set)
            ]

        if m_passes.empty:
            continue

        # Xây dựng đồ thị có hướng có trọng số (số đường chuyền u -> v) theo từng đội
        G = nx.DiGraph()
        edge_weights = m_passes.groupby(["player_id", "pass_recipient_id"]).size()

        for (u, v), w in edge_weights.items():
            G.add_edge(int(u), int(v), weight=float(w))

        if len(G) < 3:
            continue

        # Độ đo Betweenness (dùng nghịch đảo trọng số làm khoảng cách)
        for _, _, d in G.edges(data=True):
            d["distance"] = 1.0 / max(1.0, d["weight"])

        try:
            betw = nx.betweenness_centrality(G, weight="distance", normalized=True)
        except Exception:
            betw = {n: 0.0 for n in G.nodes()}

        deg_in = dict(G.in_degree(weight="weight"))
        deg_out = dict(G.out_degree(weight="weight"))

        try:
            clos = nx.closeness_centrality(G, distance="distance")
        except Exception:
            clos = {n: 0.0 for n in G.nodes()}

        try:
            eig = nx.eigenvector_centrality(G, max_iter=200, weight="weight")
        except Exception:
            eig = {n: 0.0 for n in G.nodes()}

        for n in G.nodes():
            pid = int(n)
            mins = match_minutes_map.get((mid, pid))
            if mins is None:
                if match_minutes_map:
                    p_events = m_passes[(m_passes["player_id"] == pid) | (m_passes["pass_recipient_id"] == pid)]
                    if not p_events.empty and "minute" in p_events.columns:
                        mins = max(1.0, float(p_events["minute"].max() - p_events["minute"].min() + 1.0))
                    else:
                        mins = 1.0
                    logger.warning(
                        "Cầu thủ %s trong trận %s không có trong lineups minutes map; ước tính %s phút (không gán 90.0 mặc định)",
                        pid, mid, mins
                    )
                else:
                    mins = 90.0

            rec = {
                "match_id": mid,
                "player_id": pid,
                "minutes": mins,
                "betweenness": float(betw.get(n, 0.0)),
                "degree_in": float(deg_in.get(n, 0.0)),
                "degree_out": float(deg_out.get(n, 0.0)),
                "closeness": float(clos.get(n, 0.0)),
                "eigenvector": float(eig.get(n, 0.0)),
            }
            if tid is not None:
                rec["team_id"] = tid
            match_records.append(rec)

    if not match_records:
        return pd.DataFrame()

    m_df = pd.DataFrame(match_records)

    # Lấy tên cầu thủ mới nhất
    player_names = (
        df[["player_id", "player_name"]]
        .dropna()
        .drop_duplicates("player_id")
        .set_index("player_id")["player_name"]
        .to_dict()
    )

    metrics_cols = ["betweenness", "degree_in", "degree_out", "closeness", "eigenvector"]

    if aggregate_level == "match_to_season_weighted" and "minutes" in m_df.columns:
        # Tính trung bình có trọng số số phút thi đấu: sum(metric * minutes) / sum(minutes)
        agg_rows = []
        for pid, group in m_df.groupby("player_id"):
            tot_mins = group["minutes"].sum()
            row_dict: dict[str, Any] = {
                "player_id": pid,
                "matches_played": group["match_id"].nunique(),
            }
            if tot_mins > 0:
                for col in metrics_cols:
                    row_dict[col] = float((group[col] * group["minutes"]).sum() / tot_mins)
            else:
                for col in metrics_cols:
                    row_dict[col] = float(group[col].mean())
            agg_rows.append(row_dict)
        agg = pd.DataFrame(agg_rows)
    else:
        # Tổng hợp trung bình đơn giản trên mỗi trận thi đấu
        agg_dict: dict[str, Any] = {col: "mean" for col in metrics_cols}
        agg_dict["match_id"] = "nunique"
        agg = m_df.groupby("player_id").agg(agg_dict).reset_index()
        agg.rename(columns={"match_id": "matches_played"}, inplace=True)

    agg["player_name"] = agg["player_id"].map(player_names)
    return agg.drop_duplicates(subset=["player_id"]).reset_index(drop=True)


def apply_xai_shrinkage(
    player_xai_df: pd.DataFrame,
    shrinkage_m: float = 20.0,
    min_possessions: int = 30,
    importance_col: str = "mean_importance",
    count_col: str | None = None,
) -> pd.DataFrame:
    """Áp dụng co rút toàn cục (Global Shrinkage) và lọc tối thiểu số lần tham gia (D12).

    Công thức:
        I_shrunk(i) = (n_i * bar_I_i + m * mu_global) / (n_i + m)
    Trong đó:
        - n_i: Số lần tham gia possession (n_i >= min_possessions).
        - bar_I_i: Điểm quan trọng trung bình gốc.
        - mu_global: Giá trị trung bình toàn cục của các cầu thủ.
        - m: Tham số co rút (shrinkage_m = 20).
    """
    df = player_xai_df.copy()
    if df.empty:
        return df

    # Nếu DataFrame ở cấp possession (chứa nhiều dòng trên 1 player_id), gom nhóm
    if df["player_id"].duplicated().any() or count_col is None:
        agg_rules: dict[str, Any] = {importance_col: ["mean", "count"]}
        if "total_importance" in df.columns:
            agg_rules["total_importance"] = "sum"
        if "mean_importance_no_last" in df.columns:
            agg_rules["mean_importance_no_last"] = "mean"
        if "total_importance_no_last" in df.columns:
            agg_rules["total_importance_no_last"] = "sum"

        grouped = df.groupby("player_id").agg(agg_rules)
        # Làm phẳng multi-index columns
        grouped.columns = [
            f"{c[0]}_{c[1]}" if c[1] != "" else c[0]
            for c in grouped.columns
        ]
        grouped = grouped.reset_index()

        # Chuẩn hóa tên cột
        mean_col = f"{importance_col}_mean"
        n_col = f"{importance_col}_count"
        grouped.rename(columns={mean_col: "raw_mean_importance", n_col: "n_possessions"}, inplace=True)
        if "mean_importance_no_last_mean" in grouped.columns:
            grouped.rename(columns={"mean_importance_no_last_mean": "raw_mean_no_last"}, inplace=True)
        df = grouped
    else:
        n_col = count_col if count_col in df.columns else "n_possessions"
        if n_col not in df.columns:
            df["n_possessions"] = 30
            n_col = "n_possessions"
        df["raw_mean_importance"] = df[importance_col]
        if "mean_importance_no_last" in df.columns:
            df["raw_mean_no_last"] = df["mean_importance_no_last"]

    # Lọc tối thiểu số lần tham gia possession
    filtered = df[df["n_possessions"] >= min_possessions].copy()
    if filtered.empty:
        logger.warning("Không có cầu thủ nào thỏa mãn n_possessions >= %d", min_possessions)
        return df

    # Tính mu_global
    mu_global = float(filtered["raw_mean_importance"].mean())
    n_i = filtered["n_possessions"].values
    bar_i = filtered["raw_mean_importance"].values

    shrunk = (n_i * bar_i + shrinkage_m * mu_global) / (n_i + shrinkage_m)
    filtered["shrunk_importance"] = shrunk
    filtered["mean_importance"] = shrunk

    if "raw_mean_no_last" in filtered.columns:
        mu_no_last = float(filtered["raw_mean_no_last"].mean())
        bar_no_last = filtered["raw_mean_no_last"].values
        shrunk_no_last = (n_i * bar_no_last + shrinkage_m * mu_no_last) / (n_i + shrinkage_m)
        filtered["shrunk_importance_no_last"] = shrunk_no_last
        filtered["mean_importance_no_last"] = shrunk_no_last

    logger.info(
        "Đã áp dụng shrinkage m=%.1f (mu_global=%.4f) trên %d cầu thủ (lọc từ %d cầu thủ)",
        shrinkage_m, mu_global, len(filtered), len(df)
    )
    return filtered.drop_duplicates(subset=["player_id"]).reset_index(drop=True)


def percentile_rank(series: pd.Series) -> pd.Series:
    """Chuẩn hoá phân vị trên đoạn [0, 1] theo Quyết định D14."""
    n = len(series)
    if n <= 1:
        return pd.Series(0.5, index=series.index)
    ranks = rankdata(series.values, method="average")
    return pd.Series((ranks - 1.0) / (n - 1.0), index=series.index)


def compute_hybrid_centrality(
    centrality_df: pd.DataFrame,
    xai_player_df: pd.DataFrame,
    alpha: float = 0.5,
    centrality_col: str = "betweenness",
    xai_col: str = "mean_importance",
    exclude_goalkeepers: bool = True,
    lineups_dir: str | Path | None = None,
    min_minutes: float = 90.0,
    minutes_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Tính chỉ số Hybrid Centrality kết hợp giữa Network Centrality và XAI (D12, D14).

    Quy trình chuẩn hóa chuẩn mực:
    1. Ghép centrality và XAI theo duy nhất player_id.
    2. Lọc cầu thủ đủ điều kiện (loại thủ môn và lọc minutes_played >= min_minutes)
       TRƯỚC KHI tính Percentile Rank để đảm bảo không bị méo mó phân phối phân vị.
    3. Chuẩn hóa Percentile Rank trên đoạn [0, 1] trên tập cầu thủ đã lọc.
    4. Kết hợp tuyến tính: HC = alpha * Rank_Centrality + (1 - alpha) * Rank_XAI.
    """
    cent = centrality_df.drop_duplicates(subset=["player_id"]).copy()
    xai = xai_player_df.drop_duplicates(subset=["player_id"]).copy()

    merged = pd.merge(cent, xai, on="player_id", how="inner")
    if merged.empty:
        return pd.DataFrame()

    # Ghép thông tin số phút và thủ môn nếu có
    if minutes_df is not None and not minutes_df.empty:
        cols_to_use = [c for c in ["player_id", "player_name", "team_name", "minutes_played", "is_goalkeeper"] if c in minutes_df.columns]
        m_sub = minutes_df[cols_to_use].drop_duplicates("player_id")
        merged = pd.merge(merged, m_sub, on="player_id", how="left")

    # Loại bỏ thủ môn TRƯỚC KHI tính percentile nếu được yêu cầu
    if exclude_goalkeepers:
        if "is_goalkeeper" not in merged.columns and lineups_dir is not None and Path(lineups_dir).exists():
            gk_df = compute_player_minutes(lineups_dir)[["player_id", "is_goalkeeper"]].drop_duplicates("player_id")
            merged = pd.merge(merged, gk_df, on="player_id", how="left")

        if "is_goalkeeper" in merged.columns:
            merged = merged[~merged["is_goalkeeper"].fillna(False)].copy()

    # Lọc số phút thi đấu tối thiểu TRƯỚC KHI tính percentile
    if min_minutes > 0.0 and "minutes_played" in merged.columns:
        merged = merged[merged["minutes_played"] >= min_minutes].copy()

    if merged.empty:
        return pd.DataFrame()

    c_vals = merged[centrality_col].fillna(0.0)
    x_vals = merged[xai_col].fillna(0.0)

    merged["rank_centrality"] = percentile_rank(c_vals)
    merged["rank_xai"] = percentile_rank(x_vals)

    merged["alpha"] = alpha
    merged["hybrid_centrality"] = alpha * merged["rank_centrality"] + (1.0 - alpha) * merged["rank_xai"]

    return merged.drop_duplicates(subset=["player_id"]).sort_values(
        by="hybrid_centrality", ascending=False
    ).reset_index(drop=True)


def optimize_alpha_on_set_a(
    centrality_df: pd.DataFrame,
    xai_player_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    alpha_step: float = 0.05,
    target_metric: str = "xt_per_90",
    split_half_min_reliability_ratio: float = 0.70,
    min_minutes: float = 90.0,
    exclude_goalkeepers: bool = True,
    lineups_dir: str | Path | None = None,
    half_a1_df: pd.DataFrame | None = None,
    half_a2_df: pd.DataFrame | None = None,
    minutes_df: pd.DataFrame | None = None,
) -> tuple[float, pd.DataFrame]:
    """Quét grid search alpha in [0, 1] trên Set A để tìm alpha* tối ưu (Quyết định D16).

    Có áp dụng lọc thủ môn và phút thi đấu trước khi tính percentile,
    và ràng buộc độ tin cậy chia đôi (split-half reliability constraint):
        r_rel(alpha) >= split_half_min_reliability_ratio * max_alpha(r_rel(alpha))
    """
    merged_base = pd.merge(
        centrality_df.drop_duplicates(subset=["player_id"]),
        xai_player_df.drop_duplicates(subset=["player_id"]),
        on="player_id",
        how="inner",
    )
    merged = pd.merge(
        merged_base,
        reference_df.drop_duplicates(subset=["player_id"]),
        on="player_id",
        how="inner",
    )

    if minutes_df is not None and not minutes_df.empty and "minutes_played" not in merged.columns:
        cols_to_use = [c for c in ["player_id", "minutes_played", "is_goalkeeper"] if c in minutes_df.columns]
        m_sub = minutes_df[cols_to_use].drop_duplicates("player_id")
        merged = pd.merge(merged, m_sub, on="player_id", how="left")

    # Loại bỏ thủ môn nhất quán trước khi tính percentile
    if exclude_goalkeepers:
        if "is_goalkeeper" not in merged.columns and lineups_dir is not None and Path(lineups_dir).exists():
            gk_df = compute_player_minutes(lineups_dir)[["player_id", "is_goalkeeper"]].drop_duplicates("player_id")
            merged = pd.merge(merged, gk_df, on="player_id", how="left")
        if "is_goalkeeper" in merged.columns:
            merged = merged[~merged["is_goalkeeper"].fillna(False)].copy()

    # Lọc cầu thủ có >= min_minutes phút thi đấu TRƯỚC KHI tính percentile
    if min_minutes > 0.0 and "minutes_played" in merged.columns:
        merged = merged[merged["minutes_played"] >= min_minutes].copy()

    if len(merged) < 10 or target_metric not in merged.columns:
        logger.warning("Không đủ dữ liệu trên Set A để tối ưu alpha! Trả về mặc định 0.5")
        return 0.5, pd.DataFrame()

    c_rank = percentile_rank(merged["betweenness"].fillna(0.0))
    x_rank = percentile_rank(merged["mean_importance"].fillna(0.0))
    y_target = merged[target_metric].values

    # Tính độ tin cậy chia đôi nếu có dữ liệu 2 nửa Set A
    has_split_halves = half_a1_df is not None and half_a2_df is not None
    half_reliabilities: dict[float, float] = {}

    alphas = np.arange(0.0, 1.0 + alpha_step / 2.0, alpha_step)

    if has_split_halves:
        # Lọc nghiêm ngặt tập cầu thủ của 2 nửa split-half đồng nhất với quần thể Set A (D16)
        valid_pids = set(merged["player_id"])
        common_pids = set(half_a1_df["player_id"]).intersection(set(half_a2_df["player_id"])).intersection(valid_pids)
        if len(common_pids) >= 10:
            h1 = half_a1_df[half_a1_df["player_id"].isin(common_pids)].sort_values("player_id").copy()
            h2 = half_a2_df[half_a2_df["player_id"].isin(common_pids)].sort_values("player_id").copy()

            if exclude_goalkeepers and "is_goalkeeper" in h1.columns:
                h1 = h1[~h1["is_goalkeeper"].fillna(False)]
                h2 = h2[~h2["is_goalkeeper"].fillna(False)]

            h1_c = percentile_rank(h1["betweenness"].fillna(0.0))
            h1_x = percentile_rank(h1["mean_importance"].fillna(0.0))
            h2_c = percentile_rank(h2["betweenness"].fillna(0.0))
            h2_x = percentile_rank(h2["mean_importance"].fillna(0.0))

            for a in alphas:
                hc1 = a * h1_c + (1.0 - a) * h1_x
                hc2 = a * h2_c + (1.0 - a) * h2_x
                r_rel, _ = spearmanr(hc1, hc2)
                half_reliabilities[round(float(a), 2)] = float(r_rel) if not np.isnan(r_rel) else 0.0

    max_rel = max(half_reliabilities.values()) if half_reliabilities else 1.0
    min_allowed_rel = split_half_min_reliability_ratio * max_rel if half_reliabilities else -1.0

    curve_rows = []
    best_alpha = 0.5
    best_corr = -1.0

    for a in alphas:
        a_key = round(float(a), 2)
        hc = a * c_rank + (1.0 - a) * x_rank
        corr, p_val = spearmanr(hc, y_target)
        corr_val = float(corr) if not np.isnan(corr) else 0.0

        corr_kp, _ = spearmanr(hc, merged["key_passes_per_90"]) if "key_passes_per_90" in merged else (0.0, 1.0)
        corr_xa, _ = spearmanr(hc, merged["xA_per_90"]) if "xA_per_90" in merged else (0.0, 1.0)

        rel_val = half_reliabilities.get(a_key, 1.0)
        passes_constraint = rel_val >= min_allowed_rel

        curve_rows.append({
            "alpha": a_key,
            f"spearman_{target_metric}": round(corr_val, 4),
            "p_value": round(float(p_val), 4),
            "spearman_key_passes": round(float(corr_kp), 4),
            "spearman_xa": round(float(corr_xa), 4),
            "split_half_reliability": round(rel_val, 4) if has_split_halves else 1.0,
            "meets_reliability_constraint": passes_constraint,
        })

        if passes_constraint and corr_val > best_corr:
            best_corr = corr_val
            best_alpha = a_key

    curve_df = pd.DataFrame(curve_rows)
    logger.info(
        "Hoàn tất quét alpha trên Set A! Alpha* tối ưu = %.2f (Spearman=%s: %.4f, Ràng buộc tin cậy: %s)",
        best_alpha, target_metric, best_corr, "Có" if has_split_halves else "Không"
    )
    return best_alpha, curve_df
