"""Module tính toán và tối ưu hóa chỉ số Hybrid Centrality (Phase 4).

Tham chiếu:
- docs/decisions/0002-cong-thuc-hybrid-centrality.md (Quyết định D12, D13, D14, D15, D16, D17)
Công thức:
    HC(p; alpha) = alpha * PercentileRank(C(p)) + (1 - alpha) * PercentileRank(X(p))
Trong đó:
    - C(p): Độ đo Betweenness centrality truyền thống trên đồ thị chuyền bóng cấp trận đấu (D13).
    - X(p): Điểm XAI trung bình mỗi lần tham gia hoặc bóc tách cạnh cuối (D12).
    - PercentileRank: Chuẩn hoá thứ hạng phân vị trên đoạn [0, 1] (D14).
    - alpha*: Tối ưu hoá trên tập Set A và kiểm định độc lập trên tập Set B (D15, D16).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from xai_football.utils.logging import get_logger

logger = get_logger("hybrid_centrality")


def compute_match_traditional_centrality(
    events_df: pd.DataFrame,
    match_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Xây dựng mạng lưới chuyền bóng cấp trận đấu và tính các độ đo Centrality (D13).

    Returns:
        DataFrame: [player_id, player_name, betweenness, degree_in, degree_out, closeness, eigenvector]
    """
    df = events_df
    if match_ids is not None:
        df = df[df["match_id"].isin(match_ids)]

    # Lọc các đường chuyền thành công
    passes = df[
        (df["event_type"] == "Pass")
        & (df["pass_outcome"].isna())
        & (df["player_id"].notna())
        & (df["pass_recipient_id"].notna())
    ].copy()

    if passes.empty:
        return pd.DataFrame(columns=[
            "player_id", "player_name", "betweenness", "degree_in", "degree_out", "closeness", "eigenvector"
        ])

    # Tính centrality theo từng trận đấu rồi lấy trung bình qua các trận
    match_grouped = passes.groupby("match_id")
    match_records = []

    for mid, m_passes in match_grouped:
        # Xây dựng đồ thị có hướng có trọng số (số đường chuyền u -> v)
        G = nx.DiGraph()
        edge_weights = m_passes.groupby(["player_id", "pass_recipient_id"]).size()

        for (u, v), w in edge_weights.items():
            G.add_edge(int(u), int(v), weight=float(w))

        if len(G) < 3:
            continue

        # Độ đo Betweenness (dùng nghịch đảo trọng số làm khoảng cách)
        for u, v, d in G.edges(data=True):
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
            match_records.append({
                "match_id": mid,
                "player_id": int(n),
                "betweenness": float(betw.get(n, 0.0)),
                "degree_in": float(deg_in.get(n, 0.0)),
                "degree_out": float(deg_out.get(n, 0.0)),
                "closeness": float(clos.get(n, 0.0)),
                "eigenvector": float(eig.get(n, 0.0)),
            })

    if not match_records:
        return pd.DataFrame()

    m_df = pd.DataFrame(match_records)

    # Lấy tên cầu thủ mới nhất
    player_names = df[["player_id", "player_name"]].dropna().drop_duplicates("player_id").set_index("player_id")["player_name"].to_dict()

    # Tổng hợp trung bình trên mỗi trận thi đấu
    agg = m_df.groupby("player_id").agg({
        "betweenness": "mean",
        "degree_in": "mean",
        "degree_out": "mean",
        "closeness": "mean",
        "eigenvector": "mean",
        "match_id": "nunique",
    }).reset_index()

    agg.rename(columns={"match_id": "matches_played"}, inplace=True)
    agg["player_name"] = agg["player_id"].map(player_names)

    return agg


def percentile_rank(series: pd.Series) -> pd.Series:
    """Chuẩn hoá phân vị trên đoạn [0, 1] theo Quyết định D14."""
    n = len(series)
    if n <= 1:
        return pd.Series(0.5, index=series.index)
    # rankdata: thứ hạng từ 1 đến n
    ranks = rankdata(series.values, method="average")
    return pd.Series((ranks - 1.0) / (n - 1.0), index=series.index)


def compute_hybrid_centrality(
    centrality_df: pd.DataFrame,
    xai_player_df: pd.DataFrame,
    alpha: float = 0.5,
    centrality_col: str = "betweenness",
    xai_col: str = "mean_importance",
) -> pd.DataFrame:
    """Tính chỉ số Hybrid Centrality kết hợp giữa Network Centrality và XAI (D12, D14)."""
    merged = pd.merge(centrality_df, xai_player_df, on="player_id", how="inner")
    if merged.empty:
        return pd.DataFrame()

    c_vals = merged[centrality_col].fillna(0.0)
    x_vals = merged[xai_col].fillna(0.0)

    merged["rank_centrality"] = percentile_rank(c_vals)
    merged["rank_xai"] = percentile_rank(x_vals)

    merged["alpha"] = alpha
    merged["hybrid_centrality"] = alpha * merged["rank_centrality"] + (1.0 - alpha) * merged["rank_xai"]

    return merged.sort_values(by="hybrid_centrality", ascending=False).reset_index(drop=True)


def optimize_alpha_on_set_a(
    centrality_df: pd.DataFrame,
    xai_player_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    alpha_step: float = 0.05,
    target_metric: str = "xt_per_90",
) -> tuple[float, pd.DataFrame]:
    """Quét grid search alpha in [0, 1] trên Set A để tìm alpha* tối ưu (Quyết định D16)."""
    merged_base = pd.merge(centrality_df, xai_player_df, on="player_id", how="inner")
    merged = pd.merge(merged_base, reference_df, on="player_id", how="inner")

    # Lọc cầu thủ có >= 90 phút thi đấu
    if "minutes_played" in merged.columns:
        merged = merged[merged["minutes_played"] >= 90.0]

    if len(merged) < 10 or target_metric not in merged.columns:
        logger.warning("Không đủ dữ liệu trên Set A để tối ưu alpha! Trả về mặc định 0.5")
        return 0.5, pd.DataFrame()

    c_rank = percentile_rank(merged["betweenness"].fillna(0.0))
    x_rank = percentile_rank(merged["mean_importance"].fillna(0.0))
    y_target = merged[target_metric].values

    alphas = np.arange(0.0, 1.0 + alpha_step / 2.0, alpha_step)
    curve_rows = []

    best_alpha = 0.5
    best_corr = -1.0

    for a in alphas:
        hc = a * c_rank + (1.0 - a) * x_rank
        corr, p_val = spearmanr(hc, y_target)
        corr_val = float(corr) if not np.isnan(corr) else 0.0

        # Tương quan với các chỉ số khác nếu có
        corr_kp, _ = spearmanr(hc, merged["key_passes_per_90"]) if "key_passes_per_90" in merged else (0.0, 1.0)
        corr_xa, _ = spearmanr(hc, merged["xA_per_90"]) if "xA_per_90" in merged else (0.0, 1.0)

        curve_rows.append({
            "alpha": round(float(a), 2),
            f"spearman_{target_metric}": round(corr_val, 4),
            "p_value": round(float(p_val), 4),
            "spearman_key_passes": round(float(corr_kp), 4),
            "spearman_xa": round(float(corr_xa), 4),
        })

        if corr_val > best_corr:
            best_corr = corr_val
            best_alpha = float(a)

    curve_df = pd.DataFrame(curve_rows)
    logger.info("Hoàn tất quét alpha trên Set A! Alpha* tối ưu = %.2f (Spearman=%s: %.4f)",
                best_alpha, target_metric, best_corr)
    return best_alpha, curve_df
