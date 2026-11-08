"""Unit tests for Cross-League Evaluation (EPL -> La Liga 2015/16)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data/processed/cross_league_laliga"
INTERIM_DIR = PROJECT_ROOT / "data/interim/cross_league_laliga"
RESULTS_DIR = PROJECT_ROOT / "results/cross_league"


def test_cross_league_graph_data_integrity() -> None:
    """Kiểm tra tính toàn vẹn của dataset đồ thị La Liga 2015/16."""
    index_path = PROCESSED_DIR / "graph_index.parquet"
    assert index_path.exists(), f"Không tìm thấy {index_path}"

    df_idx = pd.read_parquet(index_path)
    assert len(df_idx) >= 40000, f"Số lượng đồ thị quá ít: {len(df_idx)}"
    assert "match_id" in df_idx.columns
    assert "possession_id" in df_idx.columns
    assert "y" in df_idx.columns
    assert "n_nodes" in df_idx.columns
    assert "n_edges" in df_idx.columns

    # Tỷ lệ nhãn shot nằm trong khoảng hợp lý của bóng đá (10% - 15%)
    shot_rate = df_idx["y"].mean()
    assert 0.10 <= shot_rate <= 0.15, f"Tỷ lệ shot bất thường: {shot_rate:.3f}"

    # Kiểm tra file graphs.pt
    graphs_pt = PROCESSED_DIR / "graphs.pt"
    assert graphs_pt.exists(), f"Không tìm thấy {graphs_pt}"
    # Đọc mẫu 1 đồ thị
    graphs = torch.load(graphs_pt, map_location="cpu", weights_only=False)
    sample_g = graphs[0]
    assert sample_g.x.shape[1] == 8, f"Node features dim != 8: {sample_g.x.shape}"
    assert sample_g.edge_attr.shape[1] == 12, f"Edge features dim != 12: {sample_g.edge_attr.shape}"
    assert sample_g.edge_index.shape[0] == 2, f"Edge index dim != 2: {sample_g.edge_index.shape}"


def test_cross_league_player_reference_stats() -> None:
    """Kiểm tra ground truth thống kê cầu thủ La Liga."""
    ref_path = INTERIM_DIR / "player_reference_stats.parquet"
    assert ref_path.exists(), f"Không tìm thấy {ref_path}"

    df_ref = pd.read_parquet(ref_path)
    assert len(df_ref) >= 400
    expected_cols = [
        "player_id", "player_name", "team_name", "key_passes", "assists", "xA"
    ]
    for col in expected_cols:
        assert col in df_ref.columns, f"Thiếu cột {col}"

    # Phải có các siêu sao kiến thiết hàng đầu La Liga 2015/16
    player_names = df_ref["player_name"].str.lower().tolist()
    assert any("messi" in name for name in player_names), "Thiếu Lionel Messi"
    assert any("neymar" in name for name in player_names), "Thiếu Neymar"


def test_cross_league_model_transfer_metrics() -> None:
    """Kiểm tra kết quả RQ1 Zero-shot Model Transfer."""
    csv_path = RESULTS_DIR / "model_transfer_metrics.csv"
    assert csv_path.exists(), f"Không tìm thấy {csv_path}"

    df = pd.read_csv(csv_path)
    assert len(df) == 4
    assert set(df["model"].str.upper()) == {"MPNN", "GCN", "GATV2", "GRAPHSAGE"}

    for _, row in df.iterrows():
        assert 0.0 <= row["laliga_zero_shot_auc_pr"] <= 1.0
        assert 0.5 <= row["laliga_zero_shot_auc_roc"] <= 1.0
        assert 0.0 <= row["laliga_zero_shot_brier"] <= 1.0

    # MPNN phải có mức suy giảm (degradation) thấp nhất trong tất cả các GNN
    mpnn_deg = df[df["model"].str.upper() == "MPNN"]["degradation_pct"].values[0]
    gcn_deg = df[df["model"].str.upper() == "GCN"]["degradation_pct"].values[0]
    gat_deg = df[df["model"].str.upper() == "GATV2"]["degradation_pct"].values[0]
    sage_deg = df[df["model"].str.upper() == "GRAPHSAGE"]["degradation_pct"].values[0]

    assert abs(mpnn_deg) < abs(gcn_deg), f"MPNN ({mpnn_deg}) không bền vững hơn GCN ({gcn_deg})"
    assert abs(mpnn_deg) < abs(gat_deg), f"MPNN ({mpnn_deg}) không bền vững hơn GATv2 ({gat_deg})"
    assert abs(mpnn_deg) < abs(sage_deg), f"MPNN ({mpnn_deg}) không bền vững hơn GraphSAGE ({sage_deg})"


def test_cross_league_xai_metrics_and_gate_d17() -> None:
    """Kiểm tra kết quả RQ2 XAI Transferability và Gate D17."""
    csv_path = RESULTS_DIR / "xai_transfer_metrics.csv"
    if not csv_path.exists():
        pytest.skip(f"Chưa có file {csv_path}, pipeline có thể đang chạy")

    df = pd.read_csv(csv_path)
    assert "explainer" in df.columns
    assert "fidelity_plus_mean" in df.columns
    assert "fidelity_minus_mean" in df.columns
    assert "sparsity_mean" in df.columns
    assert "gate_d17_status" in df.columns

    # Heuristic last-edge phải vượt qua Gate D17 với CI dương
    expl_status = dict(zip(df["explainer"], df["gate_d17_status"]))
    assert expl_status.get("last_edge") == "PASS", f"Last-edge failed Gate D17: {expl_status}"
    assert "gnn_explainer" in expl_status
    assert "random" in expl_status


def test_cross_league_hybrid_centrality() -> None:
    """Kiểm tra kết quả RQ3 Hybrid Centrality và tương quan với ground truth."""
    metrics_path = RESULTS_DIR / "hybrid_centrality_metrics.csv"
    top_players_path = RESULTS_DIR / "hybrid_centrality_top_players.csv"
    if not metrics_path.exists() or not top_players_path.exists():
        pytest.skip("Chưa hoàn tất RQ3 Hybrid Centrality")

    df_metrics = pd.read_csv(metrics_path)
    assert len(df_metrics) > 0
    assert "metric" in df_metrics.columns
    assert "hybrid_centrality_spearman" in df_metrics.columns

    # Spearman tương quan với xt_per_90 phải dương và >= 0.40
    xt_row = df_metrics[df_metrics["metric"] == "xt_per_90"]
    assert len(xt_row) > 0, "Thiếu metric xt_per_90"
    rho = xt_row["hybrid_centrality_spearman"].values[0]
    assert rho > 0.4, f"Tương quan với xT quá thấp: {rho}"

    # Delta R^2 trên tất cả các chỉ số đều phải dương
    assert (df_metrics["ols_delta_r2"] > 0).all(), "Có chỉ số delta R^2 không dương"

    df_top = pd.read_csv(top_players_path)
    assert len(df_top) >= 15
    top_names = df_top["player_name"].str.lower().tolist()
    assert any("messi" in name or "marcelo" in name or "banega" in name for name in top_names)
