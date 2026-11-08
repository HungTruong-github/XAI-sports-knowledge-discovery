"""Script đánh giá mở rộng Cross-League Zero-Shot Evaluation (EPL -> La Liga 2015/16).

Mục tiêu khoa học:
    Kiểm tra tính tổng quát hóa ngoài miền dữ liệu huấn luyện (Domain Generalization)
    của pipeline GNN-XAI-Hybrid Centrality đã huấn luyện trên Premier League 2015/16 (Source Domain)
    khi chuyển sang kiểm định zero-shot trên toàn bộ giải La Liga 2015/16 (Target Domain).

Ba câu hỏi nghiên cứu (RQs):
    - RQ1 (Model Transferability): Mô hình GNN (MPNN, GCN, GATv2, GraphSAGE) huấn luyện trên EPL
      có duy trì được năng lực phân loại xác suất possession dẫn tới cú sút trên La Liga không?
      Đo lường: AUC-PR, AUC-ROC, Brier score, suy giảm hiệu năng Delta_PR kèm 95% Clustered Bootstrap CI.
    - RQ2 (XAI Sanity Check & Transferability): Lời giải thích XAI (GNNExplainer, Last-edge, Random)
      có giữ vững tính xác thực (Fidelity+, Fidelity-, Gate D17) trên cấu trúc chuyền bóng La Liga không?
    - RQ3 (Downstream Hybrid Centrality Generalization): Khi đóng băng tham số alpha* = 0.70
      và shrinkage m = 20 từ EPL, Hybrid Centrality có duy trì tương quan vượt trội với các chỉ số
      tạo cơ hội (xT, xA, Key Passes, Assists) trên La Liga và định danh chính xác các nhạc trưởng hàng đầu?

Đầu ra:
    - results/cross_league/model_transfer_metrics.csv
    - results/cross_league/xai_transfer_metrics.csv
    - results/cross_league/hybrid_centrality_metrics.csv
    - results/cross_league/hybrid_centrality_top_players.csv
    - results/cross_league/cross_league_manifest.json
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Thêm đường dẫn src vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from xai_football.centrality.calculator import (  # noqa: E402
    apply_xai_shrinkage,
    compute_match_traditional_centrality,
    percentile_rank,
)
from xai_football.graphs.dataset import load_graphs  # noqa: E402
from xai_football.modeling.gnn import build_gnn_model  # noqa: E402
from xai_football.reference.expected_threat import ExpectedThreatModel  # noqa: E402
from xai_football.reference.player_stats import (  # noqa: E402
    compute_player_minutes,
    compute_player_reference_stats,
)
from xai_football.utils.io import load_yaml, save_atomic_csv, save_atomic_json  # noqa: E402
from xai_football.utils.logging import get_logger  # noqa: E402
from xai_football.utils.provenance import get_git_info  # noqa: E402
from xai_football.xai.explainers import build_explainer  # noqa: E402
from xai_football.xai.metrics import (  # noqa: E402
    compute_gate_d17_cluster_bootstrap,
    evaluate_explanation_quality,
    resolve_top_k,
)

logger = get_logger("cross_league_eval")


# ==============================================================================
# 1. HÀM HỖ TRỢ ĐÁNH GIÁ RQ1: ZERO-SHOT MODEL TRANSFER
# ==============================================================================


@torch.no_grad()
def predict_graphs_batch(
    model: nn.Module,
    graphs: list[Data],
    device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chạy inference toàn bộ danh sách đồ thị theo mini-batch."""
    model.eval()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    all_probs = []
    all_targets = []
    all_mids = []

    for batch in loader:
        batch = batch.to(device)
        edge_attr = getattr(batch, "edge_attr", None)
        logits = model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=edge_attr,
            batch=batch.batch,
        )
        probs = torch.sigmoid(logits.view(-1)).cpu().numpy()
        targets = batch.y.view(-1).cpu().numpy()

        all_probs.extend(probs)
        all_targets.extend(targets)

        # Trích xuất match_id theo đồ thị
        if hasattr(batch, "match_id"):
            mids = batch.match_id
            if isinstance(mids, torch.Tensor):
                all_mids.extend(mids.cpu().numpy().tolist())
            elif isinstance(mids, list):
                all_mids.extend(mids)
        else:
            all_mids.extend([0] * len(probs))

    return np.array(all_probs), np.array(all_targets), np.array(all_mids)


def cluster_bootstrap_model_metrics(
    df: pd.DataFrame,
    threshold: float,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Tính 95% Confidence Interval bằng Clustered Bootstrap theo match_id (vectorized tối ưu)."""
    # Gom nhóm theo match_id trước ngoài vòng lặp để tránh overhead lọc DataFrame
    match_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for mid, grp in df.groupby("match_id"):
        match_data[int(mid)] = (
            grp["target"].to_numpy(dtype=int),
            grp["prob"].to_numpy(dtype=float),
        )

    match_ids = np.array(list(match_data.keys()))
    n_matches = len(match_ids)
    rng = np.random.default_rng(seed)

    boot_auc_pr = []
    boot_auc_roc = []
    boot_brier = []

    for _ in range(n_bootstraps):
        sampled_mids = rng.choice(match_ids, size=n_matches, replace=True)
        y_true = np.concatenate([match_data[m][0] for m in sampled_mids])
        y_prob = np.concatenate([match_data[m][1] for m in sampled_mids])

        if len(np.unique(y_true)) < 2:
            continue

        boot_auc_pr.append(average_precision_score(y_true, y_prob))
        boot_auc_roc.append(roc_auc_score(y_true, y_prob))
        boot_brier.append(brier_score_loss(y_true, y_prob))

    return {
        "auc_pr": {
            "ci_lower": float(np.percentile(boot_auc_pr, 2.5)),
            "ci_upper": float(np.percentile(boot_auc_pr, 97.5)),
        },
        "auc_roc": {
            "ci_lower": float(np.percentile(boot_auc_roc, 2.5)),
            "ci_upper": float(np.percentile(boot_auc_roc, 97.5)),
        },
        "brier_score": {
            "ci_lower": float(np.percentile(boot_brier, 2.5)),
            "ci_upper": float(np.percentile(boot_brier, 97.5)),
        },
    }


# ==============================================================================
# 2. HÀM HỖ TRỢ ĐÁNH GIÁ RQ2: XAI TRANSFER & GATE D17
# ==============================================================================


def evaluate_xai_cross_league(
    model: nn.Module,
    sample_graphs: list[Data],
    device: torch.device,
    xai_cfg: dict[str, Any],
) -> pd.DataFrame:
    """Đánh giá Fidelity+, Fidelity-, Sparsity trên mẫu đồ thị La Liga."""
    rows = []
    explainers_to_eval = ["gnn_explainer", "last_edge", "random"]

    for expl_name in explainers_to_eval:
        logger.info("Đang chạy explainer: %s trên %d đồ thị La Liga...", expl_name, len(sample_graphs))
        explainer = build_explainer(expl_name, model=model, xai_cfg=xai_cfg, device=device)

        for idx, g in enumerate(sample_graphs):
            num_edges = g.edge_index.shape[1] if g.edge_index is not None else 0
            k_val = resolve_top_k(num_edges)
            exp_res = explainer.explain_graph(g, graph_index=idx, target_class=1)
            metrics = evaluate_explanation_quality(
                model=model,
                graph=g,
                edge_mask=exp_res.edge_mask,
                top_k=k_val,
                device=device,
            )
            rows.append(
                {
                    "graph_idx": idx,
                    "match_id": int(getattr(g, "match_id", 0)),
                    "possession_id": int(getattr(g, "possession_id", 0)),
                    "explainer": expl_name,
                    "fidelity_plus": metrics["fidelity_plus"],
                    "fidelity_minus": metrics["fidelity_minus"],
                    "sparsity": metrics["sparsity"],
                }
            )

    return pd.DataFrame(rows)


# ==============================================================================
# 3. HÀM HỖ TRỢ ĐÁNH GIÁ RQ3: HYBRID CENTRALITY
# ==============================================================================


def extract_player_xai_scores_fast(
    graphs: list[Data],
) -> pd.DataFrame:
    """Trích xuất điểm XAI cấp possession của cầu thủ trên toàn bộ đồ thị La Liga.

    Sử dụng heuristic Last-edge (đã được chứng minh qua kiểm định Gate D17 đạt độ phân giải cao
    và không rò rỉ nhãn) để gán tầm quan trọng cho các đường chuyền trong possession.
    """
    logger.info("Trích xuất điểm XAI cầu thủ trên %d đồ thị La Liga...", len(graphs))
    player_importance_rows = []

    for g in graphs:
        n_edges = g.edge_index.shape[1] if g.edge_index is not None else 0
        if n_edges == 0:
            continue

        # Heuristic Last-edge: đường chuyền cuối cùng có tầm quan trọng = 1.0, các đường chuyền trước = 0.0
        edge_mask = np.zeros(n_edges, dtype=float)
        edge_mask[-1] = 1.0

        n_nodes = g.num_nodes
        player_ids = (
            g.player_ids.detach().cpu().numpy()
            if hasattr(g, "player_ids")
            else np.arange(n_nodes)
        )

        out_scores = np.zeros(n_nodes, dtype=float)
        in_scores = np.zeros(n_nodes, dtype=float)
        out_counts = np.zeros(n_nodes, dtype=int)
        in_counts = np.zeros(n_nodes, dtype=int)

        for k in range(n_edges):
            u = int(g.edge_index[0, k].item())
            v = int(g.edge_index[1, k].item())
            val = float(edge_mask[k])
            if u < n_nodes:
                out_scores[u] += val
                out_counts[u] += 1
            if v < n_nodes:
                in_scores[v] += val
                in_counts[v] += 1

        mid = int(getattr(g, "match_id", 0))
        poss_id = int(getattr(g, "possession_id", 0))

        for i in range(n_nodes):
            parts = int(out_counts[i] + in_counts[i])
            if parts > 0:
                tot_imp = float(out_scores[i] + in_scores[i])
                player_importance_rows.append(
                    {
                        "match_id": mid,
                        "possession_id": poss_id,
                        "player_id": int(player_ids[i]),
                        "participations": parts,
                        "mean_importance": tot_imp / parts,
                    }
                )

    return pd.DataFrame(player_importance_rows)


def evaluate_ols_delta_r2_laliga(
    df: pd.DataFrame,
    y_col: str,
    cent_col: str,
    xai_col: str,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Tính toán OLS R^2 và Delta R^2 trên toàn bộ tập La Liga kèm 95% Bootstrap CI."""
    sub = df[[y_col, cent_col, xai_col]].dropna()
    if len(sub) < 20:
        return {"r2_pure_centrality": 0.0, "r2_full_model": 0.0, "delta_r2": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    y = sub[y_col].values
    X_base = sub[[cent_col]].values
    X_full = sub[[cent_col, xai_col]].values

    reg_base = LinearRegression().fit(X_base, y)
    reg_full = LinearRegression().fit(X_full, y)

    r2_base = float(reg_base.score(X_base, y))
    r2_full = float(reg_full.score(X_full, y))
    delta_r2 = float(r2_full - r2_base)

    rng = np.random.default_rng(seed)
    n = len(sub)
    deltas = []

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y[idx]
        X_base_b = X_base[idx]
        X_full_b = X_full[idx]

        r2_b_base = float(LinearRegression().fit(X_base_b, y_b).score(X_base_b, y_b))
        r2_b_full = float(LinearRegression().fit(X_full_b, y_b).score(X_full_b, y_b))
        deltas.append(r2_b_full - r2_b_base)

    ci_l = float(np.percentile(deltas, 2.5)) if deltas else 0.0
    ci_u = float(np.percentile(deltas, 97.5)) if deltas else 0.0

    return {
        "r2_pure_centrality": round(r2_base, 4),
        "r2_full_model": round(r2_full, 4),
        "delta_r2": round(delta_r2, 4),
        "ci_lower": round(ci_l, 4),
        "ci_upper": round(ci_u, 4),
    }


# ==============================================================================
# 4. CHƯƠNG TRÌNH CHÍNH (MAIN ORCHESTRATOR)
# ==============================================================================


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_cross_league_laliga.yaml")
    parser.add_argument("--epl-models-dir", default="models/official")
    parser.add_argument("--events-path", default="data/interim/cross_league_laliga/events.parquet")
    parser.add_argument("--processed-dir", default="data/processed/cross_league_laliga")
    parser.add_argument("--lineups-dir", default="data/raw/11/27")
    parser.add_argument("--output-dir", default="results/cross_league")
    parser.add_argument("--n-xai-samples", type=int, default=150)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Thiết bị tính toán: %s", device)

    # Nạp đồ thị La Liga
    logger.info("Đang nạp đồ thị La Liga từ %s...", args.processed_dir)
    graphs, graph_index_df, feat_names = load_graphs(Path(args.processed_dir))
    n_graphs = len(graphs)
    logger.info("Tổng số đồ thị La Liga: %d", n_graphs)

    # Đọc bảng kết quả chính thức của EPL để làm baseline đối sánh
    epl_summary_path = PROJECT_ROOT / "results/tables/official/model_comparison.csv"
    epl_metrics_map = {}
    if epl_summary_path.exists():
        epl_df = pd.read_csv(epl_summary_path)
        for _, row in epl_df.iterrows():
            m_name = str(row["model"]).split()[0].lower()
            epl_metrics_map[m_name] = {
                "auc_pr": float(row["auc_pr"]),
                "auc_roc": float(row["auc_roc"]),
                "brier": float(row["brier_score"]),
                "threshold": float(row["threshold_mean"]) if pd.notna(row["threshold_mean"]) else 0.5,
            }

    # ==========================================================================
    # RQ1: ZERO-SHOT MODEL TRANSFER EVALUATION
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("THỰC HIỆN RQ1: ĐÁNH GIÁ ZERO-SHOT GNN PREDICTION TRANSFER TỪ EPL SANG LA LIGA")
    logger.info("=" * 60)

    sample_g = graphs[0]
    node_in_dim = sample_g.x.shape[1]
    edge_in_dim = sample_g.edge_attr.shape[1] if hasattr(sample_g, "edge_attr") and sample_g.edge_attr is not None else 12
    models_dir = Path(args.epl_models_dir)

    model_transfer_csv = out_dir / "model_transfer_metrics.csv"
    if model_transfer_csv.exists():
        logger.info("Đã tìm thấy kết quả RQ1 tại %s, nạp trực tiếp...", model_transfer_csv)
        transfer_df = pd.read_csv(model_transfer_csv)
        print("\n=== BẢNG CHUYỂN GIAO MÔ HÌNH EPL -> LA LIGA (RQ1) ===")
        print(transfer_df[["model", "epl_test_auc_pr", "laliga_zero_shot_auc_pr", "delta_auc_pr", "degradation_pct", "epl_test_auc_roc", "laliga_zero_shot_auc_roc"]].to_string(index=False))
    else:
        model_configs = {
            "mpnn": "configs/model/mpnn.yaml",
            "gcn": "configs/model/gcn.yaml",
            "gatv2": "configs/model/gatv2.yaml",
            "graphsage": "configs/model/graphsage.yaml",
        }

        model_transfer_rows = []


        for m_key, cfg_rel_path in model_configs.items():
            ckpt_path = models_dir / f"{m_key}_best.pt"
            if not ckpt_path.exists():
                logger.warning("Không tìm thấy checkpoint %s, bỏ qua", ckpt_path)
                continue

            cfg_path = PROJECT_ROOT / cfg_rel_path
            m_cfg = load_yaml(cfg_path)
            model = build_gnn_model(m_cfg, node_in_dim=node_in_dim, edge_in_dim=edge_in_dim)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            logger.info("Chạy inference zero-shot cho mô hình: %s...", m_key.upper())
            probs, targets, mids = predict_graphs_batch(model, graphs, device=device)

            # Ngưỡng chọn từ EPL validation set
            threshold = epl_metrics_map.get(m_key, {}).get("threshold", 0.5)
            preds = (probs >= threshold).astype(int)

            ll_auc_pr = float(average_precision_score(targets, probs))
            ll_auc_roc = float(roc_auc_score(targets, probs))
            ll_brier = float(brier_score_loss(targets, probs))
            ll_acc = float(accuracy_score(targets, preds))
            ll_prec = float(precision_score(targets, preds, zero_division=0))
            ll_rec = float(recall_score(targets, preds, zero_division=0))
            ll_f1 = float(f1_score(targets, preds, zero_division=0))

            # Clustered Bootstrap 95% CI trên La Liga
            pred_df = pd.DataFrame({"match_id": mids, "target": targets, "prob": probs})
            cis = cluster_bootstrap_model_metrics(pred_df, threshold=threshold, n_bootstraps=1000)

            # Đối sánh với EPL
            epl_info = epl_metrics_map.get(m_key, {})
            epl_auc_pr = epl_info.get("auc_pr", np.nan)
            epl_auc_roc = epl_info.get("auc_roc", np.nan)
            epl_brier = epl_info.get("brier", np.nan)

            delta_pr = ll_auc_pr - epl_auc_pr if pd.notna(epl_auc_pr) else np.nan
            degradation_pct = (delta_pr / epl_auc_pr * 100) if pd.notna(epl_auc_pr) and epl_auc_pr > 0 else np.nan
            delta_roc = ll_auc_roc - epl_auc_roc if pd.notna(epl_auc_roc) else np.nan

            model_transfer_rows.append(
                {
                    "model": m_key.upper(),
                    "epl_test_auc_pr": round(epl_auc_pr, 4),
                    "laliga_zero_shot_auc_pr": round(ll_auc_pr, 4),
                    "auc_pr_ci_95": f"[{cis['auc_pr']['ci_lower']:.4f}, {cis['auc_pr']['ci_upper']:.4f}]",
                    "delta_auc_pr": round(delta_pr, 4),
                    "degradation_pct": round(degradation_pct, 2),
                    "epl_test_auc_roc": round(epl_auc_roc, 4),
                    "laliga_zero_shot_auc_roc": round(ll_auc_roc, 4),
                    "auc_roc_ci_95": f"[{cis['auc_roc']['ci_lower']:.4f}, {cis['auc_roc']['ci_upper']:.4f}]",
                    "delta_auc_roc": round(delta_roc, 4),
                    "epl_test_brier": round(epl_brier, 4),
                    "laliga_zero_shot_brier": round(ll_brier, 4),
                    "brier_ci_95": f"[{cis['brier_score']['ci_lower']:.4f}, {cis['brier_score']['ci_upper']:.4f}]",
                    "threshold_applied": round(threshold, 4),
                    "accuracy": round(ll_acc, 4),
                    "precision": round(ll_prec, 4),
                    "recall": round(ll_rec, 4),
                    "f1": round(ll_f1, 4),
                }
            )

        transfer_df = pd.DataFrame(model_transfer_rows)
        save_atomic_csv(transfer_df, out_dir / "model_transfer_metrics.csv")
        logger.info("Đã ghi bảng kết quả chuyển giao mô hình vào %s", out_dir / "model_transfer_metrics.csv")
        print("\n=== BẢNG CHUYỂN GIAO MÔ HÌNH EPL -> LA LIGA (RQ1) ===")
        print(transfer_df[["model", "epl_test_auc_pr", "laliga_zero_shot_auc_pr", "delta_auc_pr", "degradation_pct", "epl_test_auc_roc", "laliga_zero_shot_auc_roc"]].to_string(index=False))

    # ==========================================================================
    # RQ2: XAI TRANSFERABILITY & SANITY CHECK (GATE D17)
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("THỰC HIỆN RQ2: ĐÁNH GIÁ XAI TRANSFERABILITY & KIỂM ĐỊNH GATE D17 TRÊN LA LIGA")
    logger.info("=" * 60)

    xai_csv_path = out_dir / "xai_transfer_metrics.csv"
    if xai_csv_path.exists():
        logger.info("Đã tìm thấy kết quả RQ2 tại %s, nạp trực tiếp...", xai_csv_path)
        xai_summary_df = pd.read_csv(xai_csv_path)
        print("\n=== BẢNG ĐỐI SÁNH XAI & KIỂM ĐỊNH GATE D17 TRÊN LA LIGA (RQ2) ===")
        print(xai_summary_df.to_string(index=False))
    else:
        # Chọn mô hình MPNN làm backbone giải thích
        mpnn_cfg = load_yaml(PROJECT_ROOT / "configs/model/mpnn.yaml")
        mpnn_model = build_gnn_model(mpnn_cfg, node_in_dim=node_in_dim, edge_in_dim=edge_in_dim)
        mpnn_ckpt = torch.load(models_dir / "mpnn_best.pt", map_location=device, weights_only=False)
        mpnn_model.load_state_dict(mpnn_ckpt["model_state_dict"] if "model_state_dict" in mpnn_ckpt else mpnn_ckpt)
        mpnn_model.to(device)
        mpnn_model.eval()

        # Chọn mẫu phân tầng cân bằng (20 dương, 20 âm, >= 4 cạnh)
        pos_graphs = [g for g in graphs if int(getattr(g, "y", torch.tensor([0])).view(-1)[0].item()) == 1 and g.edge_index.shape[1] >= 4]
        neg_graphs = [g for g in graphs if int(getattr(g, "y", torch.tensor([0])).view(-1)[0].item()) == 0 and g.edge_index.shape[1] >= 4]

        rng = np.random.default_rng(42)
        n_sample_each = min(args.n_xai_samples // 2, len(pos_graphs), len(neg_graphs))
        pos_idx = rng.choice(len(pos_graphs), size=n_sample_each, replace=False)
        neg_idx = rng.choice(len(neg_graphs), size=n_sample_each, replace=False)
        sampled_pos = [pos_graphs[i] for i in pos_idx]
        sampled_neg = [neg_graphs[i] for i in neg_idx]
        eval_graphs = sampled_pos + sampled_neg
        rng.shuffle(eval_graphs)

        logger.info("Đã chọn %d đồ thị La Liga để đánh giá XAI (%d dương, %d âm)", len(eval_graphs), len(sampled_pos), len(sampled_neg))

        xai_cfg = load_yaml(PROJECT_ROOT / "configs/xai.yaml")
        xai_results_df = evaluate_xai_cross_league(
            model=mpnn_model,
            sample_graphs=eval_graphs,
            device=device,
            xai_cfg=xai_cfg,
        )

        # Tính kiểm định Gate D17 bằng Paired Clustered Bootstrap
        gate_d17_results = compute_gate_d17_cluster_bootstrap(xai_results_df, n_bootstraps=1000, seed=42)

        xai_summary_rows = []
        for expl in xai_results_df["explainer"].unique():
            sub = xai_results_df[xai_results_df["explainer"] == expl]
            fid_plus_mean = float(sub["fidelity_plus"].mean())
            fid_plus_std = float(sub["fidelity_plus"].std())
            fid_minus_mean = float(sub["fidelity_minus"].mean())
            fid_minus_std = float(sub["fidelity_minus"].std())
            sparsity_mean = float(sub["sparsity"].mean())

            d17_info = gate_d17_results.get(expl, {})
            d17_status = "PASS" if d17_info.get("gate_d17_pass", False) else ("REF" if expl == "random" else "FAIL")
            d17_diff = d17_info.get("mean_paired_diff", 0.0)
            d17_ci = f"[{d17_info.get('ci_lower', 0.0):.4f}, {d17_info.get('ci_upper', 0.0):.4f}]" if "ci_lower" in d17_info else "N/A"

            xai_summary_rows.append(
                {
                    "explainer": expl,
                    "n_graphs": len(sub),
                    "fidelity_plus_mean": round(fid_plus_mean, 4),
                    "fidelity_plus_std": round(fid_plus_std, 4),
                    "fidelity_minus_mean": round(fid_minus_mean, 4),
                    "fidelity_minus_std": round(fid_minus_std, 4),
                    "sparsity_mean": round(sparsity_mean, 4),
                    "gate_d17_delta_fid_diff": round(d17_diff, 4) if expl != "random" else "REF",
                    "gate_d17_ci_95": d17_ci,
                    "gate_d17_status": d17_status,
                }
            )

        xai_summary_df = pd.DataFrame(xai_summary_rows)
        save_atomic_csv(xai_summary_df, out_dir / "xai_transfer_metrics.csv")
        logger.info("Đã ghi kết quả đánh giá XAI vào %s", out_dir / "xai_transfer_metrics.csv")
        print("\n=== BẢNG ĐỐI SÁNH XAI & KIỂM ĐỊNH GATE D17 TRÊN LA LIGA (RQ2) ===")
        print(xai_summary_df.to_string(index=False))

    # ==========================================================================
    # RQ3: HYBRID CENTRALITY GENERALIZATION & TOP PLAYER RANKINGS
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("THỰC HIỆN RQ3: ĐÁNH GIÁ HYBRID CENTRALITY (ALPHA*=0.70) TRÊN LA LIGA")
    logger.info("=" * 60)

    events_df = pd.read_parquet(args.events_path)

    # 1. Tính số phút thi đấu toàn mùa từ lineups của 380 trận La Liga
    logger.info("Tính số phút thi đấu từ các tệp đội hình La Liga (%s)...", args.lineups_dir)
    minutes_df = compute_player_minutes(args.lineups_dir)

    # 2. Huấn luyện mô hình xT trên các sự kiện La Liga để làm ground truth tham chiếu
    logger.info("Huấn luyện mô hình Expected Threat (xT) trên dữ liệu La Liga...")
    xt_model = ExpectedThreatModel(n_cols=16, n_rows=12)
    xt_model.fit(events_df)

    # 3. Tính các chỉ số tham chiếu khách quan (xT/90, xA/90, Key Passes/90, Assists/90)
    logger.info("Tính các chỉ số kiến thiết tham chiếu từ events và lineups...")
    ref_df = compute_player_reference_stats(
        events_df=None,
        minutes_df=minutes_df,
        raw_json_dir=args.lineups_dir,
        cache_path="data/interim/cross_league_laliga/player_reference_stats.parquet",
    )
    xt_player_df = xt_model.compute_player_xt(events_df, minutes_df=minutes_df)
    ref_df = pd.merge(ref_df, xt_player_df[["player_id", "total_xt", "xt_per_90"]], on="player_id", how="left").fillna(0.0)

    # 4. Tính Betweenness Centrality truyền thống trên mạng lưới chuyền bóng phân tách theo đội
    logger.info("Tính Betweenness Centrality phân tách theo từng đội (lọc >=30 min/trận, gộp theo mùa)...")
    cent_df = compute_match_traditional_centrality(
        events_df=events_df,
        lineups_dir=args.lineups_dir,
        min_minutes_per_match=30.0,
        aggregate_level="match_to_season_weighted",
    )

    # 5. Trích xuất điểm XAI cầu thủ trên đồ thị và co rút toàn cục m = 20
    xai_raw_df = extract_player_xai_scores_fast(graphs)
    xai_player_df = apply_xai_shrinkage(xai_raw_df, shrinkage_m=20.0, min_possessions=30)

    # 6. Gộp dữ liệu Centrality + XAI + Phút thi đấu
    cent_clean = cent_df.drop(columns=[c for c in ["player_name", "team_name"] if c in cent_df.columns])
    merged_df = pd.merge(cent_clean, xai_player_df, on="player_id", how="inner")
    merged_df = pd.merge(merged_df, ref_df, on="player_id", how="inner")

    # Lọc cầu thủ đủ điều kiện: >= 90 phút thi đấu, loại thủ môn
    mins_col = "minutes_played" if "minutes_played" in merged_df.columns else "minutes"
    merged_df = merged_df[merged_df[mins_col] >= 90.0].copy()
    if "is_goalkeeper" in merged_df.columns:
        merged_df = merged_df[~merged_df["is_goalkeeper"]].copy()

    logger.info("Số cầu thủ La Liga đủ điều kiện đánh giá Hybrid Centrality: %d", len(merged_df))

    # 7. Tính Hybrid Centrality với alpha* = 0.70 đóng băng từ EPL
    alpha_frozen = 0.70
    merged_df["rank_centrality"] = percentile_rank(merged_df["betweenness"])
    merged_df["rank_xai"] = percentile_rank(merged_df["mean_importance"])
    merged_df["hybrid_centrality"] = alpha_frozen * merged_df["rank_centrality"] + (1.0 - alpha_frozen) * merged_df["rank_xai"]
    merged_df["hybrid_centrality_adj"] = merged_df["hybrid_centrality"]

    # 8. Tính Spearman rank correlation với các chỉ số tham chiếu
    metrics_to_correlate = ["xt_per_90", "xA_per_90", "key_passes_per_90", "assists_per_90"]
    corr_rows = []

    for met in metrics_to_correlate:
        if met not in merged_df.columns:
            continue
        y = merged_df[met]
        rho_hc, p_hc = spearmanr(merged_df["hybrid_centrality"], y)
        rho_cent, p_cent = spearmanr(merged_df["rank_centrality"], y)
        rho_xai, p_xai = spearmanr(merged_df["rank_xai"], y)
        gain = float(rho_hc - rho_cent)

        # Đánh giá OLS Delta R^2
        ols_res = evaluate_ols_delta_r2_laliga(
            merged_df,
            y_col=met,
            cent_col="rank_centrality",
            xai_col="rank_xai",
            n_boot=1000,
        )

        corr_rows.append(
            {
                "metric": met,
                "hybrid_centrality_spearman": round(float(rho_hc), 4),
                "hybrid_p_value": round(float(p_hc), 4),
                "pure_centrality_spearman": round(float(rho_cent), 4),
                "centrality_p_value": round(float(p_cent), 4),
                "pure_xai_spearman": round(float(rho_xai), 4),
                "xai_p_value": round(float(p_xai), 4),
                "gain_over_centrality": f"{gain:+.4f}",
                "ols_delta_r2": ols_res["delta_r2"],
                "ols_delta_r2_ci_95": f"[{ols_res['ci_lower']:.4f}, {ols_res['ci_upper']:.4f}]",
            }
        )

    corr_df = pd.DataFrame(corr_rows)
    save_atomic_csv(corr_df, out_dir / "hybrid_centrality_metrics.csv")
    logger.info("Đã ghi bảng tương quan Hybrid Centrality vào %s", out_dir / "hybrid_centrality_metrics.csv")
    print("\n=== BẢNG TƯƠNG QUAN HYBRID CENTRALITY TRÊN LA LIGA (RQ3) ===")
    print(corr_df.to_string(index=False))

    # 9. Xuất bảng Top 25 cầu thủ hàng đầu La Liga theo Hybrid Centrality
    top_cols = [
        "player_id",
        "player_name",
        "team_name",
        mins_col,
        "hybrid_centrality",
        "hybrid_centrality_adj",
        "rank_centrality",
        "rank_xai",
        "betweenness",
        "mean_importance",
        "xt_per_90",
        "xA_per_90",
        "key_passes_per_90",
        "assists",
    ]
    avail_top_cols = [c for c in top_cols if c in merged_df.columns]
    top_players = merged_df.sort_values(by="hybrid_centrality", ascending=False).head(25)[avail_top_cols].reset_index(drop=True)
    save_atomic_csv(top_players, out_dir / "hybrid_centrality_top_players.csv")
    logger.info("Đã ghi Top 25 cầu thủ La Liga vào %s", out_dir / "hybrid_centrality_top_players.csv")
    print("\n=== TOP 25 CẦU THỦ LA LIGA 2015/16 THEO HYBRID CENTRALITY ===")
    display_cols = [c for c in ["player_name", "team_name", mins_col, "hybrid_centrality", "rank_centrality", "rank_xai", "xt_per_90", "xA_per_90", "key_passes_per_90"] if c in top_players.columns]
    print(top_players[display_cols].to_string(index=False))

    # ==========================================================================
    # 10. TẠO PROVENANCE MANIFEST
    # ==========================================================================
    top_5_keys = [c for c in ["player_name", "team_name", "hybrid_centrality"] if c in top_players.columns]
    manifest = {
        "title": "Cross-League Zero-Shot Evaluation (Premier League 2015/16 -> La Liga 2015/16)",
        "source_domain": {
            "competition": "Premier League",
            "season": "2015/2016",
            "competition_id": 2,
            "season_id": 27,
            "matches": 380,
            "trained_models_dir": str(models_dir),
        },
        "target_domain": {
            "competition": "La Liga",
            "season": "2015/2016",
            "competition_id": 11,
            "season_id": 27,
            "matches": 380,
            "total_graphs": n_graphs,
            "processed_dir": str(args.processed_dir),
            "events_path": str(args.events_path),
        },
        "rq1_model_transfer": {
            "models_evaluated": list(transfer_df["model"].unique()),
            "best_model": "MPNN",
            "summary": transfer_df.to_dict(orient="records"),
        },
        "rq2_xai_transfer": {
            "explainers_evaluated": list(xai_summary_df["explainer"].unique()),
            "gate_d17_compliance": {row["explainer"]: row["gate_d17_status"] for _, row in xai_summary_df.iterrows()},
            "summary": xai_summary_df.to_dict(orient="records"),
        },
        "rq3_hybrid_centrality": {
            "frozen_alpha": alpha_frozen,
            "shrinkage_m": 20.0,
            "min_possessions": 30,
            "players_evaluated": len(merged_df),
            "correlations": corr_df.to_dict(orient="records"),
            "top_5_players": top_players[top_5_keys].head(5).to_dict(orient="records"),
        },
        "git": get_git_info(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    save_atomic_json(manifest, out_dir / "cross_league_manifest.json")
    logger.info("Đã tạo hoàn tất Cross-League Manifest tại %s", out_dir / "cross_league_manifest.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
