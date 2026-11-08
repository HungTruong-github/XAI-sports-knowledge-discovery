"""Bước 15 — Trực quan hóa kết quả thực nghiệm Cross-League (EPL -> La Liga 2015/16).

Tạo ra 2 biểu đồ ấn phẩm khoa học:
1. results/figures/12_cross_league_performance_degradation.svg (.png)
   So sánh hiệu năng EPL Test vs La Liga Zero-shot AUC-PR và AUC-ROC trên 4 kiến trúc GNN.
   Minh chứng ưu thế vượt trội của MPNN (Edge-conditioned) trước Domain Shift.

2. results/figures/13_cross_league_laliga_top_players.svg (.png)
   Bảng xếp hạng Top 15 cầu thủ La Liga 2015/16 theo Hybrid Centrality,
   thể hiện sự dung hòa giữa Betweenness Centrality (Cấu trúc mạng chuyền)
   và XAI Explainer Importance (Đóng góp xác suất tạo cơ hội sút bóng).
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Thiết lập phong cách đồ họa học thuật chuẩn quốc tế
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.family": "sans-serif",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
CROSS_LEAGUE_DIR = Path("results/cross_league")


def plot_performance_degradation() -> None:
    csv_path = CROSS_LEAGUE_DIR / "model_transfer_metrics.csv"
    if not csv_path.exists():
        print(f"Bỏ qua: Không tìm thấy {csv_path}")
        return

    df = pd.read_csv(csv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    x = np.arange(len(df))
    width = 0.35

    # 1. Biểu đồ AUC-PR
    ax1.bar(
        x - width / 2,
        df["epl_test_auc_pr"],
        width,
        label="EPL 2015/16 (In-domain Test)",
        color="#2b5c8f",
        alpha=0.9,
    )
    ax1.bar(
        x + width / 2,
        df["laliga_zero_shot_auc_pr"],
        width,
        label="La Liga 2015/16 (Zero-shot External)",
        color="#d95f02",
        alpha=0.9,
    )

    ax1.set_ylabel("AUC-PR")
    ax1.set_title("(a) Khả năng chống suy giảm hiệu năng AUC-PR (Domain Shift)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["model"], fontweight="bold")
    ax1.set_ylim(0.2, 0.7)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, loc="upper right")

    # Thêm nhãn % suy giảm trên thanh
    for i, (_, row) in enumerate(df.iterrows()):
        deg = row["degradation_pct"]
        ax1.annotate(
            f"{deg:+.1f}%",
            xy=(x[i] + width / 2, row["laliga_zero_shot_auc_pr"] + 0.015),
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#b2182b" if deg < -10 else "#1b7837",
        )

    # 2. Biểu đồ AUC-ROC
    ax2.bar(
        x - width / 2,
        df["epl_test_auc_roc"],
        width,
        label="EPL Test AUC-ROC",
        color="#41b6c4",
        alpha=0.9,
    )
    ax2.bar(
        x + width / 2,
        df["laliga_zero_shot_auc_roc"],
        width,
        label="La Liga Zero-shot AUC-ROC",
        color="#fe9929",
        alpha=0.9,
    )

    ax2.set_ylabel("AUC-ROC")
    ax2.set_title("(b) Độ phân biệt xác suất AUC-ROC xuyên giải đấu")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["model"], fontweight="bold")
    ax2.set_ylim(0.7, 0.95)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, loc="upper right")

    for i, (_, row) in enumerate(df.iterrows()):
        diff = row["laliga_zero_shot_auc_roc"] - row["epl_test_auc_roc"]
        ax2.annotate(
            f"{diff:+.3f}",
            xy=(x[i] + width / 2, row["laliga_zero_shot_auc_roc"] + 0.007),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#1b7837" if diff >= 0 else "#b2182b",
        )

    plt.tight_layout()
    fig.savefig(FIG_DIR / "12_cross_league_performance_degradation.svg")
    fig.savefig(FIG_DIR / "12_cross_league_performance_degradation.png")
    plt.close(fig)
    print("Đã lưu 12_cross_league_performance_degradation.png/.svg")


def plot_laliga_top_players() -> None:
    csv_path = CROSS_LEAGUE_DIR / "hybrid_centrality_top_players.csv"
    if not csv_path.exists():
        print(f"Bỏ qua: Không tìm thấy {csv_path}")
        return

    df = pd.read_csv(csv_path).head(15)
    # Đảo ngược thứ tự để cầu thủ hạng 1 ở trên cùng đồ thị ngang
    df = df.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6.5))

    y = np.arange(len(df))
    height = 0.55

    bars = ax.barh(y, df["hybrid_centrality_adj"], height, color="#2b5c8f", alpha=0.85, edgecolor="#1c3b5e")

    ax.set_yticks(y)
    # Tên cầu thủ kèm đội bóng
    labels = [f"{r['player_name']} ({r['team_name']})" for _, r in df.iterrows()]
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("Điểm Hybrid Centrality điều chỉnh (Empirical Bayes Shrinkage, m=20, α=0.70)")
    ax.set_title("Top 15 Tiền vệ / Cầu thủ kiến thiết La Liga 2015/16 phát hiện bởi Mô hình EPL", fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # Thêm giá trị cụ thể và thứ hạng ở đầu mỗi thanh
    for bar, (_, r) in zip(bars, df.iterrows()):
        val = r["hybrid_centrality_adj"]
        ax.text(
            val + 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f} (xT: {r.get('xt_value', 0):.2f})",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#222222",
        )

    ax.set_xlim(0, max(df["hybrid_centrality_adj"]) * 1.25)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "13_cross_league_laliga_top_players.svg")
    fig.savefig(FIG_DIR / "13_cross_league_laliga_top_players.png")
    plt.close(fig)
    print("Đã lưu 13_cross_league_laliga_top_players.png/.svg")


if __name__ == "__main__":
    plot_performance_degradation()
    plot_laliga_top_players()
