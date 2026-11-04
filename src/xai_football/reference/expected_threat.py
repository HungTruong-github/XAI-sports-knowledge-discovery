"""Mô hình Expected Threat (xT) trên lưới Markov 16x12 theo tọa độ sân StatsBomb.

Tham chiếu:
- Karun Singh (2019): "Introducing Expected Threat (xT)"
- docs/decisions/0002-cong-thuc-hybrid-centrality.md (Quyết định D15)
Nguyên tắc:
- Lưới 16 cột (trục x: 0..120, mỗi ô 7.5m) x 12 hàng (trục y: 0..80, mỗi ô 6.67m).
- Huấn luyện CHỈ trên tập trận đấu Set A (không rò rỉ sang Set B).
- Đánh giá đóng góp xT của cầu thủ trên tập trận đấu Set B.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xai_football.utils.logging import get_logger

logger = get_logger("expected_threat")


class ExpectedThreatModel:
    """Mô hình Markov Expected Threat (xT) dạng lưới 16x12."""

    def __init__(self, n_cols: int = 16, n_rows: int = 12, max_iter: int = 25, tol: float = 1e-4) -> None:
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.n_cells = n_cols * n_rows
        self.max_iter = max_iter
        self.tol = tol

        self.cell_w = 120.0 / n_cols
        self.cell_h = 80.0 / n_rows

        self.is_fitted = False
        self.grid_values = np.zeros((n_rows, n_cols), dtype=float)

    def coord_to_cell(self, x: float, y: float) -> tuple[int, int]:
        """Chuyển đổi tọa độ sân (x, y) sang chỉ số ô lưới (row_idx, col_idx)."""
        col = int(np.clip(np.floor(x / self.cell_w), 0, self.n_cols - 1))
        row = int(np.clip(np.floor(y / self.cell_h), 0, self.n_rows - 1))
        return row, col

    def coord_to_index(self, x: float, y: float) -> int:
        """Chuyển đổi tọa độ (x, y) sang chỉ số 1D phẳng (0..191)."""
        r, c = self.coord_to_cell(x, y)
        return r * self.n_cols + c

    def coords_to_indices(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Chuyển đổi vector tọa độ (x, y) sang vector chỉ số 1D phẳng (vectorized)."""
        cols = np.clip(np.floor(xs / self.cell_w), 0, self.n_cols - 1).astype(int)
        rows = np.clip(np.floor(ys / self.cell_h), 0, self.n_rows - 1).astype(int)
        return rows * self.n_cols + cols

    def fit(self, events_df: pd.DataFrame, match_ids: set[int] | None = None) -> ExpectedThreatModel:
        """Huấn luyện mô hình xT bằng quy hoạch động Bellman trên dữ liệu sự kiện.

        BẮT BUỘC: match_ids chỉ nên chứa các trận thuộc Set A để chống rò rỉ.
        """
        df = events_df
        if match_ids is not None:
            df = df[df["match_id"].isin(match_ids)]

        logger.info("Huấn luyện mô hình xT trên %d sự kiện (%d trận)", len(df), df["match_id"].nunique())

        # Khởi tạo ma trận đếm
        total_actions = np.zeros(self.n_cells, dtype=float)
        shot_counts = np.zeros(self.n_cells, dtype=float)
        goal_counts = np.zeros(self.n_cells, dtype=float)
        move_matrix = np.zeros((self.n_cells, self.n_cells), dtype=float)

        # 1. Thu thập các hành động Shot (vectorized)
        shots = df[df["event_type"] == "Shot"]
        valid_shots = shots[shots["x"].notna() & shots["y"].notna()]
        if not valid_shots.empty:
            s_idx = self.coords_to_indices(
                valid_shots["x"].to_numpy(dtype=float),
                valid_shots["y"].to_numpy(dtype=float),
            )
            np.add.at(total_actions, s_idx, 1.0)
            np.add.at(shot_counts, s_idx, 1.0)
            is_goal = (valid_shots["shot_outcome"] == "Goal").to_numpy(dtype=bool)
            np.add.at(goal_counts, s_idx[is_goal], 1.0)

        # 2. Thu thập các hành động di chuyển bóng (Passes & Carries thành công, vectorized)
        p_mask = (
            (df["event_type"] == "Pass")
            & df["pass_outcome"].isna()
            & df["x"].notna()
            & df["y"].notna()
            & df["pass_end_x"].notna()
            & df["pass_end_y"].notna()
        )
        c_mask = (
            (df["event_type"] == "Carry")
            & df["x"].notna()
            & df["y"].notna()
            & df["pass_end_x"].notna()
            & df["pass_end_y"].notna()
        )
        moves = df[p_mask | c_mask]
        if not moves.empty:
            u_idx = self.coords_to_indices(
                moves["x"].to_numpy(dtype=float),
                moves["y"].to_numpy(dtype=float),
            )
            v_idx = self.coords_to_indices(
                moves["pass_end_x"].to_numpy(dtype=float),
                moves["pass_end_y"].to_numpy(dtype=float),
            )
            np.add.at(total_actions, u_idx, 1.0)
            np.add.at(move_matrix, (u_idx, v_idx), 1.0)

        # 3. Tính toán xác suất chuyển trạng thái
        shot_prob = np.zeros(self.n_cells, dtype=float)
        goal_prob = np.zeros(self.n_cells, dtype=float)
        trans_prob = np.zeros((self.n_cells, self.n_cells), dtype=float)

        for u in range(self.n_cells):
            tot = total_actions[u]
            if tot > 0:
                shot_prob[u] = shot_counts[u] / tot
                if shot_counts[u] > 0:
                    goal_prob[u] = goal_counts[u] / shot_counts[u]
                trans_prob[u, :] = move_matrix[u, :] / tot

        # 4. Giải phương trình quy hoạch động Bellman
        # V(u) = S(u) * G(u) + sum_v T(u, v) * V(v)
        V = np.zeros(self.n_cells, dtype=float)
        immediate_reward = shot_prob * goal_prob

        for it in range(self.max_iter):
            V_next = immediate_reward + trans_prob.dot(V)
            diff = np.max(np.abs(V_next - V))
            V = V_next
            if diff < self.tol:
                logger.info("xT hội tụ sau %d vòng lặp (max diff: %.6f)", it + 1, diff)
                break

        self.grid_values = V.reshape((self.n_rows, self.n_cols))
        self.is_fitted = True
        logger.info(
            "Huấn luyện xT hoàn tất! Giá trị min=%.4f, max=%.4f",
            self.grid_values.min(),
            self.grid_values.max(),
        )
        return self

    def get_value(self, x: float, y: float) -> float:
        """Lấy giá trị xT tại tọa độ (x, y)."""
        if not self.is_fitted:
            raise RuntimeError("Mô hình xT chưa được huấn luyện! Hãy gọi fit() trước.")
        r, c = self.coord_to_cell(x, y)
        return float(self.grid_values[r, c])

    def compute_action_xt(
        self,
        events_df: pd.DataFrame,
        match_ids: set[int] | None = None,
    ) -> pd.DataFrame:
        """Tính gia số Delta xT cho từng đường chuyền và lần dẫn bóng thành công."""
        if not self.is_fitted:
            raise RuntimeError("Mô hình xT chưa được fit!")

        df = events_df
        if match_ids is not None:
            df = df[df["match_id"].isin(match_ids)]

        # Lọc passes thành công và carries
        valid_mask = (
            ((df["event_type"] == "Pass") & df["pass_outcome"].isna())
            | (df["event_type"] == "Carry")
        ) & (df["x"].notna()) & (df["y"].notna()) & (df["pass_end_x"].notna()) & (df["pass_end_y"].notna())

        actions = df[valid_mask].copy()
        if actions.empty:
            return pd.DataFrame()

        u_idx = self.coords_to_indices(
            actions["x"].to_numpy(dtype=float),
            actions["y"].to_numpy(dtype=float),
        )
        v_idx = self.coords_to_indices(
            actions["pass_end_x"].to_numpy(dtype=float),
            actions["pass_end_y"].to_numpy(dtype=float),
        )

        v_flat = self.grid_values.flatten()
        v_start = v_flat[u_idx]
        v_end = v_flat[v_idx]

        actions["xt_start"] = v_start
        actions["xt_end"] = v_end
        # Delta xT dương (chỉ tính đóng góp tích cực tạo nguy hiểm)
        actions["delta_xt"] = np.maximum(0.0, v_end - v_start)

        return actions

    def compute_player_xt(
        self,
        events_df: pd.DataFrame,
        minutes_df: pd.DataFrame | None = None,
        match_ids: set[int] | None = None,
    ) -> pd.DataFrame:
        """Tổng hợp xT theo cầu thủ và tính xT/90."""
        actions = self.compute_action_xt(events_df, match_ids=match_ids)
        if actions.empty:
            return pd.DataFrame(columns=["player_id", "player_name", "team_name", "total_xt", "xt_per_90"])

        agg = actions.groupby(["player_id", "player_name"]).agg(
            team_name=("team_name", "last"),
            total_xt=("delta_xt", "sum"),
            action_count=("event_id", "count"),
        ).reset_index()

        agg["total_xt"] = np.round(agg["total_xt"], 4)

        if minutes_df is not None and not minutes_df.empty:
            merged = pd.merge(minutes_df, agg, on=["player_id", "player_name"], how="left")
            merged["team_name"] = merged["team_name_x"].combine_first(merged["team_name_y"])
            merged = merged.drop(columns=["team_name_x", "team_name_y"], errors="ignore")
            merged["total_xt"] = merged["total_xt"].fillna(0.0)
            merged["action_count"] = merged["action_count"].fillna(0).astype(int)

            mins = merged["minutes_played"]
            merged["xt_per_90"] = np.where(mins >= 90.0, np.round((merged["total_xt"] / mins) * 90.0, 4), 0.0)
            return merged.sort_values(by="total_xt", ascending=False).reset_index(drop=True)

        return agg.sort_values(by="total_xt", ascending=False).reset_index(drop=True)
