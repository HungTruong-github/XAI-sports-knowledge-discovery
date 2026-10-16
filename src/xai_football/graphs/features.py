"""Bước 3.1-3.2 — Đặc trưng đỉnh và cạnh của đồ thị possession.

Tham chiếu: docs/Quy_trinh_XAI_Football.md mục 3.1, 3.2; configs/graph.yaml.

Nguyên tắc:
  - player_id KHÔNG BAO GIỜ là đặc trưng số (quy ước ở data/possession.py).
  - Chuẩn hóa bằng HẰNG SỐ kích thước sân, không bằng thống kê của dữ liệu —
    tránh rò rỉ thông tin tập test sang tập train trước khi chia tập (Bước 4).
  - Tên đặc trưng sinh cùng chỗ với giá trị, để XAI (Bước 7) đọc được
    "chiều thứ k của edge_attr là gì".
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

# Gom 25 vị trí StatsBomb (position.id 1-25) về 4 vai trò. Tiền vệ cánh (Right/Left Wing)
# xếp vào FW vì hoạt động chủ yếu ở 1/3 sân tấn công.
ROLES = ("GK", "DF", "MF", "FW")
POSITION_TO_ROLE: dict[str, str] = {
    "Goalkeeper": "GK",
    "Right Back": "DF",
    "Right Center Back": "DF",
    "Center Back": "DF",
    "Left Center Back": "DF",
    "Left Back": "DF",
    "Right Wing Back": "DF",
    "Left Wing Back": "DF",
    "Right Defensive Midfield": "MF",
    "Center Defensive Midfield": "MF",
    "Left Defensive Midfield": "MF",
    "Right Midfield": "MF",
    "Right Center Midfield": "MF",
    "Center Midfield": "MF",
    "Left Center Midfield": "MF",
    "Left Midfield": "MF",
    "Right Attacking Midfield": "MF",
    "Center Attacking Midfield": "MF",
    "Left Attacking Midfield": "MF",
    "Right Wing": "FW",
    "Left Wing": "FW",
    "Right Center Forward": "FW",
    "Center Forward": "FW",
    "Left Center Forward": "FW",
    "Secondary Striker": "FW",
}

# Giá trị pass.height của StatsBomb
PASS_HEIGHTS = ("Ground Pass", "Low Pass", "High Pass")


def role_of(position_name: str | None) -> str | None:
    """Vai trò GK/DF/MF/FW của một vị trí StatsBomb; None nếu không xác định."""
    if position_name is None:
        return None
    return POSITION_TO_ROLE.get(position_name)


def _enabled(features_cfg: dict[str, Any], key: str) -> bool:
    return bool(features_cfg.get(key, False))


def node_feature_names(features_cfg: dict[str, Any]) -> list[str]:
    """Tên từng chiều của `x`, theo đúng thứ tự `compute_node_features` sinh ra."""
    if _enabled(features_cfg, "season_stats"):
        raise NotImplementedError("nodes.features.season_stats chưa được cài đặt.")
    names: list[str] = []
    if _enabled(features_cfg, "position"):
        names += [f"role_{r}" for r in ROLES]
    if _enabled(features_cfg, "mean_xy"):
        names += ["mean_x", "mean_y"]
    if _enabled(features_cfg, "touch_count"):
        names += ["n_passes_made", "n_passes_received"]
    if not names:
        raise ValueError("configs/graph.yaml: phải bật ít nhất một đặc trưng đỉnh.")
    return names


def edge_feature_names(features_cfg: dict[str, Any]) -> list[str]:
    """Tên từng chiều của `edge_attr`, theo đúng thứ tự `compute_edge_features`."""
    names: list[str] = []
    if _enabled(features_cfg, "start_end_xy"):
        names += ["start_x", "start_y", "end_x", "end_y"]
    if _enabled(features_cfg, "distance"):
        names += ["length"]
    if _enabled(features_cfg, "angle"):
        names += ["angle_sin", "angle_cos"]
    if _enabled(features_cfg, "pass_type"):
        names += ["height_ground", "height_low", "height_high"]
    if _enabled(features_cfg, "timestamp"):
        names += ["order_in_possession", "elapsed_minutes"]
    if not names:
        raise ValueError("configs/graph.yaml: phải bật ít nhất một đặc trưng cạnh.")
    return names


def _pass_seconds(p: dict[str, Any]) -> float:
    return float(p["minute"]) * 60 + float(p["second"])


def compute_node_features(
    player_ids: Sequence[int],
    passes: Sequence[dict[str, Any]],
    roles: dict[int, str | None],
    features_cfg: dict[str, Any],
    pitch_length: float = 120.0,
    pitch_width: float = 80.0,
) -> list[list[float]]:
    """Ma trận đặc trưng đỉnh, mỗi hàng ứng với `player_ids[i]`.

    Một "lần chạm bóng" của cầu thủ = điểm bắt đầu đường chuyền họ thực hiện
    hoặc điểm kết thúc đường chuyền họ nhận.
    """
    touches: dict[int, list[tuple[float, float]]] = {pid: [] for pid in player_ids}
    n_made: dict[int, int] = dict.fromkeys(player_ids, 0)
    n_received: dict[int, int] = dict.fromkeys(player_ids, 0)
    for p in passes:
        touches[p["passer_id"]].append((p["start_x"], p["start_y"]))
        touches[p["recipient_id"]].append((p["end_x"], p["end_y"]))
        n_made[p["passer_id"]] += 1
        n_received[p["recipient_id"]] += 1

    rows: list[list[float]] = []
    for pid in player_ids:
        row: list[float] = []
        if _enabled(features_cfg, "position"):
            role = roles.get(pid)
            row += [1.0 if role == r else 0.0 for r in ROLES]
        if _enabled(features_cfg, "mean_xy"):
            xs = [t[0] for t in touches[pid]]
            ys = [t[1] for t in touches[pid]]
            row += [sum(xs) / len(xs) / pitch_length, sum(ys) / len(ys) / pitch_width]
        if _enabled(features_cfg, "touch_count"):
            row += [float(n_made[pid]), float(n_received[pid])]
        rows.append(row)
    return rows


def compute_edge_features(
    passes: Sequence[dict[str, Any]],
    features_cfg: dict[str, Any],
    pitch_length: float = 120.0,
    pitch_width: float = 80.0,
) -> list[list[float]]:
    """Ma trận đặc trưng cạnh, mỗi hàng ứng với một đường chuyền (đã xếp theo thời gian).

    Độ dài và góc được tính lại từ tọa độ (cùng hệ tọa độ StatsBomb, y hướng
    xuống) thay vì đọc pass.length/pass.angle — không bao giờ thiếu giá trị và
    luôn nhất quán với x1,y1,x2,y2.
    """
    diagonal = math.hypot(pitch_length, pitch_width)
    n = len(passes)
    t0 = _pass_seconds(passes[0]) if n else 0.0

    rows: list[list[float]] = []
    for i, p in enumerate(passes):
        dx = p["end_x"] - p["start_x"]
        dy = p["end_y"] - p["start_y"]
        row: list[float] = []
        if _enabled(features_cfg, "start_end_xy"):
            row += [
                p["start_x"] / pitch_length,
                p["start_y"] / pitch_width,
                p["end_x"] / pitch_length,
                p["end_y"] / pitch_width,
            ]
        if _enabled(features_cfg, "distance"):
            row += [math.hypot(dx, dy) / diagonal]
        if _enabled(features_cfg, "angle"):
            angle = math.atan2(dy, dx)
            row += [math.sin(angle), math.cos(angle)]
        if _enabled(features_cfg, "pass_type"):
            row += [1.0 if p.get("height") == h else 0.0 for h in PASS_HEIGHTS]
        if _enabled(features_cfg, "timestamp"):
            row += [i / (n - 1) if n > 1 else 0.0, (_pass_seconds(p) - t0) / 60.0]
        rows.append(row)
    return rows
