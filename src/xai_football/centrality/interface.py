"""Bước 9 — Định nghĩa Interface và Ranh giới Phương pháp luận cho Hybrid Centrality.

Tham chiếu: configs/hybrid_centrality.yaml và docs/decisions/0002-cong-thuc-hybrid-centrality.md.

CẢNH BÁO PHƯƠNG PHÁP LUẬN:
    - Công thức toán học chính thức của Hybrid Centrality và trọng số alpha hiện VẪN ĐANG PENDING,
      chờ phê duyệt của Giảng viên hướng dẫn.
    - Module này cung cấp:
        1. Ràng buộc kiểm tra cấu hình nghiêm ngặt (validate_hybrid_centrality_config) để ngăn chặn
           việc sử dụng âm thầm các tham số null.
        2. Đặc tả dữ liệu đầu vào / đầu ra (Interface & Schemas).
        3. Hàm thực thi prototype có kiểm soát (phải bật cờ allow_provisional=True).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from xai_football.utils.logging import get_logger

logger = get_logger("centrality_interface")


def validate_hybrid_centrality_config(cfg: dict[str, Any]) -> None:
    """Kiểm tra tính hợp lệ của cấu hình hybrid_centrality.yaml.

    Ném lỗi tường minh nếu các trường bắt buộc đang null khi cố gắng thực thi chính thức.
    """
    trad = cfg.get("traditional", {})
    xai = cfg.get("xai_importance", {})
    comb = cfg.get("combination", {})

    missing_fields = []

    if trad.get("aggregate_level") is None:
        missing_fields.append("traditional.aggregate_level")
    if trad.get("combine_measures") is None:
        missing_fields.append("traditional.combine_measures")
    if xai.get("source_explainer") is None:
        missing_fields.append("xai_importance.source_explainer")
    if xai.get("aggregation") is None:
        missing_fields.append("xai_importance.aggregation")
    if comb.get("normalization") is None:
        missing_fields.append("combination.normalization")
    if comb.get("alpha") is None and comb.get("alpha_selection_criterion") is None:
        missing_fields.append("combination.alpha (hoặc alpha_selection_criterion)")

    if missing_fields:
        raise ValueError(
            f"Cấu hình Hybrid Centrality chưa hoàn thiện (Decision 0002 Pending). "
            f"Các trường bắt buộc đang null: {missing_fields}. "
            f"Không được tự ý áp dụng công thức khi chưa có văn bản phê duyệt chính thức."
        )


def compute_provisional_hybrid_centrality(
    traditional_df: pd.DataFrame,
    xai_player_df: pd.DataFrame,
    alpha: float = 0.5,
    allow_provisional: bool = False,
) -> pd.DataFrame:
    """Thực thi prototype kỹ thuật có gắn nhãn PROVISIONAL/EXPERIMENTAL.

    Args:
        traditional_df: Bảng độ đo centrality truyền thống (degree, betweenness, ...).
        xai_player_df: Bảng điểm XAI tổng hợp của cầu thủ từ Bước 7.
        alpha: Hệ số kết hợp thử nghiệm (mặc định 0.5).
        allow_provisional: Phải đặt True để xác nhận đây chỉ là prototype thử nghiệm kỹ thuật.

    Returns:
        DataFrame chứa điểm provisional_hybrid_centrality kèm cảnh báo.
    """
    if not allow_provisional:
        raise RuntimeError(
            "Chức năng tính Hybrid Centrality đang ở trạng thái PROVISIONAL (Decision 0002 Pending). "
            "Để chạy thử nghiệm kỹ thuật, phải truyền allow_provisional=True."
        )

    logger.warning(
        "ĐANG CHẠY HYBRID CENTRALITY PROTOTYPE (EXPERIMENTAL ONLY - KHÔNG PHẢI KẾT QUẢ CHÍNH THỨC)"
    )

    merged = pd.merge(traditional_df, xai_player_df, on="player_id", how="inner")
    if merged.empty:
        return pd.DataFrame()

    # Chuẩn hóa min-max kỹ thuật
    def _minmax(s: pd.Series) -> pd.Series:
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 1e-8 else pd.Series(0.5, index=s.index)

    c_norm = _minmax(merged.get("degree_in", pd.Series(0, index=merged.index)))
    x_norm = _minmax(merged.get("total_importance", pd.Series(0, index=merged.index)))

    merged["provisional_alpha"] = alpha
    merged["provisional_hybrid_centrality"] = alpha * c_norm + (1.0 - alpha) * x_norm
    merged["is_official_result"] = False
    merged["decision_0002_status"] = "pending_approval"

    return merged
