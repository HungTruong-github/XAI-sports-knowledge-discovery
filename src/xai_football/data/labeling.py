"""Bước 2.3 — Gán nhãn possession.

Tham chiếu: docs/decisions/0001-quy-tac-gan-nhan.md.

Module cài đặt **cả hai** phương án để so sánh tỉ lệ lớp dương trên dữ liệu
thật, đúng cam kết với giảng viên ở báo cáo 19/09 mục 3.1:

    Phương án A (`possession_level`) — nhãn 1 nếu possession kết thúc bằng
        một cú sút. Một possession = một đồ thị = một nhãn.
    Phương án B (`pass_level`) — nhãn 1 cho đường chuyền dẫn đến cú sút trong
        N giây tiếp theo, tương tự cách tiếp cận của các mô hình xThreat.

Quy tắc chính thức dùng cho huấn luyện lấy từ `configs/data.yaml` → `labeling`;
KHÔNG được đổi sau khi đã huấn luyện (mục 12 — Rủi ro).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .possession import Possession

Mode = Literal["possession_level", "pass_level"]
Target = Literal["shot", "goal"]


@dataclass
class LabelStats:
    """Tỉ lệ lớp dương của một cách gán nhãn — căn cứ chốt quyết định 0001."""

    mode: str
    target: str
    n_units: int
    n_positive: int
    shot_window_seconds: float | None = None

    @property
    def positive_rate(self) -> float:
        return self.n_positive / self.n_units if self.n_units else 0.0

    def as_row(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "target": self.target,
            "shot_window_seconds": self.shot_window_seconds,
            "n_units": self.n_units,
            "n_positive": self.n_positive,
            "positive_rate": round(self.positive_rate, 4),
            "positive_rate_pct": round(self.positive_rate * 100, 2),
        }


def _hit(possession: Possession, target: Target) -> bool:
    return possession.ends_with_goal if target == "goal" else possession.ends_with_shot


def label_possession(possession: Possession, target: Target = "shot") -> int:
    """Phương án A — nhãn ở mức possession."""
    return int(_hit(possession, target))


def label_passes(
    possession: Possession,
    target: Target = "shot",
    window_seconds: float = 10.0,
) -> list[int]:
    """Phương án B — nhãn ở mức đường chuyền.

    Gán 1 cho đường chuyền nếu có ÍT NHẤT MỘT cú sút (phù hợp với target)
    xảy ra SAU đường chuyền và trong vòng `window_seconds`.
    """
    valid_shots = [s for s in possession.shots if (target != "goal" or s.is_goal)]
    if not valid_shots:
        return [0] * possession.n_passes

    shot_times = [s.time_seconds for s in valid_shots]
    labels = []
    for p in possession.passes:
        pass_time = p.minute * 60 + p.second
        is_pos = any(0 <= (st - pass_time) <= window_seconds for st in shot_times)
        labels.append(int(is_pos))
    return labels


def resolve_labeling(config: dict[str, Any]) -> tuple[Mode, Target, float | None]:
    """Đọc quy tắc gán nhãn chính thức từ cấu hình, báo lỗi rõ nếu chưa chốt."""
    labeling = config.get("labeling") or {}
    mode, target = labeling.get("mode"), labeling.get("target")

    if mode is None or target is None:
        raise ValueError(
            "configs/data.yaml -> labeling.mode / labeling.target chưa được điền. "
            "Đây là quyết định nghiên cứu chặn toàn bộ pipeline — xem "
            "docs/decisions/0001-quy-tac-gan-nhan.md, phải chốt trước khi huấn luyện."
        )
    if mode not in ("possession_level", "pass_level"):
        raise ValueError(f"labeling.mode không hợp lệ: {mode!r}")
    if target not in ("shot", "goal"):
        raise ValueError(f"labeling.target không hợp lệ: {target!r}")

    window = labeling.get("shot_window_seconds")
    if mode == "pass_level" and window is None:
        raise ValueError("labeling.mode = 'pass_level' thì bắt buộc phải có shot_window_seconds.")
    return mode, target, window


def apply_labeling(
    possessions: list[Possession], config: dict[str, Any]
) -> list[tuple[Possession, int | list[int]]]:
    """Gán nhãn theo đúng quy tắc đã chốt trong cấu hình."""
    mode, target, window = resolve_labeling(config)
    if mode == "possession_level":
        return [(p, label_possession(p, target)) for p in possessions]
    return [(p, label_passes(p, target, window or 10.0)) for p in possessions]


def compare_labeling_schemes(
    possessions: list[Possession],
    pass_level_windows: tuple[float, ...] = (5.0, 10.0, 15.0),
) -> list[LabelStats]:
    """Bảng so sánh tỉ lệ lớp dương của mọi cách gán nhãn.

    Đây chính là bằng chứng định lượng nhóm đã hứa mang tới buổi 26/09 để
    chốt quyết định 0001, thay vì chọn theo cảm tính.
    """
    stats: list[LabelStats] = []

    for target in ("shot", "goal"):
        stats.append(
            LabelStats(
                mode="possession_level",
                target=target,
                n_units=len(possessions),
                n_positive=sum(label_possession(p, target) for p in possessions),
            )
        )

    n_passes = sum(p.n_passes for p in possessions)
    for target in ("shot", "goal"):
        for window in pass_level_windows:
            stats.append(
                LabelStats(
                    mode="pass_level",
                    target=target,
                    n_units=n_passes,
                    n_positive=sum(sum(label_passes(p, target, window)) for p in possessions),
                    shot_window_seconds=window,
                )
            )

    return stats
