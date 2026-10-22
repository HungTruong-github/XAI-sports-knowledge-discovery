"""Bước 2.2 và 2.4 — Làm sạch dữ liệu và chuẩn hóa không gian sân.

Tham chiếu: docs/Quy_trinh_XAI_Football.md mục 2.2, 2.4.

Nguyên tắc: sự kiện thiếu tọa độ bị **loại bỏ**, không suy diễn (impute) —
tránh đưa nhiễu vào đồ thị (`configs/data.yaml` → `impute_coordinates: false`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..utils.logging import get_logger
from .possession import Possession

logger = get_logger(__name__)

# StatsBomb: period 1-2 là hai hiệp chính, 3-4 là hiệp phụ, 5 là luân lưu.
REGULAR_PERIODS = (1, 2)


@dataclass
class CleaningReport:
    """Đếm số possession bị loại ở từng bước — phục vụ mô tả dataset."""

    n_input: int = 0
    dropped_extra_time: int = 0
    dropped_missing_coords: int = 0
    dropped_too_short: int = 0
    dropped_no_players: int = 0
    # Cùng các bước trên nhưng chỉ đếm possession CÓ SÚT (has_shot) — cho biết
    # bộ lọc làm mất bao nhiêu mẫu dương, ảnh hưởng trực tiếp tới Decision 0001.
    shot_input: int = 0
    shot_dropped_extra_time: int = 0
    shot_dropped_missing_coords: int = 0
    shot_dropped_too_short: int = 0
    shot_dropped_no_players: int = 0

    @property
    def n_output(self) -> int:
        return (
            self.n_input
            - self.dropped_extra_time
            - self.dropped_missing_coords
            - self.dropped_too_short
            - self.dropped_no_players
        )

    @property
    def shot_output(self) -> int:
        return (
            self.shot_input
            - self.shot_dropped_extra_time
            - self.shot_dropped_missing_coords
            - self.shot_dropped_too_short
            - self.shot_dropped_no_players
        )

    def as_row(self) -> dict[str, Any]:
        return {
            "possessions_input": self.n_input,
            "dropped_extra_time": self.dropped_extra_time,
            "dropped_missing_coords": self.dropped_missing_coords,
            "dropped_too_short": self.dropped_too_short,
            "dropped_no_players": self.dropped_no_players,
            "possessions_output": self.n_output,
            "shot_possessions_input": self.shot_input,
            "shot_dropped_extra_time": self.shot_dropped_extra_time,
            "shot_dropped_missing_coords": self.shot_dropped_missing_coords,
            "shot_dropped_too_short": self.shot_dropped_too_short,
            "shot_dropped_no_players": self.shot_dropped_no_players,
            "shot_possessions_output": self.shot_output,
            "shot_retention_rate": (
                round(self.shot_output / self.shot_input, 4) if self.shot_input else 0.0
            ),
        }

    def merge(self, other: CleaningReport) -> CleaningReport:
        return CleaningReport(
            n_input=self.n_input + other.n_input,
            dropped_extra_time=self.dropped_extra_time + other.dropped_extra_time,
            dropped_missing_coords=self.dropped_missing_coords + other.dropped_missing_coords,
            dropped_too_short=self.dropped_too_short + other.dropped_too_short,
            dropped_no_players=self.dropped_no_players + other.dropped_no_players,
            shot_input=self.shot_input + other.shot_input,
            shot_dropped_extra_time=self.shot_dropped_extra_time + other.shot_dropped_extra_time,
            shot_dropped_missing_coords=(
                self.shot_dropped_missing_coords + other.shot_dropped_missing_coords
            ),
            shot_dropped_too_short=self.shot_dropped_too_short + other.shot_dropped_too_short,
            shot_dropped_no_players=self.shot_dropped_no_players + other.shot_dropped_no_players,
        )


def flip_possession(possession: Possession, pitch_length: float, pitch_width: float) -> None:
    """Lật tọa độ 180 độ quanh tâm sân (sửa tại chỗ).

    Dùng khi một pha bóng được ghi theo hướng tấn công ngược với quy ước.
    """
    for p in possession.passes:
        if p.start_x is not None:
            p.start_x = pitch_length - p.start_x
        if p.start_y is not None:
            p.start_y = pitch_width - p.start_y
        if p.end_x is not None:
            p.end_x = pitch_length - p.end_x
        if p.end_y is not None:
            p.end_y = pitch_width - p.end_y
        if p.angle is not None:
            # Góc quay 180 độ, giữ trong khoảng (-pi, pi]
            import math

            p.angle = (p.angle + math.pi + math.pi) % (2 * math.pi) - math.pi


def clean_possessions(
    possessions: list[Possession],
    config: dict[str, Any],
) -> tuple[list[Possession], CleaningReport]:
    """Áp dụng toàn bộ quy tắc làm sạch của mục 2.2 và chuẩn hóa mục 2.4."""
    prep = config.get("preprocessing") or {}
    pitch = config.get("pitch") or {}

    if prep.get("impute_coordinates", False):
        # Không có code suy diễn tọa độ; báo lỗi thay vì lặng lẽ bỏ qua cấu hình.
        raise NotImplementedError(
            "preprocessing.impute_coordinates = true chưa được cài đặt "
            "(dự án chọn loại bỏ sự kiện thiếu tọa độ, không suy diễn)."
        )

    min_passes = prep.get("min_passes_per_possession", 2)
    drop_extra_time = prep.get("drop_extra_time", True)
    drop_missing_coords = prep.get("drop_missing_coordinates", True)
    pitch_length = pitch.get("length", 120)
    pitch_width = pitch.get("width", 80)

    report = CleaningReport(
        n_input=len(possessions),
        shot_input=sum(1 for p in possessions if p.has_shot),
    )
    kept: list[Possession] = []

    for possession in possessions:
        # Làm sạch chỉ lọc `passes`, không đụng `shots`, nên has_shot cố định
        # trong suốt vòng lặp.
        is_shot = possession.has_shot

        if drop_extra_time and possession.period not in REGULAR_PERIODS:
            report.dropped_extra_time += 1
            report.shot_dropped_extra_time += is_shot
            continue

        if drop_missing_coords:
            n_before = possession.n_passes
            possession.passes = [p for p in possession.passes if p.has_coordinates]
            if possession.n_passes < n_before and possession.n_passes < min_passes:
                report.dropped_missing_coords += 1
                report.shot_dropped_missing_coords += is_shot
                continue

        if possession.n_passes < min_passes:
            report.dropped_too_short += 1
            report.shot_dropped_too_short += is_shot
            continue

        # Cạnh cần cả hai đầu; đường chuyền không xác định được người nhận
        # thì không dựng được cạnh có hướng A -> B (mục 3.2).
        possession.passes = [p for p in possession.passes if p.passer_id and p.recipient_id]
        if possession.n_passes < min_passes or possession.n_players < 2:
            report.dropped_no_players += 1
            report.shot_dropped_no_players += is_shot
            continue

        kept.append(possession)

    if pitch.get("normalize_attack_direction", True):
        normalize_attack_direction(
            kept,
            pitch_length,
            pitch_width,
            force_flip=bool(pitch.get("force_flip_misoriented", False)),
        )

    return kept, report


def normalize_attack_direction(
    possessions: list[Possession],
    pitch_length: float = 120,
    pitch_width: float = 80,
    force_flip: bool = False,
) -> int:
    """Kiểm tra mọi possession đều tấn công về phía x lớn (khung thành x = 120).

    StatsBomb **đã chuẩn hóa sẵn** theo quy ước này, nên mặc định hàm chỉ
    kiểm tra và cảnh báo, KHÔNG tự lật.

    Lý do không tự lật: đã kiểm chứng trên dữ liệu thật (Premier League
    2015/16, 5 trận) rằng các possession có cú sút nhưng đường chuyền dừng ở
    nửa sân nhà là **pha dẫn bóng (carry) dài rồi sút**, không phải lỗi tọa độ.
    Tự động lật theo suy đoán sẽ tạo ra đúng rủi ro âm thầm A5 mà tài liệu rủi
    ro cảnh báo: mô hình vẫn học được, chỉ là học sai không gian.

    `force_flip=True` chỉ nên bật cho nguồn dữ liệu đã xác minh là chưa chuẩn
    hóa (ví dụ Wyscout), và phải ghi lại trong biên bản quyết định.
    """
    suspicious = [
        p
        for p in possessions
        if p.has_shot
        and (xs := [x.end_x for x in p.passes if x.end_x is not None])
        and max(xs) < pitch_length / 2
    ]

    if not suspicious:
        return 0

    share = 100 * len(suspicious) / len(possessions) if possessions else 0.0
    if not force_flip:
        logger.info(
            "%d/%d possession (%.1f%%) có cú sút nhưng đường chuyền dừng ở nửa sân nhà "
            "— dự kiến là pha dẫn bóng dài rồi sút, không lật tọa độ. "
            "Nếu tỉ lệ này vượt ~5%% thì mới nghi ngờ nguồn dữ liệu chưa chuẩn hóa.",
            len(suspicious),
            len(possessions),
            share,
        )
        return 0

    for possession in suspicious:
        flip_possession(possession, pitch_length, pitch_width)
    logger.warning("Đã lật hướng tấn công cho %d possession (force_flip=True)", len(suspicious))
    return len(suspicious)


def play_pattern_distribution(
    raw: list[Possession], clean: list[Possession]
) -> list[dict[str, Any]]:
    """Phân bố play pattern trước và sau làm sạch, kèm tỉ lệ có sút sau làm sạch.

    Căn cứ cho quyết định phạm vi open-play: bảng cũ chỉ đếm trên possession
    thô, trong khi mô hình học trên possession đã làm sạch.
    """

    def _key(p: Possession) -> str:
        return p.play_pattern or "Unknown"

    raw_counts: dict[str, int] = {}
    for p in raw:
        raw_counts[_key(p)] = raw_counts.get(_key(p), 0) + 1

    clean_counts: dict[str, int] = {}
    clean_shots: dict[str, int] = {}
    for p in clean:
        clean_counts[_key(p)] = clean_counts.get(_key(p), 0) + 1
        clean_shots[_key(p)] = clean_shots.get(_key(p), 0) + int(p.has_shot)

    n_raw, n_clean = len(raw), len(clean)
    rows = []
    patterns = sorted(
        raw_counts.keys() | clean_counts.keys(), key=lambda k: (-raw_counts.get(k, 0), k)
    )
    for pattern in patterns:
        c_raw = raw_counts.get(pattern, 0)
        c_clean = clean_counts.get(pattern, 0)
        s_clean = clean_shots.get(pattern, 0)
        rows.append(
            {
                "play_pattern": pattern,
                "count_raw": c_raw,
                "pct_raw": round(100 * c_raw / n_raw, 2) if n_raw else 0.0,
                "count_clean": c_clean,
                "pct_clean": round(100 * c_clean / n_clean, 2) if n_clean else 0.0,
                "shot_count_clean": s_clean,
                "shot_rate_clean": round(100 * s_clean / c_clean, 2) if c_clean else 0.0,
            }
        )
    return rows


def attack_direction_diagnostic(possessions: list[Possession]) -> dict[str, float]:
    """Chẩn đoán hướng tấn công — phép kiểm tra A5 đã hứa với giảng viên.

    Trả về tọa độ x trung bình của điểm nhận bóng, tách riêng nhóm possession
    dẫn đến sút và nhóm không. Nhóm dẫn đến sút phải có x trung bình **cao hơn
    rõ rệt** (gần khung thành đối phương); nếu không, tọa độ chưa đồng bộ.
    """
    shot_xs, other_xs = [], []
    for possession in possessions:
        xs = [p.end_x for p in possession.passes if p.end_x is not None]
        if not xs:
            continue
        (shot_xs if possession.has_shot else other_xs).append(sum(xs) / len(xs))

    mean_shot = sum(shot_xs) / len(shot_xs) if shot_xs else float("nan")
    mean_other = sum(other_xs) / len(other_xs) if other_xs else float("nan")
    return {
        "mean_end_x_shot_possessions": round(mean_shot, 2),
        "mean_end_x_other_possessions": round(mean_other, 2),
        "gap": round(mean_shot - mean_other, 2),
    }
