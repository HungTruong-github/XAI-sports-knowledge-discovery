"""Data quality / sanity checks — tách riêng để dùng lại được.

Các check trả về PASS / WARNING / FAIL kèm lý do.
FAIL nghiêm trọng khiến pipeline trả exit code != 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CheckResult:
    """Kết quả một phép kiểm tra."""

    check: str
    value: Any
    status: str  # "PASS" | "WARNING" | "FAIL"
    note: str = ""

    def as_row(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "value": self.value,
            "status": self.status,
            "note": self.note,
        }


def check_match_count(
    n_matches: int,
    expected: int | None = None,
    strict: bool = False,
) -> CheckResult:
    """Kiểm tra số trận.

    Parameters
    ----------
    strict : bool
        Nếu True (official run), bất kỳ sai lệch nào so với expected đều FAIL.
        Nếu False (pilot), chỉ WARNING khi lệch.
    """
    if expected is not None and n_matches != expected:
        return CheckResult(
            check=f"match_count == {expected}",
            value=n_matches,
            status="FAIL" if strict else "WARNING",
            note=f"Kỳ vọng {expected}, thực tế {n_matches}",
        )
    return CheckResult(
        check=f"match_count == {expected or n_matches}",
        value=n_matches,
        status="PASS",
    )


def check_unique_teams(
    n_teams: int,
    expected_range: tuple[int, int] = (18, 24),
) -> CheckResult:
    """Premier League: 20 đội."""
    lo, hi = expected_range
    ok = lo <= n_teams <= hi
    return CheckResult(
        check=f"unique_teams in [{lo}, {hi}]",
        value=n_teams,
        status="PASS" if ok else "WARNING",
        note="" if ok else f"Số đội = {n_teams}, ngoài khoảng [{lo}, {hi}]",
    )


def check_events_per_match(
    events_per_match: list[int],
    expected_range: tuple[int, int] = (1000, 4500),
) -> CheckResult:
    """StatsBomb event count per match thường trong khoảng 1500–3500."""
    import numpy as np

    arr = np.array(events_per_match)
    median_val = float(np.median(arr))
    lo, hi = expected_range
    ok = lo <= median_val <= hi
    return CheckResult(
        check=f"median_events_per_match in [{lo}, {hi}]",
        value=round(median_val, 1),
        status="PASS" if ok else "WARNING",
        note=f"min={int(arr.min())}, max={int(arr.max())}",
    )


def check_possessions_per_match(
    possessions_per_match: list[int],
    expected_range: tuple[int, int] = (90, 130),
) -> CheckResult:
    """Kiểm tra possession/trận."""
    import numpy as np

    arr = np.array(possessions_per_match)
    median_val = float(np.median(arr))
    lo, hi = expected_range
    ok = lo <= median_val <= hi
    return CheckResult(
        check=f"median_possessions_per_match in [{lo}, {hi}]",
        value=round(median_val, 1),
        status="PASS" if ok else "WARNING",
        note=f"min={int(arr.min())}, max={int(arr.max())}",
    )


def check_successful_passes_per_match(
    passes_per_match: list[int],
    expected_range: tuple[int, int] = (500, 1200),
) -> CheckResult:
    """Số pass thành công mỗi trận (cả hai đội)."""
    import numpy as np

    arr = np.array(passes_per_match)
    median_val = float(np.median(arr))
    lo, hi = expected_range
    ok = lo <= median_val <= hi
    return CheckResult(
        check=f"median_successful_passes_per_match in [{lo}, {hi}]",
        value=round(median_val, 1),
        status="PASS" if ok else "WARNING",
        note=f"min={int(arr.min())}, max={int(arr.max())}",
    )


def check_missing_coordinate_rate(
    n_missing: int,
    n_total: int,
    threshold: float = 0.05,
) -> CheckResult:
    """Tỉ lệ event thiếu tọa độ."""
    rate = n_missing / n_total if n_total else 0.0
    ok = rate < threshold
    return CheckResult(
        check=f"missing_coordinate_rate < {threshold}",
        value=round(rate, 4),
        status="PASS" if ok else "WARNING",
        note=f"{n_missing}/{n_total} events thiếu tọa độ",
    )


def check_missing_recipient_rate(
    n_missing: int,
    n_passes: int,
    threshold: float = 0.15,
) -> CheckResult:
    """Tỉ lệ pass thiếu người nhận."""
    rate = n_missing / n_passes if n_passes else 0.0
    ok = rate < threshold
    return CheckResult(
        check=f"missing_recipient_rate < {threshold}",
        value=round(rate, 4),
        status="PASS" if ok else "WARNING",
        note=f"{n_missing}/{n_passes} passes thiếu recipient",
    )


def check_positive_shot_rate(
    n_positive: int,
    n_total: int,
    expected_range: tuple[float, float] = (0.10, 0.25),
) -> CheckResult:
    """Tỉ lệ possession dẫn đến sút."""
    rate = n_positive / n_total if n_total else 0.0
    lo, hi = expected_range
    ok = lo <= rate <= hi
    return CheckResult(
        check=f"positive_shot_rate in [{lo}, {hi}]",
        value=round(rate, 4),
        status="PASS" if ok else "WARNING",
        note=f"{n_positive}/{n_total} possessions dẫn đến sút",
    )


def check_actual_shots_per_match(
    n_actual_shots: int,
    n_matches: int,
    expected_range: tuple[float, float] = (15.0, 35.0),
) -> CheckResult:
    """Số cú sút thực tế mỗi trận (cả hai đội).

    Lưu ý: n_actual_shots là số Shot events, KHÔNG phải số
    shot-positive possessions.
    """
    spm = n_actual_shots / n_matches if n_matches else 0.0
    lo, hi = expected_range
    ok = lo <= spm <= hi
    return CheckResult(
        check=f"actual_shots_per_match in [{lo}, {hi}]",
        value=round(spm, 1),
        status="PASS" if ok else "WARNING",
        note=f"{n_actual_shots} actual Shot events / {n_matches} matches",
    )


def check_duplicate_event_ids(
    event_ids: list[str],
) -> CheckResult:
    """Kiểm tra trùng event ID."""
    n_total = len(event_ids)
    n_unique = len(set(event_ids))
    n_dup = n_total - n_unique
    ok = n_dup == 0
    return CheckResult(
        check="no_duplicate_event_ids",
        value=n_dup,
        status="PASS" if ok else "WARNING",
        note=f"{n_dup} duplicate event IDs" if n_dup else "",
    )


def check_attack_direction(
    diagnostic: dict[str, float],
) -> CheckResult:
    """Hướng tấn công: x trung bình pha sút phải cao hơn."""
    gap = diagnostic.get("gap", float("nan"))
    import math

    ok = not math.isnan(gap) and gap > 0
    return CheckResult(
        check="attack_direction_gap > 0",
        value=round(gap, 2) if not math.isnan(gap) else "NaN",
        status="PASS" if ok else "WARNING",
        note="gap <= 0 nghĩa là tọa độ chưa chuẩn hóa đồng bộ" if not ok else "",
    )


def check_possession_team_consistency(
    events_df: pd.DataFrame,
) -> CheckResult:
    """Kiểm tra possession_team_id nhất quán trong cùng possession.

    Dùng possession_team_id (numeric) thay vì so sánh tên để tránh
    sai lệch do tên team khác nhau giữa các nguồn dữ liệu.
    """
    col = "possession_team_id"
    if events_df.empty or col not in events_df.columns:
        return CheckResult(
            check="possession_team_consistency",
            value="N/A",
            status="WARNING",
            note="Không có dữ liệu để kiểm tra",
        )
    grouped = events_df.groupby(["match_id", "possession_id"])[col].nunique()
    n_inconsistent = int((grouped > 1).sum())
    ok = n_inconsistent == 0
    return CheckResult(
        check="possession_team_consistency",
        value=n_inconsistent,
        status="PASS" if ok else "WARNING",
        note=f"{n_inconsistent} possessions có nhiều hơn 1 team" if not ok else "",
    )


def check_graph_size_proxy(
    players_per_possession: list[int],
    passes_per_possession: list[int],
) -> CheckResult:
    """Proxy cho kích thước đồ thị."""
    import numpy as np

    p_arr = np.array(players_per_possession)
    e_arr = np.array(passes_per_possession)
    return CheckResult(
        check="graph_size_proxy",
        value=f"median_players={float(np.median(p_arr)):.0f}, "
        f"median_passes={float(np.median(e_arr)):.0f}",
        status="PASS",
        note=f"players: [{int(p_arr.min())}, {int(p_arr.max())}], "
        f"passes: [{int(e_arr.min())}, {int(e_arr.max())}]",
    )


def check_duplicate_possession_keys(
    keys: list[tuple[int, int]],
) -> CheckResult:
    """Kiểm tra tính duy nhất của cặp (match_id, possession_id)."""
    n_total = len(keys)
    n_unique = len(set(keys))
    n_dup = n_total - n_unique
    ok = n_dup == 0
    return CheckResult(
        check="no_duplicate_possession_keys",
        value=n_dup,
        status="PASS" if ok else "FAIL",
        note=f"{n_dup} duplicate (match_id, possession_id) keys!" if n_dup else "",
    )


def check_successful_pass_coordinate_missingness(
    n_missing_start: int,
    n_missing_end: int,
    n_passes: int,
    threshold: float = 0.01,
) -> CheckResult:
    """Kiểm tra missing coordinate trên các đường chuyền thành công (nơi dựng cạnh đồ thị)."""
    rate = max(n_missing_start, n_missing_end) / n_passes if n_passes else 0.0
    ok = rate < threshold
    return CheckResult(
        check=f"successful_pass_coordinate_missingness < {threshold}",
        value=round(rate, 4),
        status="PASS" if ok else "WARNING",
        note=f"start_missing={n_missing_start}, end_missing={n_missing_end} / {n_passes} passes",
    )


def run_all_checks(results: list[CheckResult]) -> pd.DataFrame:
    """Tổng hợp kết quả check thành DataFrame."""
    df = pd.DataFrame([r.as_row() for r in results])
    n_fail = int((df["status"] == "FAIL").sum())
    n_warn = int((df["status"] == "WARNING").sum())
    n_pass = int((df["status"] == "PASS").sum())
    logger.info(
        "Sanity checks: %d PASS, %d WARNING, %d FAIL",
        n_pass,
        n_warn,
        n_fail,
    )
    if n_fail:
        logger.error("%d phép kiểm tra FAIL — cần xem lại trước khi tiếp tục", n_fail)
    return df


def has_critical_failure(checks_df: pd.DataFrame) -> bool:
    """Trả True nếu có bất kỳ FAIL nào."""
    return bool((checks_df["status"] == "FAIL").any())
