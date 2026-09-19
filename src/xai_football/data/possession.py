"""Bước 2.1 — Trích xuất possession sequence từ event data.

Tham chiếu: docs/Quy_trinh_XAI_Football.md mục 2.1.

Dùng trường `possession` có sẵn trong StatsBomb event data để nhóm các sự
kiện thuộc cùng một pha kiểm soát bóng của một đội. Đây là lý do chính khiến
nhóm chọn StatsBomb: các nguồn khác (ví dụ Wyscout) không có trường này và
phải tự tái tạo qua SPADL — rủi ro D4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from ..utils.io import load_json
from ..utils.paths import match_json_path


def _name(value: Any) -> str | None:
    """Lấy tên từ một trường StatsBomb có thể là dict {'id','name'} hoặc chuỗi."""
    if isinstance(value, dict):
        return value.get("name")
    return value


def _nested(event: dict[str, Any], group: str, key: str) -> Any:
    """Đọc `event[group][key]`, trả None nếu thiếu bất kỳ cấp nào."""
    sub = event.get(group)
    return sub.get(key) if isinstance(sub, dict) else None


@dataclass
class PassEvent:
    """Một đường chuyền — sẽ trở thành một cạnh có hướng ở Bước 3."""

    event_id: str
    index: int
    period: int
    minute: int
    second: int
    timestamp: str | None
    passer: str | None
    recipient: str | None
    start_x: float | None
    start_y: float | None
    end_x: float | None
    end_y: float | None
    length: float | None
    angle: float | None
    height: str | None
    pass_type: str | None
    outcome: str | None

    @property
    def is_complete(self) -> bool:
        """StatsBomb: `pass.outcome` rỗng nghĩa là chuyền thành công."""
        return self.outcome is None

    @property
    def has_coordinates(self) -> bool:
        return None not in (self.start_x, self.start_y, self.end_x, self.end_y)


@dataclass
class Possession:
    """Một pha kiểm soát bóng — sẽ trở thành một đồ thị ở Bước 3."""

    match_id: int
    competition_id: int
    season_id: int
    possession_id: int
    team: str | None
    play_pattern: str | None
    period: int
    passes: list[PassEvent] = field(default_factory=list)
    ends_with_shot: bool = False
    ends_with_goal: bool = False
    shot_second: float | None = None

    @property
    def n_passes(self) -> int:
        return len(self.passes)

    @property
    def players(self) -> list[str]:
        """Cầu thủ tham gia possession — sẽ là đỉnh của đồ thị (mục 3.1)."""
        seen: dict[str, None] = {}
        for p in self.passes:
            for name in (p.passer, p.recipient):
                if name is not None:
                    seen.setdefault(name, None)
        return list(seen)

    @property
    def n_players(self) -> int:
        return len(self.players)

    @property
    def n_unique_edges(self) -> int:
        """Số cặp (người chuyền, người nhận) khác nhau — số cạnh sau khi gộp."""
        return len({(p.passer, p.recipient) for p in self.passes})


def _to_seconds(event: dict[str, Any]) -> float:
    return float(event.get("minute") or 0) * 60 + float(event.get("second") or 0)


def parse_pass(event: dict[str, Any]) -> PassEvent:
    location = event.get("location") or [None, None]
    end_location = _nested(event, "pass", "end_location") or [None, None]

    return PassEvent(
        event_id=event.get("id", ""),
        index=int(event.get("index") or 0),
        period=int(event.get("period") or 0),
        minute=int(event.get("minute") or 0),
        second=int(event.get("second") or 0),
        timestamp=event.get("timestamp"),
        passer=_name(event.get("player")),
        recipient=_name(_nested(event, "pass", "recipient")),
        start_x=location[0],
        start_y=location[1] if len(location) > 1 else None,
        end_x=end_location[0],
        end_y=end_location[1] if len(end_location) > 1 else None,
        length=_nested(event, "pass", "length"),
        angle=_nested(event, "pass", "angle"),
        height=_name(_nested(event, "pass", "height")),
        pass_type=_name(_nested(event, "pass", "type")),
        outcome=_name(_nested(event, "pass", "outcome")),
    )


def extract_possessions(
    events: Iterable[dict[str, Any]],
    match_id: int,
    competition_id: int,
    season_id: int,
) -> list[Possession]:
    """Nhóm sự kiện của một trận thành các possession.

    Chỉ giữ đường chuyền **thành công** trong chuỗi (mục 2.1), nhưng vẫn đọc
    toàn bộ sự kiện của possession để xác định pha đó có kết thúc bằng cú sút
    hay không.
    """
    events = sorted(events, key=lambda e: int(e.get("index") or 0))
    buckets: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        pid = event.get("possession")
        if pid is None:
            continue
        buckets.setdefault(int(pid), []).append(event)

    possessions = []
    for pid, group in buckets.items():
        first = group[0]
        possession = Possession(
            match_id=match_id,
            competition_id=competition_id,
            season_id=season_id,
            possession_id=pid,
            team=_name(first.get("possession_team")),
            play_pattern=_name(first.get("play_pattern")),
            period=int(first.get("period") or 0),
        )

        for event in group:
            event_type = _name(event.get("type"))

            if event_type == "Pass":
                pass_event = parse_pass(event)
                if pass_event.is_complete:
                    possession.passes.append(pass_event)

            elif event_type == "Shot":
                # Chỉ tính cú sút của chính đội đang kiểm soát bóng —
                # tránh đếm nhầm cú sút của đội đối phương (kiểm tra 4.2).
                if _name(event.get("team")) != possession.team:
                    continue
                possession.ends_with_shot = True
                if possession.shot_second is None:
                    possession.shot_second = _to_seconds(event)
                if _name(_nested(event, "shot", "outcome")) == "Goal":
                    possession.ends_with_goal = True

        possessions.append(possession)

    return sorted(possessions, key=lambda p: p.possession_id)


def load_match_possessions(
    competition_id: int, season_id: int, match_id: int
) -> list[Possession]:
    """Đọc tệp event thô của một trận và trích ra danh sách possession."""
    events = load_json(match_json_path(competition_id, season_id, match_id))
    if isinstance(events, dict):
        events = list(events.values())
    return extract_possessions(events, match_id, competition_id, season_id)
