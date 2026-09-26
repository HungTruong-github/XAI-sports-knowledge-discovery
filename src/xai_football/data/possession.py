"""Bước 2.1 — Trích xuất possession sequence từ event data.

Tham chiếu: docs/Quy_trinh_XAI_Football.md mục 2.1.

Dùng trường `possession` có sẵn trong StatsBomb event data để nhóm các sự
kiện thuộc cùng một pha kiểm soát bóng của một đội. Đây là lý do chính khiến
nhóm chọn StatsBomb: các nguồn khác (ví dụ Wyscout) không có trường này và
phải tự tái tạo qua SPADL — rủi ro D4.

Quy ước identity:
  - player_id = identity chính (dùng nhận dạng node, aggregate, join metadata)
  - player_name = metadata / display
  - KHÔNG BAO GIỜ dùng player_id làm numerical model feature
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..utils.io import load_json
from ..utils.paths import match_json_path


def _name(value: Any) -> str | None:
    """Lấy tên từ một trường StatsBomb có thể là dict {'id','name'} hoặc chuỗi."""
    if isinstance(value, dict):
        return value.get("name")
    return value


def _id(value: Any) -> int | None:
    """Lấy id từ trường StatsBomb có thể là dict {'id','name'} hoặc int."""
    if isinstance(value, dict):
        v = value.get("id")
        return int(v) if v is not None else None
    if isinstance(value, (int, float)):
        return int(value)
    return None


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
    # Identity: player_id là chính, player_name là metadata
    passer_id: int | None
    passer_name: str | None
    recipient_id: int | None
    recipient_name: str | None
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
        """StatsBomb: `pass.outcome` rỗng (None) hoặc 'Complete' nghĩa là chuyền thành công."""
        return self.outcome is None or self.outcome == "Complete"

    @property
    def has_coordinates(self) -> bool:
        return None not in (self.start_x, self.start_y, self.end_x, self.end_y)

    @property
    def passer(self) -> int | None:
        """Node identity — dùng player_id."""
        return self.passer_id

    @property
    def recipient(self) -> int | None:
        """Node identity — dùng player_id."""
        return self.recipient_id


@dataclass
class ShotInfo:
    """Thông tin một cú sút trong possession."""

    event_id: str
    minute: int
    second: int
    outcome: str | None
    xg: float | None

    @property
    def time_seconds(self) -> float:
        return float(self.minute) * 60 + float(self.second)

    @property
    def is_goal(self) -> bool:
        return self.outcome == "Goal"


@dataclass
class Possession:
    """Một pha kiểm soát bóng — sẽ trở thành một đồ thị ở Bước 3.

    Unique key: (match_id, possession_id).
    possession_id KHÔNG phải unique key một mình — có thể trùng giữa các trận.

    Semantics rõ ràng:
      has_shot: possession CÓ CHỨA ít nhất một cú sút (của đội kiểm soát bóng)
      has_goal: possession CÓ CHỨA ít nhất một bàn thắng (của đội kiểm soát bóng)
      ends_with_shot: backward compatibility — cùng ngữ nghĩa với has_shot
      ends_with_goal: backward compatibility — cùng ngữ nghĩa với has_goal
    """

    match_id: int
    competition_id: int
    season_id: int
    possession_id: int
    team_id: int | None
    team_name: str | None
    play_pattern: str | None
    period: int
    passes: list[PassEvent] = field(default_factory=list)
    shots: list[ShotInfo] = field(default_factory=list)

    @property
    def has_shot(self) -> bool:
        """Possession chứa ít nhất một cú sút."""
        return len(self.shots) > 0

    @property
    def has_goal(self) -> bool:
        """Possession chứa ít nhất một bàn thắng."""
        return any(s.is_goal for s in self.shots)

    # Backward compatibility aliases
    @property
    def ends_with_shot(self) -> bool:
        """Alias cho has_shot — giữ backward compatibility.

        Lưu ý ngữ nghĩa: đây là "possession chứa shot", không nhất thiết
        "event cuối cùng là shot". Tên gốc có thể gây hiểu nhầm.
        """
        return self.has_shot

    @property
    def ends_with_goal(self) -> bool:
        """Alias cho has_goal — giữ backward compatibility."""
        return self.has_goal

    @property
    def shot_second(self) -> float | None:
        """Thời điểm cú sút đầu tiên (seconds). Backward compatibility."""
        if not self.shots:
            return None
        return self.shots[0].time_seconds

    @property
    def shot_times(self) -> list[float]:
        """Danh sách thời điểm của tất cả cú sút (seconds)."""
        return [s.time_seconds for s in self.shots]

    @property
    def n_passes(self) -> int:
        return len(self.passes)

    @property
    def players(self) -> list[int]:
        """Cầu thủ tham gia possession — sẽ là đỉnh của đồ thị (mục 3.1).

        Trả về danh sách player_id (identity chính).
        """
        seen: dict[int, None] = {}
        for p in self.passes:
            for pid in (p.passer_id, p.recipient_id):
                if pid is not None:
                    seen.setdefault(pid, None)
        return list(seen)

    @property
    def player_names(self) -> dict[int, str]:
        """Mapping player_id -> player_name cho display."""
        names: dict[int, str] = {}
        for p in self.passes:
            if p.passer_id is not None and p.passer_name:
                names.setdefault(p.passer_id, p.passer_name)
            if p.recipient_id is not None and p.recipient_name:
                names.setdefault(p.recipient_id, p.recipient_name)
        return names

    @property
    def n_players(self) -> int:
        return len(self.players)

    @property
    def n_unique_edges(self) -> int:
        """Số cặp (người chuyền, người nhận) khác nhau — số cạnh sau khi gộp."""
        return len({(p.passer_id, p.recipient_id) for p in self.passes})


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
        passer_id=_id(event.get("player")),
        passer_name=_name(event.get("player")),
        recipient_id=_id(_nested(event, "pass", "recipient")),
        recipient_name=_name(_nested(event, "pass", "recipient")),
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


def _parse_shot(event: dict[str, Any]) -> ShotInfo:
    return ShotInfo(
        event_id=event.get("id", ""),
        minute=int(event.get("minute") or 0),
        second=int(event.get("second") or 0),
        outcome=_name(_nested(event, "shot", "outcome")),
        xg=_nested(event, "shot", "statsbomb_xg"),
    )


def extract_possessions(
    events: Iterable[dict[str, Any]],
    match_id: int,
    competition_id: int = 0,
    season_id: int = 0,
) -> list[Possession]:
    """Nhóm sự kiện của một trận thành các possession.

    Chỉ giữ đường chuyền **thành công** trong chuỗi (mục 2.1), nhưng vẫn đọc
    toàn bộ sự kiện của possession để xác định pha đó có chứa cú sút hay không.
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
            team_id=_id(first.get("possession_team")),
            team_name=_name(first.get("possession_team")),
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
                shot_team = _name(event.get("team"))
                if shot_team != possession.team_name:
                    continue
                possession.shots.append(_parse_shot(event))

        possessions.append(possession)

    return sorted(possessions, key=lambda p: p.possession_id)


def load_match_possessions(competition_id: int, season_id: int, match_id: int) -> list[Possession]:
    """Đọc tệp event thô của một trận và trích ra danh sách possession."""
    events = load_json(match_json_path(competition_id, season_id, match_id))
    if isinstance(events, dict):
        events = list(events.values())
    return extract_possessions(events, match_id, competition_id, season_id)
