"""Bước 1 — Thu thập dữ liệu từ StatsBomb Open Data.

Tham chiếu: docs/Quy_trinh_XAI_Football.md mục 1.

Nguyên tắc:
  - Lưu nguyên JSON gốc theo cấu trúc raw/{competition_id}/{season_id}/{match_id}.json
    để truy vết lại được và không phải tải lại nhiều lần (mục 1.4).
  - Ghi log số trận, số sự kiện mỗi trận, tỉ lệ trận bị loại — phục vụ phần
    mô tả dataset trong khóa luận.
  - Số trận thực tế LUÔN lấy từ statsbombpy, không dùng lại con số ước lượng
    từ dung lượng tệp trong docs/Nguon_du_lieu_XAI_Football.md (rủi ro F5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..utils.io import load_yaml, save_json
from ..utils.logging import get_logger
from ..utils.paths import lineup_json_path, match_json_path

logger = get_logger(__name__)


@dataclass
class CollectionReport:
    """Nhật ký thu thập của một cặp giải–mùa."""

    competition_name: str
    competition_id: int
    season_id: int
    n_matches_listed: int = 0
    n_matches_attempted: int = 0
    n_matches_complete: int = 0
    n_matches_failed: int = 0
    n_events_total: int = 0
    failed_match_ids: list[int] = field(default_factory=list)

    @property
    def n_matches_downloaded(self) -> int:
        """Alias cho backward compatibility — tương đương n_matches_complete."""
        return self.n_matches_complete

    @property
    def drop_rate(self) -> float:
        if self.n_matches_listed == 0:
            return 0.0
        return self.n_matches_failed / self.n_matches_listed

    def as_row(self) -> dict[str, Any]:
        return {
            "competition": self.competition_name,
            "competition_id": self.competition_id,
            "season_id": self.season_id,
            "matches_listed": self.n_matches_listed,
            "matches_attempted": self.n_matches_attempted,
            "matches_complete": self.n_matches_complete,
            "matches_downloaded": self.n_matches_downloaded,
            "matches_failed": self.n_matches_failed,
            "drop_rate": round(self.drop_rate, 4),
            "events_total": self.n_events_total,
            "events_per_match": (
                round(self.n_events_total / self.n_matches_complete, 1)
                if self.n_matches_complete
                else 0.0
            ),
        }


def _statsbombpy():
    """Nạp statsbombpy và tắt cảnh báo NoAuthWarning cho dữ liệu mở."""
    import warnings

    from statsbombpy import sb
    from statsbombpy.api_client import NoAuthWarning

    warnings.simplefilter("ignore", NoAuthWarning)
    return sb


def list_competitions() -> pd.DataFrame:
    """Danh sách toàn bộ cặp giải–mùa có trong StatsBomb Open Data."""
    return _statsbombpy().competitions()


def resolve_competitions(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Điền `competition_id`/`season_id` còn thiếu bằng cách tra theo tên.

    Cho phép `configs/data.yaml` khai báo bằng tên giải + mùa cho dễ đọc,
    nhưng vẫn chạy đúng khi chỉ có id.
    """
    entries = config.get("competitions") or []
    need_lookup = any(
        e.get("competition_id") is None or e.get("season_id") is None for e in entries
    )
    catalogue = list_competitions() if need_lookup else None

    resolved: list[dict[str, Any]] = []
    for entry in entries:
        comp_id, season_id = entry.get("competition_id"), entry.get("season_id")

        if (comp_id is None or season_id is None) and catalogue is not None:
            mask = pd.Series(True, index=catalogue.index)
            if entry.get("name"):
                mask &= catalogue["competition_name"] == entry["name"]
            if entry.get("season"):
                mask &= catalogue["season_name"] == entry["season"]
            hits = catalogue[mask]
            if len(hits) != 1:
                raise ValueError(
                    f"Không tra được duy nhất một cặp giải-mùa cho {entry!r}: "
                    f"tìm thấy {len(hits)} kết quả. Hãy điền thẳng "
                    f"competition_id / season_id vào configs/data.yaml."
                )
            comp_id = int(hits.iloc[0]["competition_id"])
            season_id = int(hits.iloc[0]["season_id"])

        if comp_id is None or season_id is None:
            raise ValueError(
                f"Mục {entry!r} trong configs/data.yaml chưa đủ thông tin. "
                f"Cần competition_id + season_id, hoặc name + season để tra."
            )

        resolved.append(
            {
                "name": entry.get("name") or f"competition_{comp_id}",
                "season": entry.get("season"),
                "competition_id": int(comp_id),
                "season_id": int(season_id),
            }
        )
    return resolved


def download_competition(
    competition_id: int,
    season_id: int,
    competition_name: str = "",
    limit_matches: int | None = None,
    overwrite: bool = False,
    with_lineups: bool = True,
) -> CollectionReport:
    """Tải toàn bộ event (và đội hình) của một cặp giải–mùa."""
    import json

    sb = _statsbombpy()
    report = CollectionReport(
        competition_name or f"competition_{competition_id}", competition_id, season_id
    )

    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    report.n_matches_listed = len(matches)
    logger.info(
        "%s (comp=%s, season=%s): %d trận trong kho",
        report.competition_name,
        competition_id,
        season_id,
        len(matches),
    )

    match_ids = matches["match_id"].astype(int).tolist()
    if limit_matches is not None:
        match_ids = match_ids[:limit_matches]
        logger.info("Giới hạn chạy thử: chỉ tải %d trận đầu", len(match_ids))

    for match_id in match_ids:
        report.n_matches_attempted += 1
        events_path = match_json_path(competition_id, season_id, match_id)
        lineup_path = lineup_json_path(competition_id, season_id, match_id)

        events_valid = False
        if events_path.exists() and not overwrite:
            try:
                with open(events_path, encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, (list, dict)) and len(data) > 0:
                        events_valid = True
            except (OSError, ValueError):
                events_valid = False

        lineups_valid = False
        if with_lineups:
            if lineup_path.exists() and not overwrite:
                try:
                    with open(lineup_path, encoding="utf-8") as f:
                        ldata = json.load(f)
                        if isinstance(ldata, (list, dict)):
                            lineups_valid = True
                except (OSError, ValueError):
                    lineups_valid = False
        else:
            lineups_valid = True

        try:
            if not events_valid:
                events = sb.events(match_id=match_id, fmt="dict")
                records = list(events.values()) if isinstance(events, dict) else events
                save_json(records, events_path)
            else:
                with open(events_path, encoding="utf-8") as f:
                    records = json.load(f)

            if with_lineups and not lineups_valid:
                lineups = sb.lineups(match_id=match_id)
                save_json(
                    {team: df.to_dict("records") for team, df in lineups.items()},
                    lineup_path,
                )

            report.n_events_total += len(records)
            report.n_matches_complete += 1
        except Exception as exc:  # noqa: BLE001 — ghi nhận trận lỗi rồi đi tiếp
            logger.warning("Trận %s lỗi download/load, bỏ qua: %s", match_id, exc)
            report.n_matches_failed += 1
            report.failed_match_ids.append(match_id)

    logger.info(
        "%s: hoàn thành %d/%d trận, %d sự kiện, tỉ lệ loại %.2f%%",
        report.competition_name,
        report.n_matches_complete,
        len(match_ids),
        report.n_events_total,
        report.drop_rate * 100,
    )
    return report


def download_all(
    config_path: str = "data.yaml", limit_matches: int | None = None, overwrite: bool = False
) -> list[CollectionReport]:
    """Tải toàn bộ các cặp giải–mùa khai báo trong configs/data.yaml."""
    config = load_yaml(config_path)
    reports = []
    for entry in resolve_competitions(config):
        reports.append(
            download_competition(
                competition_id=entry["competition_id"],
                season_id=entry["season_id"],
                competition_name=entry["name"],
                limit_matches=limit_matches,
                overwrite=overwrite,
            )
        )
    return reports
