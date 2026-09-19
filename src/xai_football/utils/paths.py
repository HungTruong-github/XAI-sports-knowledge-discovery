"""Đường dẫn chuẩn trong dự án — một nguồn duy nhất.

Mọi module khác phải lấy đường dẫn từ đây, không tự ghép chuỗi, để khi đổi
bố cục thư mục chỉ phải sửa một chỗ.
"""

from __future__ import annotations

from pathlib import Path

# src/xai_football/utils/paths.py -> lùi 4 cấp là gốc repo
PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def match_json_path(competition_id: int, season_id: int, match_id: int) -> Path:
    """Đường dẫn tệp event thô của một trận.

    Cấu trúc `raw/{competition_id}/{season_id}/{match_id}.json` theo đúng
    mục 1.4 trong docs/Quy_trinh_XAI_Football.md — để truy vết lại được và
    không phải gọi lại API nhiều lần.
    """
    return RAW_DIR / str(competition_id) / str(season_id) / f"{match_id}.json"


def lineup_json_path(competition_id: int, season_id: int, match_id: int) -> Path:
    """Đường dẫn tệp đội hình thô của một trận (bổ sung đặc trưng vị trí đỉnh)."""
    return RAW_DIR / str(competition_id) / str(season_id) / f"{match_id}.lineups.json"


def ensure_dir(path: Path) -> Path:
    """Tạo thư mục nếu chưa có; trả lại chính path để gọi nối chuỗi được."""
    path.mkdir(parents=True, exist_ok=True)
    return path
