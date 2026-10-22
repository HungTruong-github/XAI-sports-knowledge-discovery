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
METADATA_DIR = RESULTS_DIR / "metadata"
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


def get_run_name(config: dict) -> str:
    """Lấy `run_name` từ config (mặc định 'official')."""
    return str(config.get("run_name") or "official")


def get_interim_dir(config: dict) -> Path:
    """Lấy thư mục interim theo run_name (ví dụ data/interim/official hoặc data/interim/pilot)."""
    return ensure_dir(INTERIM_DIR / get_run_name(config))


def get_tables_dir(config: dict) -> Path:
    """Lấy thư mục tables theo run_name (results/tables/official hoặc results/tables/pilot)."""
    return ensure_dir(TABLES_DIR / get_run_name(config))


def get_processed_dir(config: dict) -> Path:
    """Lấy thư mục processed theo run_name."""
    return ensure_dir(PROCESSED_DIR / get_run_name(config))


def get_metadata_dir(config: dict) -> Path:
    """Lấy thư mục metadata theo run_name."""
    return ensure_dir(METADATA_DIR / get_run_name(config))


def resolve_project_path(path: str | Path) -> Path:
    """Đường dẫn tương đối trong config được hiểu là tính từ gốc repo.

    Tránh lỗi đọc/ghi sai chỗ khi chạy script từ một thư mục khác gốc repo.
    """
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def get_split_dir(split_cfg: dict, run_name: str) -> Path:
    """Thư mục danh sách trận của từng tập: data/splits (official) hoặc data/splits/<run_name>."""
    base = resolve_project_path(split_cfg.get("output_dir") or "data/splits")
    return base if run_name == "official" else base / run_name


def get_experiments_dir(run_name: str, suffix: str = "") -> Path:
    """experiments/<run_name><suffix> — suffix ví dụ '_smoke', '_debug'."""
    return EXPERIMENTS_DIR / f"{run_name}{suffix}"


def get_models_dir(run_name: str, suffix: str = "") -> Path:
    """models/<run_name><suffix> — nơi lưu checkpoint tốt nhất của từng mô hình."""
    return MODELS_DIR / f"{run_name}{suffix}"
