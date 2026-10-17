"""Unit tests kiểm tra tính hợp lệ về định dạng (nbformat v4) và cấu trúc 10 mục của mọi Jupyter Notebook."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat
import pytest

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"
EXPECTED_NOTEBOOKS = [
    "00_environment_and_reproducibility.ipynb",
    "01_data_audit_and_collection.ipynb",
    "02_preprocessing_and_labels.ipynb",
    "03_graph_construction.ipynb",
    "04_match_level_split.ipynb",
    "05_model_training.ipynb",
    "06_model_evaluation.ipynb",
]

REQUIRED_SECTIONS = [
    "Mục tiêu",
    "Vị trí",
    "Input",
    "Config",
    "xử lý",
    "Kết quả",
    "Kiểm tra chất lượng",
    "Output",
    "tái chạy",
    "Giới hạn",
]


@pytest.mark.parametrize("nb_name", EXPECTED_NOTEBOOKS)
def test_notebook_exists_and_valid_nbformat(nb_name: str):
    """Mọi notebook phải tồn tại và tuân thủ chuẩn nbformat v4."""
    nb_path = NOTEBOOKS_DIR / nb_name
    assert nb_path.exists(), f"Thiếu notebook: {nb_path}"

    nb = nbformat.read(nb_path, as_version=4)
    assert nb.nbformat == 4
    assert len(nb.cells) > 0


@pytest.mark.parametrize("nb_name", EXPECTED_NOTEBOOKS)
def test_notebook_contains_required_sections(nb_name: str):
    """Mỗi notebook phải có đầy đủ 10 mục nội dung theo quy chuẩn nghiên cứu."""
    nb_path = NOTEBOOKS_DIR / nb_name
    nb = nbformat.read(nb_path, as_version=4)

    all_markdown = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "markdown")

    for sec in REQUIRED_SECTIONS:
        pattern = re.compile(re.escape(sec), re.IGNORECASE)
        assert (
            pattern.search(all_markdown) is not None
        ), f"Notebook {nb_name} thiếu mục liên quan đến: '{sec}'"


@pytest.mark.parametrize("nb_name", EXPECTED_NOTEBOOKS)
def test_no_hardcoded_fallback_in_code_cells(nb_name: str):
    """Đảm bảo không có các mẫu hardcoded fallback như 'else 662' hay 'or 380' trong cell code."""
    nb_path = NOTEBOOKS_DIR / nb_name
    nb = nbformat.read(nb_path, as_version=4)

    forbidden_patterns = [
        "else 662",
        "len(raw_files) or 380",
    ]

    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            for pat in forbidden_patterns:
                assert (
                    pat not in cell.source
                ), f"Notebook {nb_name} cell {i} chứa fallback bị cấm: '{pat}'"
