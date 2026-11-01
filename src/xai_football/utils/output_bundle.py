"""Tiện ích quản lý Output Evidence Bundle và Provenance Tracking (Bước 0–6).

Module này phụ trách:
1. Tạo và cấu trúc thư mục outputs/ cho từng bước và run_id:
   outputs/
   ├── README.md
   ├── run_registry.json
   └── step_XX_<name>/runs/<run_id>/
       ├── files/      (CSV, JSON, YAML)
       ├── figures/    (PNG 300 DPI, SVG)
       ├── logs/       (Nhật ký thực thi)
       ├── reports/    (Báo cáo Markdown)
       └── manifest.json
2. Xác thực và ghi manifest chuẩn ISO 8601, SHA-256, đường dẫn tương đối portable.
3. Đăng ký artifact có giới hạn kích thước, ngăn chặn sao chép file nhị phân nặng.
4. Cung cấp hàm save_figure_artifact dùng chung cho scripts và notebooks.
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import shutil
from pathlib import Path
from typing import Any

from xai_football.utils.io import save_atomic_json
from xai_football.utils.paths import PROJECT_ROOT, ensure_dir
from xai_football.utils.provenance import compute_file_sha256, get_git_info

OUTPUTS_ROOT = PROJECT_ROOT / "outputs"

STEP_DIRECTORIES = {
    0: "step_00_environment_and_reproducibility",
    1: "step_01_data_audit_and_collection",
    2: "step_02_preprocessing_and_labels",
    3: "step_03_graph_construction",
    4: "step_04_match_level_split",
    5: "step_05_model_training",
    6: "step_06_model_evaluation",
}

STEP_NAMES = {
    0: "Environment & Reproducibility Audit",
    1: "Data Audit & Collection",
    2: "Preprocessing & Labeling",
    3: "Graph Construction",
    4: "Match-Level Split",
    5: "Model Training",
    6: "Model Evaluation & Comparison",
}

RUN_STATUS_TYPES = (
    "PASS",
    "WARN",
    "FAIL",
    "NOT_RUN",
    "HISTORICAL_OFFICIAL",
    "HISTORICAL_PROVISIONAL",
    "UNVERIFIED",
    "SMOKE",
    "PARTIAL",
)


def get_step_dir_name(step: int | str) -> str:
    """Lấy tên thư mục chuẩn cho từng bước."""
    if isinstance(step, int):
        if step not in STEP_DIRECTORIES:
            raise ValueError(f"Bước không hợp lệ: {step}. Chỉ hỗ trợ 0-6.")
        return STEP_DIRECTORIES[step]
    s = str(step).strip()
    if s.isdigit():
        return get_step_dir_name(int(s))
    for k, v in STEP_DIRECTORIES.items():
        if s == v or s == f"step_{k:02d}":
            return v
    raise ValueError(f"Không nhận diện được step: '{step}'")


def get_step_run_dir(step: int | str, run_id: str, outputs_root: Path | None = None) -> Path:
    """Trả về đường dẫn thư mục runs/<run_id> của bước cụ thể."""
    base = Path(outputs_root) if outputs_root else OUTPUTS_ROOT
    step_dir = get_step_dir_name(step)
    return base / step_dir / "runs" / run_id


def init_step_run_bundle(
    step: int | str,
    run_id: str,
    outputs_root: Path | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Khởi tạo toàn bộ cấu trúc thư mục con cho một run trong một bước."""
    run_dir = get_step_run_dir(step, run_id, outputs_root)
    if run_dir.exists() and not force:
        pass  # Cho phép cập nhật / bổ sung artifact vào run hiện có nếu cùng run_id
    files_dir = ensure_dir(run_dir / "files")
    figures_dir = ensure_dir(run_dir / "figures")
    logs_dir = ensure_dir(run_dir / "logs")
    reports_dir = ensure_dir(run_dir / "reports")
    return {
        "run_dir": run_dir,
        "files": files_dir,
        "figures": figures_dir,
        "logs": logs_dir,
        "reports": reports_dir,
    }


def to_portable_rel_path(path: Path | str, base_root: Path | None = None) -> str:
    """Chuyển đổi đường dẫn sang dạng tương đối POSIX portable (không chứa ổ đĩa Windows tuyệt đối)."""
    p = Path(path).resolve()
    root = (Path(base_root) if base_root else PROJECT_ROOT).resolve()
    try:
        rel = p.relative_to(root)
        return rel.as_posix()
    except ValueError:
        # Nếu nằm ngoài root, trả về tên file tương đối
        return p.name


def register_artifact(
    target_dir: Path,
    source_path: Path | str,
    dest_name: str | None = None,
    file_type: str = "data",
    generator: str = "pipeline",
    max_copy_size_bytes: int = 50_000_000,  # 50 MB
) -> dict[str, Any]:
    """Đăng ký và sao chép an toàn một artifact vào bundle.

    Nếu file vượt quá max_copy_size_bytes hoặc là file nhị phân nặng (.pt, .parquet lớn),
    hệ thống sẽ KHÔNG sao chép để tránh phình dung lượng git, nhưng vẫn tính hash
    và lưu vết đường dẫn nguồn.
    """
    src = Path(source_path).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Artifact nguồn không tồn tại: {src}")

    size_bytes = src.stat().st_size
    sha256 = compute_file_sha256(src)
    target_name = dest_name or src.name
    dest_file = (target_dir / target_name).resolve()

    # Các đuôi file nặng không sao chép vật lý vào outputs/
    heavy_extensions = {".pt", ".pth", ".ckpt", ".h5", ".parquet"}
    is_heavy = src.suffix.lower() in heavy_extensions or size_bytes > max_copy_size_bytes

    if is_heavy:
        # Lưu file mô tả metadata pointer thay vì file nặng
        pointer_info = {
            "source_path": to_portable_rel_path(src),
            "size_bytes": size_bytes,
            "sha256": sha256,
            "reason": "Exceeds max size limit or heavy binary format, registered by reference",
        }
        pointer_file = target_dir / f"{target_name}.pointer.json"
        with open(pointer_file, "w", encoding="utf-8") as f:
            json.dump(pointer_info, f, indent=2)
        recorded_path = to_portable_rel_path(pointer_file)
    else:
        ensure_dir(dest_file.parent)
        if src.resolve() != dest_file:
            shutil.copy2(src, dest_file)
        recorded_path = to_portable_rel_path(dest_file)

    return {
        "name": target_name,
        "path": recorded_path,
        "source_path": to_portable_rel_path(src),
        "size_bytes": size_bytes,
        "sha256": sha256,
        "file_type": file_type,
        "generator": generator,
        "is_by_reference": is_heavy,
    }


def setup_notebook_figures(
    step: int | str, run_id: str | None = None, outputs_root: Path | None = None
) -> tuple[Path, Path]:
    """Khởi tạo và trả về (official_figures_dir, bundle_figures_dir) cho notebook.

    Giúp notebook vừa lưu hình vào results/figures/official/ vừa lưu vào outputs/ bundle.
    """
    effective_run_id = run_id or os.environ.get(
        "XAI_FOOTBALL_RUN_ID", "official_evaluation_20261021_184358"
    )
    official_dir = ensure_dir(PROJECT_ROOT / "results" / "figures" / "official")
    bundle = init_step_run_bundle(step, effective_run_id, outputs_root=outputs_root)
    return official_dir, bundle["figures"]


def save_figure_artifact(
    fig_name: str,
    fig_or_plt: Any = None,
    step: int | str = 0,
    run_id: str | None = None,
    save_results_official: bool = True,
    outputs_root: Path | None = None,
    dpi: int = 300,
) -> dict[str, str]:
    """Lưu biểu đồ đồng thời vào outputs/<step>/runs/<run_id>/figures/ và results/figures/official/.

    Hỗ trợ cả đối tượng matplotlib.figure.Figure và module matplotlib.pyplot (mặc định nếu fig_or_plt=None).
    Tự động xuất cả 2 định dạng PNG (300 DPI) và SVG (vector).
    """
    import matplotlib.pyplot as plt

    target = plt if fig_or_plt is None else fig_or_plt
    effective_run_id = run_id or os.environ.get(
        "XAI_FOOTBALL_RUN_ID", "official_evaluation_20261021_184358"
    )

    clean_name = fig_name.replace(".png", "").replace(".svg", "")
    saved_paths: dict[str, str] = {}

    # 1. Lưu vào results/figures/official nếu được yêu cầu (backward compatibility)
    if save_results_official:
        official_fig_dir = ensure_dir(PROJECT_ROOT / "results" / "figures" / "official")
        png_path = official_fig_dir / f"{clean_name}.png"
        svg_path = official_fig_dir / f"{clean_name}.svg"
        if hasattr(target, "savefig"):
            target.savefig(png_path, dpi=dpi, bbox_inches="tight")
            target.savefig(svg_path, bbox_inches="tight")
        else:
            plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
            plt.savefig(svg_path, bbox_inches="tight")
        saved_paths["official_png"] = to_portable_rel_path(png_path)
        saved_paths["official_svg"] = to_portable_rel_path(svg_path)

    # 2. Lưu vào outputs/<step_dir>/runs/<run_id>/figures
    if effective_run_id:
        bundle = init_step_run_bundle(step, effective_run_id, outputs_root=outputs_root)
        out_png = bundle["figures"] / f"{clean_name}.png"
        out_svg = bundle["figures"] / f"{clean_name}.svg"
        if hasattr(target, "savefig"):
            target.savefig(out_png, dpi=dpi, bbox_inches="tight")
            target.savefig(out_svg, bbox_inches="tight")
        else:
            plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
            plt.savefig(out_svg, bbox_inches="tight")
        saved_paths["bundle_png"] = to_portable_rel_path(out_png)
        saved_paths["bundle_svg"] = to_portable_rel_path(out_svg)

    return saved_paths


def build_manifest_schema(
    run_id: str,
    step: int | str,
    status: str,
    command_or_entrypoint: str,
    exit_code: int = 0,
    started_at: str | None = None,
    finished_at: str | None = None,
    configs_used: dict[str, str] | None = None,
    inputs: list[dict[str, Any]] | None = None,
    outputs: list[dict[str, Any]] | None = None,
    quality_checks: list[dict[str, Any]] | None = None,
    dependencies: list[dict[str, str]] | None = None,
    official_eligible: bool = True,
    official_eligibility_reason: str = "",
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Tạo đối tượng manifest chuẩn hóa theo schema thống nhất."""
    step_num = int(step) if str(step).isdigit() else 0
    step_id = f"step_{step_num:02d}"
    step_name = STEP_NAMES.get(step_num, str(step))

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t_start = started_at or now_iso
    t_end = finished_at or now_iso

    git_info = get_git_info(PROJECT_ROOT)

    key_packages = {}
    for pkg in ("torch", "torch_geometric", "pandas", "numpy", "scikit-learn"):
        try:
            mod = __import__(pkg.replace("-", "_"))
            key_packages[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            key_packages[pkg] = "not installed"

    manifest = {
        "schema_version": "1.0.0",
        "run_id": run_id,
        "step_id": step_id,
        "step_name": step_name,
        "status": status,
        "started_at": t_start,
        "finished_at": t_end,
        "command_or_entrypoint": command_or_entrypoint,
        "exit_code": exit_code,
        "git": {
            "branch": git_info["branch"],
            "commit_sha": git_info["commit_sha"],
            "clean_working_tree": git_info["clean_working_tree"],
        },
        "environment": {
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "architecture": platform.machine(),
            "packages": key_packages,
        },
        "configs": configs_used or {},
        "inputs": inputs or [],
        "outputs": outputs or [],
        "quality_checks": quality_checks or [],
        "dependencies": dependencies or [],
        "official_eligible": official_eligible,
        "official_eligibility_reason": official_eligibility_reason,
    }
    if extra_metadata:
        manifest["extra_metadata"] = extra_metadata
    return manifest


def write_step_manifest(run_dir: Path, manifest_data: dict[str, Any]) -> Path:
    """Ghi tệp manifest.json an toàn nguyên tử vào thư mục run."""
    dest = run_dir / "manifest.json"
    save_atomic_json(manifest_data, dest)
    return dest


def update_global_run_registry(
    run_id: str,
    run_type: str,
    step_statuses: dict[str, str],
    description: str = "",
    outputs_root: Path | None = None,
    git_commit: str | None = None,
    extra_info: dict[str, Any] | None = None,
) -> Path:
    """Cập nhật tệp outputs/run_registry.json tổng hợp toàn bộ các runs."""
    base = Path(outputs_root) if outputs_root else OUTPUTS_ROOT
    ensure_dir(base)
    reg_file = base / "run_registry.json"

    registry: dict[str, Any] = {
        "schema_version": "1.0.0",
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "runs": {},
    }
    if reg_file.exists():
        try:
            with open(reg_file, encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict) and "runs" in loaded:
                    registry = loaded
        except Exception:
            pass

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    current_git = get_git_info(PROJECT_ROOT)

    entry = {
        "run_id": run_id,
        "run_type": run_type,
        "description": description,
        "last_updated": now_iso,
        "git": {
            "branch": current_git["branch"],
            "commit_sha": git_commit or current_git["commit_sha"],
            "clean_working_tree": current_git["clean_working_tree"],
        },
        "step_statuses": step_statuses,
    }
    if extra_info:
        entry.update(extra_info)

    registry["updated_at"] = now_iso
    registry["runs"][run_id] = entry

    save_atomic_json(registry, reg_file)
    return reg_file


def generate_outputs_readme(outputs_root: Path | None = None) -> Path:
    """Sinh tệp outputs/README.md hướng dẫn cấu trúc và tra cứu bằng chứng."""
    base = Path(outputs_root) if outputs_root else OUTPUTS_ROOT
    ensure_dir(base)
    readme_path = base / "README.md"

    content = """# outputs/ — Kho Lưu Trữ Minh Chứng Thực Nghiệm (Pipeline Steps 0–6)

Thư mục này chứa toàn bộ các **bằng chứng khoa học có cấu trúc (Evidence Bundles)** được sinh ra tự động từ pipeline Bước 0 đến Bước 6, phục vụ kiểm toán khả năng tái lập và cung cấp số liệu cho bài báo khoa học / khóa luận.

---

## 1. Cấu trúc thư mục chuẩn

```
outputs/
├── README.md                 # Hướng dẫn cấu trúc & quy tắc kiểm toán
├── run_registry.json         # Danh bạ đăng ký trung tâm các run thực nghiệm
├── step_00_environment_and_reproducibility/
│   └── runs/<run_id>/
├── step_01_data_audit_and_collection/
│   └── runs/<run_id>/
├── step_02_preprocessing_and_labels/
│   └── runs/<run_id>/
├── step_03_graph_construction/
│   └── runs/<run_id>/
├── step_04_match_level_split/
│   └── runs/<run_id>/
├── step_05_model_training/
│   └── runs/<run_id>/
└── step_06_model_evaluation/
    └── runs/<run_id>/
```

Trong mỗi thư mục `runs/<run_id>/`:
- `files/`: Bảng số liệu CSV, cấu hình JSON/YAML, danh sách kiểm kê và fingerprints.
- `figures/`: Biểu đồ trực quan hóa xuất đồng thời ở định dạng PNG (300 DPI) và SVG (vector).
- `logs/`: Nhật ký thực thi thực tế.
- `reports/`: Báo cáo kết quả và bảng kiểm định chất lượng (quality checks).
- `manifest.json`: Siêu dữ liệu xuất xứ chuẩn ISO 8601 (SHA-256, commit Git, tham số, inputs, outputs).

---

## 2. Quy tắc Quản lý và Truy vết (Provenance Integrity)

1. **Định danh duy nhất**: Toàn bộ các bước thuộc cùng một lần chạy chia sẻ chung một `run_id`.
2. **Không tự động chọn run mới nhất**: Mọi script phân tích bắt buộc nhận `run_id` rõ ràng; tuyệt đối không quét thư mục theo thứ tự chữ cái hay thời gian sửa file.
3. **Phân loại rõ ràng**:
   - `HISTORICAL_OFFICIAL`: Kết quả thực nghiệm chính thức 5-seed từ commit `fc18a929` trên nhánh `main`, được bảo toàn làm minh chứng đối soát.
   - `HISTORICAL_PROVISIONAL`: Artifact lịch sử quy mô mẫu nhỏ (ví dụ run XAI 5 đồ thị từ commit `9fae6c51`).
   - `SMOKE`: Kết quả kiểm tra đường dẫn pipeline nhanh (1 epoch hoặc subset nhỏ).
   - `OFFICIAL`: Run thực nghiệm đầy đủ, hợp lệ trên mã nguồn hiện tại.
4. **Không đưa file nhị phân nặng lên Git**: Checkpoint `.pt`, đồ thị `.pt`, parquet dung lượng lớn được ghi nhận thông qua mã băm SHA-256 và file pointer `.pointer.json`.
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    return readme_path
