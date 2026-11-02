"""Bước 8 — Đối sánh chất lượng các phương pháp giải thích (XAI Evaluation & Comparison).

Tham chiếu: docs/Quy_trinh_XAI_Football.md, mục 8 và configs/xai.yaml.
So sánh:
    - GNNExplainer vs PGExplainer vs Random Baseline
    - Các độ đo: Fidelity+, Fidelity-, Sparsity, Stability, Deletion AUC, Insertion AUC.

QUY TẮC BẢO VỆ DỮ LIỆU & TRUY XUẤT PROVENANCE:
    - Bắt buộc chỉ định rõ --xai-dir hoặc --run-id. Nghiêm cấm tự động chọn thư mục "mới nhất" theo tên.
    - Mọi run trước khi tổng hợp đều phải qua kiểm tra manifest (run_id, commit, split, số mẫu).
    - Run pilot, smoke test, partial hoặc run lịch sử (historical) không được ghi đè bảng official canonical.
    - Run lịch sử khi được chỉ định rõ ràng qua --allow-historical sẽ được lưu vào bảng riêng biệt kèm tag HISTORICAL.

Cách dùng:
    # 1. Liệt kê các run khả dụng (nếu không truyền --run-id hay --xai-dir):
    python scripts/09_evaluate_xai.py --config configs/data.yaml

    # 2. Đối sánh run cụ thể:
    python scripts/09_evaluate_xai.py --config configs/data.yaml --run-id 20261017_091540 --allow-historical
    python scripts/09_evaluate_xai.py --config configs/data.yaml --xai-dir results/xai/official/mpnn/<run_id>

Đầu ra:
    results/tables/<run_name>/xai_comparison.csv (chỉ khi run official hợp lệ vượt qua kiểm tra)
    results/tables/<run_name>/xai_comparison_historical_<run_id>.csv (khi chạy phân tích lịch sử)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from xai_football.utils.io import load_yaml, save_atomic_csv  # noqa: E402
from xai_football.utils.logging import get_logger  # noqa: E402
from xai_football.utils.paths import get_tables_dir  # noqa: E402
from xai_football.utils.provenance import (  # noqa: E402
    format_run_table,
    get_git_info,
    scan_available_xai_runs,
    validate_run_manifest,
)

logger = get_logger("evaluate_xai")


def resolve_target_xai_dir(
    base_xai_dir: Path,
    xai_dir_arg: str | None,
    run_id_arg: str | None,
) -> tuple[Path | None, str]:
    """Xác định thư mục run XAI dựa trên tham số dòng lệnh."""
    if xai_dir_arg:
        p = Path(xai_dir_arg)
        if not p.is_dir():
            return None, f"Thư mục chỉ định qua --xai-dir không tồn tại: {xai_dir_arg}"
        return p, ""

    if run_id_arg:
        # Tìm thư mục khớp với run_id trong base_xai_dir hoặc toàn bộ results/xai
        matches = [d for d in base_xai_dir.glob(f"**/{run_id_arg}") if d.is_dir()]
        if not matches:
            # Tìm kiếm mở rộng trong results/xai
            alt_base = Path("results/xai")
            matches = [d for d in alt_base.glob(f"**/{run_id_arg}") if d.is_dir()]

        if not matches:
            return None, f"Không tìm thấy thư mục nào ứng với run_id '{run_id_arg}'"
        if len(matches) > 1:
            logger.warning(
                "Tìm thấy nhiều thư mục khớp với run_id '%s': %s. Chọn thư mục đầu tiên.",
                run_id_arg,
                matches,
            )
        return matches[0], ""

    return None, "Chưa chỉ định --xai-dir hoặc --run-id"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument(
        "--xai-dir", default=None, help="Đường dẫn thư mục kết quả XAI cụ thể cần đối sánh"
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Mã định danh duy nhất (run_id) của lần chạy XAI cần đối sánh",
    )
    parser.add_argument(
        "--allow-historical",
        action="store_true",
        help="Cho phép phân tích run lịch sử / thử nghiệm (kết quả lưu riêng, không ghi đè official)",
    )
    parser.add_argument(
        "--promote-official",
        action="store_true",
        help="Thao tác xác nhận cập nhật bảng canonical official sau khi vượt qua toàn bộ kiểm tra",
    )
    args = parser.parse_args()

    data_cfg = load_yaml(args.config)
    run_name = data_cfg.get("run_name", "official")
    tables_dir = get_tables_dir(data_cfg)
    base_xai = Path("results/xai") / run_name

    # 1. Bắt buộc người dùng chỉ định rõ ràng run cần đối sánh; không tự ý chọn run mới nhất
    if not args.xai_dir and not args.run_id:
        available_runs = scan_available_xai_runs(Path("results/xai"))
        logger.error(
            "YÊU CẦU CHỈ ĐỊNH RUN: Bạn phải truyền tham số '--run-id <ID>' hoặc '--xai-dir <DIR>'.\n"
            "Hệ thống không tự động chọn run mới nhất theo tên thư mục nhằm ngăn chặn rò rỉ hoặc lấy nhầm số liệu cũ.\n\n"
            "DANH SÁCH CÁC RUN XAI KHẢ DỤNG TRÊN ĐĨA:\n%s\n\n"
            "Ví dụ:\n"
            "    python scripts/09_evaluate_xai.py --run-id 20261017_091540 --allow-historical\n",
            format_run_table(available_runs),
        )
        return 1

    target_dir, err = resolve_target_xai_dir(base_xai, args.xai_dir, args.run_id)
    if target_dir is None:
        logger.error("LỖI ĐƯỜNG DẪN: %s", err)
        return 1

    logger.info("Đang kiểm tra và tổng hợp run tại: %s", target_dir)

    # 2. Đọc và kiểm tra Manifest của run
    manifest_file = target_dir / "xai_run_manifest.json"
    if not manifest_file.exists():
        logger.error(
            "TỪ CHỐI TỔNG HỢP: Thư mục %s thiếu tệp xai_run_manifest.json (UNVERIFIED).\n"
            "Mọi kết quả không rõ nguồn gốc đều bị từ chối đưa vào bảng tổng hợp.",
            target_dir,
        )
        return 1

    with open(manifest_file, encoding="utf-8") as f:
        manifest: dict[str, Any] = json.load(f)

    # 3. Kiểm định các trường provenance bắt buộc trong manifest
    run_id = manifest.get("run_id", target_dir.name)
    manifest_run_name = manifest.get("run_name", "unknown")
    split = manifest.get("split", "unknown")
    n_graphs = manifest.get("n_graphs_explained", 0)
    git_info = manifest.get("git", {})
    commit_sha = git_info.get("commit_sha", "unknown")
    checkpoint_sha = manifest.get("model_checkpoint_sha256", "unknown")
    status = manifest.get("status", "complete").lower()

    # Kiểm tra tính toàn vẹn cấu hình
    is_valid, reason = validate_run_manifest(
        manifest, expected_run_name=None, require_official=False
    )
    if not is_valid:
        logger.error("Manifest của run %s không hợp lệ: %s", run_id, reason)
        return 1

    # 4. Phân biệt run hiện hành (Official) và run lịch sử (Historical / Pilot / Partial)
    is_historical = False
    historical_reasons = []

    if status != "complete":
        is_historical = True
        historical_reasons.append(f"Trạng thái run là '{status}' (chưa hoàn thành)")

    # Nhận diện run pilot
    if manifest_run_name == "pilot":
        is_historical = True
        historical_reasons.append("Run thuộc cấu hình pilot (data_pilot.yaml)")

    # Nhận diện run kích thước mẫu nhỏ / thử nghiệm (vd: run 20261017_091540 chỉ có 5 graphs)
    if n_graphs < 50:
        is_historical = True
        historical_reasons.append(
            f"Số mẫu quá ít ({n_graphs} graphs < 50) — đây là pilot/smoke test"
        )

    # Nhận diện commit khác biệt so với HEAD hiện tại
    current_git = get_git_info()
    if commit_sha != current_git.get("commit_sha"):
        is_historical = True
        historical_reasons.append(
            f"Commit khi chạy ({commit_sha[:8]}) khác commit hiện tại ({current_git.get('commit_sha')[:8]})"
        )

    # Nếu phát hiện là run lịch sử / thử nghiệm:
    if is_historical:
        if not args.allow_historical:
            logger.error(
                "TỪ CHỐI TỔNG HỢP OFFICIAL: Run '%s' là kết quả LỊCH SỬ / THỬ NGHIỆM:\n"
                "  - %s\n"
                "Quy tắc nghiên cứu: Không được dùng kết quả pilot, smoke test hoặc run lịch sử "
                "để ghi đè bảng official canonical.\n"
                "Để trích xuất bảng phân tích lịch sử độc lập, hãy truyền cờ '--allow-historical'.",
                run_id,
                "\n  - ".join(historical_reasons),
            )
            return 1

        logger.warning(
            "CẢNH BÁO PROVENANCE: Đang xử lý run lịch sử '%s' (--allow-historical được bật).\n"
            "Kết quả sẽ được đánh dấu rõ ràng là HISTORICAL_PROVISIONAL và lưu vào tệp riêng, "
            "tuyệt đối KHÔNG ghi đè bảng kết quả chính thức.",
            run_id,
        )

    # 5. Kiểm tra tệp số liệu tóm tắt xai_metrics_summary.csv
    summary_file = target_dir / "xai_metrics_summary.csv"
    if not summary_file.exists():
        logger.error("Không tìm thấy tệp số liệu tóm tắt %s", summary_file)
        return 1

    summary_df = pd.read_csv(summary_file)
    logger.info("Đã tải tệp tổng hợp XAI từ: %s (%d dòng)", summary_file, len(summary_df))

    # Gắn metadata nguồn gốc vào DataFrame
    summary_df["run_id"] = run_id
    summary_df["run_name"] = manifest_run_name
    summary_df["split"] = split
    summary_df["git_commit"] = commit_sha[:8]
    summary_df["checkpoint_sha256"] = checkpoint_sha[:8] if checkpoint_sha else "none"
    summary_df["provenance_status"] = (
        "HISTORICAL_PROVISIONAL" if is_historical else "OFFICIAL_VERIFIED"
    )

    tables_dir.mkdir(parents=True, exist_ok=True)

    # 6. Xuất bảng an toàn (Safe Atomic Write)
    if is_historical:
        out_table = tables_dir / f"xai_comparison_historical_{run_id}.csv"
        save_atomic_csv(summary_df, out_table, index=False, encoding="utf-8-sig")
        print("\n=== BẢNG ĐỐI SÁNH XAI (HISTORICAL RUN — KHÔNG PHẢI CANONICAL OFFICIAL) ===")
    else:
        out_table = tables_dir / "xai_comparison.csv"
        save_atomic_csv(summary_df, out_table, index=False, encoding="utf-8-sig")
        print("\n=== BẢNG ĐỐI SÁNH CHẤT LƯỢNG GIẢI THÍCH XAI CHÍNH THỨC (BƯỚC 8) ===")

    print(summary_df.to_string(index=False))
    print(f"\n[THÀNH CÔNG] Đã lưu an toàn kết quả đối sánh XAI vào: {out_table}")
    print(
        f"Nguồn gốc run: run_id={run_id}, commit={commit_sha[:8]}, samples={n_graphs}, split={split}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
