"""Sinh tệp metadata môi trường và phiên bản (results/metadata/<run_name>/environment.json).

Cách dùng:
    python scripts/00_generate_environment_metadata.py --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xai_football.utils.environment import save_environment_metadata
from xai_football.utils.io import load_yaml
from xai_football.utils.logging import get_logger
from xai_football.utils.paths import get_metadata_dir

logger = get_logger("environment_metadata")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    data_cfg = load_yaml(args.config)
    meta_dir = get_metadata_dir(data_cfg)
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_file = meta_dir / "environment.json"

    meta = save_environment_metadata(out_file)
    logger.info("Đã cập nhật metadata môi trường vào: %s", out_file)
    print(f"OS: {meta['os']}")
    print(f"Python: {meta['python_version']}")
    print(f"Git commit: {meta['git']['commit_sha']} (branch: {meta['git']['branch']})")
    print(f"Clean working tree: {meta['git']['clean_working_tree']}")
    print(f"CUDA: {meta['cuda_available']} ({meta['cuda_device_name']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
