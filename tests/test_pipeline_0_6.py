"""Integration tests for scripts/run_pipeline_0_6.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from xai_football.utils.paths import PROJECT_ROOT


def test_pipeline_0_6_cli_help():
    cmd = [sys.executable, "scripts/run_pipeline_0_6.py", "--help"]
    res = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert res.returncode == 0
    assert "run_pipeline_0_6.py" in res.stdout
    assert "--mode" in res.stdout
    assert "--run-id" in res.stdout
    assert "--steps" in res.stdout


def test_pipeline_0_6_existing_mode_step_0(tmp_path: Path):
    test_run_id = "test_run_audit_00_tmp"
    cmd = [
        sys.executable,
        "scripts/run_pipeline_0_6.py",
        "--mode",
        "existing",
        "--run-id",
        test_run_id,
        "--steps",
        "0",
    ]
    res = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert res.returncode == 0, f"Error: {res.stderr}\n{res.stdout}"

    run_dir = (
        PROJECT_ROOT / "outputs" / "step_00_environment_and_reproducibility" / "runs" / test_run_id
    )
    assert run_dir.exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "files" / "environment.json").exists()
    assert (run_dir / "reports" / "environment_report.md").exists()
    assert (run_dir / "logs" / "step_00_audit.log").exists()

    with open(run_dir / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["status"] == "HISTORICAL_OFFICIAL"
    assert manifest["official_eligible"] is True

    # Cleanup temporary test directory and registry entry
    import shutil

    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)

    reg_file = PROJECT_ROOT / "outputs" / "run_registry.json"
    if reg_file.exists():
        try:
            with open(reg_file, encoding="utf-8") as f:
                rdata = json.load(f)
            if "runs" in rdata and test_run_id in rdata["runs"]:
                del rdata["runs"][test_run_id]
                with open(reg_file, "w", encoding="utf-8") as f:
                    json.dump(rdata, f, indent=2)
        except Exception:
            pass
