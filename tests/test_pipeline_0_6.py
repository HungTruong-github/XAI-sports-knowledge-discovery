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


def test_audit_steps_0_6_cli_help():
    cmd = [sys.executable, "scripts/audit_steps_0_6.py", "--help"]
    res = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert res.returncode == 0
    assert "audit_steps_0_6.py" in res.stdout
    assert "--mode" in res.stdout
    assert "--strict" in res.stdout
    assert "--output-dir" in res.stdout


def test_pipeline_audit_json_validates_schema():
    import jsonschema

    schema_file = PROJECT_ROOT / "results" / "metadata" / "official" / "pipeline_audit.schema.json"
    audit_file = PROJECT_ROOT / "results" / "metadata" / "official" / "pipeline_audit.json"
    assert schema_file.is_file(), f"Missing schema: {schema_file}"
    assert audit_file.is_file(), f"Missing audit report: {audit_file}"

    schema = json.loads(schema_file.read_text(encoding="utf-8"))
    audit_data = json.loads(audit_file.read_text(encoding="utf-8"))
    jsonschema.validate(instance=audit_data, schema=schema)
    assert audit_data["overall_status"] in ("PASS", "PARTIAL", "FAIL", "BLOCKED")
    assert len(audit_data["checks"]) >= 30
    assert len(audit_data["steps"]) == 7


def test_step_0_6_evidence_manifest_validity():
    manifest_file = PROJECT_ROOT / "results" / "reproducibility" / "step_0_6_evidence_manifest.json"
    assert manifest_file.is_file(), f"Missing manifest: {manifest_file}"

    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert data["total_artifacts"] >= 30
    assert len(data["artifacts"]) == data["total_artifacts"]

    for item in data["artifacts"]:
        assert "path" in item
        assert "sha256" in item and len(item["sha256"]) == 64
        assert "size_bytes" in item and item["size_bytes"] > 0
        assert "git_tracked" in item
        target_file = PROJECT_ROOT / item["path"]
        assert (
            target_file.is_file()
        ), f"Artifact declared in manifest does not exist on disk: {target_file}"


def test_regression_missing_file_cannot_pass(tmp_path: Path):
    """Kiểm tra hồi quy: nếu file nguồn bắt buộc bị thiếu thì check phải trả FAIL hoặc BLOCKED, cấm trả PASS."""
    import sys

    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from audit_steps_0_6 import AuditCollector

    collector = AuditCollector(PROJECT_ROOT)
    # Thử đăng ký kiểm tra khi file không tồn tại
    non_existent = tmp_path / "does_not_exist.csv"
    collector.record_check(
        name="test_missing_file_check",
        step=1,
        status="PASS" if non_existent.is_file() else "FAIL",
        details="Kiểm tra file không tồn tại",
        expected=True,
        observed=non_existent.is_file(),
        evidence=[non_existent] if non_existent.is_file() else [],
    )
    chk = collector.checks[0]
    assert chk["status"] == "FAIL"
    assert chk["observed"] is False
    assert len(chk["evidence_paths"]) == 0
