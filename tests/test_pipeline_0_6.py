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
    import hashlib

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

        # Tính lại SHA-256 và so sánh kích thước byte thực tế
        actual_sha = hashlib.sha256(target_file.read_bytes()).hexdigest()
        assert (
            actual_sha == item["sha256"]
        ), f"SHA-256 mismatch for {item['path']}: manifest={item['sha256']} vs disk={actual_sha}"
        assert (
            target_file.stat().st_size == item["size_bytes"]
        ), f"Size mismatch for {item['path']}: manifest={item['size_bytes']} vs disk={target_file.stat().st_size}"


def test_regression_missing_file_cannot_pass(tmp_path: Path):
    """Kiểm tra hồi quy: nếu file nguồn bắt buộc bị thiếu trong official mode, runner phải trả FAIL."""
    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from run_pipeline_0_6 import export_step_1, is_step_completed

    # 1. Gọi export_step_1 trong official mode khi raw data không đủ
    # Chế độ official bắt buộc 380 match files trên đĩa
    m = export_step_1("test_missing_raw", mode="official", outputs_root=tmp_path)
    # Tìm check match_count_complete
    chk = [c for c in m["quality_checks"] if c["name"] == "match_count_complete"][0]
    # Khi chạy trên máy hoặc CI mà không có 380 match files, check này bắt buộc FAIL
    # (nếu có đủ 380 thì PASS, nhưng nếu thiếu thì không được faking PASS vì collection_report)
    raw_dir = PROJECT_ROOT / "data" / "raw" / "2" / "27"
    raw_matches = (
        len([f for f in raw_dir.glob("*.json") if not f.name.endswith(".lineups.json")])
        if raw_dir.is_dir()
        else 0
    )
    if raw_matches < 380:
        assert chk["status"] == "FAIL"

    # 2. is_step_completed phải trả False nếu thiếu bất kỳ file nào trong danh sách
    assert not is_step_completed(999)


def test_runner_smoke_mode_execution(tmp_path: Path):
    """Kiểm tra chế độ smoke mode: dry-run và thực thi smoke step 0."""
    # Test dry-run
    cmd_dry = [
        sys.executable,
        "scripts/run_pipeline_0_6.py",
        "--mode",
        "smoke",
        "--dry-run",
        "--steps",
        "0,1",
    ]
    res_dry = subprocess.run(
        cmd_dry,
        cwd=PROJECT_ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert res_dry.returncode == 0
    assert "KẾ HOẠCH THỰC THI (DRY-RUN)" in res_dry.stdout

    # Test smoke step 0 execution
    cmd_smoke = [
        sys.executable,
        "scripts/run_pipeline_0_6.py",
        "--mode",
        "smoke",
        "--steps",
        "0",
        "--run-id",
        "test_smoke_unit",
        "--output-dir",
        str(tmp_path / "smoke_out"),
    ]
    res_smoke = subprocess.run(
        cmd_smoke,
        cwd=PROJECT_ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert res_smoke.returncode == 0
    assert "HOÀN TẤT THÀNH CÔNG" in res_smoke.stderr or res_smoke.returncode == 0


def test_runner_resume_logic():
    """Kiểm tra chức năng --resume: phát hiện chính xác bước đã hoàn thành."""
    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from run_pipeline_0_6 import is_step_completed

    # Bước 0 đã có environment.json và requirements.lock.txt
    assert is_step_completed(0) is True
    # Bước 6 đã có model_comparison.csv và significance_tests.csv
    assert is_step_completed(6) is True
    # Bước không tồn tại
    assert is_step_completed(99) is False
