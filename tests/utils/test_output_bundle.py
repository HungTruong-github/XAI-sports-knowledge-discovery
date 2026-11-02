"""Unit tests for Output Evidence Bundle and Provenance Tracker."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xai_football.utils.output_bundle import (
    STEP_DIRECTORIES,
    build_manifest_schema,
    generate_outputs_readme,
    get_step_dir_name,
    get_step_run_dir,
    init_step_run_bundle,
    register_artifact,
    to_portable_rel_path,
    update_global_run_registry,
    write_step_manifest,
)


def test_step_dir_names():
    assert STEP_DIRECTORIES[0] == "step_00_environment_and_reproducibility"
    assert get_step_dir_name(0) == "step_00_environment_and_reproducibility"
    assert get_step_dir_name(6) == "step_06_model_evaluation"
    assert (
        get_step_dir_name("step_01_data_audit_and_collection")
        == "step_01_data_audit_and_collection"
    )
    run_dir = get_step_run_dir(0, "run1")
    assert run_dir.name == "run1"

    with pytest.raises(ValueError):
        get_step_dir_name(7)


def test_init_step_run_bundle(tmp_path: Path):
    run_id = "test_run_123"
    bundle = init_step_run_bundle(0, run_id, outputs_root=tmp_path)

    assert bundle["run_dir"].exists()
    assert bundle["files"].is_dir()
    assert bundle["figures"].is_dir()
    assert bundle["logs"].is_dir()
    assert bundle["reports"].is_dir()
    assert bundle["run_dir"].name == run_id


def test_to_portable_rel_path(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    child = root / "data" / "splits" / "test.csv"
    child.parent.mkdir(parents=True)
    child.write_text("hello", encoding="utf-8")

    rel = to_portable_rel_path(child, base_root=root)
    assert rel == "data/splits/test.csv"
    assert "\\" not in rel  # Always POSIX forward slashes


def test_register_artifact_light_vs_heavy(tmp_path: Path):
    target_dir = tmp_path / "bundle_files"
    target_dir.mkdir()

    # 1. Light text file
    light_file = tmp_path / "summary.csv"
    light_file.write_text("col1,col2\n1,2", encoding="utf-8")
    reg_light = register_artifact(target_dir, light_file, file_type="table")

    assert not reg_light["is_by_reference"]
    assert (target_dir / "summary.csv").exists()
    assert reg_light["sha256"] is not None

    # 2. Heavy binary file (simulated .pt)
    heavy_file = tmp_path / "model.pt"
    heavy_file.write_bytes(b"\x00" * 1024)
    reg_heavy = register_artifact(target_dir, heavy_file, file_type="checkpoint")

    assert reg_heavy["is_by_reference"]
    assert not (target_dir / "model.pt").exists()
    pointer = target_dir / "model.pt.pointer.json"
    assert pointer.exists()
    with open(pointer, encoding="utf-8") as f:
        p_info = json.load(f)
    assert p_info["sha256"] == reg_heavy["sha256"]


def test_register_artifact_missing_raises(tmp_path: Path):
    target_dir = tmp_path / "bundle_files"
    target_dir.mkdir()
    missing_file = tmp_path / "nonexistent.csv"

    with pytest.raises(FileNotFoundError):
        register_artifact(target_dir, missing_file)


def test_manifest_schema_and_serialization(tmp_path: Path):
    run_id = "test_run_456"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)

    manifest_data = build_manifest_schema(
        run_id=run_id,
        step=0,
        status="PASS",
        command_or_entrypoint="python scripts/00_audit.py",
        exit_code=0,
        configs_used={"configs/data.yaml": "dummy_sha256"},
        inputs=[],
        outputs=[
            {"name": "env.json", "path": "outputs/step_00/runs/r1/files/env.json", "sha256": "123"}
        ],
        quality_checks=[{"name": "check1", "status": "PASS", "details": "ok"}],
        official_eligible=True,
    )

    manifest_file = write_step_manifest(run_dir, manifest_data)
    assert manifest_file.exists()

    with open(manifest_file, encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["schema_version"] == "1.0.0"
    assert loaded["run_id"] == run_id
    assert loaded["status"] == "PASS"
    assert loaded["official_eligible"] is True
    assert "git" in loaded
    assert "environment" in loaded
    assert len(loaded["quality_checks"]) == 1


def test_global_run_registry_update(tmp_path: Path):
    outputs_root = tmp_path / "outputs"
    run_id = "official_run_01"

    reg_file = update_global_run_registry(
        run_id=run_id,
        run_type="historical_official",
        step_statuses={"step_00": "PASS", "step_01": "PASS"},
        description="Test registry entry",
        outputs_root=outputs_root,
    )

    assert reg_file.exists()
    with open(reg_file, encoding="utf-8") as f:
        reg = json.load(f)

    assert run_id in reg["runs"]
    assert reg["runs"][run_id]["run_type"] == "historical_official"
    assert reg["runs"][run_id]["step_statuses"]["step_00"] == "PASS"


def test_generate_outputs_readme(tmp_path: Path):
    readme = generate_outputs_readme(outputs_root=tmp_path)
    assert readme.exists()
    content = readme.read_text(encoding="utf-8")
    assert "outputs/" in content
    assert "run_registry.json" in content
    assert "HISTORICAL_OFFICIAL" in content
