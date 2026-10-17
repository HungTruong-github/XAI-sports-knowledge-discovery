"""Unit tests cho module thu thập metadata môi trường và provenance."""

from xai_football.utils.environment import collect_environment_metadata
from xai_football.utils.provenance import compute_dict_sha256, get_git_info


def test_get_git_info():
    info = get_git_info()
    assert isinstance(info, dict)
    assert "branch" in info
    assert "commit_sha" in info
    assert "clean_working_tree" in info


def test_collect_environment_metadata():
    meta = collect_environment_metadata()
    assert "os" in meta
    assert "python_version" in meta
    assert "packages" in meta
    assert "torch" in meta["packages"]
    assert "git" in meta
    assert "config_hashes" in meta


def test_compute_dict_sha256():
    d1 = {"b": 2, "a": 1}
    d2 = {"a": 1, "b": 2}
    # Tính tất định độc lập với thứ tự key
    assert compute_dict_sha256(d1) == compute_dict_sha256(d2)
    assert len(compute_dict_sha256(d1)) == 64
