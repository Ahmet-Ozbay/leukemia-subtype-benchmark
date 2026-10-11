"""
Unit tests for the automated dataset acquisition pipeline (src/data/download_data.py).
Tests authentication checks, cache verification, --force purging, error handling,
and dispatcher registry without requiring active network calls.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.download_data import (
    DOWNLOADERS,
    download_geo,
    download_golub,
    ensure_kaggle_token,
)


# ==============================================================================
# 1. Authentication Tests
# ==============================================================================
def test_ensure_kaggle_token_present(monkeypatch):
    """When KAGGLE_API_TOKEN is set in the environment, ensure_kaggle_token returns True."""
    monkeypatch.setenv("KAGGLE_API_TOKEN", "KGAT_test_dummy_token_123")
    assert ensure_kaggle_token() is True


def test_ensure_kaggle_token_missing_non_interactive(monkeypatch):
    """When KAGGLE_API_TOKEN is missing and shell is non-interactive, fails gracefully."""
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert ensure_kaggle_token() is False


# ==============================================================================
# 2. Golub Downloader Tests
# ==============================================================================
def test_download_golub_cache_hit(tmp_path):
    """If all expected files exist with positive size, skips download and returns True."""
    dest_dir = tmp_path
    golub_dir = dest_dir / "golub"
    golub_dir.mkdir(parents=True)

    expected = [
        "data_set_ALL_AML_train.csv",
        "data_set_ALL_AML_independent.csv",
        "actual.csv",
    ]
    for filename in expected:
        f = golub_dir / filename
        f.write_text("dummy,content\n1,2\n")

    # Should return True without needing Kaggle token or kagglehub
    assert download_golub("crawford/gene-expression", "golub", dest_dir, force=False) is True


def test_download_golub_staged_from_kagglehub(tmp_path, monkeypatch):
    """Tests staging files when download is triggered via mocked kagglehub."""
    dest_dir = tmp_path
    monkeypatch.setenv("KAGGLE_API_TOKEN", "mock_token")

    # Mock cache dir returned by kagglehub
    fake_cache = tmp_path / "fake_kaggle_cache"
    fake_cache.mkdir(parents=True)
    for filename in [
        "data_set_ALL_AML_train.csv",
        "data_set_ALL_AML_independent.csv",
        "actual.csv",
    ]:
        (fake_cache / filename).write_text("gene,expr\n100,200\n")

    mock_kagglehub = MagicMock()
    mock_kagglehub.dataset_download.return_value = str(fake_cache)
    monkeypatch.setitem(__import__("sys").modules, "kagglehub", mock_kagglehub)

    success = download_golub("crawford/gene-expression", "golub", dest_dir, force=True)
    assert success is True

    # Check files are copied into dest_dir/golub
    for filename in ["data_set_ALL_AML_train.csv", "data_set_ALL_AML_independent.csv", "actual.csv"]:
        assert (dest_dir / "golub" / filename).exists()
        assert (dest_dir / "golub" / filename).stat().st_size > 0


def test_download_golub_missing_files_fails(tmp_path, monkeypatch):
    """If kagglehub archive is missing expected files, returns False."""
    dest_dir = tmp_path
    monkeypatch.setenv("KAGGLE_API_TOKEN", "mock_token")

    fake_cache = tmp_path / "fake_empty_cache"
    fake_cache.mkdir(parents=True)
    # Only create one file out of three
    (fake_cache / "actual.csv").write_text("id,cancer\n")

    mock_kagglehub = MagicMock()
    mock_kagglehub.dataset_download.return_value = str(fake_cache)
    monkeypatch.setitem(__import__("sys").modules, "kagglehub", mock_kagglehub)

    success = download_golub("crawford/gene-expression", "golub", dest_dir, force=True)
    assert success is False


# ==============================================================================
# 3. GEO Downloader Tests
# ==============================================================================
def test_download_geo_cache_hit(tmp_path):
    """If a valid non-empty archive (>1KB) exists, skips download and returns True."""
    dest_dir = tmp_path
    geo_dir = dest_dir / "armstrong_2002"
    geo_dir.mkdir(parents=True)

    archive = geo_dir / "GSE28497_family.soft.gz"
    archive.write_bytes(b"x" * 2048)  # 2KB file

    assert download_geo("GSE28497", "armstrong_2002", dest_dir, force=False) is True


def test_download_geo_force_unlinks_existing(tmp_path, monkeypatch):
    """When force=True, existing files are deleted before GEOparse download."""
    dest_dir = tmp_path
    geo_dir = dest_dir / "armstrong_2002"
    geo_dir.mkdir(parents=True)

    old_file = geo_dir / "GSE28497_family.soft.gz"
    old_file.write_bytes(b"x" * 2048)

    mock_geoparse = MagicMock()

    def fake_get_geo(geo, destdir, silent=False):
        # Simulate creating newly downloaded file
        (Path(destdir) / f"{geo}_new.soft.gz").write_bytes(b"new" * 1024)
        return MagicMock()

    mock_geoparse.get_GEO.side_effect = fake_get_geo
    monkeypatch.setitem(__import__("sys").modules, "GEOparse", mock_geoparse)

    success = download_geo("GSE28497", "armstrong_2002", dest_dir, force=True)
    assert success is True
    # The old file should have been unlinked
    assert not old_file.exists()


# ==============================================================================
# 4. Dispatcher Registry Completeness
# ==============================================================================
def test_downloaders_registry_completeness():
    """Ensures all three required cohorts are present in the DOWNLOADERS dispatch table."""
    expected_cohorts = {"golub", "armstrong", "mile"}
    assert set(DOWNLOADERS.keys()) == expected_cohorts
