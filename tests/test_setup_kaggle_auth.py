"""
Unit tests for the Kaggle API authentication setup utility (scripts/setup_kaggle_auth.py).
Tests interactive token entry, existing token retention, overwriting, error handling,
and .env persistence using temporary isolated environments.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import setup_kaggle_auth
from scripts.setup_kaggle_auth import prompt_kaggle_token


@pytest.fixture
def mock_env(tmp_path, monkeypatch):
    """Provides an isolated .env path and clears KAGGLE_API_TOKEN from environment."""
    test_env_path = tmp_path / ".env"
    monkeypatch.setattr(setup_kaggle_auth, "ENV_PATH", test_env_path)
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    return test_env_path


def test_prompt_kaggle_token_new_entry(mock_env, monkeypatch):
    """When no token exists, prompts for token, saves to .env, and sets os.environ."""
    test_token = "KGAT_test_valid_token_12345"
    monkeypatch.setattr("builtins.input", lambda prompt="": test_token)

    success = prompt_kaggle_token()
    assert success is True

    # Check os.environ updated
    assert os.environ.get("KAGGLE_API_TOKEN") == test_token

    # Check .env file contents
    assert mock_env.exists()
    content = mock_env.read_text(encoding="utf-8")
    assert f"KAGGLE_API_TOKEN={test_token}" in content


def test_prompt_kaggle_token_existing_in_environ_keep(mock_env, monkeypatch):
    """When a token exists in os.environ and user declines overwrite, keeps existing token."""
    existing_token = "KGAT_existing_env_token_999"
    monkeypatch.setenv("KAGGLE_API_TOKEN", existing_token)

    # User answers 'n' to overwrite prompt
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")

    success = prompt_kaggle_token(force_overwrite=False)
    assert success is True
    assert os.environ.get("KAGGLE_API_TOKEN") == existing_token
    # .env was not created or modified
    assert not mock_env.exists()


def test_prompt_kaggle_token_existing_in_file_keep(mock_env, monkeypatch):
    """When a token is stored in .env and user declines overwrite, loads into os.environ."""
    existing_token = "KGAT_file_token_555"
    mock_env.write_text(f"KAGGLE_API_TOKEN={existing_token}\n", encoding="utf-8")

    # User answers 'n' to overwrite prompt
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")

    success = prompt_kaggle_token(force_overwrite=False)
    assert success is True
    assert os.environ.get("KAGGLE_API_TOKEN") == existing_token


def test_prompt_kaggle_token_overwrite_existing(mock_env, monkeypatch):
    """When a token exists and user confirms overwrite, updates .env and os.environ."""
    old_token = "KGAT_old_token"
    new_token = "KGAT_brand_new_token"
    mock_env.write_text(f"OTHER_SETTING=true\nKAGGLE_API_TOKEN={old_token}\n", encoding="utf-8")
    monkeypatch.setenv("KAGGLE_API_TOKEN", old_token)

    # First input: 'y' for overwrite; second input: new token
    inputs = iter(["y", new_token])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    success = prompt_kaggle_token(force_overwrite=False)
    assert success is True
    assert os.environ.get("KAGGLE_API_TOKEN") == new_token

    content = mock_env.read_text(encoding="utf-8")
    assert "OTHER_SETTING=true" in content
    assert f"KAGGLE_API_TOKEN={new_token}" in content
    assert old_token not in content


def test_prompt_kaggle_token_force_overwrite(mock_env, monkeypatch):
    """When force_overwrite=True, does not prompt for confirmation and asks directly for token."""
    old_token = "KGAT_old_token"
    new_token = "KGAT_forced_new_token"
    monkeypatch.setenv("KAGGLE_API_TOKEN", old_token)

    # Only single input needed since confirmation is bypassed
    monkeypatch.setattr("builtins.input", lambda prompt="": new_token)

    success = prompt_kaggle_token(force_overwrite=True)
    assert success is True
    assert os.environ.get("KAGGLE_API_TOKEN") == new_token
    assert f"KAGGLE_API_TOKEN={new_token}" in mock_env.read_text(encoding="utf-8")


def test_prompt_kaggle_token_user_abort_keyboard_interrupt(mock_env, monkeypatch):
    """Handles KeyboardInterrupt / cancellation cleanly and returns False."""
    def raise_interrupt(prompt=""):
        raise KeyboardInterrupt()

    monkeypatch.setattr("builtins.input", raise_interrupt)

    success = prompt_kaggle_token()
    assert success is False


def test_prompt_kaggle_token_user_abort_eof(mock_env, monkeypatch):
    """Handles EOFError cleanly and returns False."""
    def raise_eof(prompt=""):
        raise EOFError()

    monkeypatch.setattr("builtins.input", raise_eof)

    success = prompt_kaggle_token()
    assert success is False


def test_prompt_kaggle_token_empty_input_reprompts(mock_env, monkeypatch):
    """Re-prompts user if they provide an empty token string."""
    valid_token = "KGAT_eventual_valid_token"
    inputs = iter(["", "   ", valid_token])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    success = prompt_kaggle_token()
    assert success is True
    assert os.environ.get("KAGGLE_API_TOKEN") == valid_token
