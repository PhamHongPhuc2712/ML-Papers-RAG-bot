"""Configuration validation that needs no services."""

from __future__ import annotations

import os

import pytest

from copilot.config import Settings, SettingsError


@pytest.fixture(autouse=True)
def _no_ambient_data_dir(monkeypatch):
    for name in ("DATA_DIR", "COPILOT_DATA_DIR"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("environment", ["development", "test", "production"])
def test_unset_data_dir_is_rejected_in_every_environment(environment):
    with pytest.raises(SettingsError, match="DATA_DIR"):
        Settings(
            _env_file=None,
            environment=environment,
            secret_key="not-a-default-secret",
            cors_origins=["http://localhost:3000"],
            database_url="postgresql+psycopg://u:p@localhost:5432/test_copilot",
            qdrant_collection_prefix="test_",
        )


def test_data_dir_is_read_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert Settings(_env_file=None).data_dir == tmp_path


@pytest.mark.parametrize("separator", ["\\", "/", os.sep])
def test_documented_trailing_separator_spelling_is_normalized(tmp_path, separator):
    settings = Settings(_env_file=None, data_dir=str(tmp_path) + separator)
    assert settings.data_dir == tmp_path


def test_relative_data_dir_is_rejected():
    with pytest.raises(SettingsError, match="absolute"):
        Settings(_env_file=None, data_dir="relative/data")


def test_test_environment_requires_the_test_subdirectory(tmp_path):
    common = {
        "environment": "test",
        "database_url": "postgresql+psycopg://u:p@localhost:5432/test_copilot",
        "qdrant_collection_prefix": "test_",
    }
    with pytest.raises(SettingsError, match="test_subdirectory"):
        Settings(_env_file=None, data_dir=tmp_path, **common)
    assert Settings(_env_file=None, data_dir=tmp_path / "test", **common).data_dir.name == "test"


def test_comma_separated_cors_origins_parse_from_dotenv(tmp_path):
    env_file = tmp_path / ".env"
    lines = [f"DATA_DIR={tmp_path}", "CORS_ORIGINS=http://localhost:3000, https://app.example"]
    env_file.write_text(os.linesep.join(lines) + os.linesep, encoding="utf-8")
    settings = Settings(_env_file=env_file)
    assert settings.cors_origins == ["http://localhost:3000", "https://app.example"]
    assert settings.data_dir == tmp_path
