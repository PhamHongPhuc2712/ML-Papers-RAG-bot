"""Validated application configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import AliasChoices, Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

SettingsError = ValidationError


class Settings(BaseSettings):
    """Environment-backed configuration for the API and local services."""

    model_config = SettingsConfigDict(
        env_file=(".env", "backend/.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    environment: Literal["development", "test", "production"] = Field(
        default="development",
        validation_alias=AliasChoices("COPILOT_ENV", "ENVIRONMENT", "APP_ENV"),
    )
    # Single root for all persistent local data (spec §3). No code default: the
    # documented Windows default lives in .env.example, other platforms supply
    # their own, and startup fails in every environment when it is unset.
    data_dir: Path = Field(validation_alias=AliasChoices("COPILOT_DATA_DIR", "DATA_DIR"))
    database_url: str = Field(
        default="postgresql+psycopg://copilot:copilot@localhost:5432/copilot",
        validation_alias=AliasChoices(
            "COPILOT_DATABASE_URL", "DATABASE_URL", "TEST_DATABASE_URL"
        ),
    )
    qdrant_url: str = Field(
        default="http://localhost:6333",
        validation_alias=AliasChoices("COPILOT_QDRANT_URL", "QDRANT_URL", "TEST_QDRANT_URL"),
    )
    qdrant_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("COPILOT_QDRANT_API_KEY", "QDRANT_API_KEY"),
    )
    qdrant_collection_prefix: str = Field(
        default="dev_",
        validation_alias=AliasChoices(
            "COPILOT_QDRANT_COLLECTION_PREFIX",
            "QDRANT_COLLECTION_PREFIX",
            "TEST_QDRANT_COLLECTION_PREFIX",
        ),
    )
    secret_key: str = Field(
        default="change-me",
        validation_alias=AliasChoices("COPILOT_SECRET_KEY", "SECRET_KEY"),
    )
    model_mode: Literal["real", "mock"] = Field(
        default="real",
        validation_alias=AliasChoices("COPILOT_MODEL_MODE", "MODEL_MODE"),
    )
    mock_mode: bool = Field(
        default=False,
        validation_alias=AliasChoices("COPILOT_MOCK_MODE", "MOCK_MODE", "mock_mode"),
    )
    ready_timeout_seconds: float = Field(
        default=2.0,
        gt=0.0,
        le=30.0,
        validation_alias=AliasChoices("COPILOT_READY_TIMEOUT_SECONDS", "READY_TIMEOUT_SECONDS"),
    )
    cors_origins: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("COPILOT_CORS_ORIGINS", "CORS_ORIGINS"),
    )

    @field_validator("data_dir", mode="before")
    @classmethod
    def parse_data_dir(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise ValueError("data_dir_required")
            # Accept the documented trailing-separator spelling, e.g. C:\ml-copilot-data\.
            return Path(stripped.rstrip("\\/") or stripped)
        return value

    @field_validator("data_dir")
    @classmethod
    def validate_data_dir(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("data_dir_must_be_absolute")
        return value

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"postgresql", "postgresql+psycopg", "postgres"}:
            raise ValueError("database_url_must_use_postgresql")
        if not parsed.path or parsed.path == "/":
            raise ValueError("database_url_requires_database_name")
        return value

    @field_validator("qdrant_url")
    @classmethod
    def validate_qdrant_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("qdrant_url_must_be_http")
        return value.rstrip("/")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @model_validator(mode="after")
    def validate_environment(self) -> Settings:
        if self.environment == "test":
            database_name = urlparse(self.database_url).path.removeprefix("/")
            if not database_name.startswith("test_"):
                raise ValueError("test_database_name_must_start_with_test_")
            if not self.qdrant_collection_prefix.startswith("test_"):
                raise ValueError("test_qdrant_collection_prefix_must_start_with_test_")
            # Test files live only under <DATA_DIR>/test so cleanup cannot reach developer data.
            if self.data_dir.name != "test":
                raise ValueError("test_data_dir_must_be_the_test_subdirectory")

        if self.environment == "production":
            if self.model_mode == "mock" or self.mock_mode:
                raise ValueError("mock_model_mode_forbidden_in_production")
            if self.secret_key.strip().lower() in {
                "",
                "change-me",
                "change_me",
                "dev-secret",
                "dev_secret",
                "test-secret",
                "test_secret",
                "replace-this-secret",
                "replace-this-with-a-random-development-secret",
            }:
                raise ValueError("default_secret_forbidden_in_production")
            if not self.cors_origins or "*" in self.cors_origins:
                raise ValueError("unrestricted_cors_forbidden_in_production")

        return self
