"""Application settings read from environment variables. No secrets have defaults."""

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    provider: str = field(default_factory=lambda: _env("DOCVERIFY_PROVIDER", "mock"))
    openai_api_key: str | None = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY") or None, repr=False)
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4.1-mini"))
    openai_timeout_s: float = field(default_factory=lambda: float(_env("OPENAI_TIMEOUT_S", "30")))
    openai_max_attempts: int = field(default_factory=lambda: int(_env("OPENAI_MAX_ATTEMPTS", "3")))
    # Empty string means "do not send temperature" (some reasoning models reject it).
    openai_temperature: str = field(default_factory=lambda: _env("OPENAI_TEMPERATURE", "0"))
    db_path: Path = field(default_factory=lambda: Path(_env("DOCVERIFY_DB_PATH", str(PROJECT_ROOT / "var" / "docverify.db"))))
    policy_path: Path = field(
        default_factory=lambda: Path(_env("DOCVERIFY_POLICY_PATH", str(PROJECT_ROOT / "data" / "policies" / "demo_policy_v1.json")))
    )
    golden_dir: Path = field(default_factory=lambda: Path(_env("DOCVERIFY_GOLDEN_DIR", str(PROJECT_ROOT / "data" / "golden"))))
    reports_dir: Path = field(default_factory=lambda: Path(_env("DOCVERIFY_REPORTS_DIR", str(PROJECT_ROOT / "reports"))))
    max_upload_bytes: int = field(default_factory=lambda: int(_env("DOCVERIFY_MAX_UPLOAD_BYTES", str(5 * 1024 * 1024))))
    max_files_per_case: int = field(default_factory=lambda: int(_env("DOCVERIFY_MAX_FILES", "10")))
    max_pdf_pages: int = 20
    max_text_chars: int = 50_000
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))


def get_settings() -> Settings:
    return Settings()
