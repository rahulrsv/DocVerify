"""Provider selection. The rest of the app depends only on the ExtractionProvider protocol."""

from app.core.config import Settings
from app.extraction.base import ExtractionProvider
from app.extraction.mock_provider import RuleBasedExtractionProvider


def build_provider(settings: Settings, name: str | None = None) -> ExtractionProvider:
    choice = (name or settings.provider).lower()
    if choice == "mock":
        return RuleBasedExtractionProvider()
    if choice == "openai":
        from app.extraction.openai_provider import OpenAIExtractionProvider

        temperature = float(settings.openai_temperature) if settings.openai_temperature.strip() else None
        return OpenAIExtractionProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            timeout_s=settings.openai_timeout_s,
            max_attempts=settings.openai_max_attempts,
            temperature=temperature,
        )
    raise ValueError(f"Unknown extraction provider '{choice}'. Use 'mock' or 'openai'.")
