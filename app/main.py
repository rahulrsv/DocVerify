"""FastAPI application factory and dependency wiring."""

from fastapi import FastAPI

from app.api.routes import router
from app.audit.store import AuditStore
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.core.pipeline import CaseProcessor
from app.extraction.base import ExtractionProvider
from app.extraction.factory import build_provider
from app.policies.config import load_policy


def create_app(settings: Settings | None = None, provider: ExtractionProvider | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    audit = AuditStore(settings.db_path)
    processor = CaseProcessor(
        provider or build_provider(settings),
        load_policy(settings.policy_path),
        audit,
        max_pdf_pages=settings.max_pdf_pages,
        max_text_chars=settings.max_text_chars,
    )
    app = FastAPI(
        title="IncomeLens AI",
        version="0.1.0",
        description=(
            "Auditable income-document underwriting **simulation**. Synthetic data only. "
            "Outcomes reflect demonstration rules and are not credit decisions."
        ),
    )
    app.state.settings = settings
    app.state.audit = audit
    app.state.processor = processor
    app.include_router(router)
    return app

