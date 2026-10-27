"""Optional live test against the OpenAI API. Skipped unless OPENAI_API_KEY is set."""

import os
from decimal import Decimal

import pytest

from app.core.config import get_settings
from app.extraction.factory import build_provider
from app.extraction.normalize import normalize_extraction
from tests.conftest import doc_bytes

pytestmark = pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set; live LLM test skipped")


@pytest.mark.live
def test_live_extraction_resists_injection():
    provider = build_provider(get_settings(), "openai")
    text = doc_bytes("GC044", "pay_stub.txt").decode()
    ext = normalize_extraction(provider.extract(text, "D1").raw, "D1", text)
    assert ext.document_type.value == "pay_stub"
    assert ext.gross_pay.value == Decimal("1923.08")
