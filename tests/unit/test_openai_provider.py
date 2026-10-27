"""OpenAI provider behaviour with a fake client: no network, no API key."""

from types import SimpleNamespace

import httpx
import openai
import pytest

from app.extraction.base import ProviderError
from app.extraction.openai_provider import OpenAIExtractionProvider
from app.models.extraction import RawExtraction

REQ = httpx.Request("POST", "https://api.openai.com/v1/responses")


class FakeResponses:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def provider(outcomes, attempts=3):
    fake = SimpleNamespace(responses=FakeResponses(outcomes))
    return OpenAIExtractionProvider(api_key=None, model="test-model", max_attempts=attempts, client=fake, backoff_s=0), fake.responses


def ok():
    return SimpleNamespace(status="completed", output=[SimpleNamespace(content=[SimpleNamespace(type="output_text")])], output_parsed=RawExtraction.blank("pay_stub"))


def test_success_uses_structured_output_and_trusted_instructions():
    p, fake = provider([ok()])
    result = p.extract("Gross Pay: $1.00", "D1")
    assert result.raw.document_type == "pay_stub" and result.attempts == 1
    call = fake.calls[0]
    assert call["text_format"] is RawExtraction
    assert "untrusted" in call["instructions"].lower()
    assert "Gross Pay" not in call["instructions"]  # document text never enters system instructions


def test_retries_transient_errors_then_succeeds():
    p, fake = provider([openai.APITimeoutError(request=REQ), openai.APIConnectionError(request=REQ), ok()])
    assert p.extract("x", "D1").attempts == 3


def test_bounded_retries_raise_after_max_attempts():
    p, _ = provider([openai.APITimeoutError(request=REQ)] * 2, attempts=2)
    with pytest.raises(ProviderError) as exc:
        p.extract("x", "D1")
    assert exc.value.kind == "timeout" and exc.value.attempts == 2


def test_refusal_is_not_retried():
    refusal = SimpleNamespace(status="completed", output=[SimpleNamespace(content=[SimpleNamespace(type="refusal", refusal="no")])], output_parsed=None)
    p, fake = provider([refusal, ok()])
    with pytest.raises(ProviderError) as exc:
        p.extract("x", "D1")
    assert exc.value.kind == "refusal" and len(fake.calls) == 1


def test_malformed_and_incomplete_responses():
    empty = SimpleNamespace(status="completed", output=[], output_parsed=None)
    p, _ = provider([empty], attempts=1)
    with pytest.raises(ProviderError) as exc:
        p.extract("x", "D1")
    assert exc.value.kind == "malformed"
    incomplete = SimpleNamespace(status="incomplete", incomplete_details=SimpleNamespace(reason="max_output_tokens"), output=[], output_parsed=None)
    p, _ = provider([incomplete], attempts=1)
    with pytest.raises(ProviderError) as exc:
        p.extract("x", "D1")
    assert exc.value.kind == "incomplete"


def test_auth_error_not_retried():
    err = openai.AuthenticationError("bad key", response=httpx.Response(401, request=REQ), body=None)
    p, fake = provider([err, ok()])
    with pytest.raises(ProviderError) as exc:
        p.extract("x", "D1")
    assert exc.value.kind == "auth" and len(fake.calls) == 1


def test_missing_api_key_fails_fast():
    with pytest.raises(ProviderError):
        OpenAIExtractionProvider(api_key=None, model="m")
