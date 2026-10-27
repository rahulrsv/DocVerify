"""OpenAI structured-output extraction provider.

Uses the Responses API with a Pydantic ``text_format`` so the model is constrained to
the ``RawExtraction`` JSON schema. Handles refusals, malformed/incomplete output and
API failures explicitly, with bounded retries for transient errors only.
"""

import logging
import time
from typing import Any

import openai
from pydantic import ValidationError

from app.extraction.base import ProviderError, ProviderResult
from app.extraction.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_user_message
from app.models.extraction import RawExtraction

log = logging.getLogger(__name__)

_RETRYABLE = (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)


class OpenAIExtractionProvider:
    name = "openai"
    prompt_version = PROMPT_VERSION

    def __init__(
        self,
        api_key: str | None,
        model: str,
        timeout_s: float = 30.0,
        max_attempts: int = 3,
        temperature: float | None = 0.0,
        client: Any | None = None,
        backoff_s: float = 1.0,
    ) -> None:
        if client is None:
            if not api_key:
                raise ProviderError("auth", "OPENAI_API_KEY is not set; use INCOMELENS_PROVIDER=mock to run offline")
            # SDK-level retries disabled so every attempt is visible and bounded here.
            client = openai.OpenAI(api_key=api_key, timeout=timeout_s, max_retries=0)
        self._client = client
        self.model = model
        self._max_attempts = max(1, max_attempts)
        self._temperature = temperature
        self._backoff_s = backoff_s

    def extract(self, text: str, document_id: str) -> ProviderResult:
        started = time.perf_counter()
        last_error: ProviderError | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                raw = self._call(text, document_id)
                return ProviderResult(raw=raw, attempts=attempt, latency_ms=int((time.perf_counter() - started) * 1000))
            except _RETRYABLE as exc:
                kind = "timeout" if isinstance(exc, openai.APITimeoutError) else "api_error"
                last_error = ProviderError(kind, f"Transient provider error: {type(exc).__name__}", attempt)
            except ValidationError as exc:
                # Structured outputs make this rare, but never trust the wire format blindly.
                last_error = ProviderError("malformed", f"Response failed schema validation: {exc.error_count()} errors", attempt)
            except openai.AuthenticationError as exc:
                raise ProviderError("auth", "Authentication with the provider failed", attempt) from exc
            except openai.APIStatusError as exc:  # 4xx other than rate limit: not retryable
                raise ProviderError("api_error", f"Provider rejected the request (HTTP {exc.status_code})", attempt) from exc
            except ProviderError as exc:
                exc.attempts = attempt
                if exc.kind == "refusal":
                    raise  # retrying a refusal is pointless
                last_error = exc
            log.warning(
                "extraction attempt failed",
                extra={"ctx": {"document_id": document_id, "attempt": attempt, "kind": last_error.kind}},
            )
            if attempt < self._max_attempts:
                time.sleep(self._backoff_s * (2 ** (attempt - 1)))
        assert last_error is not None
        raise last_error

    def _call(self, text: str, document_id: str) -> RawExtraction:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": SYSTEM_PROMPT,
            "input": [{"role": "user", "content": build_user_message(document_id, text)}],
            "text_format": RawExtraction,
        }
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        response = self._client.responses.parse(**kwargs)

        for item in getattr(response, "output", None) or []:
            for part in getattr(item, "content", None) or []:
                if getattr(part, "type", None) == "refusal":
                    raise ProviderError("refusal", f"Model refused: {getattr(part, 'refusal', '')[:200]}")
        if getattr(response, "status", "completed") == "incomplete":
            reason = getattr(getattr(response, "incomplete_details", None), "reason", "unknown")
            raise ProviderError("incomplete", f"Response incomplete: {reason}")
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise ProviderError("malformed", "Response contained no parsed output")
        if not isinstance(parsed, RawExtraction):
            parsed = RawExtraction.model_validate(parsed)
        return parsed
