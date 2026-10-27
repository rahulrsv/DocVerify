# Architecture

IncomeLens AI is a single Python service (FastAPI) with a Streamlit client, a SQLite
store and an offline evaluation harness. There are no queues, microservices or agent
frameworks; each concern is a small module with typed inputs and outputs.

```
             +-----------------------------------------------------------------+
upload ----> | loader -> safety scan -> provider.extract -> normalize/ground   |
(PDF/TXT)    |   |            |              (LLM or mock)        |            |
             |   |            v                                   v            |
             |   |      injection signals                DocumentExtraction    |
             |   v                                    (Decimal, ISO dates,     |
             | DocumentRecord (hashes)                 evidence per field)     |
             |        \____________________   ___________________/           |
             |                             v v                                |
             |                   calculations (deterministic)                 |
             |                   -> DerivedValue + Finding                    |
             |                             |                                  |
             |                   policy engine (JSON config)                  |
             |                   -> RuleResult[] -> Decision                  |
             +-----------------------------|----------------------------------+
                                           v
                       SQLite: processing_runs + append-only audit_events
```

## Modules

| Module | Responsibility |
|---|---|
| `app/documents/loader.py` | Upload validation (type, size, PDF magic bytes) and safe text extraction with PyMuPDF (page and character limits, encrypted/no-text-layer detection). Computes the file hash and a whitespace-normalized text hash. |
| `app/documents/safety.py` | Deterministic prompt-injection signal scanner. |
| `app/documents/classifier.py` | Keyword classifier (used by the mock provider). |
| `app/extraction/base.py` | `ExtractionProvider` protocol and `ProviderError`. |
| `app/extraction/openai_provider.py` | OpenAI Responses API with `text_format=RawExtraction` (strict structured output), explicit refusal/incomplete/malformed handling, bounded retries with exponential backoff for transient errors only, per-request timeout, SDK retries disabled so attempts are visible. |
| `app/extraction/mock_provider.py` | Deterministic regex extractor. Not an LLM; labeled as such in every report. |
| `app/extraction/prompts.py` | Versioned system prompt; document text wrapped in a per-request delimiter. |
| `app/extraction/normalize.py` | Converts the raw wire schema into the canonical schema, grounds every value in the source text and records validation issues. Shared by all providers. |
| `app/calculations/income.py` | Annualization, consistency checks, duplicates, date plausibility, deposit corroboration. Emits facts, never decisions. |
| `app/policies/` | Rule functions, JSON policy config, aggregation into an outcome. |
| `app/audit/store.py` | SQLite persistence of runs and hash-chained append-only events. |
| `app/core/pipeline.py` | Orchestrates one case end to end and writes the audit trail. |
| `app/api/routes.py` | Thin HTTP handlers. |
| `app/evals/` | Golden-set loader, metrics, baseline comparison, report rendering, CLI. |

## Key decisions and trade-offs

**Two schemas: wire vs canonical.** The LLM fills `RawExtraction`, where every value is a
string plus an evidence excerpt. Strict JSON-schema structured outputs map cleanly onto
strings, and parsing money and dates in our own code keeps conversion deterministic and
testable. `DocumentExtraction` holds `Decimal`, `date` and enums. The cost is a
normalization layer; the benefit is that LLM and mock outputs obey exactly the same rules.

**Grounding as the main anti-hallucination control.** A value is kept only if its
evidence string occurs verbatim in the document (after whitespace and case
normalization), and, for money and names, the value appears in that evidence. Otherwise
the value becomes `null` with an `EVIDENCE_NOT_GROUNDED` or `VALUE_NOT_IN_EVIDENCE` issue.
A model that invents "$15,000.00" cannot make it reach the policy engine. Trade-off: a
model that paraphrases evidence loses correct values. That is the safe failure direction,
and it shows up in eval metrics as a recall loss.

**Ambiguity becomes null.** `09/10/2026` is rejected as `AMBIGUOUS_DATE` instead of being
guessed. Frequency is never inferred from period length. Gross is never back-calculated
from net or YTD.

**Facts vs decisions.** Calculations emit `Finding`s (for example
`PAY_PERIOD_FREQUENCY_MISMATCH`) and `DerivedValue`s with method, inputs and assumptions.
Rules read those and apply thresholds from `data/policies/demo_policy_v1.json`. Changing a
threshold, disabling a rule or changing a rule's fail effect is a config change with a new
`policy_version`; prompts are untouched.

**The model has no authority.** The wire schema has no decision field. Rules never read
model output except validated facts. The one exception is the model's
`suspicious_instructions_found` flag, which can only *escalate* a case to review (R013),
never relax a rule.

**Outcome precedence.** `REFER_FOR_REVIEW > INSUFFICIENT_EVIDENCE > APPROVE` (configurable).
A risk signal such as manipulation, conflict or a duplicate needs a person even if documents
are also missing, because requesting more documents alone would not resolve it.

**Recurring income method.** Annualized income comes from the most recent non-duplicate
pay stub's gross × periods per year (52/26/24/12). It is computed only when every pay stub
has gross and frequency, all frequencies agree, none is irregular, and gross variation
(max−min)/mean is ≤ 10%. Otherwise income is `VARIABLE` or `UNKNOWN` and is not annualized.
Other assumptions: period lengths weekly 6–8 days, biweekly 13–15, semimonthly 12–17,
monthly 27–32 (inclusive, with tolerance); pay date within −3/+15 days of period end;
salary tolerance 5%; pay stub recency ≤ 60 days from a case-level `as_of_date`. The
date is an input, not `today()`, so runs are reproducible.

**Duplicates.** Three levels: identical bytes (sha256), identical normalized text, and
identical pay stub facts (employer, period, pay date, gross). Duplicates are excluded from
income calculations and trigger review.

**Idempotency.** A run's fingerprint is the hash of document hashes, as-of date, provider,
model, prompt version, schema version and policy version. Re-submitting the same
case and fingerprint returns the stored run (HTTP 200, `idempotent_replay: true`) and logs a
`PROCESSING_REPLAYED` event. `force=true` records a new run. A changed policy or prompt
yields a new fingerprint and therefore a new run, never an overwrite.

**Upload validation vs content failure.** Wrong type, empty, oversized or fake-PDF uploads
are rejected (415/400/413). A real PDF that is corrupt, encrypted or has no text layer is
*processed*: the document gets a `load_error`, the failure is audited, and policy refers the
case. Silent rejection would leave no trace.

**Synchronous processing.** Simple and enough for a demo. With a real LLM a multi-document
case takes several seconds, and `POST /evals/run` with `provider=openai` blocks for minutes.
A job queue is listed under next improvements.

## Audit design (MVP, not compliant or tamper-proof)

* `processing_runs`: one immutable row per run (an UPDATE trigger aborts), holding the full
  `CaseResult` JSON (extractions with evidence, validation issues, derived values,
  rule results, decision, provider/model/prompt/schema/policy versions, attempts, errors,
  duration, evaluation run ID).
* `audit_events`: append-only (UPDATE/DELETE triggers abort). Each event stores
  `prev_hash` and `event_hash = sha256(prev_hash, fields)`. `verify_chain()` detects edits
  made by bypassing the triggers. Anyone with write access to the file can still rebuild the
  whole chain, so this is tamper-evident at best, not tamper-proof.
* Event payloads hold IDs, hashes, codes and outcomes, not raw document text. API keys are
  never logged, and the JSON log formatter redacts known secret keys.

## Security posture

Document text is untrusted data. Layers: (1) trusted instructions only in the system
prompt; (2) document wrapped in a per-request random delimiter that the document cannot
close; (3) strict output schema without decision fields; (4) grounding; (5) deterministic
injection scanner that forces review; (6) deterministic policy outside the model. Layers 1
to 3 reduce risk and are not guarantees. Layers 4 to 6 limit what a successful injection can
achieve.
