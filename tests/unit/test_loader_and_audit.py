import sqlite3

import pymupdf
import pytest

from app.audit.store import AuditStore
from app.documents.loader import DocumentRejected, load_document, validate_upload
from tests.conftest import doc_bytes


def test_validate_upload_rejections():
    with pytest.raises(DocumentRejected) as e:
        validate_upload("a.docx", b"x", 100)
    assert e.value.reason == "unsupported_type"
    with pytest.raises(DocumentRejected) as e:
        validate_upload("a.txt", b"", 100)
    assert e.value.reason == "empty"
    with pytest.raises(DocumentRejected) as e:
        validate_upload("a.txt", b"x" * 101, 100)
    assert e.value.reason == "too_large"
    with pytest.raises(DocumentRejected):
        validate_upload("fake.pdf", b"not a pdf", 100)
    assert validate_upload("ok.PDF", b"%PDF-1.7 ...", 100) == "application/pdf"


def test_pdf_text_extraction():
    loaded = load_document(0, "pay_stub.pdf", doc_bytes("GC007", "pay_stub.pdf"))
    assert loaded.record.load_error is None
    assert "Gross Pay: $2,916.67" in loaded.text
    assert loaded.record.page_count == 1


@pytest.mark.parametrize(("case", "filename", "error"), [("GC036", "pay_stub_corrupt.pdf", ""), ("GC037", "pay_stub_scanned.pdf", "NO_TEXT_LAYER"), ("GC038", "blank.txt", "EMPTY_TEXT")])
def test_unusable_content_becomes_load_error(case, filename, error):
    loaded = load_document(0, filename, doc_bytes(case, filename))
    assert loaded.record.load_error and error in loaded.record.load_error
    assert loaded.text == ""


def test_encrypted_pdf_rejected():
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret")
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
    assert "ENCRYPTED_PDF" in load_document(0, "x.pdf", data).record.load_error


def test_non_utf8_text_rejected():
    assert "NOT_UTF8" in load_document(0, "x.txt", b"\xff\xfe\x00bad").record.load_error


def test_hashes_are_stable_and_text_hash_ignores_whitespace():
    a = load_document(0, "a.txt", b"Gross Pay:  $1.00\n")
    b = load_document(1, "b.txt", b"gross pay: $1.00")
    assert a.record.sha256 != b.record.sha256
    assert a.record.text_sha256 == b.record.text_sha256
    assert load_document(0, "a.txt", b"Gross Pay:  $1.00\n").record.document_id == a.record.document_id


def test_audit_events_append_only_and_chained(tmp_path):
    store = AuditStore(tmp_path / "a.db")
    store.append_event("C1", "CASE_RECEIVED", {"n": 1})
    store.append_event("C1", "DECISION_RECORDED", {"outcome": "APPROVE"})
    store.append_event("C2", "CASE_RECEIVED", {"n": 2})
    events = store.events_for_case("C1")
    assert [e["event_type"] for e in events] == ["CASE_RECEIVED", "DECISION_RECORDED"]
    assert events[1]["prev_hash"] == events[0]["event_hash"]
    assert store.verify_chain() == (True, None)

    conn = sqlite3.connect(tmp_path / "a.db")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("UPDATE audit_events SET payload = '{}' WHERE seq = 1")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("DELETE FROM audit_events")


def test_processing_runs_cannot_be_deleted(tmp_path):
    AuditStore(tmp_path / "a.db")
    conn = sqlite3.connect(tmp_path / "a.db")
    conn.execute("INSERT INTO processing_runs VALUES ('r1','C1','t','f','[]','mock','m','p','s','pol','APPROVE','api',NULL,1,'[]','{}')")
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute("DELETE FROM processing_runs")


def test_tampering_detected_by_chain(tmp_path):
    store = AuditStore(tmp_path / "a.db")
    store.append_event("C1", "CASE_RECEIVED", {"n": 1})
    store.append_event("C1", "DECISION_RECORDED", {"outcome": "REFER_FOR_REVIEW"})
    conn = sqlite3.connect(tmp_path / "a.db")
    conn.execute("DROP TRIGGER audit_events_no_update")  # simulate someone bypassing the guard
    conn.execute("UPDATE audit_events SET payload = '{\"outcome\": \"APPROVE\"}' WHERE seq = 2")
    conn.commit()
    assert store.verify_chain() == (False, 2)
