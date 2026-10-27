"""Upload validation and safe text extraction for PDF and TXT documents.

Two failure classes are kept distinct:

* ``DocumentRejected``: the upload itself is unacceptable (wrong type, too large,
  empty). The API rejects the request with a 4xx status.
* ``load_error`` on the record: the file is the right kind but its content cannot be
  used (corrupt PDF, no text layer, encrypted, not UTF-8). The case is still processed
  so the failure is visible in the audit trail and the policy outcome.
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import PurePath

import pymupdf

from app.models.case import DocumentRecord

SUPPORTED_EXTENSIONS = {".pdf": "application/pdf", ".txt": "text/plain"}
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class DocumentRejected(Exception):
    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason  # unsupported_type | too_large | empty


@dataclass(frozen=True)
class LoadedDocument:
    record: DocumentRecord
    text: str


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def normalized_text_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).casefold().encode()).hexdigest()


def validate_upload(filename: str, content: bytes, max_bytes: int) -> str:
    """Return the media type or raise ``DocumentRejected``."""
    ext = PurePath(filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DocumentRejected("unsupported_type", f"Unsupported file type '{ext or 'none'}'. Allowed: .pdf, .txt")
    if len(content) == 0:
        raise DocumentRejected("empty", f"File '{filename}' is empty")
    if len(content) > max_bytes:
        raise DocumentRejected("too_large", f"File '{filename}' exceeds the {max_bytes} byte limit")
    if ext == ".pdf" and not content.startswith(b"%PDF-"):
        raise DocumentRejected("unsupported_type", f"File '{filename}' has a .pdf extension but is not a PDF")
    return SUPPORTED_EXTENSIONS[ext]


def _extract_pdf_text(content: bytes, max_pages: int) -> tuple[str, int]:
    # PyMuPDF parses from memory; nothing is executed (no JavaScript, no external links followed).
    with pymupdf.open(stream=content, filetype="pdf") as doc:
        if doc.needs_pass:
            raise ValueError("ENCRYPTED_PDF: password-protected PDFs are not supported")
        if doc.page_count > max_pages:
            raise ValueError(f"TOO_MANY_PAGES: {doc.page_count} pages exceeds limit of {max_pages}")
        text = "\n".join(page.get_text("text") for page in doc)
        return text, doc.page_count


def load_document(index: int, filename: str, content: bytes, *, max_pages: int = 20, max_chars: int = 50_000) -> LoadedDocument:
    """Extract text from an already validated upload. Content problems become ``load_error``."""
    digest = sha256_bytes(content)
    media_type = SUPPORTED_EXTENSIONS.get(PurePath(filename).suffix.lower(), "application/octet-stream")
    record = DocumentRecord(
        document_id=f"D{index + 1}-{digest[:8]}",
        filename=PurePath(filename).name,
        media_type=media_type,
        sha256=digest,
    )
    text = ""
    try:
        if media_type == "application/pdf":
            text, record.page_count = _extract_pdf_text(content, max_pages)
        else:
            try:
                text = content.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise ValueError("NOT_UTF8: text files must be UTF-8 encoded") from exc
        text = _CONTROL_CHARS.sub("", text.replace("\r\n", "\n"))
        if len(text) > max_chars:
            raise ValueError(f"TOO_MUCH_TEXT: {len(text)} characters exceeds limit of {max_chars}")
        if not text.strip():
            if media_type == "application/pdf":
                raise ValueError("NO_TEXT_LAYER: no extractable text (scanned PDFs need OCR, which is out of scope)")
            raise ValueError("EMPTY_TEXT: document contains no text")
    except ValueError as exc:
        record.load_error = str(exc)
        text = ""
    except Exception as exc:  # malformed PDFs raise a variety of library errors
        record.load_error = f"UNREADABLE_DOCUMENT: {type(exc).__name__}"
        text = ""

    record.char_count = len(text)
    record.text_sha256 = normalized_text_hash(text) if text else None
    return LoadedDocument(record=record, text=text)
