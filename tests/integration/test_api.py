from tests.conftest import doc_bytes


def files(case_id, *names):
    return [("files", (n, doc_bytes(case_id, n), "application/octet-stream")) for n in names]


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["provider"] == "mock" and body["policy_version"] == "demo-policy-1.0"


def test_process_get_and_audit(client):
    r = client.post("/documents/process", files=files("GC001", "pay_stub.txt", "employment_letter.txt"), data={"case_id": "demo-1", "as_of_date": "2026-10-15"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["decision"]["outcome"] == "APPROVE"
    assert body["extractions"][0]["gross_pay"]["value"] == "3076.92"

    case = client.get("/cases/demo-1")
    assert case.status_code == 200 and case.json()["run_id"] == body["run_id"]

    audit = client.get("/cases/demo-1/audit").json()
    assert audit["chain_verified"] is True
    assert audit["runs"][0]["run_id"] == body["run_id"]
    assert audit["events"][0]["event_type"] == "CASE_RECEIVED"
    assert client.get("/cases").json()[0]["case_id"] == "demo-1"


def test_duplicate_upload_is_idempotent(client):
    payload = dict(files=files("GC001", "pay_stub.txt", "employment_letter.txt"), data={"case_id": "idem", "as_of_date": "2026-10-15"})
    first = client.post("/documents/process", **payload)
    second = client.post("/documents/process", **payload)
    assert first.status_code == 201 and second.status_code == 200
    assert second.json()["idempotent_replay"] is True
    assert second.json()["run_id"] == first.json()["run_id"]
    forced = client.post("/documents/process", files=payload["files"], data={**payload["data"], "force": "true"})
    assert forced.status_code == 201 and forced.json()["run_id"] != first.json()["run_id"]


def test_api_error_cases(client):
    assert client.get("/cases/unknown-case").status_code == 404
    assert client.get("/cases/unknown-case/audit").status_code == 404
    assert client.get("/cases/bad id!").status_code in (404, 422)
    assert client.post("/documents/process", data={"case_id": "x"}).status_code == 422  # no files
    bad_type = client.post("/documents/process", files=[("files", ("a.docx", b"hello", "application/octet-stream"))])
    assert bad_type.status_code == 415
    fake_pdf = client.post("/documents/process", files=[("files", ("a.pdf", b"hello", "application/pdf"))])
    assert fake_pdf.status_code == 415
    empty = client.post("/documents/process", files=[("files", ("a.txt", b"", "text/plain"))])
    assert empty.status_code == 400
    big = client.post("/documents/process", files=[("files", ("a.txt", b"x" * 300_000, "text/plain"))])
    assert big.status_code == 413
    bad_id = client.post("/documents/process", files=files("GC001", "pay_stub.txt"), data={"case_id": "no spaces allowed"})
    assert bad_id.status_code == 422
    bad_date = client.post("/documents/process", files=files("GC001", "pay_stub.txt"), data={"as_of_date": "not-a-date"})
    assert bad_date.status_code == 422


def test_corrupt_pdf_is_processed_not_rejected(client):
    r = client.post("/documents/process", files=files("GC036", "pay_stub_corrupt.pdf", "employment_letter.txt"), data={"case_id": "corrupt", "as_of_date": "2026-10-15"})
    assert r.status_code == 201
    assert r.json()["decision"]["outcome"] == "REFER_FOR_REVIEW"
    assert r.json()["documents"][0]["load_error"]


def test_evals_endpoints(client):
    assert client.get("/evals/latest").status_code == 404
    r = client.post("/evals/run", json={"provider": "mock", "case_ids": ["GC001", "GC044"]})
    assert r.status_code == 200, r.text
    summary = r.json()
    assert summary["cases"] == 2 and summary["provider"] == "mock"
    assert "MOCK MODE" in summary["mode_label"]
    latest = client.get("/evals/latest").json()
    assert latest["metadata"]["eval_run_id"] == summary["eval_run_id"]
    # eval runs are visible in the audit trail, tagged with the eval run id
    runs = client.get("/cases/GC044/audit").json()["runs"]
    assert runs[-1]["evaluation_run_id"] == summary["eval_run_id"]


def test_openai_eval_without_key_is_503(client):
    assert client.post("/evals/run", json={"provider": "openai"}).status_code == 503
