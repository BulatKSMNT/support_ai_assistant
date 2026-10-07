import io
from uuid import uuid4

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


def test_pdf_source_preserves_page_number(client):
    writer = PdfWriter()
    for sentence in [
        "Basic safety instructions for equipment.",
        "Delivery requires the destination city and shipping address.",
    ]:
        page = writer.add_blank_page(width=400, height=200)
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {
                        NameObject("/F1"): DictionaryObject(
                            {
                                NameObject("/Type"): NameObject("/Font"),
                                NameObject("/Subtype"): NameObject("/Type1"),
                                NameObject("/BaseFont"): NameObject("/Helvetica"),
                            }
                        )
                    }
                )
            }
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 10 Tf 20 100 Td ({sentence}) Tj ET".encode())
        page[NameObject("/Contents")] = stream
    content = io.BytesIO()
    writer.write(content)
    result = client.post(
        "/v1/documents/upload",
        data={"source_key": "pdf-handbook", "title": "Customer handbook"},
        files={"file": ("handbook.pdf", content.getvalue(), "application/pdf")},
    )
    assert result.status_code == 200, result.text
    hits = client.post("/v1/search", json={"query": "Delivery destination address"}).json()
    assert hits[0]["page"] == 2
    assert "shipping address" in hits[0]["text"]


def test_protected_routes_and_public_liveness(client):
    for path in ["/v1/documents", "/health/ready", "/metrics"]:
        response = client.get(path, headers={"Authorization": "Bearer wrong"})
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
    assert client.get("/health/live", headers={"Authorization": ""}).status_code == 200
    assert client.get("/").status_code == 200
    schema = client.get("/openapi.json").json()
    assert schema["paths"]["/v1/drafts"]["post"]["security"] == [{"HTTPBearer": []}]


def test_idempotent_ingest_update_and_delete(client, document):
    first = client.post("/v1/documents", json=document)
    assert first.status_code == 200, first.text
    first = first.json()
    assert first["document"]["version"] == 1
    duplicate = client.post("/v1/documents", json=document).json()
    assert not duplicate["changed"]
    assert duplicate["document"]["updated_at"] == first["document"]["updated_at"]
    hits = client.post("/v1/search", json={"query": "Как рассчитывается доставка?"}).json()
    assert hits[0]["document_id"] == first["document"]["id"]
    document["text"] = (
        "Оформление возврата требует номера заказа и фотографии повреждённого товара."
    )
    updated = client.post("/v1/documents", json=document).json()
    assert updated["document"]["version"] == 2
    assert updated["document"]["id"] == first["document"]["id"]
    # Changed title is also a versioned change, even if body remains the same.
    document["title"] = "Правила возврата"
    assert client.post("/v1/documents", json=document).json()["document"]["version"] == 3
    hits = client.post("/v1/search", json={"query": "Фотографии повреждённого товара"}).json()
    assert hits[0]["version"] == 3
    assert "Срок поставки" not in hits[0]["text"]
    identifier = first["document"]["id"]
    assert client.delete("/v1/documents/" + identifier).status_code == 204
    assert client.post("/v1/search", json={"query": "возврат товара"}).json() == []
    assert client.delete("/v1/documents/" + identifier).status_code == 404


def test_empty_knowledge_abstains_and_extractive_mode_is_explicit(client, document):
    empty = client.post("/v1/drafts", json={"question": "Какие сроки доставки?"}).json()
    assert empty["status"] == "insufficient_evidence"
    assert empty["draft"] == ""
    client.post("/v1/documents", json=document)
    identifier = str(uuid4())
    result = client.post(
        "/v1/drafts",
        json={"question": "Как уточнить доставку?"},
        headers={"X-Request-ID": identifier},
    )
    assert result.status_code == 200, result.text
    data = result.json()
    assert data["generation_mode"] == "extractive"
    assert data["status"] == "ready"
    assert data["draft"] == data["sources"][0]["text"] + " [S1]"
    assert data["request_id"] == result.headers["X-Request-ID"] == identifier
    assert "не создавался" in data["missing_information"][0]
    assert "no-store" in result.headers["Cache-Control"]


def test_upload_utf8_limits_and_pdf_without_text(client, document):
    fields = {"source_key": "uploaded", "title": "Инструкция доставки"}
    response = client.post(
        "/v1/documents/upload",
        data=fields,
        files={"file": ("инструкция.md", document["text"].encode(), "text/markdown")},
    )
    assert response.status_code == 200, response.text
    bad = client.post(
        "/v1/documents/upload", data=fields, files={"file": ("file.txt", b"\xff\xfe", "text/plain")}
    )
    assert bad.status_code == 422
    bad = client.post(
        "/v1/documents/upload",
        data=fields,
        files={"file": ("file.exe", b"MZ" * 50, "application/octet-stream")},
    )
    assert bad.status_code == 415
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    content = io.BytesIO()
    writer.write(content)
    empty = client.post(
        "/v1/documents/upload",
        data=fields,
        files={"file": ("scan.pdf", content.getvalue(), "application/pdf")},
    )
    assert empty.status_code == 422
    assert "OCR" in empty.json()["error"]["message"]


def test_chunked_request_is_bounded_before_parsing(client):
    def chunks():
        for _ in range(22):
            yield b"x" * 100_000

    result = client.post(
        "/v1/documents", content=chunks(), headers={"Content-Type": "application/json"}
    )
    assert result.status_code == 413


def test_validation_does_not_echo_personal_data(client):
    secret = "private-customer-data"
    result = client.post("/v1/drafts", json={"question": "valid question", "unexpected": secret})
    assert result.status_code == 422
    assert secret not in result.text


def test_metrics_and_readiness(client, document):
    ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json()["generation_mode"] == "extractive"
    client.post("/v1/documents", json=document)
    client.post("/v1/drafts", json={"question": "Что уточнить для доставки?"})
    metrics = client.get("/metrics")
    assert "rag_drafts_total" in metrics.text
    assert 'mode="extractive",status="ready"' in metrics.text
    assert document["text"] not in metrics.text
    assert (
        client.post("/v1/search", json={"query": "доставка", "mode": "hybrid"}).status_code == 409
    )
