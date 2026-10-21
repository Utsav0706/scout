from fastapi.testclient import TestClient
from sqlalchemy import func, select

from scout.models import Chunk
from tests.conftest import auth_headers, requires_postgres
from tests.fakes import make_pdf

POLICY_PDF = make_pdf(
    [
        "Refund policy. Customers can request a refund within 30 days of purchase.",
        "Shipping policy. Orders ship within 2 working days from the Leeds warehouse.",
    ]
)
RECIPE_PDF = make_pdf(["Pancake recipe. Mix flour, eggs and milk, then fry in butter."])


def _upload(client: TestClient, headers: dict, data: bytes, name: str = "policy.pdf"):
    return client.post(
        "/documents", headers=headers, files={"file": (name, data, "application/pdf")}
    )


# --- upload + list (run everywhere) ---


def test_upload_stores_document_and_chunks(client: TestClient) -> None:
    resp = _upload(client, auth_headers(client), POLICY_PDF)
    assert resp.status_code == 201, resp.text
    assert resp.json()["pages"] == 2
    with client.app.state.session_factory() as db:
        assert db.scalar(select(func.count()).select_from(Chunk)) == 2


def test_upload_requires_login(client: TestClient) -> None:
    assert _upload(client, {}, POLICY_PDF).status_code == 401


def test_non_pdf_rejected_even_if_named_pdf(client: TestClient) -> None:
    resp = _upload(client, auth_headers(client), b"just text, not a pdf", "fake.pdf")
    assert resp.status_code == 415


def test_too_large_rejected(client: TestClient) -> None:
    client.app.state.config.rag.max_upload_mb = 1
    resp = _upload(client, auth_headers(client), b"%PDF-" + b"0" * (1024 * 1024))
    assert resp.status_code == 413


def test_duplicate_upload_conflicts_with_existing_id(client: TestClient) -> None:
    headers = auth_headers(client)
    first = _upload(client, headers, POLICY_PDF).json()
    again = _upload(client, headers, POLICY_PDF)
    assert again.status_code == 409
    assert again.json()["detail"]["id"] == first["id"]


def test_path_stripped_from_filename(client: TestClient) -> None:
    resp = _upload(client, auth_headers(client), POLICY_PDF, "../../etc/evil.pdf")
    assert resp.json()["filename"] == "evil.pdf"


def test_users_only_list_their_own_documents(client: TestClient) -> None:
    uv, other = auth_headers(client), auth_headers(client, "other@example.com")
    _upload(client, uv, POLICY_PDF)
    _upload(client, other, RECIPE_PDF)
    assert [d["filename"] for d in client.get("/documents", headers=uv).json()] == ["policy.pdf"]
    assert len(client.get("/documents", headers=other).json()) == 1


# --- search (Postgres + pgvector only) ---


def _search(client: TestClient, headers: dict, query: str, **extra) -> list[dict]:
    resp = client.post("/documents/search", headers=headers, json={"query": query, **extra})
    assert resp.status_code == 200, resp.text
    return resp.json()


@requires_postgres
def test_search_returns_best_chunk_and_page(client: TestClient) -> None:
    headers = auth_headers(client)
    _upload(client, headers, POLICY_PDF)
    hits = _search(client, headers, "how many days to get a refund")
    assert hits[0]["page"] == 1
    assert "30 days" in hits[0]["text"]
    assert hits[0]["filename"] == "policy.pdf"


@requires_postgres
def test_keyword_search_finds_exact_names(client: TestClient) -> None:
    headers = auth_headers(client)
    _upload(client, headers, POLICY_PDF)
    hits = _search(client, headers, "Leeds")
    assert hits[0]["page"] == 2


@requires_postgres
def test_search_never_returns_other_users_documents(client: TestClient) -> None:
    uv, other = auth_headers(client), auth_headers(client, "other@example.com")
    _upload(client, other, RECIPE_PDF)
    assert _search(client, uv, "pancake recipe flour eggs") == []


@requires_postgres
def test_search_filters_by_document_ids(client: TestClient) -> None:
    headers = auth_headers(client)
    policy = _upload(client, headers, POLICY_PDF).json()
    _upload(client, headers, RECIPE_PDF)
    hits = _search(client, headers, "policy recipe", document_ids=[policy["id"]])
    assert hits and {h["document_id"] for h in hits} == {policy["id"]}


@requires_postgres
def test_top_k_limits_results(client: TestClient) -> None:
    headers = auth_headers(client)
    _upload(client, headers, POLICY_PDF)
    assert len(_search(client, headers, "policy", top_k=1)) == 1


def test_search_on_sqlite_says_not_supported(client: TestClient) -> None:
    if client.app.state.engine.dialect.name == "postgresql":
        return
    resp = client.post("/documents/search", headers=auth_headers(client), json={"query": "x"})
    assert resp.status_code == 501
