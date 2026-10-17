import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import select

from scout.auth import (
    InvalidTokenError,
    create_access_token,
    decode_access_token,
    hash_password,
    require_jwt_secret,
    verify_password,
)
from scout.models import User
from tests.conftest import TEST_JWT_SECRET

CREDS = {"email": "uv@example.com", "password": "correct-horse-battery"}


def _register(client: TestClient, **overrides: str) -> dict:
    resp = client.post("/auth/register", json={**CREDS, **overrides})
    assert resp.status_code == 201, resp.text
    return resp.json()


def _token(client: TestClient) -> str:
    resp = client.post("/auth/login", json=CREDS)
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


# --- unit: hashing and tokens ---


def test_password_hash_round_trip() -> None:
    h = hash_password("s3cret-password")
    assert h != "s3cret-password"
    assert verify_password("s3cret-password", h)
    assert not verify_password("wrong-password", h)


def test_verify_password_rejects_garbage_hash() -> None:
    assert not verify_password("anything", "not-a-real-hash")


def test_token_round_trip() -> None:
    uid = uuid.uuid4()
    assert decode_access_token(create_access_token(uid, TEST_JWT_SECRET, 5), TEST_JWT_SECRET) == uid


def test_expired_token_rejected() -> None:
    past = datetime.now(UTC) - timedelta(minutes=10)
    token = jwt.encode(
        {"sub": str(uuid.uuid4()), "iat": past, "exp": past + timedelta(minutes=1)},
        TEST_JWT_SECRET,
        algorithm="HS256",
    )
    with pytest.raises(InvalidTokenError):
        decode_access_token(token, TEST_JWT_SECRET)


def test_token_signed_with_other_secret_rejected() -> None:
    token = create_access_token(uuid.uuid4(), "another-secret-" + "y" * 40, 5)
    with pytest.raises(InvalidTokenError):
        decode_access_token(token, TEST_JWT_SECRET)


def test_unsigned_alg_none_token_rejected() -> None:
    token = jwt.encode({"sub": str(uuid.uuid4()), "exp": 9999999999}, None, algorithm="none")
    with pytest.raises(InvalidTokenError):
        decode_access_token(token, TEST_JWT_SECRET)


@pytest.mark.parametrize("secret", [None, SecretStr("too-short")])
def test_weak_or_missing_jwt_secret_fails_fast(secret: SecretStr | None) -> None:
    with pytest.raises(RuntimeError, match="JWT_SECRET"):
        require_jwt_secret(secret)


# --- API ---


def test_register_returns_user_without_password(client: TestClient) -> None:
    body = _register(client)
    assert body["email"] == CREDS["email"]
    assert "password" not in body and "password_hash" not in body


def test_password_is_stored_hashed(client: TestClient) -> None:
    _register(client)
    with client.app.state.session_factory() as db:
        user = db.scalar(select(User))
    assert user is not None
    assert user.password_hash != CREDS["password"]
    assert user.password_hash.startswith("$argon2id$")


def test_email_is_normalised(client: TestClient) -> None:
    body = _register(client, email="  UV@Example.COM ")
    assert body["email"] == "uv@example.com"


def test_duplicate_email_conflicts(client: TestClient) -> None:
    _register(client)
    resp = client.post("/auth/register", json={**CREDS, "email": "UV@example.com"})
    assert resp.status_code == 409


@pytest.mark.parametrize(
    "payload",
    [
        {"email": "not-an-email", "password": "long-enough-pw"},
        {"email": "a@example.com", "password": "short"},
        {"email": "a@example.com", "password": "x" * 129},
        {"email": "a@example.com", "password": "long-enough-pw", "is_admin": True},
    ],
)
def test_register_validates_input(client: TestClient, payload: dict) -> None:
    assert client.post("/auth/register", json=payload).status_code == 422


def test_login_and_me(client: TestClient) -> None:
    _register(client)
    resp = client.post("/auth/login", json=CREDS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] == 60 * 60

    me = client.get("/auth/me", headers={"Authorization": f"Bearer {body['access_token']}"})
    assert me.status_code == 200
    assert me.json()["email"] == CREDS["email"]


def test_wrong_password_and_unknown_email_look_the_same(client: TestClient) -> None:
    _register(client)
    wrong_pw = client.post("/auth/login", json={**CREDS, "password": "wrong-password"})
    no_user = client.post("/auth/login", json={**CREDS, "email": "nobody@example.com"})
    assert wrong_pw.status_code == no_user.status_code == 401
    assert wrong_pw.json() == no_user.json()


@pytest.mark.parametrize("header", [None, "Bearer not-a-jwt", "Basic dXY6cHc="])
def test_me_requires_valid_token(client: TestClient, header: str | None) -> None:
    headers = {"Authorization": header} if header else {}
    assert client.get("/auth/me", headers=headers).status_code == 401


def test_token_for_deleted_user_rejected(client: TestClient) -> None:
    _register(client)
    token = _token(client)
    with client.app.state.session_factory() as db:
        db.delete(db.scalar(select(User)))
        db.commit()
    resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401
