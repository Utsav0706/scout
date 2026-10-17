"""Password hashing and JWT login tokens.

Argon2id over bcrypt: it's OWASP's first recommendation, is memory-hard (expensive to
crack on GPUs), and has no 72-byte password limit. JWT over server-side sessions: the
API stays stateless, so the API and worker can verify a user without a session store.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from pydantic import SecretStr

_ALGORITHM = "HS256"
_MIN_SECRET_LEN = 32
_hasher = PasswordHasher()


class InvalidTokenError(Exception):
    """Token is missing, expired, tampered with or malformed."""


def require_jwt_secret(secret: SecretStr | None) -> str:
    """Fail fast at startup rather than on the first login."""
    if secret is None or len(secret.get_secret_value()) < _MIN_SECRET_LEN:
        raise RuntimeError(
            f"JWT_SECRET must be set and at least {_MIN_SECRET_LEN} characters (see .env.example)"
        )
    return secret.get_secret_value()


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerificationError, InvalidHashError):
        return False


def create_access_token(user_id: uuid.UUID, secret: str, ttl_minutes: int) -> str:
    now = datetime.now(UTC)
    claims = {"sub": str(user_id), "iat": now, "exp": now + timedelta(minutes=ttl_minutes)}
    return jwt.encode(claims, secret, algorithm=_ALGORITHM)


def decode_access_token(token: str, secret: str) -> uuid.UUID:
    try:
        # Pinning algorithms stops the classic "alg: none" / algorithm-swap attacks.
        claims = jwt.decode(
            token, secret, algorithms=[_ALGORITHM], options={"require": ["sub", "exp"]}
        )
        return uuid.UUID(claims["sub"])
    except (jwt.PyJWTError, ValueError) as exc:
        raise InvalidTokenError(str(exc)) from exc
