"""Shared FastAPI dependencies."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from scout.auth import InvalidTokenError, decode_access_token
from scout.db import get_db
from scout.models import User

_bearer = HTTPBearer(auto_error=False)

DbSession = Annotated[Session, Depends(get_db)]


def get_current_user(
    request: Request,
    db: DbSession,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> User:
    # Same error for every failure, so the response doesn't tell an attacker
    # whether the token was expired, forged, or for a deleted user.
    unauthorised = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if creds is None:
        raise unauthorised
    try:
        user_id = decode_access_token(creds.credentials, request.app.state.jwt_secret)
    except InvalidTokenError:
        raise unauthorised from None
    user = db.get(User, user_id)
    if user is None:
        raise unauthorised
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
