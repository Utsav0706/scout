"""Sign up, log in, and "who am I"."""

from __future__ import annotations

import hmac
import logging

from fastapi import APIRouter, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from scout.api.deps import CurrentUser, DbSession
from scout.auth import create_access_token, hash_password, verify_password
from scout.models import User
from scout.schemas import Credentials, Registration, TokenOut, UserOut

log = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])

# Verifying against a real hash when the email doesn't exist keeps login time the same
# either way, so response timing can't be used to discover which emails have accounts.
_DUMMY_HASH = hash_password("timing-equaliser-not-a-real-password")


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(body: Registration, request: Request, db: DbSession) -> User:
    code: str | None = request.app.state.registration_code
    # compare_digest: the check takes the same time however much of the code is right.
    if code and not hmac.compare_digest((body.invite_code or "").encode(), code.encode()):
        log.info("registration refused: wrong invite code")
        raise HTTPException(status.HTTP_403_FORBIDDEN, "A valid invite code is needed to sign up")
    user = User(email=body.email, password_hash=hash_password(body.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered") from None
    db.refresh(user)
    log.info("user registered", extra={"user_id": str(user.id)})
    return user


@router.post("/login", response_model=TokenOut)
def login(body: Credentials, request: Request, db: DbSession) -> TokenOut:
    user = db.scalar(select(User).where(User.email == body.email))
    password_ok = verify_password(body.password, user.password_hash if user else _DUMMY_HASH)
    if user is None or not password_ok:
        log.info("login failed")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid email or password")

    ttl = request.app.state.config.auth.token_ttl_minutes
    token = create_access_token(user.id, request.app.state.jwt_secret, ttl)
    log.info("login ok", extra={"user_id": str(user.id)})
    return TokenOut(access_token=token, expires_in=ttl * 60)


@router.get("/me", response_model=UserOut)
def me(user: CurrentUser) -> User:
    return user
