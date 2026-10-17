"""Database tables."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from scout.db import Base


class User(Base):
    __tablename__ = "users"

    # UUIDs, not 1, 2, 3...: ids show up in URLs and tokens, and sequential ids leak
    # how many users exist and invite guessing other users' ids.
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
