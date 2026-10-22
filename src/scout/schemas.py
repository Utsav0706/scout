"""API request and response bodies."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator


class Credentials(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    # Upper bound stops someone sending a 10 MB "password" to burn hashing CPU.
    password: str = Field(min_length=8, max_length=128)

    @field_validator("email")
    @classmethod
    def normalise_email(cls, v: str) -> str:
        return v.strip().lower()


class TokenOut(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int  # seconds


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    created_at: datetime


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    status: Literal["processing", "ready", "failed"]
    error: str | None = None
    pages: int  # 0 until processed
    created_at: datetime


class SearchIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=1000)
    document_ids: list[uuid.UUID] | None = Field(default=None, max_length=100)
    top_k: int | None = Field(default=None, gt=0, le=50)


class SearchHitOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page: int
    text: str
    score: float


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    db: Literal["ok", "down"]
    version: str


class ResearchIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=5, max_length=1000)
    use_web: bool = True
    use_docs: bool = True

    @field_validator("question")
    @classmethod
    def strip_question(cls, v: str) -> str:
        v = " ".join(v.split())
        if len(v) < 5:
            raise ValueError("question is too short")
        return v

    @model_validator(mode="after")
    def some_source(self) -> ResearchIn:
        if not (self.use_web or self.use_docs):
            raise ValueError("enable at least one of use_web / use_docs")
        return self


class JobCreatedOut(BaseModel):
    job_id: uuid.UUID
    status: str
    status_url: str


class JobEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    node: str
    message: str
    created_at: datetime


class JobOut(BaseModel):
    id: uuid.UUID
    question: str
    status: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    cost_gbp: float
    error: str | None
    events: list[JobEventOut]
    report: dict | None  # the full report JSON once finished
