"""Configuration: non-secret settings from config.yaml, secrets from the environment.

Two sources on purpose: config.yaml is committed and reviewed like code
(models, limits), while secrets only ever come from env vars / .env.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, get_args

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Agent nodes that call an LLM. The finaliser only formats output, so it has no model.
NodeName = Literal["planner", "researcher", "reader", "writer", "verifier", "reviser"]
# Tag sent with every LLM request so Gearbox can route on it later.
TaskType = Literal["plan", "extract", "write", "verify"]


Effort = Literal["low", "medium", "high"]


class NodeModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    task_type: TaskType
    effort: Effort = "medium"
    # Hard ceiling per call (thinking tokens included), so one call can't run up a bill.
    max_tokens: int = Field(gt=0, le=16_000)
    fallback: bool = False


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str | None = None
    request_timeout_s: float = Field(default=120, gt=0, le=600)
    nodes: dict[NodeName, NodeModel]

    @model_validator(mode="after")
    def every_node_configured(self) -> LLMConfig:
        missing = set(get_args(NodeName)) - set(self.nodes)
        if missing:
            raise ValueError(f"config.yaml is missing models for nodes: {sorted(missing)}")
        return self


class ModelPrice(BaseModel):
    """USD per million tokens."""

    model_config = ConfigDict(extra="forbid")

    input: float = Field(ge=0)
    output: float = Field(ge=0)
    cache_read: float = Field(ge=0)
    cache_write: float = Field(ge=0)


class Limits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_steps: int = Field(default=25, gt=0)
    max_cost_gbp: float = Field(default=0.20, gt=0)
    timeout_s: int = Field(default=180, gt=0)


class AuthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token_ttl_minutes: int = Field(default=60, gt=0, le=24 * 60)


class RagConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedding_model: str = Field(min_length=1)
    reranker_model: str = Field(min_length=1)
    chunk_chars: int = Field(default=1500, ge=200, le=8000)
    chunk_overlap_chars: int = Field(default=200, ge=0)
    candidates: int = Field(default=20, gt=0, le=200)
    top_k: int = Field(default=5, gt=0, le=50)
    max_upload_mb: int = Field(default=20, gt=0, le=100)
    max_pages: int = Field(default=300, gt=0)

    @model_validator(mode="after")
    def overlap_smaller_than_chunk(self) -> RagConfig:
        if self.chunk_overlap_chars >= self.chunk_chars // 2:
            raise ValueError("chunk_overlap_chars must be less than half of chunk_chars")
        if self.top_k > self.candidates:
            raise ValueError("top_k cannot be larger than candidates")
        return self


class ToolsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_timeout_s: float = Field(default=30, gt=0, le=120)
    http_timeout_s: float = Field(default=15, gt=0, le=60)
    search_max_results: int = Field(default=5, gt=0, le=20)
    fetch_max_bytes: int = Field(default=2_000_000, gt=0, le=20_000_000)
    fetch_max_chars: int = Field(default=20_000, gt=0, le=200_000)
    max_redirects: int = Field(default=5, ge=0, le=10)


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_sub_questions: int = Field(default=4, ge=1, le=5)
    max_tool_turns: int = Field(default=3, ge=1, le=10)
    max_sources_to_read: int = Field(default=8, ge=1, le=30)
    max_source_chars: int = Field(default=8000, ge=500, le=50_000)
    max_notes_per_source: int = Field(default=5, ge=1, le=20)
    max_retry_questions: int = Field(default=2, ge=0, le=5)
    max_retry_sources: int = Field(default=4, ge=1, le=20)
    retry_min_budget_gbp: float = Field(default=0.10, ge=0)
    finish_reserve_gbp: float = Field(default=0.08, ge=0)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm: LLMConfig
    auth: AuthConfig
    rag: RagConfig
    tools: ToolsConfig
    agent: AgentConfig
    limits: Limits
    prices_usd_per_mtok: dict[str, ModelPrice]
    usd_to_gbp: float = Field(gt=0)

    @model_validator(mode="after")
    def every_model_has_a_price(self) -> AppConfig:
        # Without a price, cost tracking (and the max_cost_gbp guard) would silently read 0.
        unpriced = {n.model for n in self.llm.nodes.values()} - set(self.prices_usd_per_mtok)
        if unpriced:
            raise ValueError(f"prices_usd_per_mtok is missing: {sorted(unpriced)}")
        if self.agent.finish_reserve_gbp >= self.limits.max_cost_gbp:
            raise ValueError("agent.finish_reserve_gbp must be less than limits.max_cost_gbp")
        return self


class Secrets(BaseSettings):
    """Read from env vars, falling back to a local .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    database_url: str = "postgresql+psycopg://scout:scout@localhost:5432/scout"
    # Signs login tokens. Checked at startup (see auth.require_jwt_secret), not here,
    # so config-only tools and tests can run without one.
    jwt_secret: SecretStr | None = None
    log_level: str = "INFO"
    config_path: Path = Path("config.yaml")


def load_config(path: Path) -> AppConfig:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig.model_validate(raw)


@lru_cache
def get_secrets() -> Secrets:
    return Secrets()


@lru_cache
def get_config() -> AppConfig:
    return load_config(get_secrets().config_path)
