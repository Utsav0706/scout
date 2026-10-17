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


class NodeModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    task_type: TaskType
    # Hard cap so a single call can never produce a surprise bill.
    max_tokens: int = Field(gt=0, le=4000)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str | None = None
    nodes: dict[NodeName, NodeModel]

    @model_validator(mode="after")
    def every_node_configured(self) -> LLMConfig:
        missing = set(get_args(NodeName)) - set(self.nodes)
        if missing:
            raise ValueError(f"config.yaml is missing models for nodes: {sorted(missing)}")
        return self


class Limits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_steps: int = Field(default=25, gt=0)
    max_cost_gbp: float = Field(default=0.20, gt=0)
    timeout_s: int = Field(default=180, gt=0)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm: LLMConfig
    limits: Limits
    usd_to_gbp: float = Field(gt=0)


class Secrets(BaseSettings):
    """Read from env vars, falling back to a local .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
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
