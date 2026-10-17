from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from scout.config import AppConfig, Secrets, load_config

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def _raw() -> dict:
    return yaml.safe_load(REPO_CONFIG.read_text(encoding="utf-8"))


def test_repo_config_is_valid() -> None:
    cfg = load_config(REPO_CONFIG)
    assert cfg.limits.max_steps == 25
    assert cfg.llm.nodes["reader"].task_type == "extract"


def test_missing_node_is_rejected() -> None:
    raw = _raw()
    del raw["llm"]["nodes"]["verifier"]
    with pytest.raises(ValidationError, match="verifier"):
        AppConfig.model_validate(raw)


def test_max_tokens_cap_is_enforced() -> None:
    raw = _raw()
    raw["llm"]["nodes"]["writer"]["max_tokens"] = 50_000
    with pytest.raises(ValidationError):
        AppConfig.model_validate(raw)


def test_unknown_keys_are_rejected() -> None:
    raw = _raw()
    raw["limits"]["max_stpes"] = 10  # typo must fail loudly, not be ignored
    with pytest.raises(ValidationError):
        AppConfig.model_validate(raw)


def test_secrets_come_from_env_and_stay_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-123")
    secrets = Secrets(_env_file=None)
    assert secrets.tavily_api_key is not None
    assert secrets.tavily_api_key.get_secret_value() == "tvly-test-123"
    assert "tvly-test-123" not in repr(secrets)
