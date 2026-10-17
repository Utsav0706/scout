import json
import logging

import pytest

from scout.log import setup_logging


def test_logs_are_json_with_extra_fields(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging("INFO")
    logging.getLogger("scout.test").info("tool called", extra={"tool": "web_search", "ms": 42})

    line = capsys.readouterr().out.strip().splitlines()[-1]
    entry = json.loads(line)
    assert entry["msg"] == "tool called"
    assert entry["level"] == "INFO"
    assert entry["tool"] == "web_search"
    assert entry["ms"] == 42


def test_level_filters_debug(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging("WARNING")
    logging.getLogger("scout.test").info("should not appear")
    assert capsys.readouterr().out == ""
