from __future__ import annotations

import os

import pytest

from core.config import (
    IngestionApiSettings,
    _env_bool,
    _env_int,
    _env_str,
    configure_process_environment,
)


@pytest.mark.parametrize("raw", ["1", "true", "YES", "on"])
def test_env_bool_true_values(monkeypatch, raw: str):
    monkeypatch.setenv("BOOL_TEST", raw)
    assert _env_bool("BOOL_TEST", False) is True


@pytest.mark.parametrize("raw", ["0", "false", "NO", "off"])
def test_env_bool_false_values(monkeypatch, raw: str):
    monkeypatch.setenv("BOOL_TEST", raw)
    assert _env_bool("BOOL_TEST", True) is False


def test_env_bool_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("BOOL_TEST", "maybe")
    with pytest.raises(ValueError, match="deve essere booleano"):
        _env_bool("BOOL_TEST", False)


def test_env_int_and_string(monkeypatch):
    monkeypatch.setenv("INT_TEST", " 42 ")
    monkeypatch.setenv("STR_TEST", " value ")
    assert _env_int("INT_TEST", 0) == 42
    assert _env_str("STR_TEST", "fallback") == "value"


def test_env_int_rejects_non_integer(monkeypatch):
    monkeypatch.setenv("INT_TEST", "4.2")
    with pytest.raises(ValueError, match="deve essere un intero"):
        _env_int("INT_TEST", 0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"api_prefix": "api/v1"}, "deve iniziare con /"),
        ({"default_max_jobs": 0}, "deve essere > 0"),
        ({"max_jobs_per_run": 0}, "deve essere > 0"),
        ({"default_max_jobs": 5, "max_jobs_per_run": 4}, "supera il massimo"),
        ({"run_history_limit": 0}, "deve essere > 0"),
    ],
)
def test_settings_validation(kwargs, message: str):
    with pytest.raises(ValueError, match=message):
        IngestionApiSettings(**kwargs)


def test_configure_process_environment_preserves_explicit_values(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "9")
    monkeypatch.setenv("EMBED_CPU_THREADS", "3")
    configure_process_environment()
    assert os.environ["OMP_NUM_THREADS"] == "9"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
