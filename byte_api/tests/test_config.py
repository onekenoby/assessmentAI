from __future__ import annotations

import pytest

import core.config as config


def test_defaults_are_valid():
    value = config.ByteApiSettings()
    assert value.service_name
    assert value.api_prefix.startswith("/")
    assert value.max_file_bytes > 0
    assert value.max_concurrent_uploads > 0


@pytest.mark.parametrize("raw", ["1", "true", "YES", "on"])
def test_env_bool_true(monkeypatch, raw):
    monkeypatch.setenv("FLAG", raw)
    assert config._env_bool("FLAG", False) is True


@pytest.mark.parametrize("raw", ["0", "false", "NO", "off"])
def test_env_bool_false(monkeypatch, raw):
    monkeypatch.setenv("FLAG", raw)
    assert config._env_bool("FLAG", True) is False


def test_env_bool_invalid(monkeypatch):
    monkeypatch.setenv("FLAG", "sometimes")
    with pytest.raises(ValueError):
        config._env_bool("FLAG", False)


def test_env_int(monkeypatch):
    monkeypatch.setenv("N", "42")
    assert config._env_int("N", 1) == 42


def test_env_int_invalid(monkeypatch):
    monkeypatch.setenv("N", "x")
    with pytest.raises(ValueError):
        config._env_int("N", 1)


def test_env_str_uses_default_for_blank(monkeypatch):
    monkeypatch.setenv("S", "   ")
    assert config._env_str("S", "default") == "default"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"api_prefix": "api"}, "iniziare"),
        ({"max_file_bytes": 0}, "> 0"),
        ({"max_concurrent_uploads": 0}, "> 0"),
        ({"api_key_header": ""}, "vuoto"),
    ],
)
def test_settings_reject_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        config.ByteApiSettings(**kwargs)
