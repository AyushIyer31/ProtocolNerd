"""Offline provider-startup checks. All credentials below are synthetic test values."""
import importlib.util
import logging
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

BACKEND = Path(__file__).resolve().parents[1] / "protocolnerd-backend"
sys.path.insert(0, str(BACKEND))


@pytest.fixture
def providers(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("Network is not permitted in provider-startup tests")
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    # Do not read a developer's dotenv files when importing the real provider code.
    monkeypatch.setitem(sys.modules, "dotenv", SimpleNamespace(load_dotenv=lambda *a, **k: None))
    spec = importlib.util.spec_from_file_location("provider_startup_under_test", BACKEND / "llm_providers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("key", [
    "TEST_SECRET_SHORT",
    "TEST_SECRET_EXACT_THIRTY_CHARS_!",
    "TEST_SECRET_" + "A" * 80,
    "TEST_SECRET_" + "B" * 200,
    "TEST_SECRET_UNICODE_éçñ",
    "TEST_SECRET_NEWLINE\nSECOND_LINE",
    "TEST_SECRET_QUOTES_'_\"_END",
    "TEST_SECRET_PERCENT_%s_%r_END",
    "TEST_SECRET_TAB\tEND",
    "TEST_SECRET_SPACES WITHIN VALUE",
    "TEST_SECRET_BACKSLASH_\\n_END",
    "TEST_SECRET_DOLLAR_$HOME_END",
], ids=["short", "thirty", "long", "very-long", "unicode", "newline", "quotes", "percent", "tab", "spaces", "backslash", "dollar"])
def test_claude_initialization_does_not_log_credential(providers, monkeypatch, caplog, key):
    client = object()
    factory = Mock(return_value=client)
    monkeypatch.setenv("ANTHROPIC_API_KEY", key)
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=factory))
    caplog.set_level(logging.DEBUG)
    assert providers.ClaudeProvider()._get_client() is client
    factory.assert_called_once_with()
    assert key not in caplog.text
    assert key[:30] not in caplog.text
    assert "TEST_SECRET_" not in caplog.text


def test_missing_key_is_unavailable_without_client_creation(providers, monkeypatch, caplog):
    factory = Mock()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=factory))
    caplog.set_level(logging.DEBUG)
    assert providers.ClaudeProvider()._get_client() is None
    factory.assert_not_called()
    assert "not set" in caplog.text


def test_initialized_client_is_reused(providers, monkeypatch, caplog):
    client = object()
    factory = Mock(return_value=client)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "TEST_SECRET_REUSE")
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=factory))
    caplog.set_level(logging.DEBUG)
    provider = providers.ClaudeProvider()
    assert provider._get_client() is client
    assert provider._get_client() is client
    factory.assert_called_once_with()
    assert "TEST_SECRET_" not in caplog.text


def test_client_constructor_failure_stays_unavailable(providers, monkeypatch, caplog):
    factory = Mock(side_effect=RuntimeError("synthetic constructor failure"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "TEST_SECRET_FAILURE")
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=factory))
    caplog.set_level(logging.DEBUG)
    assert providers.ClaudeProvider()._get_client() is None
    assert "TEST_SECRET_" not in caplog.text
    assert "Failed to create Anthropic client" in caplog.text
