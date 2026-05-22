from pathlib import Path

import setup_wizard


def test_default_llm_provider_maps_supported_aliases() -> None:
    assert setup_wizard._default_llm_provider({"LLM_PROVIDER": "openai"}) == "openai-compatible"
    assert setup_wizard._default_llm_provider({"LLM_PROVIDER": "compatible"}) == "openai-compatible"


def test_default_llm_provider_uses_mock_model_when_provider_is_invalid() -> None:
    existing = {"LLM_PROVIDER": "not-a-provider", "LLM_MODEL": "mock"}

    assert setup_wizard._default_llm_provider(existing) == "mock"


def test_existing_llm_api_key_recovers_misplaced_provider_secret() -> None:
    existing = {"LLM_PROVIDER": "sk-test_key_value_1234567890"}

    key, recovered = setup_wizard._existing_llm_api_key(existing)

    assert key == "sk-test_key_value_1234567890"
    assert recovered is True


def test_safe_number_defaults_ignore_invalid_existing_values() -> None:
    existing = {"REMOTE_POLL_INTERVAL_SECONDS": "soon", "SIM_ROUNDS": "many"}

    assert setup_wizard._safe_float(existing, "REMOTE_POLL_INTERVAL_SECONDS", 2.0) == 2.0
    assert setup_wizard._safe_int(existing, "SIM_ROUNDS", 3) == 3


def test_save_env_preserves_unknown_existing_settings(tmp_path: Path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    monkeypatch.setattr(setup_wizard, "_ENV_PATH", env_path)

    setup_wizard._save_env(
        {
            "LLM_PROVIDER": "mock",
            "LLM_MODEL": "mock",
            "DEFAULT_MODE": "demo",
        },
        existing={
            "LLM_PROVIDER": "bad-old-value",
            "CUSTOM_SETTING": "keep-me",
            "POLYMARKET_BASE": "https://clob.polymarket.com",
        },
    )

    written = env_path.read_text()

    assert "LLM_PROVIDER=mock" in written
    assert "bad-old-value" not in written
    assert "CUSTOM_SETTING=keep-me" in written
    assert "POLYMARKET_BASE=https://clob.polymarket.com" in written
