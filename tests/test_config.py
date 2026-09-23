from pathlib import Path

import pytest
from pydantic import SecretStr

from blue_machines_baseline.config import ApiSettings, ConfigurationError, Settings


def valid_environment() -> dict[str, str]:
    return {
        "LIVEKIT_URL": "wss://example.livekit.cloud",
        "LIVEKIT_API_KEY": "lk_api_key",
        "LIVEKIT_API_SECRET": "lk_api_secret",
        "GROQ_API_KEY": "groq_api_key",
        "GEMINI_API_KEY": "gemini_api_key",
    }


def test_settings_loads_provider_configuration_without_exposing_secret_values() -> None:
    settings = Settings.from_env(valid_environment())

    assert settings.livekit_url == "wss://example.livekit.cloud"
    assert settings.livekit_agent_name == "blue-machines-baseline"
    assert settings.livekit_api_key == SecretStr("lk_api_key")
    assert settings.stt_provider == "livekit_inference"
    assert settings.livekit_stt_model == "google/gemini-3.5-transcribe-live"
    assert settings.llm_provider == "gemini"
    assert settings.gemini_api_key == SecretStr("gemini_api_key")
    assert settings.gemini_model == "gemini-2.5-flash"
    assert settings.openrouter_fallback_models == (
        "google/gemma-4-26b-a4b-it:free",
        "liquid/lfm-2.5-2.6b:free",
    )
    assert settings.tts_provider == "livekit_inference"
    assert settings.event_log_path == Path("outputs/baseline-events.jsonl")
    assert "lk_api_secret" not in repr(settings)


def test_groq_key_is_only_required_when_groq_stt_is_selected() -> None:
    values = valid_environment()
    values.pop("GROQ_API_KEY")
    assert Settings.from_env(values).groq_api_key is None

    values["STT_PROVIDER"] = "groq"
    with pytest.raises(ConfigurationError, match="GROQ_API_KEY"):
        Settings.from_env(values)


def test_settings_loads_optional_jev_configuration() -> None:
    values = valid_environment()
    values["TYPESAFE_API_KEY"] = "jev_api_key"

    settings = Settings.from_env(values)

    assert settings.typesafe_api_key == SecretStr("jev_api_key")
    assert settings.jev_model == "jev-latest"
    assert settings.jev_timeout_seconds == 2.5


def test_settings_reports_all_missing_required_variables() -> None:
    with pytest.raises(ConfigurationError, match="LIVEKIT_URL.*GEMINI_API_KEY"):
        Settings.from_env({})


def test_settings_rejects_non_websocket_livekit_url() -> None:
    values = valid_environment()
    values["LIVEKIT_URL"] = "https://example.livekit.cloud"

    with pytest.raises(ConfigurationError, match="ws:// or wss://"):
        Settings.from_env(values)


def test_settings_parses_openrouter_fallback_models() -> None:
    values = valid_environment()
    values["OPENROUTER_FALLBACK_MODELS"] = " model-a:free,model-b:free ,, "

    settings = Settings.from_env(values)

    assert settings.openrouter_fallback_models == ("model-a:free", "model-b:free")


def test_settings_can_name_the_livekit_agent_dispatch() -> None:
    values = valid_environment()
    values["LIVEKIT_AGENT_NAME"] = "blue-machines-staging"

    settings = Settings.from_env(values)

    assert settings.livekit_agent_name == "blue-machines-staging"


def test_settings_rejects_more_than_two_openrouter_fallback_models() -> None:
    values = valid_environment()
    values["OPENROUTER_FALLBACK_MODELS"] = "model-a:free,model-b:free,model-c:free"

    with pytest.raises(ConfigurationError, match="at most 2 models"):
        Settings.from_env(values)


def test_settings_requires_elevenlabs_key_only_for_elevenlabs_tts() -> None:
    values = valid_environment()
    values["TTS_PROVIDER"] = "elevenlabs"

    with pytest.raises(ConfigurationError, match="ELEVENLABS_API_KEY"):
        Settings.from_env(values)

    values["ELEVENLABS_API_KEY"] = "elevenlabs_api_key"
    settings = Settings.from_env(values)
    assert settings.tts_provider == "elevenlabs"
    assert settings.elevenlabs_api_key == SecretStr("elevenlabs_api_key")


def test_settings_rejects_unknown_tts_provider() -> None:
    values = valid_environment()
    values["TTS_PROVIDER"] = "unknown"

    with pytest.raises(ConfigurationError, match="TTS_PROVIDER must be one of"):
        Settings.from_env(values)


def test_settings_rejects_unknown_llm_provider() -> None:
    values = valid_environment()
    values["LLM_PROVIDER"] = "unknown"

    with pytest.raises(ConfigurationError, match="LLM_PROVIDER must be one of"):
        Settings.from_env(values)


def test_settings_can_select_openrouter_explicitly() -> None:
    values = valid_environment()
    values["LLM_PROVIDER"] = "openrouter"
    values["OPENROUTER_API_KEY"] = "openrouter_api_key"

    settings = Settings.from_env(values)

    assert settings.llm_provider == "openrouter"
    assert settings.openrouter_api_key == SecretStr("openrouter_api_key")


def test_api_settings_validates_port() -> None:
    assert ApiSettings.from_env({"API_HOST": "0.0.0.0", "API_PORT": "9000"}) == ApiSettings(
        host="0.0.0.0", port=9000
    )
    with pytest.raises(ConfigurationError, match="API_PORT must be a number"):
        ApiSettings.from_env({"API_PORT": "not-a-port"})


def test_gemini_key_is_kept_when_only_speech_uses_gemini() -> None:
    """The Gemini key is optional for the LLM but required by its TTS adapter.

    Deriving it from the LLM's requirements dropped it for any other LLM provider,
    so groq LLM + gemini speech could not start at all.
    """

    settings = Settings.from_env(
        {
            "LIVEKIT_URL": "wss://example.livekit.cloud",
            "LIVEKIT_API_KEY": "lk_api_key",
            "LIVEKIT_API_SECRET": "lk_api_secret",
            "GROQ_API_KEY": "groq_api_key",
            "GEMINI_API_KEY": "gemini_api_key",
            "LLM_PROVIDER": "groq",
            "TTS_PROVIDER": "gemini_tts",
        }
    )

    assert settings.gemini_api_key is not None
    assert settings.gemini_api_key.get_secret_value() == "gemini_api_key"


def test_llm_and_speech_can_use_different_providers() -> None:
    from blue_machines_baseline.agent import create_llm, create_tts

    settings = Settings.from_env(
        {
            "LIVEKIT_URL": "wss://example.livekit.cloud",
            "LIVEKIT_API_KEY": "lk_api_key",
            "LIVEKIT_API_SECRET": "lk_api_secret",
            "GROQ_API_KEY": "groq_api_key",
            "GEMINI_API_KEY": "gemini_api_key",
            "LLM_PROVIDER": "groq",
            "TTS_PROVIDER": "gemini_tts",
        }
    )

    assert create_llm(settings).model == "openai/gpt-oss-120b"
    assert create_tts(settings).model == "gemini-3.8-flash-tts"
