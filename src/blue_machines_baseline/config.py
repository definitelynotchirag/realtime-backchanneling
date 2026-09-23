"""Environment-backed settings for the baseline agent."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from os import environ
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class ConfigurationError(ValueError):
    """Raised when the agent cannot start with the supplied environment."""


TTSProvider = Literal["livekit_inference", "elevenlabs", "gemini_tts", "groq_tts"]
LLMProvider = Literal["gemini", "openrouter", "groq"]
STTProvider = Literal["livekit_inference", "groq", "groq_interim"]
EotDetector = Literal["livekit_inference", "final_transcript"]

BATCH_ONLY_STT_PROVIDERS = frozenset({"groq"})
"""Providers that only transcribe after the fact. ``groq_interim`` is not one of
them: it transcribes the same endpoint on a cadence, so it does emit interims."""
"""Providers that only transcribe after the fact, so they never emit interim
transcripts. Jev mode needs them, which is why it is rejected for these."""

JEV_INTERIM_STT_ERROR = (
    "Jev mode needs interim transcripts while the user speaks, and "
    "STT_PROVIDER={provider} transcribes in batches only. Use "
    "STT_PROVIDER=groq_interim (interim transcripts on the Groq endpoint) or "
    "switch to the Timer policy."
)
"""One message for both the worker and the token endpoint, so the UI and the
worker log never disagree about why Jev mode was refused."""

DEFAULT_OPENROUTER_FALLBACK_MODELS = (
    "google/gemma-4-26b-a4b-it:free",
    "liquid/lfm-2.5-2.6b:free",
)


class Settings(BaseModel):
    """Validated settings used by the LiveKit worker and event recorder."""

    model_config = ConfigDict(frozen=True)

    livekit_url: str = Field(description="LiveKit websocket URL")
    livekit_agent_name: str = "blue-machines-baseline"
    livekit_api_key: SecretStr
    livekit_api_secret: SecretStr
    stt_provider: STTProvider = "livekit_inference"
    livekit_stt_model: str = "google/gemini-3.5-transcribe-live"
    groq_api_key: SecretStr | None = None
    groq_stt_model: str = "whisper-large-v3-turbo"
    groq_interim_interval_seconds: float = 1.2
    backchannel_enabled: bool = False
    backchannel_text: str = "mm-hmm"
    backchannel_delay_seconds: float = 1.4
    backchannel_cooldown_seconds: float = 4.0
    collision_window_seconds: float = 0.5
    eot_detector: EotDetector = "livekit_inference"
    typesafe_api_key: SecretStr | None = None
    jev_model: str = "jev-latest"
    jev_timeout_seconds: float = 2.5
    jev_min_interval_seconds: float = 0.75
    jev_approval_threshold: float = 0.55
    llm_provider: LLMProvider = "gemini"
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-2.5-flash"
    groq_llm_model: str = "openai/gpt-oss-120b"
    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "stealth/union-alpha"
    openrouter_fallback_models: tuple[str, ...] = DEFAULT_OPENROUTER_FALLBACK_MODELS
    tts_provider: TTSProvider = "livekit_inference"
    livekit_tts_model: str = "inworld/inworld-tts-2"
    livekit_tts_voice: str = "Ashley"
    gemini_tts_model: str = "gemini-3.8-flash-tts"
    gemini_tts_voice: str = "Kore"
    groq_tts_model: str = "canopylabs/orpheus-v1-english"
    groq_tts_voice: str = "autumn"
    elevenlabs_api_key: SecretStr | None = None
    elevenlabs_tts_model: str = "eleven_turbo_v2_5"
    elevenlabs_voice_id: str = "ODq5zmih8GrVes37Dizd"
    scenario_audio_dir: Path = Path("assets/scenarios")
    agent_instructions: str = (
        "You are a friendly voice assistant. Keep spoken answers concise and natural."
    )
    event_log_path: Path = Path("outputs/baseline-events.jsonl")

    @classmethod
    def from_env(cls, values: Mapping[str, str] | None = None) -> Settings:
        """Build settings from environment variables without logging secret values."""

        source = environ if values is None else values
        required = {
            "LIVEKIT_URL": source.get("LIVEKIT_URL", "").strip(),
            "LIVEKIT_API_KEY": source.get("LIVEKIT_API_KEY", "").strip(),
            "LIVEKIT_API_SECRET": source.get("LIVEKIT_API_SECRET", "").strip(),
        }
        stt_provider = source.get("STT_PROVIDER", "livekit_inference").strip().lower()
        if stt_provider in {"groq", "groq_interim"}:
            required["GROQ_API_KEY"] = source.get("GROQ_API_KEY", "").strip()
        elif stt_provider != "livekit_inference":
            raise ConfigurationError(
                "STT_PROVIDER must be one of: livekit_inference, groq, groq_interim"
            )
        llm_provider = source.get("LLM_PROVIDER", "gemini").strip().lower()
        if llm_provider == "gemini":
            required["GEMINI_API_KEY"] = source.get("GEMINI_API_KEY", "").strip()
        elif llm_provider == "openrouter":
            required["OPENROUTER_API_KEY"] = source.get("OPENROUTER_API_KEY", "").strip()
        elif llm_provider == "groq":
            required["GROQ_API_KEY"] = source.get("GROQ_API_KEY", "").strip()
        else:
            raise ConfigurationError("LLM_PROVIDER must be one of: gemini, openrouter, groq")
        tts_provider = source.get("TTS_PROVIDER", "livekit_inference").strip().lower()
        if tts_provider == "elevenlabs" and not source.get("ELEVENLABS_API_KEY", "").strip():
            required["ELEVENLABS_API_KEY (when TTS_PROVIDER=elevenlabs)"] = ""
        if tts_provider == "gemini_tts" and not source.get("GEMINI_API_KEY", "").strip():
            required["GEMINI_API_KEY (when TTS_PROVIDER=gemini_tts)"] = ""
        if tts_provider == "groq_tts" and not source.get("GROQ_API_KEY", "").strip():
            required["GROQ_API_KEY (when TTS_PROVIDER=groq_tts)"] = ""
        missing = [name for name, value in required.items() if not value]
        if missing:
            names = ", ".join(missing)
            raise ConfigurationError(f"Missing required environment variable(s): {names}")

        if tts_provider not in {
            "livekit_inference",
            "elevenlabs",
            "gemini_tts",
            "groq_tts",
        }:
            raise ConfigurationError(
                "TTS_PROVIDER must be one of: livekit_inference, elevenlabs, gemini_tts, groq_tts"
            )

        livekit_url = required["LIVEKIT_URL"]
        parsed = urlparse(livekit_url)
        if parsed.scheme not in {"ws", "wss"} or not parsed.netloc:
            raise ConfigurationError("LIVEKIT_URL must be a valid ws:// or wss:// URL")

        openrouter_base_url = source.get(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ).strip()
        openrouter_parsed = urlparse(openrouter_base_url)
        if openrouter_parsed.scheme not in {"http", "https"} or not openrouter_parsed.netloc:
            raise ConfigurationError("OPENROUTER_BASE_URL must be a valid http:// or https:// URL")

        fallback_models = tuple(
            model.strip()
            for model in source.get(
                "OPENROUTER_FALLBACK_MODELS", ",".join(DEFAULT_OPENROUTER_FALLBACK_MODELS)
            ).split(",")
            if model.strip()
        )
        if len(fallback_models) > 2:
            raise ConfigurationError(
                "OPENROUTER_FALLBACK_MODELS can contain at most 2 models "
                "(3 models total including OPENROUTER_MODEL)"
            )

        eot_detector = source.get("EOT_DETECTOR", "livekit_inference").strip().lower()
        if eot_detector not in {"livekit_inference", "final_transcript"}:
            raise ConfigurationError(
                "EOT_DETECTOR must be one of: livekit_inference, final_transcript"
            )

        def parse_bool(name: str, default: bool) -> bool:
            value = source.get(name, str(default)).strip().lower()
            if value not in {"true", "false", "1", "0", "yes", "no"}:
                raise ConfigurationError(f"{name} must be true or false")
            return value in {"true", "1", "yes"}

        def parse_float(name: str, default: float) -> float:
            try:
                value = float(source.get(name, str(default)))
            except ValueError as exc:
                raise ConfigurationError(f"{name} must be a number") from exc
            if value < 0:
                raise ConfigurationError(f"{name} must be non-negative")
            return value

        event_log_path = Path(
            source.get("EVENT_LOG_PATH", "outputs/baseline-events.jsonl")
        ).expanduser()
        collision_window_seconds = parse_float("BACKCHANNEL_COLLISION_WINDOW_SECONDS", 0.5)
        if collision_window_seconds <= 0:
            raise ConfigurationError("BACKCHANNEL_COLLISION_WINDOW_SECONDS must be positive")
        return cls(
            livekit_url=livekit_url,
            livekit_agent_name=source.get("LIVEKIT_AGENT_NAME", "blue-machines-baseline").strip(),
            livekit_api_key=required["LIVEKIT_API_KEY"],
            livekit_api_secret=required["LIVEKIT_API_SECRET"],
            stt_provider=stt_provider,
            livekit_stt_model=source.get(
                "LIVEKIT_STT_MODEL", "google/gemini-3.5-transcribe-live"
            ).strip(),
            groq_api_key=(
                SecretStr(source["GROQ_API_KEY"].strip())
                if source.get("GROQ_API_KEY", "").strip()
                else None
            ),
            groq_stt_model=source.get("GROQ_STT_MODEL", "whisper-large-v3-turbo").strip(),
            groq_interim_interval_seconds=parse_float("GROQ_INTERIM_INTERVAL_SECONDS", 1.2),
            backchannel_enabled=parse_bool("BACKCHANNEL_ENABLED", False),
            backchannel_text=source.get("BACKCHANNEL_TEXT", "mm-hmm").strip(),
            backchannel_delay_seconds=parse_float("BACKCHANNEL_DELAY_SECONDS", 1.4),
            backchannel_cooldown_seconds=parse_float("BACKCHANNEL_COOLDOWN_SECONDS", 4.0),
            collision_window_seconds=collision_window_seconds,
            eot_detector=eot_detector,
            typesafe_api_key=(
                SecretStr(source["TYPESAFE_API_KEY"].strip())
                if source.get("TYPESAFE_API_KEY", "").strip()
                else None
            ),
            jev_model=source.get("JEV_MODEL", "jev-latest").strip(),
            jev_timeout_seconds=parse_float("JEV_TIMEOUT_SECONDS", 2.5),
            jev_min_interval_seconds=parse_float("JEV_MIN_INTERVAL_SECONDS", 0.75),
            jev_approval_threshold=parse_float("JEV_APPROVAL_THRESHOLD", 0.55),
            llm_provider=llm_provider,
            gemini_api_key=SecretStr(required["GEMINI_API_KEY"])
            if required.get("GEMINI_API_KEY")
            else None,
            gemini_model=source.get("GEMINI_MODEL", "gemini-2.5-flash").strip(),
            groq_llm_model=source.get("GROQ_LLM_MODEL", "openai/gpt-oss-120b").strip(),
            openrouter_api_key=(
                SecretStr(required["OPENROUTER_API_KEY"])
                if required.get("OPENROUTER_API_KEY")
                else None
            ),
            openrouter_base_url=openrouter_base_url,
            openrouter_model=source.get("OPENROUTER_MODEL", "stealth/union-alpha").strip(),
            openrouter_fallback_models=fallback_models,
            tts_provider=tts_provider,
            livekit_tts_model=source.get("LIVEKIT_TTS_MODEL", "inworld/inworld-tts-2").strip(),
            livekit_tts_voice=source.get("LIVEKIT_TTS_VOICE", "Ashley").strip(),
            gemini_tts_model=source.get("GEMINI_TTS_MODEL", "gemini-3.8-flash-tts").strip(),
            gemini_tts_voice=source.get("GEMINI_TTS_VOICE", "Kore").strip(),
            groq_tts_model=source.get("GROQ_TTS_MODEL", "canopylabs/orpheus-v1-english").strip(),
            groq_tts_voice=source.get("GROQ_TTS_VOICE", "autumn").strip(),
            elevenlabs_api_key=(
                SecretStr(source["ELEVENLABS_API_KEY"].strip())
                if source.get("ELEVENLABS_API_KEY", "").strip()
                else None
            ),
            elevenlabs_tts_model=source.get("ELEVENLABS_TTS_MODEL", "eleven_turbo_v2_5").strip(),
            elevenlabs_voice_id=source.get("ELEVENLABS_VOICE_ID", "ODq5zmih8GrVes37Dizd").strip(),
            scenario_audio_dir=Path(
                source.get("SCENARIO_AUDIO_DIR", "assets/scenarios")
            ).expanduser(),
            agent_instructions=source.get(
                "AGENT_INSTRUCTIONS",
                "You are a friendly voice assistant. Keep spoken answers concise and natural.",
            ).strip(),
            event_log_path=event_log_path,
        )


def event_log_path_from_env(values: Mapping[str, str] | None = None) -> Path:
    """Return the event path for the credentials-free observability API."""

    source = environ if values is None else values
    return Path(source.get("EVENT_LOG_PATH", "outputs/baseline-events.jsonl")).expanduser()


@dataclass(frozen=True)
class ApiSettings:
    """Non-secret settings used by the local FastAPI process."""

    host: str = "127.0.0.1"
    port: int = 8000

    @classmethod
    def from_env(cls, values: Mapping[str, str] | None = None) -> ApiSettings:
        source = environ if values is None else values
        raw_port = source.get("API_PORT", "8000")
        try:
            port = int(raw_port)
        except ValueError as exc:
            raise ConfigurationError("API_PORT must be a number") from exc
        if not 1 <= port <= 65535:
            raise ConfigurationError("API_PORT must be between 1 and 65535")
        return cls(host=source.get("API_HOST", "127.0.0.1"), port=port)
