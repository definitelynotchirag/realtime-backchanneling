"""Environment-backed settings for the baseline agent."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from os import environ
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from .cue_bank import CUE_TEXTS


class ConfigurationError(ValueError):
    """Raised when the agent cannot start with the supplied environment."""


TTSProvider = Literal[
    "livekit_inference", "elevenlabs", "gemini_tts", "groq_tts", "openrouter_tts", "deepgram_tts"
]
LLMProvider = Literal["gemini", "openrouter", "groq"]
STTProvider = Literal["livekit_inference", "groq", "groq_interim", "deepgram"]
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
    deepgram_stt_model: str = "nova-3"
    deepgram_stt_language: str = "en"
    deepgram_stt_endpointing_ms: int = 300
    groq_interim_interval_seconds: float = 3.0
    backchannel_enabled: bool = False
    backchannel_texts: tuple[str, ...] = ("mm-hmm",)
    """The cues the timer policy may use, in rotation. A list, because a policy with no
    classifier cannot choose between them: repeating one sound for a whole session is
    what makes an agent sound mechanical. Every entry must be a cue in the bank, which
    is where the safety reasoning lives; the neutral group is the only one a timer
    policy has any business using (`cue_bank.NEUTRAL_CUES`)."""
    backchannel_clip_source: Literal["tts", "assets"] = "tts"
    """Where acknowledgement audio comes from. "tts" renders the cue bank once per
    session in the configured voice and falls back to the committed clips if the
    provider cannot; "assets" replays the committed clips only, which is the
    deterministic choice for a measurement sweep."""
    backchannel_delay_seconds: float = 1.4
    backchannel_cooldown_seconds: float = 4.0
    collision_window_seconds: float = 0.5
    eot_detector: EotDetector = "livekit_inference"
    typesafe_api_key: SecretStr | None = None
    jev_model: str = "jev-latest"
    # The classifier's own latency is usually 0.4-0.9 s but occasionally spikes past
    # 2.5 s; a timeout means silence, so the limit is generous. The cadence is the real
    # gate on how early in a turn Jev can approve - at 1.5 s an approval about a short
    # turn arrives as the turn ends, where the end-of-turn guard refuses to cue.
    jev_timeout_seconds: float = 4.0
    jev_min_interval_seconds: float = 0.8
    jev_approval_threshold: float = 0.55
    jev_helpful_threshold: float = 0.50
    jev_helpful_threshold_semantic: float = 0.52
    jev_expects_answer_ceiling: float = 0.60
    jev_expects_answer_ceiling_semantic: float = 0.50
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
    openrouter_tts_model: str = "deepgram/flux-tts:free"
    openrouter_tts_voice: str = "flux-alexis-en"
    deepgram_api_key: SecretStr | None = None
    deepgram_tts_model: str = "aura-2-thalia-en"
    elevenlabs_api_key: SecretStr | None = None
    elevenlabs_tts_model: str = "eleven_turbo_v2_5"
    elevenlabs_voice_id: str = "ODq5zmih8GrVes37Dizd"
    scenario_audio_dir: Path = Path("assets/scenarios")
    scenario_tts_model: str | None = None
    """Voice/model for the *user's* side in scripted runs. For Deepgram the model
    carries the voice (`aura-2-*`); unset means "same as the agent's"."""

    scenario_tts_voice: str | None = None
    """Voice for the user's side where the provider separates voice from model."""
    # Speech is billed per audio token by some providers (Groq's Orpheus allows
    # 3600 per day), so the default answer length is deliberately tiny.
    agent_instructions: str = (
        "You are a friendly voice assistant. Reply in one short sentence of at most ten words."
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
        backchannel_texts = tuple(
            item.strip()
            for item in source.get("BACKCHANNEL_TEXT", "mm-hmm").split(",")
            if item.strip()
        )
        if not backchannel_texts:
            raise ConfigurationError("BACKCHANNEL_TEXT must name at least one cue")
        unknown_cues = [cue for cue in backchannel_texts if cue not in CUE_TEXTS]
        if unknown_cues:
            raise ConfigurationError(
                f"BACKCHANNEL_TEXT has unknown cue(s) {', '.join(unknown_cues)}; "
                f"the bank is {', '.join(CUE_TEXTS)}"
            )

        backchannel_clip_source = source.get("BACKCHANNEL_CLIP_SOURCE", "tts").strip().lower()
        if backchannel_clip_source not in {"tts", "assets"}:
            raise ConfigurationError("BACKCHANNEL_CLIP_SOURCE must be 'tts' or 'assets'")

        if stt_provider in {"groq", "groq_interim"}:
            required["GROQ_API_KEY"] = source.get("GROQ_API_KEY", "").strip()
        elif stt_provider == "deepgram":
            required["DEEPGRAM_API_KEY"] = source.get("DEEPGRAM_API_KEY", "").strip()
        elif stt_provider != "livekit_inference":
            raise ConfigurationError(
                "STT_PROVIDER must be one of: livekit_inference, groq, groq_interim, deepgram"
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
        if tts_provider == "openrouter_tts" and not source.get("OPENROUTER_API_KEY", "").strip():
            required["OPENROUTER_API_KEY (when TTS_PROVIDER=openrouter_tts)"] = ""
        if tts_provider == "deepgram_tts" and not source.get("DEEPGRAM_API_KEY", "").strip():
            required["DEEPGRAM_API_KEY (when TTS_PROVIDER=deepgram_tts)"] = ""
        missing = [name for name, value in required.items() if not value]
        if missing:
            names = ", ".join(missing)
            raise ConfigurationError(f"Missing required environment variable(s): {names}")

        if tts_provider not in {
            "livekit_inference",
            "elevenlabs",
            "gemini_tts",
            "groq_tts",
            "openrouter_tts",
            "deepgram_tts",
        }:
            raise ConfigurationError(
                "TTS_PROVIDER must be one of: livekit_inference, elevenlabs, gemini_tts, "
                "groq_tts, openrouter_tts, deepgram_tts"
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

        def parse_int(name: str, default: int) -> int:
            try:
                value = int(source.get(name, str(default)))
            except ValueError as exc:
                raise ConfigurationError(f"{name} must be a whole number") from exc
            if value <= 0:
                raise ConfigurationError(f"{name} must be positive")
            return value

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
            deepgram_stt_model=source.get("DEEPGRAM_STT_MODEL", "nova-3").strip(),
            deepgram_stt_language=source.get("DEEPGRAM_STT_LANGUAGE", "en").strip(),
            deepgram_stt_endpointing_ms=parse_int("DEEPGRAM_STT_ENDPOINTING_MS", 300),
            groq_interim_interval_seconds=parse_float("GROQ_INTERIM_INTERVAL_SECONDS", 3.0),
            backchannel_enabled=parse_bool("BACKCHANNEL_ENABLED", False),
            backchannel_texts=backchannel_texts,
            backchannel_clip_source=backchannel_clip_source,
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
            jev_timeout_seconds=parse_float("JEV_TIMEOUT_SECONDS", 4.0),
            jev_min_interval_seconds=parse_float("JEV_MIN_INTERVAL_SECONDS", 0.8),
            jev_approval_threshold=parse_float("JEV_APPROVAL_THRESHOLD", 0.55),
            jev_helpful_threshold=parse_float("JEV_HELPFUL_THRESHOLD", 0.50),
            jev_helpful_threshold_semantic=parse_float("JEV_HELPFUL_THRESHOLD_SEMANTIC", 0.52),
            jev_expects_answer_ceiling=parse_float("JEV_EXPECTS_ANSWER_CEILING", 0.60),
            jev_expects_answer_ceiling_semantic=parse_float(
                "JEV_EXPECTS_ANSWER_CEILING_SEMANTIC", 0.50
            ),
            llm_provider=llm_provider,
            # Read from the environment, not from `required`: the key is only
            # required when Gemini is the *LLM*, but the Gemini TTS adapter also
            # needs it. Deriving it from `required` silently dropped it for any
            # other LLM provider, so groq LLM + gemini speech failed to start.
            gemini_api_key=(
                SecretStr(source["GEMINI_API_KEY"].strip())
                if source.get("GEMINI_API_KEY", "").strip()
                else None
            ),
            gemini_model=source.get("GEMINI_MODEL", "gemini-2.5-flash").strip(),
            groq_llm_model=source.get("GROQ_LLM_MODEL", "openai/gpt-oss-120b").strip(),
            # Same reasoning as gemini_api_key: required only when OpenRouter is
            # the LLM, but its speech endpoint uses the key too.
            openrouter_api_key=(
                SecretStr(source["OPENROUTER_API_KEY"].strip())
                if source.get("OPENROUTER_API_KEY", "").strip()
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
            openrouter_tts_model=source.get(
                "OPENROUTER_TTS_MODEL", "deepgram/flux-tts:free"
            ).strip(),
            openrouter_tts_voice=source.get("OPENROUTER_TTS_VOICE", "flux-alexis-en").strip(),
            deepgram_api_key=(
                SecretStr(source["DEEPGRAM_API_KEY"].strip())
                if source.get("DEEPGRAM_API_KEY", "").strip()
                else None
            ),
            deepgram_tts_model=source.get("DEEPGRAM_TTS_MODEL", "aura-2-thalia-en").strip(),
            # Blank means unset, not "an empty voice": the driver then uses the
            # agent's own voice for the user's side.
            scenario_tts_model=source.get("SCENARIO_TTS_MODEL", "").strip() or None,
            scenario_tts_voice=source.get("SCENARIO_TTS_VOICE", "").strip() or None,
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
                "You are a friendly voice assistant. Reply in one short sentence of at most"
                " ten words.",
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
