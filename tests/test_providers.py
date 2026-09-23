import pickle

from livekit.agents import inference
from livekit.plugins import elevenlabs, google, groq

from blue_machines_baseline import gemini_tts, groq_stt
from blue_machines_baseline.agent import create_llm, create_stt, create_tts, entrypoint
from blue_machines_baseline.config import Settings


def settings_for_provider_tests(**overrides: str) -> Settings:
    values = {
        "LIVEKIT_URL": "wss://example.livekit.cloud",
        "LIVEKIT_API_KEY": "lk_api_key",
        "LIVEKIT_API_SECRET": "lk_api_secret",
        "GROQ_API_KEY": "groq_api_key",
        "GEMINI_API_KEY": "gemini_api_key",
        "OPENROUTER_API_KEY": "openrouter_api_key",
        **overrides,
    }
    return Settings.from_env(values)


def test_provider_factories_use_the_selected_models() -> None:
    settings = settings_for_provider_tests(LLM_PROVIDER="openrouter")

    stt = create_stt(settings)
    llm = create_llm(settings)
    tts = create_tts(settings)

    assert isinstance(stt, inference.STT)
    assert stt.model == "google/gemini-3.5-transcribe-live"
    assert stt.capabilities.streaming is True
    assert stt.capabilities.interim_results is True
    assert llm.model == "stealth/union-alpha"
    assert llm._opts.extra_body == {
        "models": [
            "stealth/union-alpha",
            "google/gemma-4-26b-a4b-it:free",
            "liquid/lfm-2.5-2.6b:free",
        ]
    }
    assert isinstance(tts, inference.TTS)
    assert tts.model == "inworld/inworld-tts-2"


def test_provider_factory_keeps_groq_as_a_non_streaming_fallback() -> None:
    settings = settings_for_provider_tests(STT_PROVIDER="groq")

    stt = create_stt(settings)

    assert isinstance(stt, groq.STT)
    assert stt.capabilities.streaming is False
    assert stt.capabilities.interim_results is False


def test_provider_factory_uses_gemini_flash_by_default() -> None:
    settings = settings_for_provider_tests()

    llm = create_llm(settings)

    assert isinstance(llm, google.LLM)
    assert llm.model == "gemini-2.5-flash"


def test_provider_factory_can_select_elevenlabs_tts() -> None:
    settings = settings_for_provider_tests(
        TTS_PROVIDER="elevenlabs",
        ELEVENLABS_API_KEY="elevenlabs_api_key",
    )

    tts = create_tts(settings)

    assert isinstance(tts, elevenlabs.TTS)
    assert tts.model == "eleven_turbo_v2_5"


def test_room_entrypoint_is_pickle_safe_for_livekit_job_processes() -> None:
    assert pickle.loads(pickle.dumps(entrypoint)) is entrypoint


def test_groq_rest_is_selected_as_a_batch_stt_provider() -> None:
    settings = settings_for_provider_tests(
        STT_PROVIDER="groq_rest", GROQ_STT_MODEL="whisper-large-v3"
    )

    stt = create_stt(settings)

    assert isinstance(stt, groq_stt.STT)
    assert stt.model == "whisper-large-v3"
    assert stt.provider == "groq"
    assert stt.capabilities.streaming is False
    assert stt.capabilities.interim_results is False


def test_gemini_tts_is_selected_as_a_direct_tts_provider() -> None:
    settings = settings_for_provider_tests(
        TTS_PROVIDER="gemini_tts",
        GEMINI_TTS_MODEL="gemini-2.5-pro-preview-tts",
        GEMINI_TTS_VOICE="Puck",
    )

    tts = create_tts(settings)

    assert isinstance(tts, gemini_tts.TTS)
    assert tts.model == "gemini-2.5-pro-preview-tts"
    assert tts.voice == "Puck"
    assert tts.provider == "gemini"
    assert tts.sample_rate == 24000
    assert tts.capabilities.streaming is False
