import pickle

from livekit.agents import inference
from livekit.plugins import elevenlabs, google, groq

from blue_machines_baseline import (
    deepgram_stt,
    deepgram_tts,
    gemini_tts,
    groq_interim_stt,
    openrouter_tts,
)
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
        "DEEPGRAM_API_KEY": "deepgram_api_key",
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


def test_deepgram_stt_streams_with_interims() -> None:
    settings = settings_for_provider_tests(STT_PROVIDER="deepgram")

    stt = create_stt(settings)

    assert isinstance(stt, deepgram_stt.STT)
    assert stt.capabilities.streaming is True
    assert stt.capabilities.interim_results is True  # what Jev mode needs


def test_groq_stt_is_batch_only() -> None:
    settings = settings_for_provider_tests(STT_PROVIDER="groq", GROQ_STT_MODEL="whisper-large-v3")

    stt = create_stt(settings)

    assert isinstance(stt, groq.STT)
    assert stt.model == "whisper-large-v3"
    # Batch transcription through the OpenAI-compatible endpoint: no interim
    # transcripts, which is why Jev mode cannot run on this provider.
    assert stt.capabilities.streaming is False
    assert stt.capabilities.interim_results is False


def test_groq_interim_stt_reports_interim_capability() -> None:
    settings = settings_for_provider_tests(STT_PROVIDER="groq_interim")

    stt = create_stt(settings)

    assert isinstance(stt, groq_interim_stt.STT)
    assert stt.provider == "groq"
    # Interim transcripts are the whole point: Jev mode depends on them.
    assert stt.capabilities.streaming is True
    assert stt.capabilities.interim_results is True


def test_openrouter_tts_uses_the_free_deepgram_voice() -> None:
    settings = settings_for_provider_tests(TTS_PROVIDER="openrouter_tts")

    tts = create_tts(settings)

    assert isinstance(tts, openrouter_tts.TTS)
    assert tts.model == "deepgram/flux-tts:free"
    assert tts.provider == "openrouter"


def test_deepgram_tts_is_selected_for_streaming_speech() -> None:
    settings = settings_for_provider_tests(
        TTS_PROVIDER="deepgram_tts", DEEPGRAM_API_KEY="deepgram_api_key"
    )

    tts = create_tts(settings)

    assert isinstance(tts, deepgram_tts.TTS)
    assert tts.model == "aura-2-thalia-en"
    assert tts.provider == "deepgram"


def test_groq_tts_uses_the_bundled_plugin() -> None:
    settings = settings_for_provider_tests(TTS_PROVIDER="groq_tts")

    tts = create_tts(settings)

    assert isinstance(tts, groq.TTS)
    assert tts.model == "canopylabs/orpheus-v1-english"


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


def test_groq_llm_can_run_the_whole_stack() -> None:
    settings = settings_for_provider_tests(LLM_PROVIDER="groq")

    llm = create_llm(settings)

    assert isinstance(llm, groq.LLM)
    assert llm.model == "openai/gpt-oss-120b"
