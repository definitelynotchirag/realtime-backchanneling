import asyncio
import json

import pytest
from livekit import rtc
from livekit.agents import stt

from blue_machines_baseline import deepgram_stt


def message(*, transcript: str, is_final: bool, speech_final: bool = False) -> dict:
    return {
        "type": "Results",
        "duration": 1.5,
        "start": 5.0,
        "is_final": is_final,
        "speech_final": speech_final,
        "channel": {"alternatives": [{"transcript": transcript, "confidence": 0.98}]},
    }


def test_an_interim_result_is_reported_as_an_interim_transcript() -> None:
    events = deepgram_stt.events_from_message(
        message(transcript="so last week", is_final=False), language="en"
    )

    assert [event.type for event in events] == [stt.SpeechEventType.INTERIM_TRANSCRIPT]
    assert events[0].alternatives[0].text == "so last week"


def test_a_final_result_carries_its_timing() -> None:
    events = deepgram_stt.events_from_message(
        message(transcript="So last week we shipped.", is_final=True), language="en"
    )

    assert events[0].type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert events[0].alternatives[0].start_time == 5.0
    assert events[0].alternatives[0].end_time == 6.5


def test_speech_final_marks_the_end_of_the_turn() -> None:
    events = deepgram_stt.events_from_message(
        message(transcript="That is the whole story.", is_final=True, speech_final=True),
        language="en",
    )

    assert [event.type for event in events] == [
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
    ]


def test_empty_and_non_result_messages_produce_nothing() -> None:
    assert (
        deepgram_stt.events_from_message(message(transcript="", is_final=True), language="en") == []
    )
    assert deepgram_stt.events_from_message({"type": "Metadata"}, language="en") == []
    assert deepgram_stt.events_from_message({"type": "Results"}, language="en") == []
    assert deepgram_stt.events_from_message({"type": "Results", "channel": {}}, language="en") == []


def test_capabilities_advertise_streaming_interims() -> None:
    client = deepgram_stt.STT(api_key="key")

    assert client.capabilities.streaming is True
    assert client.capabilities.interim_results is True
    assert client.provider == "deepgram"
    assert client.model == "nova-3"


def test_the_query_carries_the_settings_that_matter() -> None:
    client = deepgram_stt.STT(api_key="key", model="nova-2", language="en-GB", endpointing_ms=250)
    url = deepgram_stt._listen_url(client._opts)

    assert "model=nova-2" in url
    assert "language=en-GB" in url
    assert "interim_results=true" in url  # without this there is no Jev signal
    assert "endpointing=250" in url
    assert "encoding=linear16" in url
    assert "sample_rate=16000" in url


def test_an_api_key_is_required() -> None:
    with pytest.raises(ValueError):
        deepgram_stt.STT(api_key="")


def _events(stream) -> list:
    """Drain what the stream has emitted so far."""

    drained = []
    while not stream._event_ch.empty():
        drained.append(stream._event_ch.recv_nowait())
    return drained


def _usage(stream) -> list[float]:
    return [
        event.recognition_usage.audio_duration
        for event in _events(stream)
        if event.type == stt.SpeechEventType.RECOGNITION_USAGE
    ]


def _push(stream, frames: int) -> None:
    for _ in range(frames):
        stream.push_frame(
            rtc.AudioFrame(
                data=b"\x00" * 640, sample_rate=16000, num_channels=1, samples_per_channel=320
            )
        )


def test_stream_reports_the_audio_each_final_transcript_covered() -> None:
    """The pipeline has no speech-to-text usage for this provider without this.

    Deepgram's live API does not return usage, so the adapter counts the audio it
    pushed and reports it per final transcript, which the SDK turns into STTMetrics.
    """

    async def scenario() -> None:
        stream = deepgram_stt.STT(api_key="key").stream()
        _push(stream, 25)  # half a second of 20 ms frames
        stream._handle_text(
            json.dumps(message(transcript="nine words of speech here", is_final=True))
        )

        assert _usage(stream) == [pytest.approx(0.5, abs=0.01)]

        # A second transcript reports only the audio since the last one.
        _push(stream, 1)
        stream._handle_text(json.dumps(message(transcript="and another final line", is_final=True)))

        assert _usage(stream) == [pytest.approx(0.02, abs=0.005)]

    asyncio.run(scenario())


def test_interims_do_not_report_usage() -> None:
    async def scenario() -> None:
        stream = deepgram_stt.STT(api_key="key").stream()
        _push(stream, 1)
        stream._handle_text(json.dumps(message(transcript="still talking", is_final=False)))

        assert _usage(stream) == []

    asyncio.run(scenario())
