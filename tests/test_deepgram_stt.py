import pytest
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
