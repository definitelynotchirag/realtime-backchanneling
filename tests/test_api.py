import base64
import json

from fastapi.testclient import TestClient

from blue_machines_baseline.api import app


def test_health_endpoint_is_credentials_free(monkeypatch) -> None:
    monkeypatch.setenv("STT_PROVIDER", "groq_interim")
    monkeypatch.setenv("TTS_PROVIDER", "groq_tts")
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    # The Jev pre-check reads this process's own env, so the effective providers
    # must be visible: a worker configured differently is a silent split brain.
    assert response.json()["stt_provider"] == "groq_interim"
    assert response.json()["tts_provider"] == "groq_tts"
    assert response.json()["status"] == "ok"


def test_replay_endpoint_returns_report_and_writes_inspection_artifacts(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("BENCHMARK_EVENT_PATH", str(tmp_path / "replay.jsonl"))
    monkeypatch.setenv("BENCHMARK_REPORT_PATH", str(tmp_path / "report.json"))

    response = TestClient(app).post(
        "/benchmark/replay",
        json={"scenario_ids": ["short_answer"], "repeats": 1},
    )

    assert response.status_code == 200
    assert response.json()["run_count"] == 3
    assert (tmp_path / "replay.jsonl").exists()
    assert (tmp_path / "report.json").exists()


def test_report_prefers_provider_measurements_and_keeps_replay_separate(
    tmp_path, monkeypatch
) -> None:
    event_path = tmp_path / "live-events.jsonl"
    report_path = tmp_path / "report.json"
    monkeypatch.setenv("EVENT_LOG_PATH", str(event_path))
    monkeypatch.setenv("BENCHMARK_REPORT_PATH", str(report_path))
    monkeypatch.setenv("BENCHMARK_PROVIDER_REPORT_PATH", str(tmp_path / "no-provider-report.json"))
    monkeypatch.setenv("STT_PROVIDER", "groq_interim")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("TTS_PROVIDER", "groq_tts")
    records = [
        {
            "name": "session_started",
            "elapsed_ms": 0.0,
            "scenario_id": "short_answer",
            "mode": "baseline",
            "run_id": "live-run-1",
            "data": {
                "stt_provider": "groq_interim",
                "llm_provider": "groq",
                "tts_provider": "groq_tts",
            },
        },
        {
            "name": "user_speech_ended",
            "elapsed_ms": 800.0,
            "scenario_id": "short_answer",
            "mode": "baseline",
            "run_id": "live-run-1",
            "data": {},
        },
        {
            "name": "agent_response_started",
            "elapsed_ms": 1300.0,
            "scenario_id": "short_answer",
            "mode": "baseline",
            "run_id": "live-run-1",
            "data": {},
        },
    ]
    event_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    report_path.write_text(json.dumps({"source": "deterministic-policy-replay"}))

    response = TestClient(app).get("/benchmark/report")

    assert response.status_code == 200
    payload = response.json()
    # Measured sessions exist, so the measured report is the headline and the
    # synthetic replay is only an aside - never the other way around.
    assert payload["source"].startswith("event-log:")
    assert payload["provenance"]["kind"] == "provider_event_log"
    assert payload["provenance"]["provider_latency_available"] is True
    assert payload["run_count"] == 1
    assert payload["replay_observation"]["source"] == "deterministic-policy-replay"
    assert payload["replay_observation"]["provenance"]["kind"] == "synthetic_replay"


def test_the_run_stamp_records_the_cue_rotation_that_was_live() -> None:
    """A launch environment can override .env, so the run says what it used.

    This was a real failure: the worker ran with BACKCHANNEL_TEXT=mm-hmm from its
    original launch environment while .env asked for three cues, so the rotation
    looked broken in every session and nothing in the log disagreed.
    """

    from blue_machines_baseline.agent import session_stamp_fields
    from blue_machines_baseline.config import Settings

    settings = Settings.from_env(
        {
            "LIVEKIT_URL": "wss://example.livekit.cloud",
            "LIVEKIT_API_KEY": "k",
            "LIVEKIT_API_SECRET": "s",
            "GEMINI_API_KEY": "g",
            "STT_PROVIDER": "deepgram",
            "DEEPGRAM_API_KEY": "d",
            "TTS_PROVIDER": "deepgram_tts",
            "BACKCHANNEL_TEXT": "mm-hmm,mm,hmm",
        }
    )

    fields = session_stamp_fields(settings)

    assert fields["backchannel_texts"] == ["mm-hmm", "mm", "hmm"]
    assert fields["backchannel_clip_source"] == "tts"
    assert fields["stt_provider"] == "deepgram"
    assert fields["agent_instructions"] == settings.agent_instructions


def test_report_falls_back_to_the_persisted_replay_without_labelled_sessions(
    tmp_path, monkeypatch
) -> None:
    event_path = tmp_path / "events.jsonl"
    report_path = tmp_path / "report.json"
    monkeypatch.setenv("EVENT_LOG_PATH", str(event_path))
    monkeypatch.setenv("BENCHMARK_PROVIDER_REPORT_PATH", str(tmp_path / "no-provider.json"))
    monkeypatch.setenv("BENCHMARK_REPORT_PATH", str(report_path))
    event_path.write_text(
        json.dumps({"name": "session_started", "elapsed_ms": 0.0, "data": {}}) + "\n"
    )
    report_path.write_text(json.dumps({"source": "deterministic-policy-replay"}))

    payload = TestClient(app).get("/benchmark/report").json()

    assert payload["source"] == "deterministic-policy-replay"
    assert payload["provenance"]["kind"] == "synthetic_replay"
    assert "replay_observation" not in payload


def test_report_is_truthfully_blocked_without_report_or_labelled_sessions(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("EVENT_LOG_PATH", str(tmp_path / "missing-events.jsonl"))
    monkeypatch.setenv("BENCHMARK_REPORT_PATH", str(tmp_path / "missing-report.json"))
    monkeypatch.setenv("BENCHMARK_PROVIDER_REPORT_PATH", str(tmp_path / "missing-provider.json"))

    response = TestClient(app).get("/benchmark/report")

    assert response.status_code == 200
    payload = response.json()
    assert payload["provenance"]["kind"] == "blocked"
    assert payload["provenance"]["provider_latency_available"] is False
    assert payload["run_count"] == 0


def test_livekit_token_endpoint_keeps_credentials_server_side(monkeypatch) -> None:
    monkeypatch.setenv("LIVEKIT_URL", "wss://example.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "12345678901234567890123456789012")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "abcdefghijklmnopqrstuvwxyz123456")

    response = TestClient(app).post(
        "/livekit/token",
        json={"scenario_id": "middle_pause", "mode": "jev_backchannel"},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["server_url"] == "wss://example.livekit.cloud"
    assert payload["room_name"].startswith("blue-machines-middle_pause-")
    assert "abcdefghijklmnopqrstuvwxyz123456" not in payload["participant_token"]


def test_livekit_token_endpoint_accepts_all_three_experiment_modes(monkeypatch) -> None:
    monkeypatch.setenv("LIVEKIT_URL", "wss://example.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "12345678901234567890123456789012")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "abcdefghijklmnopqrstuvwxyz123456")

    for mode in ("baseline", "backchannel", "jev_backchannel"):
        response = TestClient(app).post(
            "/livekit/token",
            json={"scenario_id": "short_answer", "mode": mode},
        )
        assert response.status_code == 201


def test_jev_mode_is_refused_before_the_room_is_created(monkeypatch) -> None:
    """Jev needs interim transcripts; a batch-only STT would leave a silent room."""

    monkeypatch.setenv("STT_PROVIDER", "groq")
    monkeypatch.setenv("LIVEKIT_URL", "wss://example.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "12345678901234567890123456789012")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "abcdefghijklmnopqrstuvwxyz123456")

    response = TestClient(app).post(
        "/livekit/token", json={"scenario_id": "short_answer", "mode": "jev_backchannel"}
    )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "interim transcripts" in detail
    assert "groq" in detail


def test_timer_mode_is_allowed_with_a_batch_only_stt(monkeypatch) -> None:
    monkeypatch.setenv("STT_PROVIDER", "groq")
    monkeypatch.setenv("LIVEKIT_URL", "wss://example.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "12345678901234567890123456789012")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "abcdefghijklmnopqrstuvwxyz123456")

    response = TestClient(app).post(
        "/livekit/token", json={"scenario_id": "short_answer", "mode": "backchannel"}
    )

    assert response.status_code == 201
    assert response.json()["room_name"].startswith("blue-machines-short_answer-")


def test_jev_mode_is_allowed_with_an_interim_stt(monkeypatch) -> None:
    monkeypatch.setenv("STT_PROVIDER", "groq_interim")
    monkeypatch.setenv("LIVEKIT_URL", "wss://example.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "12345678901234567890123456789012")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "abcdefghijklmnopqrstuvwxyz123456")

    response = TestClient(app).post(
        "/livekit/token", json={"scenario_id": "short_answer", "mode": "jev_backchannel"}
    )

    assert response.status_code == 201
    assert response.json()["room_name"].startswith("blue-machines-short_answer-")


def _run_records(run_id: str, *, stack: dict[str, str]) -> list[dict]:
    return [
        {
            "name": "session_started",
            "elapsed_ms": 0.0,
            "scenario_id": "short_answer",
            "mode": "baseline",
            "run_id": run_id,
            "data": stack,
        },
        {
            "name": "user_speech_ended",
            "elapsed_ms": 800.0,
            "scenario_id": "short_answer",
            "mode": "baseline",
            "run_id": run_id,
            "data": {},
        },
        {
            "name": "agent_response_started",
            "elapsed_ms": 1300.0,
            "scenario_id": "short_answer",
            "mode": "baseline",
            "run_id": run_id,
            "data": {},
        },
    ]


def test_report_shows_the_committed_evidence_over_the_whole_live_log(tmp_path, monkeypatch) -> None:
    """The UI panel must agree with the README, which cites the committed evidence.

    Rebuilding from the live log answers a different question - everything ever
    recorded on this machine, across provider stacks - and showing that under the
    same heading made the dashboard disagree with the documented numbers.
    """

    event_path = tmp_path / "live-events.jsonl"
    provider_path = tmp_path / "benchmark-report-provider.json"
    monkeypatch.setenv("EVENT_LOG_PATH", str(event_path))
    monkeypatch.setenv("BENCHMARK_PROVIDER_REPORT_PATH", str(provider_path))
    monkeypatch.setenv("BENCHMARK_REPORT_PATH", str(tmp_path / "replay.json"))
    event_path.write_text("")
    provider_path.write_text(
        json.dumps(
            {
                "source": "event-log:outputs/benchmark-events.jsonl",
                "run_count": 80,
                "scenario_count": 8,
                "overall": {},
            }
        )
    )

    payload = TestClient(app).get("/benchmark/report").json()

    assert payload["run_count"] == 80
    assert payload["source"] == "event-log:outputs/benchmark-events.jsonl"


def test_report_ignores_runs_recorded_on_another_stack(tmp_path, monkeypatch) -> None:
    """A batch-STT run must not be averaged into a streaming stack's table."""

    event_path = tmp_path / "live-events.jsonl"
    monkeypatch.setenv("EVENT_LOG_PATH", str(event_path))
    monkeypatch.setenv("BENCHMARK_PROVIDER_REPORT_PATH", str(tmp_path / "no-committed.json"))
    monkeypatch.setenv("BENCHMARK_REPORT_PATH", str(tmp_path / "no-replay.json"))
    monkeypatch.setenv("STT_PROVIDER", "deepgram")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("TTS_PROVIDER", "deepgram_tts")
    records = _run_records(
        "on-stack",
        stack={"stt_provider": "deepgram", "llm_provider": "groq", "tts_provider": "deepgram_tts"},
    )
    records += _run_records(
        "old-stack",
        stack={"stt_provider": "groq_interim", "llm_provider": "groq", "tts_provider": "groq_tts"},
    )
    records += _run_records("unstamped", stack={})
    event_path.write_text("".join(json.dumps(record) + "\n" for record in records))

    payload = TestClient(app).get("/benchmark/report").json()

    assert payload["run_count"] == 1
    assert payload["provenance"]["kind"] == "provider_event_log"


def test_report_is_blocked_when_only_other_stacks_have_runs(tmp_path, monkeypatch) -> None:
    event_path = tmp_path / "live-events.jsonl"
    monkeypatch.setenv("EVENT_LOG_PATH", str(event_path))
    monkeypatch.setenv("BENCHMARK_PROVIDER_REPORT_PATH", str(tmp_path / "no-committed.json"))
    monkeypatch.setenv("BENCHMARK_REPORT_PATH", str(tmp_path / "no-replay.json"))
    monkeypatch.setenv("STT_PROVIDER", "deepgram")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("TTS_PROVIDER", "deepgram_tts")
    records = _run_records(
        "old-stack",
        stack={"stt_provider": "groq", "llm_provider": "gemini", "tts_provider": "groq_tts"},
    )
    event_path.write_text("".join(json.dumps(record) + "\n" for record in records))

    payload = TestClient(app).get("/benchmark/report").json()

    assert payload["run_count"] == 0
    assert payload["provenance"]["kind"] == "blocked"


def test_token_uses_the_active_fallback_project(tmp_path, monkeypatch) -> None:
    """A dead primary must mint tokens for the fallback the worker uses."""

    monkeypatch.setenv("LIVEKIT_URL", "wss://primary.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "key-primary")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret-primary")
    monkeypatch.setenv("LIVEKIT_URL_2", "wss://fallback.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY_2", "key-fallback")
    monkeypatch.setenv("LIVEKIT_API_SECRET_2", "secret-fallback")
    monkeypatch.setenv("LIVEKIT_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(
        "blue_machines_baseline.livekit_endpoints._probe",
        lambda endpoint: endpoint.url.endswith("fallback.livekit.cloud"),
    )

    response = TestClient(app).post(
        "/livekit/token", json={"scenario_id": "short_answer", "mode": "baseline"}
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["server_url"] == "wss://fallback.livekit.cloud"
    encoded = payload["participant_token"].split(".")[1]
    claims = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
    assert claims["iss"] == "key-fallback"


def test_token_endpoint_reports_when_no_project_answers(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LIVEKIT_URL", "wss://primary.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "key-primary")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret-primary")
    monkeypatch.setenv("LIVEKIT_URL_2", "wss://fallback.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY_2", "key-fallback")
    monkeypatch.setenv("LIVEKIT_API_SECRET_2", "secret-fallback")
    monkeypatch.setenv("LIVEKIT_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr("blue_machines_baseline.livekit_endpoints._probe", lambda _endpoint: False)

    response = TestClient(app).post(
        "/livekit/token", json={"scenario_id": "short_answer", "mode": "baseline"}
    )

    assert response.status_code == 503
    assert "answered" in response.json()["detail"]


def test_run_summary_endpoint_recomputes_one_live_run(tmp_path, monkeypatch) -> None:
    event_path = tmp_path / "live-events.jsonl"
    monkeypatch.setenv("EVENT_LOG_PATH", str(event_path))
    records = [
        {"name": "session_started", "elapsed_ms": 0.0, "data": {}},
        {"name": "user_speech_started", "elapsed_ms": 500.0, "data": {}},
        {"name": "user_speech_ended", "elapsed_ms": 2000.0, "data": {}},
        {"name": "agent_response_started", "elapsed_ms": 2600.0, "data": {}},
    ]
    event_path.write_text(
        "".join(
            json.dumps(
                {
                    **record,
                    "scenario_id": "long_monologue",
                    "mode": "backchannel",
                    "run_id": "live-run-1",
                }
            )
            + "\n"
            for record in records
        )
    )

    response = TestClient(app).get("/runs/live-run-1/summary")

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == "live-run-1"
    assert payload["scenario_id"] == "long_monologue"
    assert payload["mode"] == "backchannel"
    assert payload["provenance"]["kind"] == "live_run"
    assert payload["summary"]["response_latencies_ms"] == [600.0]


def test_run_summary_endpoint_reports_unknown_runs(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EVENT_LOG_PATH", str(tmp_path / "missing-events.jsonl"))

    response = TestClient(app).get("/runs/never-recorded/summary")

    assert response.status_code == 404
    assert "never-recorded" in response.json()["detail"]
