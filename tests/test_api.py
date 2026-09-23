import json

from fastapi.testclient import TestClient

from blue_machines_baseline.api import app


def test_health_endpoint_is_credentials_free() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
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
    records = [
        {
            "name": "session_started",
            "elapsed_ms": 0.0,
            "scenario_id": "short_answer",
            "mode": "baseline",
            "run_id": "live-run-1",
            "data": {},
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


def test_report_falls_back_to_the_persisted_replay_without_labelled_sessions(
    tmp_path, monkeypatch
) -> None:
    event_path = tmp_path / "events.jsonl"
    report_path = tmp_path / "report.json"
    monkeypatch.setenv("EVENT_LOG_PATH", str(event_path))
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
