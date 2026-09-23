"""Local API for event inspection, benchmark analysis, and browser sessions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from livekit import api as livekit_api
from pydantic import BaseModel, Field

from .benchmark import (
    REQUIRED_SCENARIO_IDS,
    SCENARIO_BY_ID,
    build_report,
    load_jsonl,
    report_provenance,
    run_replay_benchmark,
    summarize_event_log,
    write_jsonl,
    write_report,
)
from .config import BATCH_ONLY_STT_PROVIDERS, JEV_INTERIM_STT_ERROR, event_log_path_from_env

load_dotenv()

app = FastAPI(
    title="Blue Machines Baseline API",
    version="0.2.0",
    description="LiveKit lifecycle, benchmark, and browser-session API.",
)


Mode = Literal["baseline", "backchannel", "jev_backchannel"]


class LiveKitTokenRequest(BaseModel):
    """The browser may choose an experiment label, never provider credentials."""

    scenario_id: str = "short_answer"
    mode: Mode = "baseline"
    participant_identity: str | None = None


class ReplayRequest(BaseModel):
    scenario_ids: list[str] = Field(default_factory=lambda: list(REQUIRED_SCENARIO_IDS))
    repeats: int = Field(default=3, ge=1, le=20)
    seed: int = 7


def _benchmark_report_path() -> Path:
    return Path(
        os.environ.get("BENCHMARK_REPORT_PATH", "outputs/benchmark-report.json")
    ).expanduser()


def _benchmark_event_path() -> Path:
    return Path(
        os.environ.get("BENCHMARK_EVENT_PATH", "outputs/benchmark-replay-events.jsonl")
    ).expanduser()


def _add_provenance(report: dict[str, Any]) -> dict[str, Any]:
    report.setdefault("provenance", report_provenance(str(report.get("source", "unknown"))))
    return report


def _load_report() -> dict[str, Any]:
    """Return the benchmark report the UI should show.

    Precedence: a report built from provider-measured sessions in the event log
    wins whenever such sessions exist, because the headline comparison must be
    real measurements. A persisted report (the deterministic replay) is attached
    as ``replay_observation`` instead of standing in for the measurements - an
    earlier version returned the synthetic file as primary and hid the real
    observation under a key nothing rendered.
    """

    records = load_jsonl(event_log_path_from_env())
    actual_summaries = summarize_event_log(records)
    actual_report = None
    if any(
        summary.scenario_id != "unlabelled" and not summary.run_id.startswith("legacy-")
        for summary in actual_summaries
    ):
        actual_report = build_report(
            actual_summaries,
            source=f"event-log:{event_log_path_from_env()}",
        )
    report_path = _benchmark_report_path()
    persisted: dict[str, Any] | None = None
    if report_path.exists():
        try:
            value = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict):
            persisted = _add_provenance(value)
    if actual_report is not None:
        primary = _add_provenance(actual_report)
        if persisted is not None:
            primary["replay_observation"] = persisted
        return primary
    if persisted is not None:
        return persisted
    return _add_provenance(
        build_report(
            actual_summaries,
            source="unavailable: no benchmark report or labelled provider sessions found",
        )
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Health plus the providers this process believes are configured.

    The Jev pre-check reads this process's STT_PROVIDER, so a worker started with
    a different value is a split brain: the API would allow a mode the worker
    refuses. Reporting the effective settings makes that mismatch one request away
    from being obvious instead of a silent empty room.
    """

    return {
        "status": "ok",
        "service": "blue-machines-baseline",
        "stt_provider": os.environ.get("STT_PROVIDER", "livekit_inference").strip().lower(),
        "llm_provider": os.environ.get("LLM_PROVIDER", "gemini").strip().lower(),
        "tts_provider": os.environ.get("TTS_PROVIDER", "livekit_inference").strip().lower(),
    }


@app.get("/events")
def events(
    limit: int = Query(default=100, ge=1, le=5000),
    scenario_id: str | None = None,
    mode: Mode | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    records = load_jsonl(event_log_path_from_env())
    if scenario_id or mode or run_id:
        records = [
            record
            for record in records
            if (scenario_id is None or record.get("scenario_id") == scenario_id)
            and (mode is None or record.get("mode") == mode)
            and (run_id is None or record.get("run_id") == run_id)
        ]
    return {"path": str(event_log_path_from_env()), "events": records[-limit:]}


@app.get("/benchmark/report")
def benchmark_report() -> dict[str, Any]:
    return _load_report()


@app.get("/benchmark/events")
def benchmark_events(limit: int = Query(default=5000, ge=1, le=10000)) -> dict[str, Any]:
    path = _benchmark_event_path()
    return {"path": str(path), "events": load_jsonl(path)[-limit:]}


@app.post("/benchmark/replay")
def benchmark_replay(request: ReplayRequest) -> dict[str, Any]:
    unknown = [
        scenario_id for scenario_id in request.scenario_ids if scenario_id not in SCENARIO_BY_ID
    ]
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown scenario(s): {', '.join(unknown)}")
    records, _, report = run_replay_benchmark(
        scenario_ids=request.scenario_ids,
        repeats=request.repeats,
        seed=request.seed,
    )
    write_jsonl(_benchmark_event_path(), records)
    write_report(_benchmark_report_path(), report)
    return report


@app.post("/livekit/token", status_code=201)
def livekit_token(request: LiveKitTokenRequest) -> dict[str, str]:
    """Issue a short-lived join token and explicit agent dispatch metadata."""

    if request.scenario_id not in SCENARIO_BY_ID:
        raise HTTPException(status_code=422, detail="Unknown benchmark scenario")
    stt_provider = os.environ.get("STT_PROVIDER", "livekit_inference").strip().lower()
    # Refuse the mode before the room exists: the worker raises the same error
    # and never joins, which otherwise looks like "the agent is not connecting".
    if request.mode == "jev_backchannel" and stt_provider in BATCH_ONLY_STT_PROVIDERS:
        raise HTTPException(
            status_code=409,
            detail=JEV_INTERIM_STT_ERROR.format(provider=stt_provider),
        )
    api_key = os.environ.get("LIVEKIT_API_KEY", "").strip()
    api_secret = os.environ.get("LIVEKIT_API_SECRET", "").strip()
    server_url = os.environ.get("LIVEKIT_URL", "").strip()
    if not api_key or not api_secret or not server_url:
        raise HTTPException(
            status_code=503,
            detail="LiveKit credentials are not configured in the Python API environment",
        )

    run_id = uuid4().hex[:12]
    room_name = f"blue-machines-{request.scenario_id}-{run_id}"
    participant_identity = request.participant_identity or f"browser-{uuid4().hex[:10]}"
    metadata = json.dumps(
        {
            "scenario_id": request.scenario_id,
            "mode": request.mode,
            "run_id": run_id,
        },
        separators=(",", ":"),
    )
    token = (
        livekit_api.AccessToken(api_key, api_secret)
        .with_identity(participant_identity)
        .with_name("Blue Machines browser")
        .with_grants(
            livekit_api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .with_room_config(
            livekit_api.RoomConfiguration(
                agents=[
                    livekit_api.RoomAgentDispatch(
                        agent_name=os.environ.get("LIVEKIT_AGENT_NAME", "blue-machines-baseline"),
                        metadata=metadata,
                    )
                ]
            )
        )
    )
    return {
        "server_url": server_url,
        "participant_token": token.to_jwt(),
        "room_name": room_name,
        "run_id": run_id,
    }
