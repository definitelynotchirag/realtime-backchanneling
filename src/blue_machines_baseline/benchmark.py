"""Scenario replay, event-log analysis, and comparison metrics.

The analyzer is deliberately independent of LiveKit. A real room session and the
offline replay use the same event contract, so the dashboard can inspect either
recorded provider timings or a deterministic policy smoke test.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

Mode = Literal["baseline", "backchannel", "jev_backchannel"]
MODES: tuple[Mode, ...] = ("baseline", "backchannel", "jev_backchannel")

logger = logging.getLogger("blue-machines-benchmark")

MAX_RESPONSE_WINDOW_MS = 30_000.0
"""How long after a user turn ends a response may still be attributed to it.

Beyond this the gap is idle time, not response latency: sessions are real
conversations and a turn that never got an answer must not borrow the next
turn's answer.
"""

LONG_TURN_SECONDS = 3.0
"""A user turn at least this long is long enough to deserve an acknowledgement."""

MONOTONIC_TOLERANCE_MS = 1.0
"""Slack allowed before a backwards elapsed_ms counts as a new session origin."""


def report_provenance(source: str) -> dict[str, Any]:
    """Return a truthful label for the measurements represented by a report."""

    if source == "deterministic-policy-replay":
        return {
            "kind": "synthetic_replay",
            "label": "Deterministic policy replay",
            "provider_latency_available": False,
            "note": "Policy smoke timings only; no STT, LLM, or TTS provider was called.",
        }
    if source.startswith("event-log:"):
        return {
            "kind": "provider_event_log",
            "label": "Live provider event log",
            "provider_latency_available": True,
            "note": "Measured from labelled worker sessions recorded in the event log.",
        }
    if source.startswith("unavailable:"):
        return {
            "kind": "blocked",
            "label": "Benchmark unavailable",
            "provider_latency_available": False,
            "note": source.removeprefix("unavailable:").strip(),
        }
    return {
        "kind": "unknown",
        "label": source,
        "provider_latency_available": False,
        "note": "The report source is not identified as a provider measurement.",
    }


@dataclass(frozen=True)
class BenchmarkScenario:
    """A repeatable speaking pattern used for both agent configurations."""

    scenario_id: str
    description: str
    speech_duration_seconds: float
    pause_durations_seconds: tuple[float, ...] = ()


SCENARIOS: tuple[BenchmarkScenario, ...] = (
    BenchmarkScenario("short_answer", "Short answer that should not need an acknowledgement.", 0.8),
    BenchmarkScenario("long_monologue", "Long continuous explanation.", 12.0),
    BenchmarkScenario(
        "approaching_end_of_turn",
        "Speech that pauses near the expected end of a turn.",
        3.5,
        (0.8,),
    ),
    BenchmarkScenario("middle_pause", "A pause in the middle of one sentence.", 8.0, (0.5,)),
    BenchmarkScenario("fast_speaker", "Fast, uninterrupted speech.", 7.0),
    BenchmarkScenario("noisy_audio", "Speech with background noise.", 8.0),
    BenchmarkScenario(
        "multiple_backchannels",
        "Long speech where several acknowledgements could be considered.",
        20.0,
    ),
    BenchmarkScenario(
        "stop_before_ack", "The user stops just before an acknowledgement would play.", 1.2
    ),
)

REQUIRED_SCENARIO_IDS = tuple(scenario.scenario_id for scenario in SCENARIOS)
SCENARIO_BY_ID = {scenario.scenario_id: scenario for scenario in SCENARIOS}


@dataclass(frozen=True)
class RunContext:
    """Stable labels shared by every event in one experiment run."""

    scenario_id: str
    mode: Mode
    run_id: str
    greet: bool = True


def parse_run_context(metadata: str | None) -> RunContext | None:
    """Decode the small JSON payload attached to an explicit agent dispatch."""

    if not metadata:
        return None
    try:
        value = json.loads(metadata)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, Mapping):
        return None
    scenario_id = value.get("scenario_id")
    mode = value.get("mode")
    run_id = value.get("run_id")
    if not all(isinstance(item, str) and item.strip() for item in (scenario_id, mode, run_id)):
        return None
    if mode not in MODES:
        return None
    greet = value.get("greet")
    return RunContext(
        scenario_id=scenario_id,
        mode=mode,
        run_id=run_id,
        greet=True if greet is None else bool(greet),
    )


@dataclass(frozen=True)
class RunSummary:
    """Metrics calculated from one recorded or replayed run."""

    scenario_id: str
    mode: Mode
    run_id: str
    response_latencies_ms: tuple[float, ...]
    response_p50_ms: float | None
    response_p95_ms: float | None
    backchannel_count: int
    cancelled_backchannels: int
    end_of_turn_risks: int
    overlapping_backchannels: int
    response_delta_p50_ms: float | None = None
    response_delta_p95_ms: float | None = None
    response_samples: int = 0
    response_stdev_ms: float | None = None
    response_min_ms: float | None = None
    response_max_ms: float | None = None
    unpaired_turns: int = 0
    audible_backchannels: int = 0
    collision_events: int = 0
    delayed_responses: int = 0
    long_user_turns: int = 0
    backchannels_per_long_turn: float | None = None
    backchannel_latencies_ms: tuple[float, ...] = ()
    backchannel_latency_p50_ms: float | None = None
    backchannel_latency_p95_ms: float | None = None
    backchannel_latency_samples: int = 0
    backchannel_latency_stdev_ms: float | None = None
    llm_ttft_ms: tuple[float, ...] = ()
    llm_ttft_p50_ms: float | None = None
    llm_ttft_p95_ms: float | None = None
    tts_ttfb_ms: tuple[float, ...] = ()
    tts_ttfb_p50_ms: float | None = None
    tts_ttfb_p95_ms: float | None = None
    stt_duration_ms: tuple[float, ...] = ()
    stt_duration_p50_ms: float | None = None
    eot_delays_ms: tuple[float, ...] = ()
    eot_delay_p50_ms: float | None = None
    interim_transcripts: int = 0
    final_transcripts: int = 0
    jev_decision_latencies_ms: tuple[float, ...] = ()
    jev_decision_latency_p50_ms: float | None = None
    jev_decision_latency_p95_ms: float | None = None
    jev_timeouts: int = 0
    jev_errors: int = 0

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in (
            "response_latencies_ms",
            "backchannel_latencies_ms",
            "llm_ttft_ms",
            "tts_ttfb_ms",
            "stt_duration_ms",
            "eot_delays_ms",
            "jev_decision_latencies_ms",
        ):
            result[key] = list(result[key])
        return result


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 3)


def _stdev(values: Sequence[float]) -> float | None:
    """Population standard deviation, or None when it is not defined."""

    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return round(variance**0.5, 3)


def _event_value(event: object, key: str, default: Any = None) -> Any:
    if isinstance(event, Mapping):
        return event.get(key, default)
    return getattr(event, key, default)


def _event_data(event: object) -> Mapping[str, Any]:
    data = _event_value(event, "data", {})
    return data if isinstance(data, Mapping) else {}


def _event_name(event: object) -> str:
    return str(_event_value(event, "name", ""))


def _event_time(event: object) -> float:
    try:
        return float(_event_value(event, "elapsed_ms", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _event_wall_time(event: object) -> float:
    """Wall-clock time in milliseconds, or 0 when the record has none."""

    raw = _event_value(event, "timestamp", None)
    if not isinstance(raw, str):
        return 0.0
    try:
        return datetime.fromisoformat(raw).timestamp() * 1000
    except ValueError:
        return 0.0


def _order_events(events: Sequence[object]) -> list[object]:
    """Order one session's events, preferring the monotonic offset.

    ``elapsed_ms`` is relative to each recorder's own start, so two recorders
    writing one session (a reconnect under the same run id, for example) produce
    incomparable offsets. Falling back to wall-clock order keeps the pairing
    meaningful instead of silently sorting unrelated timelines together.
    """

    by_elapsed = sorted(events, key=_event_time)
    if not _starts_new_monotonic_origin([record for record in by_elapsed]):
        return by_elapsed
    logger.debug("session has mixed monotonic origins; ordering by wall clock")
    return sorted(events, key=_event_wall_time)


def _metric_values(events: Sequence[object], metric_type: str, field: str) -> list[float]:
    values: list[float] = []
    for event in events:
        if _event_name(event) != "pipeline_metric":
            continue
        data = _event_data(event)
        metric_name = str(data.get("type", data.get("metric_type", ""))).lower()
        if metric_type not in metric_name:
            continue
        value = data.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(round(float(value) * 1000, 3))
    return values


def _backchannel_latencies(events: Sequence[object]) -> list[float]:
    decisions = [
        _event_time(event) for event in events if _event_name(event) == "backchannel_decision"
    ]
    audio_starts = [
        _event_time(event) for event in events if _event_name(event) == "backchannel_audio_started"
    ]
    # Older recordings only had backchannel_started. Keep those runs analyzable,
    # while exposing the newer audible-start marker whenever it exists.
    if not audio_starts:
        audio_starts = [
            _event_time(event) for event in events if _event_name(event) == "backchannel_started"
        ]
    latencies: list[float] = []
    audio_index = 0
    for decision in decisions:
        while audio_index < len(audio_starts) and audio_starts[audio_index] < decision:
            audio_index += 1
        if audio_index < len(audio_starts):
            latencies.append(round(audio_starts[audio_index] - decision, 3))
            audio_index += 1
    return latencies


@dataclass(frozen=True)
class UserTurn:
    """One span of user speech, used to attribute responses and cues."""

    start: float
    end: float

    @property
    def duration_ms(self) -> float:
        return self.end - self.start


def _user_turns(events: Sequence[object]) -> list[UserTurn]:
    """Collect user speech spans from the event stream in order."""

    turns: list[UserTurn] = []
    started_at: float | None = None
    for event in events:
        name = _event_name(event)
        if name == "user_speech_started":
            started_at = _event_time(event)
        elif name == "user_speech_ended":
            turns.append(
                UserTurn(
                    start=started_at if started_at is not None else _event_time(event),
                    end=_event_time(event),
                )
            )
            started_at = None
    return turns


def _has_collision_event(events: Sequence[object], cue_start: float) -> bool:
    """Whether the engine already reported a collision for this cue."""

    for event in events:
        if _event_name(event) != "backchannel_collision":
            continue
        reported_at = _event_time(event)
        if 0 <= reported_at - cue_start <= MAX_RESPONSE_WINDOW_MS:
            return True
    return False


def _pair_responses_with_turns(
    events: Sequence[object], turns: Sequence[UserTurn]
) -> tuple[list[float], int, int]:
    """Pair each answered user turn with the response it caused.

    The previous implementation paired every ``user_speech_ended`` with the next
    ``agent_response_started`` anywhere in the run. In a long conversation that
    cursor drifts: turns that never got a response swallowed idle time, and the
    measured "response latency" grew without bound (a real recording produced a
    107 second baseline P50). A response only counts for a turn when it starts
    after that turn ended, before the next turn begins, and within
    ``MAX_RESPONSE_WINDOW_MS``.

    Returns ``(latencies_ms, unpaired_turns, delayed_responses)``.
    """

    response_starts = [
        _event_time(event) for event in events if _event_name(event) == "agent_response_started"
    ]
    cue_starts = [
        _event_time(event) for event in events if _event_name(event) == "backchannel_audio_started"
    ]
    cue_ends = sorted(
        [
            _event_time(event)
            for event in events
            if _event_name(event) in {"backchannel_completed", "backchannel_cancelled"}
        ]
    )

    latencies: list[float] = []
    unpaired = 0
    delayed = 0
    used: set[int] = set()
    for index, turn in enumerate(turns):
        next_start = turns[index + 1].start if index + 1 < len(turns) else None
        answer: float | None = None
        for position, start in enumerate(response_starts):
            if position in used or start <= turn.end:
                continue
            if next_start is not None and start >= next_start:
                break
            if start - turn.end > MAX_RESPONSE_WINDOW_MS:
                break
            answer = start
            used.add(position)
            break
        if answer is None:
            unpaired += 1
            continue
        latencies.append(round(answer - turn.end, 3))
        # A cue that was still audible when the turn ended is a cue the user had
        # to talk over, and it is the clearest way the acknowledgement can delay
        # the real answer.
        if _cue_audible_at(turn.end, cue_starts, cue_ends):
            delayed += 1
    return latencies, unpaired, delayed


def _cue_audible_at(moment: float, cue_starts: Sequence[float], cue_ends: Sequence[float]) -> bool:
    """Whether an acknowledgement was audible at ``moment``."""

    for start in cue_starts:
        if start > moment:
            continue
        ended = next((end for end in cue_ends if end >= start), None)
        if ended is None or ended > moment:
            return True
    return False


def summarize_run(
    events: Sequence[object], *, scenario_id: str, mode: Mode, run_id: str
) -> RunSummary:
    """Calculate timing and behaviour metrics from recorder-compatible events."""

    ordered = _order_events(events)
    turns = _user_turns(ordered)
    (
        response_latencies,
        unpaired_turns,
        delayed_responses,
    ) = _pair_responses_with_turns(ordered, turns)

    backchannel_starts = [
        _event_time(event) for event in ordered if _event_name(event) == "backchannel_started"
    ]
    audible_cues = [
        _event_time(event) for event in ordered if _event_name(event) == "backchannel_audio_started"
    ]
    cancelled = sum(_event_name(event) == "backchannel_cancelled" for event in ordered)
    collisions = sum(_event_name(event) == "backchannel_collision" for event in ordered)
    end_of_turn_risks = collisions
    overlapping = 0
    for start in backchannel_starts:
        preceding_starts = [
            _event_time(event)
            for event in ordered
            if _event_name(event) == "user_speech_started" and _event_time(event) <= start
        ]
        preceding_ends = [
            _event_time(event)
            for event in ordered
            if _event_name(event) == "user_speech_ended" and _event_time(event) <= start
        ]
        user_started = max(preceding_starts, default=None)
        user_ended = max(preceding_ends, default=None)
        if user_started is not None and (user_ended is None or user_started > user_ended):
            overlapping += 1
            next_end = next(
                (
                    _event_time(event)
                    for event in ordered
                    if _event_name(event) == "user_speech_ended" and _event_time(event) > start
                ),
                None,
            )
            # Fallback for recordings made before the engine emitted its own
            # collision event: a cue that the user talked over into their turn
            # end. Recordings that carry backchannel_collision are not counted
            # twice because that event already accounts for the same cue.
            if (
                next_end is not None
                and next_end - start <= 500
                and not _has_collision_event(ordered, start)
            ):
                end_of_turn_risks += 1

    long_turns = [turn for turn in turns if turn.duration_ms >= LONG_TURN_SECONDS * 1000]
    cues_in_long_turns = sum(
        1 for cue in audible_cues for turn in long_turns if turn.start <= cue <= turn.end
    )

    backchannel_latencies = _backchannel_latencies(ordered)
    llm_ttft = _metric_values(ordered, "llm", "ttft")
    tts_ttfb = _metric_values(ordered, "tts", "ttfb")
    stt_duration = _metric_values(ordered, "stt", "duration")
    eot_delays = _metric_values(ordered, "eou", "end_of_utterance_delay")
    jev_decision_latencies = [
        float(value)
        for event in ordered
        if _event_name(event) == "jev_decision"
        and isinstance((value := _event_data(event).get("latency_ms")), (int, float))
        and not isinstance(value, bool)
    ]
    return RunSummary(
        scenario_id=scenario_id,
        mode=mode,
        run_id=run_id,
        response_latencies_ms=tuple(response_latencies),
        response_p50_ms=_percentile(response_latencies, 0.50),
        response_p95_ms=_percentile(response_latencies, 0.95),
        response_samples=len(response_latencies),
        response_stdev_ms=_stdev(response_latencies),
        response_min_ms=min(response_latencies) if response_latencies else None,
        response_max_ms=max(response_latencies) if response_latencies else None,
        unpaired_turns=unpaired_turns,
        backchannel_count=len(backchannel_starts),
        audible_backchannels=len(audible_cues),
        collision_events=collisions,
        delayed_responses=delayed_responses,
        long_user_turns=len(long_turns),
        backchannels_per_long_turn=(
            round(cues_in_long_turns / len(long_turns), 3) if long_turns else None
        ),
        cancelled_backchannels=cancelled,
        end_of_turn_risks=end_of_turn_risks,
        overlapping_backchannels=overlapping,
        backchannel_latencies_ms=tuple(backchannel_latencies),
        backchannel_latency_p50_ms=_percentile(backchannel_latencies, 0.50),
        backchannel_latency_p95_ms=_percentile(backchannel_latencies, 0.95),
        backchannel_latency_samples=len(backchannel_latencies),
        backchannel_latency_stdev_ms=_stdev(backchannel_latencies),
        llm_ttft_ms=tuple(llm_ttft),
        llm_ttft_p50_ms=_percentile(llm_ttft, 0.50),
        llm_ttft_p95_ms=_percentile(llm_ttft, 0.95),
        tts_ttfb_ms=tuple(tts_ttfb),
        tts_ttfb_p50_ms=_percentile(tts_ttfb, 0.50),
        tts_ttfb_p95_ms=_percentile(tts_ttfb, 0.95),
        stt_duration_ms=tuple(stt_duration),
        stt_duration_p50_ms=_percentile(stt_duration, 0.50),
        eot_delays_ms=tuple(eot_delays),
        eot_delay_p50_ms=_percentile(eot_delays, 0.50),
        interim_transcripts=sum(
            _event_name(event) == "stt_transcript"
            and not bool(_event_data(event).get("is_final", False))
            for event in ordered
        ),
        final_transcripts=sum(
            _event_name(event) == "stt_transcript"
            and bool(_event_data(event).get("is_final", False))
            for event in ordered
        ),
        jev_decision_latencies_ms=tuple(jev_decision_latencies),
        jev_decision_latency_p50_ms=_percentile(jev_decision_latencies, 0.50),
        jev_decision_latency_p95_ms=_percentile(jev_decision_latencies, 0.95),
        jev_timeouts=sum(_event_name(event) == "jev_decision_timeout" for event in ordered),
        jev_errors=sum(_event_name(event) == "jev_decision_error" for event in ordered),
    )


def _aggregate_selected(selected: Sequence[RunSummary], *, scenario_id: str) -> RunSummary:
    mode: Mode = selected[0].mode if selected else "baseline"
    latencies = [latency for run in selected for latency in run.response_latencies_ms]
    backchannel_latencies = [
        latency for run in selected for latency in run.backchannel_latencies_ms
    ]
    llm_ttft = [value for run in selected for value in run.llm_ttft_ms]
    tts_ttfb = [value for run in selected for value in run.tts_ttfb_ms]
    stt_duration = [value for run in selected for value in run.stt_duration_ms]
    eot_delays = [value for run in selected for value in run.eot_delays_ms]
    jev_decision_latencies = [value for run in selected for value in run.jev_decision_latencies_ms]
    return RunSummary(
        scenario_id=scenario_id,
        mode=mode,
        run_id="aggregate",
        response_latencies_ms=tuple(latencies),
        response_p50_ms=_percentile(latencies, 0.50),
        response_p95_ms=_percentile(latencies, 0.95),
        response_samples=len(latencies),
        response_stdev_ms=_stdev(latencies),
        response_min_ms=min(latencies) if latencies else None,
        response_max_ms=max(latencies) if latencies else None,
        unpaired_turns=sum(run.unpaired_turns for run in selected),
        backchannel_count=sum(run.backchannel_count for run in selected),
        audible_backchannels=sum(run.audible_backchannels for run in selected),
        collision_events=sum(run.collision_events for run in selected),
        delayed_responses=sum(run.delayed_responses for run in selected),
        long_user_turns=sum(run.long_user_turns for run in selected),
        backchannels_per_long_turn=_pooled_per_long_turn(selected),
        cancelled_backchannels=sum(run.cancelled_backchannels for run in selected),
        end_of_turn_risks=sum(run.end_of_turn_risks for run in selected),
        overlapping_backchannels=sum(run.overlapping_backchannels for run in selected),
        backchannel_latencies_ms=tuple(backchannel_latencies),
        backchannel_latency_p50_ms=_percentile(backchannel_latencies, 0.50),
        backchannel_latency_p95_ms=_percentile(backchannel_latencies, 0.95),
        backchannel_latency_samples=len(backchannel_latencies),
        backchannel_latency_stdev_ms=_stdev(backchannel_latencies),
        llm_ttft_ms=tuple(llm_ttft),
        llm_ttft_p50_ms=_percentile(llm_ttft, 0.50),
        llm_ttft_p95_ms=_percentile(llm_ttft, 0.95),
        tts_ttfb_ms=tuple(tts_ttfb),
        tts_ttfb_p50_ms=_percentile(tts_ttfb, 0.50),
        tts_ttfb_p95_ms=_percentile(tts_ttfb, 0.95),
        stt_duration_ms=tuple(stt_duration),
        stt_duration_p50_ms=_percentile(stt_duration, 0.50),
        eot_delays_ms=tuple(eot_delays),
        eot_delay_p50_ms=_percentile(eot_delays, 0.50),
        interim_transcripts=sum(run.interim_transcripts for run in selected),
        final_transcripts=sum(run.final_transcripts for run in selected),
        jev_decision_latencies_ms=tuple(jev_decision_latencies),
        jev_decision_latency_p50_ms=_percentile(jev_decision_latencies, 0.50),
        jev_decision_latency_p95_ms=_percentile(jev_decision_latencies, 0.95),
        jev_timeouts=sum(run.jev_timeouts for run in selected),
        jev_errors=sum(run.jev_errors for run in selected),
    )


def _pooled_per_long_turn(selected: Sequence[RunSummary]) -> float | None:
    """Re-pool the cue ratio instead of averaging per-run ratios.

    Averaging ratios would weight a run with one long turn the same as a run
    with twenty, so the pooled counts are divided instead.
    """

    long_turns = sum(run.long_user_turns for run in selected)
    if not long_turns:
        return None
    cues = 0
    for run in selected:
        if run.backchannels_per_long_turn is not None:
            cues += round(run.backchannels_per_long_turn * run.long_user_turns)
    return round(cues / long_turns, 3)


def aggregate_runs(runs: Sequence[RunSummary]) -> dict[Mode, RunSummary]:
    """Aggregate repeated runs by mode and add the experiment's deltas."""

    aggregates: dict[Mode, RunSummary] = {
        mode: _aggregate_selected([run for run in runs if run.mode == mode], scenario_id="all")
        for mode in MODES
    }
    baseline_p50 = aggregates["baseline"].response_p50_ms
    baseline_p95 = aggregates["baseline"].response_p95_ms
    for mode in MODES[1:]:
        experiment = aggregates[mode]
        updates: dict[str, float] = {}
        if baseline_p50 is not None and experiment.response_p50_ms is not None:
            updates["response_delta_p50_ms"] = round(experiment.response_p50_ms - baseline_p50, 3)
        if baseline_p95 is not None and experiment.response_p95_ms is not None:
            updates["response_delta_p95_ms"] = round(experiment.response_p95_ms - baseline_p95, 3)
        if updates:
            aggregates[mode] = RunSummary(**{**experiment.__dict__, **updates})
    return aggregates


def build_report(runs: Sequence[RunSummary], *, source: str) -> dict[str, Any]:
    """Build the JSON shape consumed by the results and timeline UI."""

    scenario_ids = list(REQUIRED_SCENARIO_IDS)
    scenario_ids.extend(
        sorted({run.scenario_id for run in runs if run.scenario_id not in scenario_ids})
    )
    scenario_rows: list[dict[str, Any]] = []
    for scenario_id in scenario_ids:
        selected = [run for run in runs if run.scenario_id == scenario_id]
        comparison = aggregate_runs(selected)
        scenario = SCENARIO_BY_ID.get(scenario_id)
        scenario_rows.append(
            {
                "scenario_id": scenario_id,
                "description": (
                    scenario.description if scenario else "Events without a scenario label."
                ),
                "runs": [run.as_dict() for run in selected],
                "comparison": {
                    mode: summary.as_dict() if summary.response_latencies_ms or selected else None
                    for mode, summary in comparison.items()
                },
            }
        )
    overall = aggregate_runs(runs)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": source,
        "provenance": report_provenance(source),
        "run_count": len(runs),
        "scenario_count": len([row for row in scenario_rows if row["runs"]]),
        "scenarios": scenario_rows,
        "overall": {mode: summary.as_dict() for mode, summary in overall.items()},
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read valid JSON object records, ignoring a partially written final line."""

    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def _record_context(record: Mapping[str, Any]) -> tuple[str | None, Mode | None, str | None]:
    data = record.get("data") if isinstance(record.get("data"), Mapping) else {}
    scenario_id = record.get("scenario_id") or data.get("scenario_id")
    raw_mode = record.get("mode") or data.get("mode")
    if raw_mode not in MODES:
        raw_mode = "backchannel" if data.get("backchannel_enabled") else "baseline"
    run_id = record.get("run_id") or data.get("run_id")
    room_name = data.get("room_name")
    return (
        str(scenario_id) if scenario_id else None,
        raw_mode,
        str(run_id or room_name) if (run_id or room_name) else None,
    )


def _starts_new_monotonic_origin(group: Sequence[object]) -> bool:
    """Whether the newest record belongs to a different session timeline.

    ``elapsed_ms`` is relative to each recorder's own start, so a second session
    in the same file restarts near zero. Within one session the value never goes
    backwards, which makes a regression the reliable signal that two timelines
    have been mixed into one group.
    """

    if len(group) < 2:
        return False
    latest = _event_time(group[-1])
    previous_max = max(_event_time(record) for record in group[:-1])
    return latest + MONOTONIC_TOLERANCE_MS < previous_max


def summarize_event_log(records: Sequence[Mapping[str, Any]]) -> list[RunSummary]:
    """Split a JSONL stream into sessions and summarize each session."""

    groups: list[tuple[str, list[Mapping[str, Any]]]] = []
    current: list[Mapping[str, Any]] = []
    current_key = ""
    for record in records:
        _, _, explicit_run_id = _record_context(record)
        if explicit_run_id:
            if current and current_key != explicit_run_id:
                groups.append((current_key or f"legacy-{len(groups) + 1}", current))
                current = []
            current_key = explicit_run_id
            current.append(record)
        else:
            if record.get("name") == "session_started" and current:
                groups.append((current_key or f"legacy-{len(groups) + 1}", current))
                current = []
                current_key = ""
            current.append(record)
            if record.get("name") == "session_stopped":
                groups.append((current_key or f"legacy-{len(groups) + 1}", current))
                current = []
                current_key = ""
            elif explicit_run_id is None and _starts_new_monotonic_origin(current):
                # A still-running recorder can keep writing while a second
                # session starts in the same file. Its elapsed_ms origin is
                # different, so continuing the group would sort unrelated
                # timelines together and silently corrupt every latency. Only
                # unlabelled groups are split: a labelled run keeps its run_id,
                # and mixed origins inside it are ordered by wall clock instead.
                stale = current[:-1]
                if stale:
                    groups.append((current_key or f"legacy-{len(groups) + 1}", stale))
                current = [record]
                current_key = ""
    if current:
        groups.append((current_key or f"legacy-{len(groups) + 1}", current))

    # A worker can still be flushing the end of one session while the next run
    # starts, so one run's records arrive in more than one fragment. The
    # fragments share a recorder timeline, so they belong to the same run:
    # merging them keeps one summary per run_id (and stops the UI from showing
    # the same run twice) without mixing any other run's timeline into it.
    merged: dict[str, list[Mapping[str, Any]]] = {}
    order: list[str] = []
    for key, group in groups:
        if key not in merged:
            merged[key] = []
            order.append(key)
        merged[key].extend(group)
    groups = [(key, merged[key]) for key in order]

    summaries: list[RunSummary] = []
    for index, (group_key, group) in enumerate(groups, start=1):
        scenario_id, mode, run_id = next(
            (
                context
                for context in (_record_context(record) for record in group)
                if context[0] is not None or context[2] is not None
            ),
            (None, None, None),
        )
        resolved_mode: Mode = mode or "baseline"
        summaries.append(
            summarize_run(
                group,
                scenario_id=scenario_id or "unlabelled",
                mode=resolved_mode,
                run_id=run_id or group_key or f"legacy-{index}",
            )
        )
    return summaries


def replay_scenario_events(
    scenario: BenchmarkScenario,
    *,
    mode: Mode,
    run_id: str,
    repeat_index: int,
    seed: int = 7,
) -> list[dict[str, Any]]:
    """Generate a deterministic policy-level replay for smoke testing the experiment.

    This is not a substitute for provider measurements. It makes the runner and
    dashboard usable before credentials are available and is labelled as a replay
    in the resulting report.
    """

    # Two streams on purpose: the response model must draw the same numbers for
    # every mode (common random numbers), so a paired baseline-vs-experiment
    # comparison is not polluted by independent noise, while the cue timings are
    # mode-specific by definition.
    response_rng = random.Random(f"{seed}:{scenario.scenario_id}:{repeat_index}")
    rng = random.Random(f"{seed}:{scenario.scenario_id}:{mode}:{repeat_index}")
    context = {
        "scenario_id": scenario.scenario_id,
        "mode": mode,
        "run_id": run_id,
    }
    start = 250.0
    end = start + scenario.speech_duration_seconds * 1000
    events: list[dict[str, Any]] = [
        {"name": "session_started", "elapsed_ms": 0.0, "data": {}, **context},
        {"name": "user_speech_started", "elapsed_ms": start, "data": {}, **context},
        {
            "name": "stt_transcript",
            "elapsed_ms": start + 180,
            "data": {"is_final": False, "character_count": 34, "word_count": 7},
            **context,
        },
    ]
    if scenario.scenario_id == "approaching_end_of_turn":
        events.append(
            {
                "name": "eot_prediction",
                "elapsed_ms": end - 500,
                "data": {"probability": 0.86, "threshold": 0.8},
                **context,
            }
        )

    opportunities = 0
    if mode in {"backchannel", "jev_backchannel"}:
        if scenario.scenario_id in {
            "long_monologue",
            "middle_pause",
            "fast_speaker",
            "noisy_audio",
        }:
            opportunities = 1
        elif scenario.scenario_id == "multiple_backchannels":
            opportunities = 3
        elif scenario.scenario_id == "stop_before_ack":
            opportunities = 1
        if mode == "jev_backchannel":
            # The semantic policy is intentionally a little more selective in
            # the synthetic replay: it suppresses one ambiguous cue while
            # retaining the long-form acknowledgement opportunities.
            if scenario.scenario_id == "approaching_end_of_turn":
                opportunities = 0
            elif scenario.scenario_id == "multiple_backchannels":
                opportunities = 2
        for opportunity in range(opportunities):
            decision = start + 1400 + opportunity * 5200
            if mode == "jev_backchannel":
                events.append(
                    {
                        "name": "jev_decision",
                        "elapsed_ms": decision - 120,
                        "data": {
                            "allowed": True,
                            "confidence": round(0.84 - opportunity * 0.04, 3),
                        },
                        **context,
                    }
                )
            if decision >= end - 350:
                # The user's turn ends before this cue could become audible, so
                # the policy drops it and nothing is played. That is the whole
                # point of the stop_before_ack scenario: no cue, and therefore
                # no cancellation event either - the engine only reports a
                # cancellation for a cue that actually reached the audio output.
                continue
            audio = decision + 38 + rng.uniform(4, 26)
            events.extend(
                [
                    {"name": "backchannel_decision", "elapsed_ms": decision, "data": {}, **context},
                    {"name": "backchannel_started", "elapsed_ms": audio, "data": {}, **context},
                    {
                        "name": "backchannel_audio_started",
                        "elapsed_ms": audio + 24,
                        "data": {},
                        **context,
                    },
                ]
            )
    events.extend(
        [
            {
                "name": "user_speech_ended",
                "elapsed_ms": end,
                "data": {"next_state": "listening"},
                **context,
            },
            {
                "name": "stt_transcript",
                "elapsed_ms": end + 60,
                "data": {"is_final": True, "character_count": 92, "word_count": 18},
                **context,
            },
            {
                "name": "pipeline_metric",
                "elapsed_ms": end + 180,
                "data": {
                    "type": "llm_metrics",
                    "ttft": 0.43 + response_rng.random() * 0.12,
                },
                **context,
            },
        ]
    )
    # The response model is identical for every mode on purpose. An earlier
    # version added a mode-specific offset (baseline answered "10 ms slower"),
    # which meant the synthetic report was constructed to show backchanneling as
    # faster. A replay must not encode the conclusion it is used to check.
    response_start = end + 500 + response_rng.uniform(-35, 45)
    events.extend(
        [
            {
                "name": "agent_response_started",
                "elapsed_ms": response_start,
                "data": {"previous_state": "thinking"},
                **context,
            },
            {
                "name": "pipeline_metric",
                "elapsed_ms": response_start + 420,
                "data": {"type": "tts_metrics", "ttfb": 0.31 + response_rng.random() * 0.1},
                **context,
            },
            {
                "name": "agent_response_ended",
                "elapsed_ms": response_start + 1300,
                "data": {"next_state": "listening"},
                **context,
            },
            {"name": "session_stopped", "elapsed_ms": response_start + 1450, "data": {}, **context},
        ]
    )
    return events


def run_replay_benchmark(
    *,
    scenario_ids: Sequence[str] = REQUIRED_SCENARIO_IDS,
    repeats: int = 3,
    seed: int = 7,
) -> tuple[list[dict[str, Any]], list[RunSummary], dict[str, Any]]:
    """Run all experiment modes for every selected scenario and return a report."""

    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    events: list[dict[str, Any]] = []
    summaries: list[RunSummary] = []
    for scenario_id in scenario_ids:
        if scenario_id not in SCENARIO_BY_ID:
            raise ValueError(f"unknown scenario: {scenario_id}")
        scenario = SCENARIO_BY_ID[scenario_id]
        for repeat_index in range(1, repeats + 1):
            for mode in MODES:
                run_id = f"replay-{scenario_id}-{mode}-{repeat_index:02d}"
                run_events = replay_scenario_events(
                    scenario,
                    mode=mode,
                    run_id=run_id,
                    repeat_index=repeat_index,
                    seed=seed,
                )
                events.extend(run_events)
                summaries.append(
                    summarize_run(
                        run_events,
                        scenario_id=scenario_id,
                        mode=mode,
                        run_id=run_id,
                    )
                )
    return events, summaries, build_report(summaries, source="deterministic-policy-replay")


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(dict(record), sort_keys=True) + "\n")


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze or replay voice benchmark runs.")
    parser.add_argument("--events", type=Path, default=Path("outputs/baseline-events.jsonl"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/benchmark-replay-events.jsonl")
    )
    parser.add_argument("--report", type=Path, default=Path("outputs/benchmark-report.json"))
    parser.add_argument(
        "--replay", action="store_true", help="Run the deterministic policy replay."
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--scenario", action="append", dest="scenarios")
    args = parser.parse_args()

    if args.replay:
        scenario_ids = args.scenarios or REQUIRED_SCENARIO_IDS
        records, _, report = run_replay_benchmark(
            scenario_ids=scenario_ids,
            repeats=args.repeats,
        )
        write_jsonl(args.output, records)
        write_report(args.report, report)
    else:
        records = load_jsonl(args.events)
        report = build_report(
            summarize_event_log(records),
            source=f"event-log:{args.events}",
        )
        write_report(args.report, report)


if __name__ == "__main__":
    main()
