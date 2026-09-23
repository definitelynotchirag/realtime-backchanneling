#!/usr/bin/env python
"""Extract the committed benchmark evidence from a live event log.

The measurement pipeline is: drive real rooms -> ``outputs/baseline-events.jsonl``
(every event, gitignored because it is large) -> this script -> the committed
``outputs/benchmark-events.jsonl`` -> ``blue-machines-benchmark`` -> the committed
report. Keeping the filter here means a reviewer can rebuild the evidence from the
raw log and check that nothing was hand-picked.

Included: runs whose ids match the scripted sweep, whose ``session_started`` event
names the same provider stack as ``--stack``, and that contain both
``user_speech_ended`` and ``agent_response_started`` - i.e. runs on one known stack
where a user turn ended and the agent actually answered. Runs from another stack, or
without a stack stamp, or that never got an answer are excluded rather than reported
as measurements for the wrong pipeline.

Usage:
    uv run python scripts/extract-benchmark-runs.py --stack stt=deepgram,llm=groq,tts=deepgram_tts
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from pathlib import Path

SCENARIOS = [
    "short_answer",
    "long_monologue",
    "approaching_end_of_turn",
    "middle_pause",
    "fast_speaker",
    "noisy_audio",
    "multiple_backchannels",
    "stop_before_ack",
]
MODES = ["baseline", "backchannel", "jev_backchannel"]
RUN_PATTERN = re.compile(
    r"^(?:" + "|".join(SCENARIOS) + r")-(?:" + "|".join(MODES) + r")-\d{2}-[0-9a-f]{6}$"
)
REQUIRED_EVENTS = ("user_speech_ended", "agent_response_started")


def scripted_run_ids(events: Iterable[dict]) -> set[str]:
    """Run ids that look like the scripted sweep, not dashboard or probe rooms."""

    ids: set[str] = set()
    for event in events:
        run_id = str(event.get("run_id") or "")
        if RUN_PATTERN.match(run_id):
            ids.add(run_id)
    return ids


def stack_of(events: Iterable[dict], run_id: str) -> dict[str, object]:
    """The provider names the agent reported when it joined this run."""

    for event in events:
        if event.get("run_id") == run_id and event.get("name") == "session_started":
            data = event.get("data") or {}
            return {
                key: data.get(key)
                for key in ("stt_provider", "llm_provider", "tts_provider", "eot_detector")
            }
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=Path("outputs/baseline-events.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("outputs/benchmark-events.jsonl"))
    parser.add_argument(
        "--stack",
        default="stt=deepgram,llm=groq,tts=deepgram_tts",
        help="comma-separated provider names a run must match, e.g. stt=groq,tts=groq",
    )
    args = parser.parse_args()
    # "stt=deepgram" names the session_started field "stt_provider"; the detector is
    # the one field that is not a provider, so it keeps its own name.
    wanted: dict[str, str] = {}
    for part in args.stack.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        wanted["eot_detector" if key == "eot" else f"{key}_provider"] = value

    if not args.events.exists():
        print(f"no event log at {args.events}; run the sweep first")
        return 1

    all_events = [json.loads(line) for line in args.events.read_text().splitlines() if line.strip()]
    candidates = scripted_run_ids(all_events)

    seen: dict[str, set[str]] = {run_id: set() for run_id in candidates}
    for event in all_events:
        run_id = str(event.get("run_id") or "")
        if run_id in seen:
            seen[run_id].add(str(event.get("name")))

    complete = {
        run_id
        for run_id, names in seen.items()
        if all(required in names for required in REQUIRED_EVENTS)
    }
    stacks = {run_id: stack_of(all_events, run_id) for run_id in complete}
    matching = {
        run_id
        for run_id in complete
        if all(stacks[run_id].get(key) == value for key, value in wanted.items())
    }
    off_stack = len(complete) - len(matching)
    if not matching:
        print("no runs match; leaving any existing evidence file untouched")
        return 1
    kept = [event for event in all_events if event.get("run_id") in matching]

    args.out.write_text("".join(json.dumps(event) + "\n" for event in kept))
    for run_id, stack in sorted(stacks.items()):
        if run_id not in matching:
            print(f"  off-stack, skipped: {run_id} {stack}")

    by_scenario_mode: dict[str, int] = {}
    for run_id in sorted(matching):
        scenario_mode = run_id.rsplit("-", 2)[0]
        by_scenario_mode[scenario_mode] = by_scenario_mode.get(scenario_mode, 0) + 1

    print(
        f"{len(matching)} complete runs on {args.stack}"
        f" ({len(candidates) - len(complete)} incomplete skipped,"
        f" {off_stack} from another or unnamed stack)"
    )
    for scenario_mode, count in sorted(by_scenario_mode.items()):
        print(f"  {scenario_mode:44s} {count}")
    print(f"wrote {len(kept)} events to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
