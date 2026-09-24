#!/usr/bin/env python
"""Rewrite the README's measured-results table from a benchmark report.

The table is the one place in the README that has to agree with committed evidence, and
it changes whenever a sweep is re-run. Regenerating it here means it cannot drift from
``outputs/benchmark-report-provider.json``, and a reviewer can check the claim by running
this script and diffing.

Usage:
    uv run python scripts/update-readme-results.py [--report PATH] [--readme PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

START = "<!-- results:start -->"
END = "<!-- results:end -->"
ARMS = ("baseline", "backchannel", "jev_backchannel")
LABELS = {"baseline": "Baseline", "backchannel": "Timer backchannel", "jev_backchannel": "Jev"}


def _number(value: object, *, unit: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, int) and unit == "ms":
        return f"{value:,} ms"
    if isinstance(value, float) and unit == "ms":
        return f"{value:,.0f} ms" if abs(value) >= 10 else f"{value:,.1f} ms"
    if isinstance(value, float):
        return f"{value:.2f}"
    return f"{value}"


def _row(label: str, key: str, overall: dict, *, unit: str = "", samples: str | None = None) -> str:
    cells = []
    for arm in ARMS:
        summary = overall[arm]
        text = _number(summary.get(key), unit=unit)
        if samples and summary.get(samples) is not None:
            text += f" *(n={summary[samples]})*"
        cells.append(text)
    return f"| {label} | " + " | ".join(cells) + " |"


def build_table(report: dict) -> str:
    overall = report["overall"]
    lines = [
        f"Measured over **{report['run_count']} runs** across {report['scenario_count']} "
        "scenarios and three modes - baseline, timer backchannel, Jev backchannel - three "
        "repeats each (a cell is re-run when a run is lost), in real rooms:",
        "",
        "| Metric | "
        f"{LABELS['baseline']} | {LABELS['backchannel']} | {LABELS['jev_backchannel']} |",
        "|---|---|---|---|",
        _row("Response floor (min)", "response_min_ms", overall, unit="ms"),
        _row("Response P50", "response_p50_ms", overall, unit="ms", samples="response_samples"),
        _row("Response P95", "response_p95_ms", overall, unit="ms"),
        _row("Response σ", "response_stdev_ms", overall, unit="ms"),
        _row("Audible cues", "audible_backchannels", overall),
        _row("Cues per long turn", "backchannels_per_long_turn", overall),
        _row("Decision → audible", "backchannel_latency_p50_ms", overall, unit="ms"),
        _row("EOT delay P50", "eot_delay_p50_ms", overall, unit="ms"),
        _row("LLM TTFT P50", "llm_ttft_p50_ms", overall, unit="ms"),
        _row("Speech TTFB P50", "tts_ttfb_p50_ms", overall, unit="ms"),
        _row("Delayed responses", "delayed_responses", overall),
        _row("Cue-blocked waits", "cue_delays_attributed", overall),
        _row("End-of-turn collisions", "collision_events", overall),
        _row("EOT-suppressed cues", "eot_suppressed_cues", overall),
        _row("Cancelled cues", "cancelled_backchannels", overall),
        _row("Unpaired turns", "unpaired_turns", overall),
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=Path("outputs/benchmark-report-provider.json")
    )
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    readme = args.readme.read_text()
    if START not in readme or END not in readme:
        print(f"{args.readme} has no {START} / {END} block")
        return 1
    head, _, rest = readme.partition(START)
    _, _, tail = rest.partition(END)
    args.readme.write_text(f"{head}{START}\n{build_table(report)}\n{END}{tail}")
    print(f"README table updated from {args.report} ({report['run_count']} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
