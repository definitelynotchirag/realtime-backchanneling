from blue_machines_baseline.benchmark import (
    REQUIRED_SCENARIO_IDS,
    aggregate_runs,
    build_report,
    run_replay_benchmark,
    summarize_event_log,
    summarize_run,
)


def test_response_latency_never_borrows_a_later_turn_s_response() -> None:
    # Regression for the pairing bug that produced a 107 second baseline P50 on
    # the real event log: a turn with no answer used to consume the next
    # response in the run, however far away it was.
    events = [
        {"name": "user_speech_started", "elapsed_ms": 1000, "data": {}},
        {"name": "user_speech_ended", "elapsed_ms": 2000, "data": {}},
        {"name": "user_speech_started", "elapsed_ms": 3000, "data": {}},
        {"name": "user_speech_ended", "elapsed_ms": 4000, "data": {}},
        {"name": "agent_response_started", "elapsed_ms": 95000, "data": {}},
    ]

    result = summarize_run(events, scenario_id="long_monologue", mode="baseline", run_id="r1")

    assert result.response_latencies_ms == ()
    assert result.response_p50_ms is None
    assert result.unpaired_turns == 2


def test_a_response_outside_the_pairing_window_is_not_attributed() -> None:
    events = [
        {"name": "user_speech_started", "elapsed_ms": 0, "data": {}},
        {"name": "user_speech_ended", "elapsed_ms": 1000, "data": {}},
        {"name": "agent_response_started", "elapsed_ms": 31001, "data": {}},
    ]

    result = summarize_run(events, scenario_id="short_answer", mode="baseline", run_id="r1")

    assert result.response_latencies_ms == ()
    assert result.unpaired_turns == 1


def test_turns_are_paired_in_order_within_the_window() -> None:
    events = [
        {"name": "user_speech_started", "elapsed_ms": 0, "data": {}},
        {"name": "user_speech_ended", "elapsed_ms": 1000, "data": {}},
        {"name": "agent_response_started", "elapsed_ms": 1500, "data": {}},
        {"name": "user_speech_started", "elapsed_ms": 4000, "data": {}},
        {"name": "user_speech_ended", "elapsed_ms": 9000, "data": {}},
        {"name": "agent_response_started", "elapsed_ms": 9400, "data": {}},
    ]

    result = summarize_run(events, scenario_id="long_monologue", mode="backchannel", run_id="r1")

    assert result.response_latencies_ms == (500.0, 400.0)
    assert result.unpaired_turns == 0
    assert result.long_user_turns == 1


def test_audible_cues_are_counted_separately_from_handle_creation() -> None:
    events = [
        {"name": "user_speech_started", "elapsed_ms": 0, "data": {}},
        {"name": "backchannel_started", "elapsed_ms": 1500, "data": {}},
        {"name": "backchannel_audio_started", "elapsed_ms": 1560, "data": {}},
        {"name": "backchannel_completed", "elapsed_ms": 2060, "data": {"duration_ms": 500.0}},
        {"name": "backchannel_started", "elapsed_ms": 2600, "data": {}},
        {"name": "backchannel_cancelled", "elapsed_ms": 2610, "data": {"reason": "user_stopped"}},
        {"name": "user_speech_ended", "elapsed_ms": 5000, "data": {}},
    ]

    result = summarize_run(
        events, scenario_id="multiple_backchannels", mode="backchannel", run_id="r1"
    )

    assert result.backchannel_count == 2
    assert result.audible_backchannels == 1
    assert result.cancelled_backchannels == 1


def test_cues_per_long_turn_ignores_short_turns() -> None:
    events = [
        # short turn with a cue: must not count towards the long-turn ratio
        {"name": "user_speech_started", "elapsed_ms": 0, "data": {}},
        {"name": "backchannel_audio_started", "elapsed_ms": 900, "data": {}},
        {"name": "user_speech_ended", "elapsed_ms": 1000, "data": {}},
        # long turn with two cues
        {"name": "user_speech_started", "elapsed_ms": 5000, "data": {}},
        {"name": "backchannel_audio_started", "elapsed_ms": 7000, "data": {}},
        {"name": "backchannel_audio_started", "elapsed_ms": 11000, "data": {}},
        {"name": "user_speech_ended", "elapsed_ms": 14000, "data": {}},
    ]

    result = summarize_run(
        events, scenario_id="multiple_backchannels", mode="backchannel", run_id="r1"
    )

    assert result.long_user_turns == 1
    assert result.backchannels_per_long_turn == 2.0
    assert result.audible_backchannels == 3


def test_delayed_responses_counts_a_cue_the_user_had_to_talk_over() -> None:
    events = [
        {"name": "user_speech_started", "elapsed_ms": 0, "data": {}},
        {"name": "backchannel_audio_started", "elapsed_ms": 1500, "data": {}},
        {"name": "user_speech_ended", "elapsed_ms": 1800, "data": {}},
        {"name": "agent_response_started", "elapsed_ms": 2600, "data": {}},
        # a clean second turn: the cue finished before the user stopped
        {"name": "user_speech_started", "elapsed_ms": 6000, "data": {}},
        {"name": "backchannel_audio_started", "elapsed_ms": 7000, "data": {}},
        {"name": "backchannel_completed", "elapsed_ms": 7400, "data": {"duration_ms": 400.0}},
        {"name": "user_speech_ended", "elapsed_ms": 9000, "data": {}},
        {"name": "agent_response_started", "elapsed_ms": 9400, "data": {}},
    ]

    result = summarize_run(
        events, scenario_id="multiple_backchannels", mode="backchannel", run_id="r1"
    )

    assert result.delayed_responses == 1
    assert result.response_latencies_ms == (800.0, 400.0)


def test_cue_outliving_the_turn_end_attributes_its_overhang_to_the_response() -> None:
    events = [
        {"name": "user_speech_started", "elapsed_ms": 0, "data": {}},
        {"name": "backchannel_audio_started", "elapsed_ms": 1500, "data": {}},
        {"name": "user_speech_ended", "elapsed_ms": 1800, "data": {}},
        {"name": "backchannel_completed", "elapsed_ms": 2600, "data": {"duration_ms": 1100.0}},
        {"name": "agent_response_started", "elapsed_ms": 3400, "data": {}},
    ]

    result = summarize_run(events, scenario_id="middle_pause", mode="backchannel", run_id="r1")

    assert result.delayed_responses == 1
    assert result.cue_delays_attributed == 1
    # the cue held the agent's speech channel for 800 ms past the turn end
    assert result.attributed_cue_delays_ms == (800.0,)


def test_cue_that_finished_before_the_turn_end_is_not_attributed() -> None:
    events = [
        {"name": "user_speech_started", "elapsed_ms": 0, "data": {}},
        {"name": "backchannel_audio_started", "elapsed_ms": 1000, "data": {}},
        {"name": "backchannel_completed", "elapsed_ms": 1400, "data": {"duration_ms": 400.0}},
        {"name": "user_speech_ended", "elapsed_ms": 3000, "data": {}},
        {"name": "agent_response_started", "elapsed_ms": 3400, "data": {}},
    ]

    result = summarize_run(events, scenario_id="middle_pause", mode="backchannel", run_id="r1")

    assert result.delayed_responses == 0
    assert result.cue_delays_attributed == 0
    assert result.attributed_cue_delays_ms == ()


def test_synthesis_sentinels_and_zero_timings_are_not_measurements() -> None:
    events = [
        {"name": "pipeline_metric", "elapsed_ms": 0, "data": {"type": "tts_metrics", "ttfb": -1.0}},
        {"name": "pipeline_metric", "elapsed_ms": 0, "data": {"type": "tts_metrics", "ttfb": 0.42}},
        {"name": "pipeline_metric", "elapsed_ms": 0, "data": {"type": "llm_metrics", "ttft": -1.0}},
    ]

    result = summarize_run(events, scenario_id="short_answer", mode="baseline", run_id="r1")

    assert result.tts_ttfb_ms == (420.0,)
    assert result.llm_ttft_ms == ()


def test_eot_suppressed_cues_are_counted_from_the_engine_signal() -> None:
    events = [
        {"name": "user_speech_started", "elapsed_ms": 0, "data": {}},
        {"name": "backchannel_suppressed_eot", "elapsed_ms": 1200, "data": {}},
        {"name": "user_speech_started", "elapsed_ms": 5000, "data": {}},
        {"name": "backchannel_suppressed_eot", "elapsed_ms": 6100, "data": {}},
    ]

    result = summarize_run(
        events, scenario_id="approaching_end_of_turn", mode="backchannel", run_id="r1"
    )

    assert result.eot_suppressed_cues == 2


def test_engine_collisions_are_folded_into_eot_risk_without_double_counting() -> None:
    events = [
        {"name": "user_speech_started", "elapsed_ms": 0, "data": {}},
        {"name": "backchannel_started", "elapsed_ms": 1500, "data": {}},
        {"name": "backchannel_audio_started", "elapsed_ms": 1560, "data": {}},
        {"name": "backchannel_collision", "elapsed_ms": 1900, "data": {"ms_since_audible": 340.0}},
        {"name": "backchannel_cancelled", "elapsed_ms": 1901, "data": {"reason": "user_stopped"}},
        {"name": "user_speech_ended", "elapsed_ms": 1901, "data": {}},
    ]

    result = summarize_run(
        events, scenario_id="approaching_end_of_turn", mode="backchannel", run_id="r1"
    )

    assert result.collision_events == 1
    assert result.end_of_turn_risks == 1


def test_legacy_collision_heuristic_still_flags_older_recordings() -> None:
    events = [
        {"name": "user_speech_started", "elapsed_ms": 0, "data": {}},
        {"name": "backchannel_started", "elapsed_ms": 1500, "data": {}},
        {"name": "user_speech_ended", "elapsed_ms": 1900, "data": {}},
        {"name": "backchannel_cancelled", "elapsed_ms": 1901, "data": {}},
    ]

    result = summarize_run(
        events, scenario_id="approaching_end_of_turn", mode="backchannel", run_id="r1"
    )

    assert result.collision_events == 0
    assert result.end_of_turn_risks == 1


def test_aggregates_report_variance_and_both_response_deltas() -> None:
    baseline = summarize_run(
        [
            {"name": "user_speech_started", "elapsed_ms": 0, "data": {}},
            {"name": "user_speech_ended", "elapsed_ms": 1000, "data": {}},
            {"name": "agent_response_started", "elapsed_ms": 1500, "data": {}},
            {"name": "user_speech_started", "elapsed_ms": 4000, "data": {}},
            {"name": "user_speech_ended", "elapsed_ms": 5000, "data": {}},
            {"name": "agent_response_started", "elapsed_ms": 6000, "data": {}},
        ],
        scenario_id="s",
        mode="baseline",
        run_id="b1",
    )
    experiment = summarize_run(
        [
            {"name": "user_speech_started", "elapsed_ms": 0, "data": {}},
            {"name": "user_speech_ended", "elapsed_ms": 1000, "data": {}},
            {"name": "agent_response_started", "elapsed_ms": 1600, "data": {}},
            {"name": "user_speech_started", "elapsed_ms": 4000, "data": {}},
            {"name": "user_speech_ended", "elapsed_ms": 5000, "data": {}},
            {"name": "agent_response_started", "elapsed_ms": 6200, "data": {}},
        ],
        scenario_id="s",
        mode="backchannel",
        run_id="e1",
    )

    aggregates = aggregate_runs([baseline, experiment])

    assert aggregates["baseline"].response_samples == 2
    assert aggregates["baseline"].response_stdev_ms == 250.0
    assert aggregates["baseline"].response_min_ms == 500.0
    assert aggregates["baseline"].response_max_ms == 1000.0
    assert aggregates["baseline"].response_delta_p95_ms is None
    assert aggregates["backchannel"].response_delta_p50_ms == 150.0
    assert aggregates["backchannel"].response_delta_p95_ms == 195.0


def test_records_with_a_second_monotonic_origin_are_split_into_separate_runs() -> None:
    records = [
        {"name": "session_started", "elapsed_ms": 0.2, "data": {"room_name": "room-a"}},
        {"name": "user_speech_started", "elapsed_ms": 1000.0, "data": {}},
        {"name": "user_speech_ended", "elapsed_ms": 2000.0, "data": {}},
        {"name": "agent_response_started", "elapsed_ms": 2500.0, "data": {}},
        # A second recorder starts while the first one is still writing.
        {"name": "session_started", "elapsed_ms": 0.3, "data": {"room_name": "room-b"}},
        {"name": "user_speech_started", "elapsed_ms": 900.0, "data": {}},
        {"name": "user_speech_ended", "elapsed_ms": 1500.0, "data": {}},
        {"name": "agent_response_started", "elapsed_ms": 2100.0, "data": {}},
        {"name": "session_stopped", "elapsed_ms": 3000.0, "data": {}},
    ]

    summaries = summarize_event_log(records)

    assert [summary.run_id for summary in summaries] == ["room-a", "room-b"]
    assert [summary.response_latencies_ms for summary in summaries] == [(500.0,), (600.0,)]


def test_required_scenarios_cover_assignment_cases() -> None:
    assert len(REQUIRED_SCENARIO_IDS) == 8
    assert {
        "short_answer",
        "long_monologue",
        "approaching_end_of_turn",
        "middle_pause",
        "fast_speaker",
        "noisy_audio",
        "multiple_backchannels",
        "stop_before_ack",
    } == set(REQUIRED_SCENARIO_IDS)


def test_summarize_run_pairs_each_user_end_with_next_real_response() -> None:
    events = [
        {"name": "user_speech_ended", "elapsed_ms": 1000, "data": {}},
        {"name": "agent_response_started", "elapsed_ms": 1480, "data": {}},
        {"name": "user_speech_ended", "elapsed_ms": 3000, "data": {}},
        {"name": "agent_response_started", "elapsed_ms": 3710, "data": {}},
    ]

    result = summarize_run(events, scenario_id="short_answer", mode="baseline", run_id="r1")

    assert result.response_latencies_ms == (480.0, 710.0)
    assert result.response_p50_ms == 595.0
    assert result.response_p95_ms == 698.5
    assert result.backchannel_count == 0


def test_summarize_run_counts_cancellations_and_end_of_turn_risk() -> None:
    events = [
        {"name": "user_speech_started", "elapsed_ms": 1000, "data": {}},
        {"name": "backchannel_started", "elapsed_ms": 2200, "data": {}},
        {"name": "user_speech_ended", "elapsed_ms": 2300, "data": {}},
        {"name": "backchannel_cancelled", "elapsed_ms": 2301, "data": {}},
        {"name": "agent_response_started", "elapsed_ms": 2900, "data": {}},
    ]

    result = summarize_run(events, scenario_id="stop_before_ack", mode="backchannel", run_id="r1")

    assert result.backchannel_count == 1
    assert result.cancelled_backchannels == 1
    assert result.end_of_turn_risks == 1
    assert result.response_latencies_ms == (600.0,)


def test_aggregate_runs_compares_modes() -> None:
    baseline = summarize_run(
        [
            {"name": "user_speech_ended", "elapsed_ms": 0, "data": {}},
            {"name": "agent_response_started", "elapsed_ms": 500, "data": {}},
        ],
        scenario_id="short_answer",
        mode="baseline",
        run_id="b1",
    )
    experiment = summarize_run(
        [
            {"name": "user_speech_ended", "elapsed_ms": 0, "data": {}},
            {"name": "agent_response_started", "elapsed_ms": 550, "data": {}},
        ],
        scenario_id="short_answer",
        mode="backchannel",
        run_id="e1",
    )
    jev_experiment = summarize_run(
        [
            {"name": "user_speech_ended", "elapsed_ms": 0, "data": {}},
            {"name": "agent_response_started", "elapsed_ms": 525, "data": {}},
        ],
        scenario_id="short_answer",
        mode="jev_backchannel",
        run_id="j1",
    )

    comparison = aggregate_runs([baseline, experiment, jev_experiment])

    assert comparison["baseline"].response_p50_ms == 500.0
    assert comparison["backchannel"].response_p50_ms == 550.0
    assert comparison["backchannel"].response_delta_p50_ms == 50.0
    assert comparison["jev_backchannel"].response_p50_ms == 525.0
    assert comparison["jev_backchannel"].response_delta_p50_ms == 25.0


def test_summarize_run_extracts_provider_timings_and_audible_backchannel_latency() -> None:
    result = summarize_run(
        [
            {"name": "backchannel_decision", "elapsed_ms": 1000, "data": {}},
            {"name": "backchannel_started", "elapsed_ms": 1010, "data": {}},
            {"name": "backchannel_audio_started", "elapsed_ms": 1080, "data": {}},
            {
                "name": "jev_decision",
                "elapsed_ms": 1090,
                "data": {"latency_ms": 321.5, "approved": True},
            },
            {"name": "jev_decision_timeout", "elapsed_ms": 1100, "data": {}},
            {"name": "user_speech_ended", "elapsed_ms": 2000, "data": {}},
            {
                "name": "pipeline_metric",
                "elapsed_ms": 2100,
                "data": {"type": "llm_metrics", "ttft": 0.4},
            },
            {
                "name": "pipeline_metric",
                "elapsed_ms": 2200,
                "data": {"type": "tts_metrics", "ttfb": 0.2},
            },
            {
                "name": "pipeline_metric",
                "elapsed_ms": 2300,
                "data": {"type": "eou_metrics", "end_of_utterance_delay": 0.15},
            },
            {"name": "agent_response_started", "elapsed_ms": 2500, "data": {}},
        ],
        scenario_id="middle_pause",
        mode="backchannel",
        run_id="r1",
    )

    assert result.backchannel_latency_p50_ms == 80.0
    assert result.llm_ttft_p50_ms == 400.0
    assert result.tts_ttfb_p50_ms == 200.0
    assert result.eot_delay_p50_ms == 150.0
    assert result.jev_decision_latency_p50_ms == 321.5
    assert result.jev_timeouts == 1


def test_replay_runner_produces_paired_runs_for_each_scenario() -> None:
    records, summaries, report = run_replay_benchmark(
        scenario_ids=("short_answer", "long_monologue"), repeats=2
    )

    assert len(summaries) == 12
    assert len(records) > len(summaries)
    assert report["run_count"] == 12
    assert report["scenario_count"] == 2
    assert report["scenarios"][1]["comparison"]["backchannel"]["response_p50_ms"] is not None
    jev_delta = report["scenarios"][1]["comparison"]["jev_backchannel"]["response_delta_p50_ms"]
    assert jev_delta is not None
    assert set(report["overall"]) == {"baseline", "backchannel", "jev_backchannel"}


def test_report_keeps_empty_required_scenarios_visible() -> None:
    report = build_report([], source="empty")

    assert len(report["scenarios"]) == 8
    assert report["scenario_count"] == 0
