# Blue Machines LiveKit Baseline

This repository contains the LiveKit voice agent, timed and Jev-assisted backchannel
experiments, and an experiment workspace for inspecting the assignment's eight speaking
situations. The same event contract powers real room measurements, a repeatable
offline policy replay, the comparison table, and the visual run timeline.

## What is included

- Python 3.12 project managed with `uv`.
- LiveKit Agents `1.8.2` with public `AgentServer`, `AgentSession`, and `RoomOptions` APIs.
- Streaming speech-to-text -> LLM -> streaming speech providers, chosen by environment
  variable rather than by code path. The shipped default is LiveKit Inference (Gemini
  Transcribe, Gemini Flash, LiveKit TTS); **the measurements in this README were taken on
  the Deepgram stack** (`STT_PROVIDER=deepgram`, `LLM_PROVIDER=groq`,
  `TTS_PROVIDER=deepgram_tts`), and every run records which stack produced it, so a number
  can never be read against the wrong pipeline.
- Optional OpenRouter model failover and direct ElevenLabs TTS.
- Silero voice activity detection.
- JSONL lifecycle events for speech boundaries, transcript metadata, EOT signals,
  backchannel decisions, audible acknowledgement starts, provider metrics, and errors.
- Eight repeatable benchmark scenarios plus an analyzer for response P50/P95,
  acknowledgement latency, provider timings, cancellations, overlap, and EOT risk.
- Three report modes: Baseline, Timer backchannel, and Jev backchannel. Jev is the
  semantic decision layer for the third experiment; its API key is never sent to
  the browser.
- A Next.js workspace with a real LiveKit browser conversation, a three-mode policy
  selector, scenario prompts, comparison results, and run timelines. The policy is
  chosen before a run and locked for its duration, so every measured run has exactly
  one policy.
- A deterministic replay runner for validating the inspection system before provider
  credentials are available. Replay numbers are labelled and are not provider data.
- A scripted scenario driver (`blue-machines-scenario`) that drives every
  (scenario, mode, repeat) through real LiveKit rooms with real audio - no human at the
  microphone, and no dependence on speaking twice. The user's line is either the committed
  clip for the scenario or, with `--live-audio`, synthesized per run by the *configured*
  speech provider and streamed into the room as it is produced (`SCENARIO_TTS_MODEL` /
  `SCENARIO_TTS_VOICE` give that voice its own identity, so the two speakers in a room are
  distinguishable). A missing clip falls back to live synthesis rather than failing.
  `--generate-audio` renders the clips with the same configured provider.
- Provider options for environments where LiveKit Inference is unavailable or rate
  limited: Deepgram live STT and Aura-2 speech (`STT_PROVIDER=deepgram`,
  `TTS_PROVIDER=deepgram_tts`), the OpenRouter-hosted Deepgram voice
  (`TTS_PROVIDER=openrouter_tts`), native Groq for all three roles
  (`STT_PROVIDER=groq_interim`, `LLM_PROVIDER=groq`, `TTS_PROVIDER=groq_tts`), and a direct
  Gemini TTS adapter (`TTS_PROVIDER=gemini_tts`). Groq's speech model additionally requires
  a one-time terms acceptance in their console; the API returns `model_terms_required`
  until then, with the acceptance link in the error.
- A real end-of-turn probability for the policy, from LiveKit's public streaming turn
  detector running on the user's audio, with an automatic final-transcript fallback. Each
  `eot_prediction` event names the model that answered (`turn-detector-v1-mini` locally,
  `turn-detector-v1` when the gateway serves it).
- Speech-to-text comes in two flavours. `STT_PROVIDER=deepgram` uses Deepgram's live
  websocket: native interim transcripts while the user speaks, finals that land *during*
  the turn, and an end-of-speech marker for the turn boundary - measured on a real
  utterance, ten interims and three finals, the first final arriving while the user was
  still talking. `STT_PROVIDER=groq_interim` reaches the same place from a batch-only
  provider: it segments turns with a VAD (the pipeline forwards audio continuously and
  never flushes one stream per turn) and snapshots interims inside each segment, with a
  per-process request budget that spends on interims only while finals stay guaranteed -
  dropping an interim costs a little policy context, dropping a final would leave the
  turn open. Both satisfy Jev mode's requirement for interim transcripts; plain
  `STT_PROVIDER=groq` does not, and the API refuses that combination before a room is
  created.
- FastAPI `/health`, `/events`, `/benchmark/report`, `/benchmark/replay`, and
  `/livekit/token` endpoints.
- Explicit tests for cooldown, EOT suppression, failed TTS, rapid transitions, and
  shutdown cleanup.

The normal flow is:

```text
browser microphone -> LiveKit room -> Silero VAD -> streaming LiveKit Gemini STT -> Gemini Flash -> LiveKit Inference TTS -> browser audio
```

The backchannel is a listener signal, not an assistant turn. It watches public user
speaking state and interim/final transcript events. A final transcript is treated as
a public end-of-turn safety signal; this version does not reach into LiveKit private
objects for EOT predictions. After a short delay it may play one brief acknowledgement,
subject to one-per-turn and cooldown gates. If the user stops, the agent needs the
floor, the room is shut down, or the browser toggles the policy off, the
acknowledgement's speech handle is interrupted immediately. It is sent with
`add_to_chat_ctx=False`, so it does not become part of the Gemini conversation.

## Local setup

From this directory:

```bash
uv sync --extra dev
cp .env.example .env
```

Edit `.env` and set these required variables:

```text
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
STT_PROVIDER=livekit_inference
LIVEKIT_STT_MODEL=google/gemini-3.5-transcribe-live
LLM_PROVIDER=gemini
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash
```

To run the Jev-assisted mode, add `TYPESAFE_API_KEY` to the server environment.
The optional `JEV_MODEL`, `JEV_TIMEOUT_SECONDS`, `JEV_MIN_INTERVAL_SECONDS`, and
`JEV_APPROVAL_THRESHOLD` settings control the classifier model, its fail-closed
timeout, how often interim transcript snapshots may be classified, and how
conservative approval is. Jev decisions must be asynchronous: timeouts and stale
answers fall back to silence and never delay the real response.
When Jev approves a cue, it chooses from a small safe vocabulary based on the
partial transcript: `mm-hmm` for plain continuation, `uh-huh` for a clear list
or story, and `I see` for an explanation or context. Thinking, uncertainty,
questions, endings, and ambiguous speech stay silent. Agreement-heavy words
such as `yes`, `yeah`, and `right` are intentionally excluded so the agent does
not endorse an unverified claim.
Jev also classifies the speech function before selecting a cue: plain
continuation, list or story, explanation or context, thinking or uncertainty,
explicitly continuing, question or ending, and emotional or ambiguous speech.
The last two categories are fail-silent, and the code maps the approved speech
function to its matching phrase instead of accepting an arbitrary phrase.
The three phrases are pre-rendered, not synthesized during the turn: the worker loads
`assets/backchannels/{mm-hmm,uh-huh,i-see}.wav` once at start, which is why a cue is
audible ~20 ms after the policy approves it and costs no speech request.
`assets/backchannels/manifest.json` records the provider, model and duration that produced
them. Regenerate them in the worker's own voice with

```bash
uv run python scripts/generate-backchannel-clips.py
```

The script follows `TTS_PROVIDER` and the agent's voice, trims leading and trailing
silence, normalizes level, writes the rate the room plays out, and refuses to write a cue
longer than 1.5 s - a long acknowledgement stops being an acknowledgement and holds the
floor. The worker must be restarted afterwards: the clips are read once per process.

During a genuinely long continuation, Jev may approve up to four additional cues after
substantial new speech (the controller allows five per turn in total, with roughly seven
new words required between cues); short turns still get a single cue. It stays silent
near the end of a turn, when the user asks a direct question, or when its result
is late or uncertain.
This slice adds the third mode to the browser/API/reporting contracts and replay;
the semantic classifier and worker gating remain in the protected core agent files.

`LIVEKIT_AGENT_NAME` defaults to `blue-machines-baseline`. The optional variables in
`.env.example` select STT, LLM, TTS, assistant instructions, event paths, and
benchmark output paths. `BACKCHANNEL_ENABLED` remains the worker default for
non-browser clients; the browser mode is carried in explicit dispatch metadata.
Secrets are read at startup and are never written to the event log.

## Run the agent

The easiest way to start the complete local stack is:

```bash
./scripts/dev-stack.sh start
```

This starts one worker, the local API on port 8000, and the dashboard on port
3001. It saves only the processes it starts, their logs, and their PID files in
`.runtime/`. Stop that stack with `./scripts/dev-stack.sh stop`, or check it with
`./scripts/dev-stack.sh status`.

To start the worker on its own:

Start the worker in development mode:

```bash
uv run blue-machines-agent dev
```

The browser workspace creates a unique room, explicitly dispatches
`blue-machines-baseline`, and attaches `scenario_id`, `mode`, and `run_id` as job
metadata. The agent greets the user, listens through the chained STT/LLM/TTS
pipeline, and speaks the response back into the room.

## Run the local API

In a second terminal:

```bash
uv run uvicorn blue_machines_baseline.api:app --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000/health` or
`http://127.0.0.1:8000/events`. Event inspection and replay analysis do not require
provider credentials. `/livekit/token` requires the LiveKit URL and key pair, but
never exposes them to the browser.

## Run the voice workspace

With Node.js 20.9 or newer installed, open a separate terminal:

```bash
cd dashboard
npm ci
npm run dev
```

Open `http://localhost:3001` when port 3000 is already occupied (or use the port
printed by Next.js). Select a scenario and choose **Baseline**, **Timer**, or
**Backchannel + Jev**. **Start conversation** requests a browser token, joins a unique
LiveKit room, publishes the microphone, and subscribes to agent audio. The selected mode
travels in the dispatch metadata and is locked by the worker for that run, so every
measured run has exactly one policy; **End conversation** releases the room and
microphone, and the selector unlocks for the next run. (The worker also accepts a
`blue-machines-control` data-channel toggle - `backchannel_toggle` or `experiment_mode` -
for other clients, and rejects any mode that contradicts the run it was dispatched with.)

**Check microphone** is a separate local-only input check. It does not record,
upload, transcribe, or play back audio. Microphone access requires localhost or
HTTPS. A pending check can be cancelled; any input subsequently granted to that
cancelled check is immediately released.

**Benchmark results** shows real P50/P95 values when labelled worker sessions have
been recorded. **Run 3x replay** executes all three modes for all eight scenarios using
deterministic policy-level timings; it is useful for demonstrating the UI and
checking analyzer regressions, but it must not be presented as provider latency.
The timeline and raw feed remain available for event-level inspection.

For an API running elsewhere, set the optional server-only `BASELINE_API_URL` in
the Next.js process environment or `dashboard/.env.local`, then restart Next.js:

```text
BASELINE_API_URL=http://127.0.0.1:8000
```

Do not add a `NEXT_PUBLIC_` prefix or put provider credentials in the frontend.
Fonts are bundled locally; the workspace does not require a remote font service.

For a production build, from `dashboard/`:

```bash
npm run build
npm start
```

Keep the Python API running separately on port 8000.

Run exactly one worker for the demo. If a worker is already registered as
`blue-machines-baseline`, reuse that process instead of starting a second one;
duplicate workers can make room dispatch and benchmark labels ambiguous. Stop only
the worker you started from its own terminal with Ctrl-C; do not kill processes by a
generic name or PID.

## Events and timing

Each JSONL record contains:

- `timestamp`: UTC wall-clock time for display;
- `elapsed_ms`: monotonic offset from the session start for timing comparisons;
- `name`: event name;
- `scenario_id`, `mode`, and `run_id`: labels for paired experiment runs;
- `data`: small structured metadata.

The actual response latency is the time from `user_speech_ended` to the next
`agent_response_started` **that belongs to that turn**. A response only counts when it
starts after the turn ends, before the next turn begins, and within 30 seconds; a turn
that never got an answer is reported as an `unpaired_turns` count instead of borrowing a
later turn's answer. (An earlier version paired each turn with the next response anywhere
in the run, which turned idle time into "latency" - a real recording reported a 107 second
baseline P50. The regression test for this is
`test_response_latency_never_borrows_a_later_turn_s_response`.)

Backchannel latency is `backchannel_decision` to `backchannel_audio_started`; the latter
comes from the agent audio output playback marker, not merely creation of a speech handle.
`pipeline_metric` records safe scalar fields exposed by LiveKit metrics, such as EOU delay,
LLM TTFT, STT duration, and TTS TTFB. Transcript text itself is not stored; only
final/interim status and counts are recorded.

End-of-turn probability is measured for real. `eot_prediction` records the probability, the
threshold it was compared against, and the detector model that produced it, from LiveKit's
public streaming turn detector running on the user's audio
(`EOT_DETECTOR=livekit_inference`). Verified directly by pushing recorded speech and
trailing silence into the detector stream and reading the predictions back. If that detector cannot be
used, the session emits `eot_detector_unavailable` once and the policy keeps working on the
binary final-transcript fallback.

Backchannel outcomes carry the information needed to judge them:

- `backchannel_cancelled` includes a `reason` (`user_stopped`, `agent_busy`, `disabled`,
  `shutdown`, `semantic_rejected`, `mode_switch`, `post_play_race`);
- `backchannel_completed` includes the audible `duration_ms`;
- `backchannel_collision` is emitted when an acknowledgement was still audible and the user
  yielded within `BACKCHANNEL_COLLISION_WINDOW_SECONDS` of it becoming audible.

## Run and compare benchmarks

Analyze the current worker log and write a report:

```bash
uv run blue-machines-benchmark --events outputs/baseline-events.jsonl \
  --report outputs/benchmark-report-provider.json
```

Run the deterministic inspection harness before provider credentials are available:

```bash
uv run blue-machines-benchmark --replay --repeats 3
```

The replay writes `outputs/benchmark-replay-events.jsonl` and
`outputs/benchmark-report.json`. It pairs every selected scenario in baseline, timer
backchannel, and Jev backchannel mode with the same scripted speech duration and
repeat count. Replay timings are policy smoke numbers, not measurements.

**The API serves measured data as the headline.** `/benchmark/report` builds the report
from labelled provider sessions in the event log and returns that as the primary payload
when any exist; a persisted replay report is attached as `replay_observation` instead of
standing in for the measurements. When no measured sessions exist, the replay is returned
with its own provenance so nothing is ever presented as provider latency.

### Scripted scenarios, without a human at the microphone

`blue-machines-scenario` drives the same interaction through real rooms for every
configuration, so results do not depend on someone speaking twice:

```bash
# once: render the scenario utterances with the configured TTS provider
uv run blue-machines-scenario --generate-audio

# for every (scenario, mode, repeat): real room, real audio, real worker
uv run blue-machines-scenario --scenarios all --modes baseline,backchannel --repeats 3
```

The driver mints a token that dispatches the worker with the usual
`scenario_id`/`mode`/`run_id` metadata, joins as a participant, waits for the worker to
finish its greeting, publishes the scenario's clip as microphone audio (inserting the
scenario's scripted pauses inside the sentence), and waits for the run to end. Clips live
in `assets/scenarios/` with a sidecar recording the text, voice and measured duration, so
a re-run needs no provider call for the user's side. The worker must be running
(`uv run blue-machines-agent dev`); timings come from its event log, not from the driver.

The committed sweep was recorded on one provider stack, after LiveKit Inference returned
HTTP 429 and ElevenLabs 402 from this machine:

```bash
STT_PROVIDER=groq LLM_PROVIDER=groq TTS_PROVIDER=groq_tts EOT_DETECTOR=livekit_inference \
  uv run blue-machines-agent dev
uv run blue-machines-scenario --scenarios all --modes baseline,backchannel --repeats 3
```

Jev mode is not part of that sweep, but it is runnable: use
`STT_PROVIDER=groq_interim`, which transcribes Groq's endpoint on a cadence so interim
transcripts arrive while the user speaks. With plain `STT_PROVIDER=groq` (batch, final-only)
the Python API refuses `jev_backchannel` with a 409 before the room exists, so the UI
explains the problem instead of leaving an empty room - which means the API and the worker
must be started with the same `STT_PROVIDER`. With a streaming STT configured, add `jev_backchannel` to `--modes` and the
same driver covers all three policies.

## Fairness and race analysis

### What counts as a bad backchannel

An acknowledgement is **bad** when it costs the user something rather than signalling
listening. In order of severity:

1. **It takes the floor away from the real answer.** If a cue is still audible when the
   user's turn ends, the reply cannot begin until it clears. This is counted as
   `delayed_responses` and is the one failure that can make the agent measurably slower.
   Threshold: any cue audible at `user_speech_ended` counts.
2. **It lands where the turn was already ending.** A cue that becomes audible and is then
   talked over within `BACKCHANNEL_COLLISION_WINDOW_SECONDS` (default 500 ms) is reported as
   `backchannel_collision`. The 500 ms figure is the length of a short acknowledgement
   itself, so a cue inside that window has no room to be heard as a cue rather than as an
   interruption.
3. **It becomes noise.** More than one acknowledgement per short turn, or two cues inside
   the cooldown window, is bad regardless of latency; these are prevented by the per-turn
   cap and the cooldown, and the surviving `audible_backchannels` count is reported per
   run so the experiment can show restraint rather than eagerness.

The reasoning: a backchannel is only useful while it is *cheap* - short, quiet, and
strictly inside the user's own turn. All three definitions are therefore about not
occupying the floor the user's answer needs, which is also why the engine interrupts the
cue the moment the user yields.

### How the comparison is kept fair

- Baseline, timer backchannel, and Jev backchannel runs use the same scenario,
  provider configuration, room creation path, and repeat count. Only the policy mode
  changes. Both experimental columns report their delta against Baseline.
- Backchannel audio is excluded from LLM context and the acknowledgement is never
  counted as an actual agent response. This prevents the experiment from making
  the response metric look better by changing conversation history.
- `response_samples`, `response_stdev_ms`, `response_min_ms` and `response_max_ms` are
  reported next to P50/P95, so a small delta can be judged against the spread of the
  measurements instead of being read as a result on its own.
- A generation counter invalidates timers from older turns. Active speech handles
  are interrupted on user floor changes, agent floor ownership, runtime disable,
  and shutdown. Late provider callbacks are ignored after the recorder closes.
- EOT risk is counted when an acknowledgement overlaps the user and the user yields
  within the short collision window. Cancellations and overlaps are shown beside
  latency so a faster-looking run cannot hide worse turn-taking.
- A positive response delta means the backchannel configuration started the real
  answer later. A negative delta is not automatically an improvement; inspect EOT
  risk, cancellations, and the timeline before drawing a conclusion.

### Whose voice the user speaks with

The committed evidence was recorded with the clips: every run heard byte-identical audio,
which is what keeps the three arms comparable. `--live-audio` trades a little of that for
audibility in the stack actually under test - the user's side then comes from the configured
provider instead of a Gemini render from months ago. It does not move the measured latency
either way: the number is anchored at the end of the user's line (`user_speech_ended`), so
how that line was produced is upstream of the measurement. What it costs is one synthesis per
run (about a second of time-to-first-audio, then real-time playback) and a slightly different
waveform per repeat.

### From rooms to the report

```
blue-machines-scenario --scenarios all --modes baseline,backchannel,jev_backchannel --repeats 3
        -> outputs/baseline-events.jsonl        every event, gitignored
scripts/extract-benchmark-runs.py --stack stt=deepgram,llm=groq,tts=deepgram_tts
        -> outputs/benchmark-events.jsonl       committed evidence
blue-machines-benchmark --events ... --report ...
        -> outputs/benchmark-report-provider.json
```

The extraction keeps only scripted runs that match the stack being reported and that
actually got an answer (`user_speech_ended` and `agent_response_started`). Every run
records the providers it used in its `session_started` event, so a table cannot
quietly mix a batch-STT measurement with a streaming one, and the extraction says out
loud how many runs it skipped and why.

## What became slower, and what I would change before production

### The measured result

`outputs/benchmark-report-provider.json` (built from `outputs/benchmark-events.jsonl`, the
scripted sweep on the **streaming stack**: Deepgram `nova-3` streaming STT, Groq
`openai/gpt-oss-120b`, Deepgram Aura-2 streaming speech, LiveKit's turn detector) reads.
Every scenario runs in all three modes, three repeats each, in real rooms - and each run
records the stack it used, so the table cannot mix providers.

<!-- results:start -->
Measured over **80 runs** across 8 scenarios and three modes - baseline, timer backchannel, Jev backchannel - three repeats each (a cell is re-run when a run is lost), in real rooms:

| Metric | Baseline | Timer backchannel | Jev |
|---|---|---|---|
| Response floor (min) | 1,008 ms | 1,121 ms | 1,209 ms |
| Response P50 | 1,802 ms *(n=24)* | 3,364 ms *(n=24)* | 3,087 ms *(n=32)* |
| Response P95 | 7,979 ms | 7,724 ms | 6,518 ms |
| Response σ | 2,443 ms | 2,043 ms | 1,551 ms |
| Audible cues | 0 | 19 | 28 |
| Cues per long turn | 0 | 1 | 1.08 |
| Decision → audible | — | 19 ms | 21 ms |
| EOT delay P50 | 577 ms | 578 ms | 577 ms |
| LLM TTFT P50 | 427 ms | 341 ms | 430 ms |
| Speech TTFB P50 | 901 ms | 914 ms | 870 ms |
| Delayed responses | 0 | 0 | 0 |
| End-of-turn collisions | 0 | 0 | 0 |
| Cancelled cues | 0 | 0 | 3 |
| Unpaired turns | 4 | 4 | 6 |
<!-- results:end -->

**Did backchanneling make the agent slower? Not through the pipeline.** The stage the policy
could touch is the turn handoff, and it is identical in every arm: the turn the user ended is
committed within a few milliseconds of the end-of-utterance metric (P50 577 ms of detector
delay in all three arms, the metric landing 5 ms later), and the providers behind it are the
same to within noise (LLM TTFT 427/341/430 ms, speech TTFB 901/914/870 ms). The fastest answer
in each arm is also within 200 ms of the others - 1,008 / 1,121 / 1,209 ms - so a cue does not
add to the floor.

What the aggregate P50 column does **not** show is a policy effect either way, and reading it
as one would be a mistake. Response latency in these rooms is provider-dominated: a quarter to
a third of turns exceed 4 s in *every* arm, and the per-scenario breakdown swings in both
directions (on `noisy_audio` the cue arms are faster, 3.3-3.9 s against 4.2-6.8 s; on
`long_monologue` the timer arm is slower, 6.9-7.9 s against 1.7-3.7 s). With three repeats per
scenario and arm, that variance decides the P50, which is why the earlier 495 ms "faster at
P50" claim from the previous stack is not carried over here: what this sweep can support is
that the policy is not on the critical path, not a sub-second claim in either direction. More
repeats, or a latency-stable provider, is what a tighter claim would need.

The cue path itself is clean in this sweep: 19 timer cues and 28 Jev cues became audible, each
19-21 ms after the decision, with 0 collisions, 0 delayed responses and 3 cancelled cues (Jev
cues still audible when the user took the floor).

Limits of this sweep, stated plainly: one worker drives one room at a time, so runs are
sequential and `n` per arm is the number of scripted repeats, not a production sample. Six
recorded runs were excluded because they produced no answer at all - the driver used to end a
run when any agent audio went quiet, and a cue is agent audio, so it disconnected while the
model was still generating; that is fixed and the three `stop_before_ack` repeats it killed now
answer. The driver also reports "produced agent audio" and "answered" separately, because
reading the first as the second is how that hid. Jev's classifier occasionally exceeds its 4 s
deadline (the decision is about a partial transcript, so a late answer is worthless) and fails
closed, which costs a cue rather than producing a late one. Every run is a real room with real
STT, LLM and speech calls, and scripted runs skip the greeting (`--greet` re-enables it) to
keep a sweep cheap.

**Speech synthesis is the latency floor.** With the streaming reader above, the wait
before the agent's first audible word is the provider's own time-to-first-audio: measured
at 0.96 s for a five-word reply and 1.11 s for a nine-word reply through the Deepgram
adapter, against 37-96 ms for providers built for real-time agents (Rime, Cartesia). What
is left after that is the pipeline's own turn-taking, which the backchannel policy is
deliberately kept out of: the policy never sits between the user's turn ending and the
agent's answer starting.

**Where any extra latency comes from.** The policy itself is not on the response path:
timers, the cue, and its cancellation all live beside the conversation, and the
acknowledgement never enters chat context. The only way backchanneling can slow the real
answer down is by **holding the floor**: the cue is played with `allow_interruptions=False`
(so it can speak over the user), which means an answer that arrives while the cue is still
audible has to wait for it to clear. That is why the engine interrupts the cue on the
user's floor change *and* reports every cue that was still audible when the user yielded
(`delayed_responses`, `backchannel_collision`). If a run shows a response delta, look at
those two counters first: a delta without them is provider variance, not the policy.

**Before running this at production scale I would change:**

1. **Generate the cues for the session's voice.** The acknowledgement clips are
   pre-rendered, which is what makes a cue audible ~2 ms after the decision - but they are
   fixed files, so they do not match a different agent voice. At scale they should be
   synthesised once per voice at session start, or streamed on first use, and cached.
2. **Feed the real turn detector's probability into the EOU decision, not just the policy.**
   Today the detector runs deliberately in parallel with LiveKit's own end-of-turn logic so
   the measurement is independent; at scale it should be one shared detector to avoid
   paying for two.
3. **Make the cue a side channel.** Publishing the acknowledgement on its own audio track
   (rather than as agent speech) would remove the floor-holding hazard structurally instead
   of managing it with interrupts. The current design was kept because LiveKit's playback
   marker gives an honest "became audible" timestamp, and because `add_to_chat_ctx=False`
   already keeps the cue out of the conversation.
4. **Per-language thresholds and cue banks.** The turn detector publishes separate
   `unlikely_threshold` and `backchannel_threshold` values per language; the policy
   currently uses one threshold for all sessions.
5. **Capacity and cost.** Each benchmark run is a real room with real STT, LLM and TTS
   calls: budget for the provider quota (the local sweep needed 48 rooms), and pin the
   worker count - one worker per agent name, as the setup notes require - so dispatch stays
   unambiguous.
6. **Observability at fleet level.** The event contract already carries everything needed
   (decision, audible start, cancellation reason, collision, response pairing); what is
   missing at scale is aggregation across sessions, which is exactly what
   `blue-machines-benchmark` does offline.

## Checks

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run python -m compileall -q src tests
uv run python -c "import blue_machines_baseline.agent; import blue_machines_baseline.api"
```

These checks run without provider credentials (143 tests). A real room conversation needs
valid LiveKit, an LLM key, and a speech provider: Deepgram speech, the OpenRouter-hosted
Deepgram voice, native Groq speech (after a one-time terms acceptance), LiveKit Inference,
ElevenLabs, or the direct Gemini TTS adapter. Jev mode additionally needs `TYPESAFE_API_KEY`
and a streaming STT with interim results - `STT_PROVIDER=deepgram` or `groq_interim`; plain
Groq STT is batch-only, so Jev mode is rejected with a clear error when it is selected with
that provider. The replay runner, analyzer, API report, and browser build remain usable
without any provider service.
