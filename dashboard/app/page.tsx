"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Room, RoomEvent, Track, type RemoteTrack } from "livekit-client";

const scenarios = [
  { id: "short_answer", title: "Short answer", tag: "WARM-UP", description: "Brief answer, under one second.", action: 'Say: "I\'m doing well today."' },
  { id: "long_monologue", title: "Long monologue", tag: "STAY WITH ME", description: "Explain an idea continuously for 10–15 seconds.", action: "Tell a story about a recent project." },
  { id: "approaching_end_of_turn", title: "Near the ending", tag: "READ THE ROOM", description: "Pause near the end, then finish your thought.", action: "Leave a small pause before your last sentence." },
  { id: "middle_pause", title: "Middle pause", tag: "HOLD THE THREAD", description: "Pause in the middle, then continue.", action: "Think out loud about a difficult decision." },
  { id: "fast_speaker", title: "Fast speaker", tag: "QUICK TEMPO", description: "Speak quickly without giving up the floor.", action: "List three things you want to improve." },
  { id: "noisy_audio", title: "Noisy audio", tag: "SIGNAL / NOISE", description: "Speak while gentle background noise is present.", action: "Describe the room around you." },
  { id: "multiple_backchannels", title: "Many openings", tag: "RESTRAINT", description: "Speak long enough for several possible cues.", action: "Give a 20-second explanation without stopping." },
  { id: "stop_before_ack", title: "Stop before cue", tag: "LAST SECOND", description: "Stop just before an acknowledgement would play.", action: "Start a sentence, then stop after one second." },
] as const;

type Mode = "baseline" | "backchannel" | "jev_backchannel";
type MicState = "idle" | "pending" | "active";
type LiveState = "idle" | "connecting" | "connected" | "error";
type IconName = "arrow" | "mic" | "stop" | "refresh" | "lock" | "play";

type MicResources = {
  stream: MediaStream | null;
  context: AudioContext | null;
  source: MediaStreamAudioSourceNode | null;
  frame: number;
};
type LiveSignalResources = {
  context: AudioContext | null;
  source: MediaStreamAudioSourceNode | null;
  frame: number;
};

type WorkerEvent = {
  timestamp?: string;
  name: string;
  elapsed_ms?: number;
  scenario_id?: string;
  mode?: Mode;
  run_id?: string;
  data: Record<string, unknown>;
};

type Summary = { [key: string]: unknown };
type ReportScenario = { scenario_id: string; description: string; runs: Summary[]; comparison: Record<Mode, Summary | null> };
type BenchmarkReport = {
  generated_at?: string;
  source: string;
  run_count: number;
  scenario_count: number;
  scenarios: ReportScenario[];
  overall: Record<Mode, Summary | null>;
  provenance?: { kind: string; label: string; provider_latency_available: boolean; note: string };
  replay_observation?: BenchmarkReport;
};
type LiveKitToken = { server_url: string; participant_token: string; room_name: string; run_id: string };
type ApiFailure = { error?: string; detail?: string };

type IconProps = { name: IconName; size?: number };
function Icon({ name, size = 15 }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="square" strokeLinejoin="miter" aria-hidden="true">
      {name === "arrow" && <><path d="M4 12h16M13 5l7 7-7 7" /></>}
      {name === "mic" && <><rect x="9" y="3" width="6" height="12" /><path d="M6 11v1a6 6 0 0 0 12 0v-1M12 18v3M9 21h6" /></>}
      {name === "stop" && <rect x="6" y="6" width="12" height="12" fill="currentColor" stroke="none" />}
      {name === "refresh" && <><path d="M20 7v5h-5M4 17v-5h5" /><path d="M6 6a8 8 0 0 1 13 3M5 15a8 8 0 0 0 13 3" /></>}
      {name === "lock" && <><rect x="6" y="10" width="12" height="10" /><path d="M9 10V7a3 3 0 0 1 6 0v3M12 14v2" /></>}
      {name === "play" && <path d="m8 5 11 7-11 7V5Z" fill="currentColor" stroke="none" />}
    </svg>
  );
}

function readEvents(value: unknown): WorkerEvent[] {
  if (!value || typeof value !== "object" || !("events" in value) || !Array.isArray(value.events)) throw new Error("The event API returned an unreadable response.");
  return value.events.flatMap((event: unknown) => {
    if (!event || typeof event !== "object" || !("name" in event) || typeof event.name !== "string") return [];
    const record = event as Record<string, unknown>;
    const data = record.data && typeof record.data === "object" ? record.data as Record<string, unknown> : {};
    return [{
      name: event.name,
      timestamp: typeof record.timestamp === "string" ? record.timestamp : undefined,
      elapsed_ms: typeof record.elapsed_ms === "number" && Number.isFinite(record.elapsed_ms) ? record.elapsed_ms : undefined,
      scenario_id: typeof record.scenario_id === "string" ? record.scenario_id : undefined,
      mode: record.mode === "baseline" || record.mode === "backchannel" || record.mode === "jev_backchannel" ? record.mode : undefined,
      run_id: typeof record.run_id === "string" ? record.run_id : undefined,
      data,
    } satisfies WorkerEvent];
  });
}

function eventTime(timestamp: string | undefined) {
  if (!timestamp) return "NO TIMESTAMP";
  const date = new Date(timestamp);
  return Number.isNaN(date.getTime()) ? "NO TIMESTAMP" : date.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
}

function eventSortValue(event: WorkerEvent) {
  if (event.timestamp) {
    const timestamp = Date.parse(event.timestamp);
    if (Number.isFinite(timestamp)) return timestamp;
  }
  return event.elapsed_ms ?? 0;
}

function eventLabel(name: string) {
  const labels: Record<string, string> = {
    session_started: "SESSION STARTED",
    session_stopped: "SESSION STOPPED",
    session_error: "SESSION ERROR",
    user_speech_started: "USER SPEECH STARTED",
    user_speech_ended: "USER SPEECH ENDED",
    stt_transcript: "STT TRANSCRIPT",
    pipeline_metric: "PIPELINE METRIC",
    eot_signal: "EOT SIGNAL",
    eot_prediction: "EOT PREDICTION",
    backchannel_decision: "BACKCHANNEL DECISION",
    jev_decision: "JEV DECISION",
    backchannel_started: "BACKCHANNEL STARTED",
    backchannel_audio_started: "BACKCHANNEL AUDIO",
    backchannel_completed: "BACKCHANNEL COMPLETE",
    backchannel_cancelled: "BACKCHANNEL CANCELLED",
    agent_response_started: "AGENT RESPONSE STARTED",
    agent_response_ended: "AGENT RESPONSE ENDED",
  };
  return labels[name] || name.replaceAll("_", " ").toUpperCase();
}

function shortMode(mode: Mode | undefined) {
  return mode === "jev_backchannel" ? "JEV" : mode === "backchannel" ? "TIMER" : mode === "baseline" ? "BASE" : "—";
}

function formatMs(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? `${Math.round(value * (value < 20 ? 1000 : 1))} ms` : "—";
}

function formatCount(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? Math.round(value).toLocaleString() : "—";
}

function metricLabel(event: WorkerEvent | undefined) {
  if (!event) return "NO PROVIDER METRIC";
  const type = typeof event.data.metric_type === "string" ? event.data.metric_type : typeof event.data.type === "string" ? event.data.type : "UNKNOWN METRIC";
  const value = event.data.ttft ?? event.data.ttfb ?? event.data.duration;
  return `${type.toUpperCase()} ${value === undefined ? "" : formatMs(value)}`.trim();
}

function transcriptLabel(event: WorkerEvent | undefined) {
  if (!event) return "NO TRANSCRIPT EVENT RECORDED";
  const finality = event.data.is_final === true ? "FINAL" : event.data.is_final === false ? "INTERIM" : "UNKNOWN";
  const words = formatCount(event.data.word_count);
  const chars = formatCount(event.data.character_count);
  return `${finality} / ${words} WORDS / ${chars} CHARS`;
}

const comparisonRows = [
  { label: "RESP P50", key: "response_p50_ms", unit: "ms" },
  { label: "RESP P95", key: "response_p95_ms", unit: "ms" },
  { label: "RESP σ", key: "response_stdev_ms", unit: "ms" },
  { label: "RESP SAMPLES", key: "response_samples", unit: "count" },
  { label: "BACKCHANNEL P50", key: "backchannel_latency_p50_ms", unit: "ms" },
  { label: "JEV DECISION P50", key: "jev_decision_latency_p50_ms", unit: "ms" },
  { label: "LLM TTFT P50", key: "llm_ttft_p50_ms", unit: "ms" },
  { label: "TTS TTFB P50", key: "tts_ttfb_p50_ms", unit: "ms" },
  { label: "EOT DELAY P50", key: "eot_delay_p50_ms", unit: "ms" },
  { label: "BACKCHANNELS", key: "backchannel_count", unit: "count" },
  { label: "AUDIBLE CUES", key: "audible_backchannels", unit: "count" },
  { label: "CUES / LONG TURN", key: "backchannels_per_long_turn", unit: "ratio" },
  { label: "DELAYED RESPONSES", key: "delayed_responses", unit: "count" },
  { label: "COLLISIONS", key: "collision_events", unit: "count" },
  { label: "CANCELLED", key: "cancelled_backchannels", unit: "count" },
  { label: "JEV TIMEOUTS", key: "jev_timeouts", unit: "count" },
  { label: "EOT RISK", key: "end_of_turn_risks", unit: "count" },
  { label: "OVERLAP", key: "overlapping_backchannels", unit: "count" },
  { label: "UNPAIRED TURNS", key: "unpaired_turns", unit: "count" },
] as const;

type MetricUnit = "ms" | "count" | "ratio";

function reportMetric(summary: Summary | null | undefined, key: string) {
  const value = summary?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function formatReportMetric(value: number | null, unit: MetricUnit) {
  if (value === null) return "—";
  if (unit === "ms") return `${Math.round(value)}ms`;
  if (unit === "ratio") return value.toFixed(2);
  return String(Math.round(value));
}

function reportDelta(baseline: Summary | null | undefined, experiment: Summary | null | undefined, key: string, unit: MetricUnit) {
  const base = reportMetric(baseline, key);
  const next = reportMetric(experiment, key);
  if (base === null || next === null) return "—";
  const delta = next - base;
  const sign = delta > 0 ? "+" : "";
  if (unit === "ms") return `${sign}${Math.round(delta)}ms`;
  if (unit === "ratio") return `${sign}${delta.toFixed(2)}`;
  return `${sign}${Math.round(delta)}`;
}

export const LANES = ["user", "stt", "eot", "decision", "audio", "agent"] as const;
export type Lane = (typeof LANES)[number];

export type TraceMark = {
  key: string;
  kind: string;
  lane: Lane;
  left: number;
  width?: number;
  title: string;
};

/** Which lane a lifecycle event belongs to.
 *
 *  DECISION holds what the policy decided (including a suppressed or rejected
 *  cue); AUDIO holds what actually reached the user's ears. A previous version
 *  routed by a loose `includes("eot")` test, which hid suppressions among the
 *  end-of-turn signals and put them on the audio lane.
 */
export function laneFor(event: WorkerEvent): Lane | null {
  const name = event.name;
  if (name === "backchannel_decision" || name === "backchannel_suppressed_eot") return "decision";
  if (name.startsWith("backchannel_semantic")) return "decision";
  if (name.startsWith("backchannel")) return "audio";
  if (name.startsWith("user_speech")) return "user";
  if (name === "stt_transcript") return "stt";
  if (name.startsWith("eot")) return "eot";
  if (name.includes("decision") || name.startsWith("jev")) return "decision";
  if (name.startsWith("agent_response")) return "agent";
  return null;
}

export function markKind(event: WorkerEvent): string {
  if (event.name === "stt_transcript") return event.data.is_final === true ? "stt-final" : "stt-interim";
  return event.name.replaceAll("_", "-");
}

/** Build the per-lane marks (points plus real duration spans) for one run. */
export function buildTrace(events: WorkerEvent[], max: number) {
  const lanes: Record<Lane, TraceMark[]> = { user: [], stt: [], eot: [], decision: [], audio: [], agent: [] };
  const pct = (value: number) => (value / max) * 100;
  let index = 0;

  const spanFor = (startName: string, endName: string, label: string, lane: Lane) => {
    let start: WorkerEvent | undefined;
    for (const event of events) {
      if (event.name === startName) {
        start = event;
      } else if (event.name === endName && start) {
        const from = start.elapsed_ms || 0;
        const to = event.elapsed_ms || 0;
        lanes[lane].push({
          key: `span-${lane}-${index++}`,
          kind: `${lane}-span`,
          lane,
          left: pct(from),
          width: Math.max(0.4, pct(Math.max(0, to - from))),
          title: `${label} / ${Math.round(to - from)}ms`,
        });
        start = undefined;
      }
    }
  };

  spanFor("user_speech_started", "user_speech_ended", "USER SPEECH", "user");
  spanFor("agent_response_started", "agent_response_ended", "AGENT RESPONSE", "agent");
  spanFor("backchannel_audio_started", "backchannel_completed", "BACKCHANNEL AUDIO", "audio");

  for (const event of events) {
    const lane = laneFor(event);
    if (!lane) continue;
    const offset = event.elapsed_ms || 0;
    const reason = typeof event.data.reason === "string" ? event.data.reason : undefined;
    const duration = typeof event.data.duration_ms === "number" ? `${Math.round(event.data.duration_ms)}ms audible` : undefined;
    const probability = typeof event.data.probability === "number" ? `p=${event.data.probability.toFixed(2)}` : undefined;
    lanes[lane].push({
      key: `mark-${lane}-${index++}`,
      kind: markKind(event),
      lane,
      left: pct(offset),
      title: `${eventLabel(event.name)} / ${Math.round(offset)}ms${reason ? ` / ${reason}` : ""}${duration ? ` / ${duration}` : ""}${probability ? ` / ${probability}` : ""}`,
    });
  }
  return lanes;
}

function ComparisonTable({ scope, comparison, provenanceLabel }: { scope: string; comparison: Record<Mode, Summary | null> | undefined; provenanceLabel: string }) {
  return (
    <div className="comparison-block">
      <div className="comparison-scope"><span>{scope}</span><span className="provenance-chip">{provenanceLabel}</span></div>
      <div className="comparison-grid" role="table" aria-label={`Benchmark comparison (${scope})`}>
        <div className="comparison-grid-row comparison-grid-head" role="row"><span>METRIC</span><span>BASE</span><span>TIMER</span><span>JEV</span><span>ΔT</span><span>ΔJ</span></div>
        {comparisonRows.map((row) => <div className="comparison-grid-row" role="row" key={row.key}><span>{row.label}</span><span>{formatReportMetric(reportMetric(comparison?.baseline, row.key), row.unit)}</span><span>{formatReportMetric(reportMetric(comparison?.backchannel, row.key), row.unit)}</span><span>{formatReportMetric(reportMetric(comparison?.jev_backchannel, row.key), row.unit)}</span><span>{reportDelta(comparison?.baseline, comparison?.backchannel, row.key, row.unit)}</span><span>{reportDelta(comparison?.baseline, comparison?.jev_backchannel, row.key, row.unit)}</span></div>)}
      </div>
    </div>
  );
}

/** Load every recorded event for one run.
 *
 *  The live feed and the replay feed only hold a window of recent events, so a
 *  run selected from the report has to be fetched by id to draw its trace.
 */
async function fetchRunEvents(runId: string): Promise<WorkerEvent[]> {
  const response = await fetch(`/api/events?run_id=${encodeURIComponent(runId)}&limit=5000`, { cache: "no-store" });
  if (!response.ok) return [];
  return readEvents(await response.json());
}

export default function Workspace() {
  const [selected, setSelected] = useState(0);
  const [mode, setMode] = useState<Mode>("backchannel");
  const [micState, setMicState] = useState<MicState>("idle");
  const [micError, setMicError] = useState("");
  const [liveState, setLiveState] = useState<LiveState>("idle");
  const [liveMessage, setLiveMessage] = useState("Not connected to a LiveKit room");
  const [liveError, setLiveError] = useState("");
  const [agentSpeaking, setAgentSpeaking] = useState(false);
  const [userSpeaking, setUserSpeaking] = useState(false);
  const [feedView, setFeedView] = useState<"live" | "replay">("live");
  const [audioBlocked, setAudioBlocked] = useState(false);
  const [roomName, setRoomName] = useState<string | null>(null);
  const [liveRunId, setLiveRunId] = useState<string | null>(null);
  const [compareRunId, setCompareRunId] = useState<string | null>(null);
  const [fetchedRunEvents, setFetchedRunEvents] = useState<WorkerEvent[]>([]);
  const [fetchedCompareEvents, setFetchedCompareEvents] = useState<WorkerEvent[]>([]);
  const [events, setEvents] = useState<WorkerEvent[]>([]);
  const [replayEvents, setReplayEvents] = useState<WorkerEvent[]>([]);
  const [eventState, setEventState] = useState<"loading" | "ready" | "offline">("loading");
  const [eventError, setEventError] = useState("");
  const [report, setReport] = useState<BenchmarkReport | null>(null);
  const [reportState, setReportState] = useState<"loading" | "ready" | "offline">("loading");
  const [reportError, setReportError] = useState("");
  const [benchmarkState, setBenchmarkState] = useState<"idle" | "running" | "error">("idle");
  const [benchmarkError, setBenchmarkError] = useState("");
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [updatedAt, setUpdatedAt] = useState<string | null>(null);
  const [signalLevel, setSignalLevel] = useState<number | null>(null);
  const orb = useRef<HTMLDivElement>(null);
  const waveform = useRef<HTMLCanvasElement>(null);
  const audioHost = useRef<HTMLDivElement>(null);
  const generation = useRef(0);
  const signalUpdated = useRef(0);
  const resources = useRef<MicResources>({ stream: null, context: null, source: null, frame: 0 });
  const roomRef = useRef<Room | null>(null);
  const liveSignalResources = useRef<LiveSignalResources>({ context: null, source: null, frame: 0 });
  const scenario = scenarios[selected];

  const releaseMicrophone = useCallback(() => {
    const current = resources.current;
    resources.current = { stream: null, context: null, source: null, frame: 0 };
    cancelAnimationFrame(current.frame);
    current.source?.disconnect();
    current.stream?.getTracks().forEach((track) => track.stop());
    if (current.context && current.context.state !== "closed") void current.context.close().catch(() => {});
    orb.current?.style.removeProperty("--signal");
    setSignalLevel(null);
    const canvas = waveform.current;
    canvas?.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
  }, []);

  const releaseLiveSignal = useCallback(() => {
    const current = liveSignalResources.current;
    liveSignalResources.current = { context: null, source: null, frame: 0 };
    cancelAnimationFrame(current.frame);
    current.source?.disconnect();
    if (current.context && current.context.state !== "closed") void current.context.close().catch(() => {});
    orb.current?.style.removeProperty("--signal");
    setSignalLevel(null);
    const canvas = waveform.current;
    canvas?.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
  }, []);

  const startLiveSignal = useCallback(async (track: MediaStreamTrack) => {
    releaseLiveSignal();
    try {
      const context = new AudioContext();
      const analyser = context.createAnalyser();
      analyser.fftSize = 512;
      const source = context.createMediaStreamSource(new MediaStream([track]));
      source.connect(analyser);
      await context.resume();
      liveSignalResources.current = { context, source, frame: 0 };
      const samples = new Uint8Array(analyser.fftSize);
      const canvas = waveform.current;
      const drawing = canvas?.getContext("2d");
      function draw() {
        if (liveSignalResources.current.context !== context) return;
        analyser.getByteTimeDomainData(samples);
        let energy = 0;
        for (const sample of samples) energy += ((sample - 128) / 128) ** 2;
        const signal = Math.min(1, Math.sqrt(energy / samples.length) * 5);
        orb.current?.style.setProperty("--signal", String(signal));
        if (performance.now() - signalUpdated.current > 100) {
          signalUpdated.current = performance.now();
          setSignalLevel(signal);
        }
        if (canvas && drawing) {
          drawing.clearRect(0, 0, canvas.width, canvas.height);
          drawing.lineWidth = 2;
          drawing.strokeStyle = "#111111";
          drawing.beginPath();
          for (let index = 0; index < samples.length; index += 1) {
            const x = (index / (samples.length - 1)) * canvas.width;
            const y = canvas.height / 2 + ((samples[index] - 128) / 128) * canvas.height * 0.45;
            if (index === 0) drawing.moveTo(x, y);
            else drawing.lineTo(x, y);
          }
          drawing.stroke();
        }
        liveSignalResources.current.frame = requestAnimationFrame(draw);
      }
      draw();
    } catch {
      releaseLiveSignal();
    }
  }, [releaseLiveSignal]);

  const stopMicrophone = useCallback(() => {
    generation.current += 1;
    releaseMicrophone();
    setMicState("idle");
  }, [releaseMicrophone]);

  async function startMicrophone() {
    const request = ++generation.current;
    setMicError("");
    setMicState("pending");
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error("Microphone access needs localhost or HTTPS.");
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (request !== generation.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      resources.current.stream = stream;
      const context = new AudioContext();
      resources.current.context = context;
      const analyser = context.createAnalyser();
      analyser.fftSize = 512;
      const source = context.createMediaStreamSource(stream);
      resources.current.source = source;
      source.connect(analyser);
      await context.resume();
      if (request !== generation.current) return;
      if (context.state !== "running") throw new Error("Audio could not start. Try the microphone check again.");
      const ended = () => {
        if (request !== generation.current) return;
        stopMicrophone();
        setMicError("Microphone access ended. Reconnect the input device and try again.");
      };
      stream.getAudioTracks().forEach((track) => track.addEventListener("ended", ended, { once: true }));
      const samples = new Uint8Array(analyser.fftSize);
      const canvas = waveform.current;
      const drawing = canvas?.getContext("2d");
      function draw() {
        if (request !== generation.current) return;
        analyser.getByteTimeDomainData(samples);
        let energy = 0;
        for (const sample of samples) energy += ((sample - 128) / 128) ** 2;
        const signal = Math.min(1, Math.sqrt(energy / samples.length) * 5);
        orb.current?.style.setProperty("--signal", String(signal));
        if (performance.now() - signalUpdated.current > 100) {
          signalUpdated.current = performance.now();
          setSignalLevel(signal);
        }
        if (canvas && drawing) {
          drawing.clearRect(0, 0, canvas.width, canvas.height);
          drawing.lineWidth = 2;
          drawing.strokeStyle = "#111111";
          drawing.beginPath();
          for (let i = 0; i < samples.length; i += 1) {
            const x = (i / (samples.length - 1)) * canvas.width;
            const y = canvas.height / 2 + ((samples[i] - 128) / 128) * canvas.height * 0.45;
            if (i === 0) drawing.moveTo(x, y);
            else drawing.lineTo(x, y);
          }
          drawing.stroke();
        }
        resources.current.frame = requestAnimationFrame(draw);
      }
      setMicState("active");
      draw();
    } catch (error) {
      if (request !== generation.current) return;
      generation.current += 1;
      releaseMicrophone();
      setMicState("idle");
      const name = error instanceof DOMException ? error.name : "";
      setMicError(name === "NotAllowedError" ? "Microphone permission denied. Allow access in site settings." : name === "NotFoundError" ? "No microphone found. Connect an input device." : error instanceof Error ? error.message : "Could not open the microphone.");
    }
  }

  const stopConversation = useCallback(() => {
    const room = roomRef.current;
    roomRef.current = null;
    if (room) void room.disconnect();
    releaseLiveSignal();
    audioHost.current?.replaceChildren();
    setLiveState("idle");
    setLiveMessage("Not connected to a LiveKit room");
    setRoomName(null);
    setLiveRunId(null);
    setAgentSpeaking(false);
    setUserSpeaking(false);
    setAudioBlocked(false);
  }, [releaseLiveSignal]);

  async function startConversation() {
    if (liveState === "connecting" || liveState === "connected") return;
    if (micState !== "idle") stopMicrophone();
    setLiveState("connecting");
    setLiveError("");
    setLiveMessage("Requesting fresh room and agent dispatch...");
    try {
      const response = await fetch("/api/livekit/token", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ scenario_id: scenario.id, mode }), cache: "no-store" });
      const payload = await response.json() as Partial<LiveKitToken> & ApiFailure;
      // FastAPI reports failures as `detail`; the proxy passes it through unchanged.
      const failure = payload.detail || payload.error;
      if (!response.ok || !payload.server_url || !payload.participant_token || !payload.room_name) throw new Error(failure || `LiveKit token request failed (HTTP ${response.status}).`);
      const room = new Room({ adaptiveStream: true, dynacast: true });
      roomRef.current = room;
      room.on(RoomEvent.TrackSubscribed, (track: RemoteTrack) => {
        if (track.kind !== Track.Kind.Audio) return;
        const element = track.attach();
        element.setAttribute("aria-hidden", "true");
        element.className = "remote-audio";
        audioHost.current?.appendChild(element);
        void element.play().catch(() => setAudioBlocked(true));
      });
      room.on(RoomEvent.TrackUnsubscribed, (track: RemoteTrack) => track.detach().forEach((element) => element.remove()));
      room.on(RoomEvent.ParticipantConnected, () => setLiveMessage("Agent connected; input channel open."));
      room.on(RoomEvent.ActiveSpeakersChanged, (speakers) => {
        setUserSpeaking(speakers.some((participant) => participant.identity === room.localParticipant.identity));
        setAgentSpeaking(speakers.some((participant) => participant.identity !== room.localParticipant.identity));
      });
      room.on(RoomEvent.AudioPlaybackStatusChanged, () => setAudioBlocked(!room.canPlaybackAudio));
      room.on(RoomEvent.Disconnected, () => { if (roomRef.current === room) stopConversation(); });
      await room.connect(payload.server_url, payload.participant_token);
      await room.localParticipant.setMicrophoneEnabled(true);
      const localPublication = room.localParticipant.getTrackPublication(Track.Source.Microphone);
      const localTrack = localPublication?.track?.mediaStreamTrack;
      if (localTrack) await startLiveSignal(localTrack);
      await room.startAudio().catch(() => setAudioBlocked(true));
      setRoomName(payload.room_name);
      const nextRunId = typeof payload.run_id === "string" ? payload.run_id : null;
      setLiveRunId(nextRunId);
      setSelectedRunId(nextRunId);
      setFeedView("live");
      setLiveState("connected");
      setLiveMessage("LiveKit connected; room is listening.");
      // A worker can refuse the room (for example Jev mode without an interim
      // STT) and then nothing ever joins. Say so instead of leaving the room
      // silently empty.
      window.setTimeout(() => {
        if (roomRef.current !== room) return;
        const agentPresent = Array.from(room.remoteParticipants.values()).some((participant) => participant.identity !== room.localParticipant.identity);
        if (!agentPresent) {
          setLiveMessage("Connected, but no agent joined this room.");
          setLiveError("No agent joined. Check that the worker is running and that your policy is supported by the configured STT (Jev needs interim transcripts).");
        }
      }, 12000);
    } catch (error) {
      roomRef.current?.disconnect();
      roomRef.current = null;
      setLiveState("error");
      setLiveMessage("LiveKit did not connect");
      setLiveError(error instanceof Error ? error.message : "Could not connect to LiveKit.");
    }
  }

  async function changeMode(nextMode: Mode) {
    const label = nextMode === "baseline" ? "Baseline" : nextMode === "backchannel" ? "Timer backchannel" : "Backchannel + Jev";
    if (liveState === "connecting") {
      setLiveMessage("Wait for the room to connect before changing mode.");
      return;
    }
    if (liveState === "connected") {
      setLiveMessage(`${label} is available on the next run. End this conversation to change mode.`);
      return;
    }
    setMode(nextMode);
    setLiveMessage(`${label} selected.`);
  }

  const refreshEvents = useCallback(async () => {
    try {
      const response = await fetch("/api/events?limit=5000", { cache: "no-store" });
      if (!response.ok) throw new Error("The lifecycle event API is unavailable.");
      setEvents(readEvents(await response.json()));
      setEventState("ready");
      setUpdatedAt(new Date().toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" }));
    } catch (error) {
      setEventError(error instanceof Error ? error.message : "Could not load lifecycle events.");
      setEventState("offline");
    }
  }, []);

  const loadExperimentData = useCallback(async () => {
    setEventState("loading");
    setReportState("loading");
    setEventError("");
    setReportError("");
    try {
      const [reportResponse, eventsResponse, replayResponse] = await Promise.all([
        fetch("/api/benchmark/report", { cache: "no-store" }),
        fetch("/api/events?limit=5000", { cache: "no-store" }),
        fetch("/api/benchmark/events?limit=5000", { cache: "no-store" }),
      ]);
      if (!reportResponse.ok) throw new Error("Benchmark results are unavailable. Start the Python API on port 8000.");
      const nextReport = await reportResponse.json() as BenchmarkReport;
      setReport(nextReport);
      setReportState("ready");
      if (!eventsResponse.ok) throw new Error("The lifecycle event API is unavailable.");
      setEvents(readEvents(await eventsResponse.json()));
      setReplayEvents(replayResponse.ok ? readEvents(await replayResponse.json()) : []);
      setEventState("ready");
      setUpdatedAt(new Date().toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" }));
      const firstRun = nextReport.scenarios.find((row) => row.runs.length > 0)?.runs[0];
      setSelectedRunId((current) => current || (typeof firstRun?.run_id === "string" ? firstRun.run_id : null));
    } catch (error) {
      const message = error instanceof Error ? error.message : "Could not load experiment data.";
      setReportError(message);
      setEventError(message);
      setReportState("offline");
      setEventState("offline");
    }
  }, []);

  async function runReplay() {
    setBenchmarkState("running");
    setBenchmarkError("");
    try {
      const response = await fetch("/api/benchmark/replay", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ scenario_ids: scenarios.map((item) => item.id), repeats: 3, seed: 7 }) });
      const payload = await response.json() as BenchmarkReport & { error?: string };
      if (!response.ok) throw new Error(payload.error || "The replay runner failed.");
      setReport(payload);
      setReportState("ready");
      setBenchmarkState("idle");
      const replayResponse = await fetch("/api/benchmark/events?limit=5000", { cache: "no-store" });
      if (replayResponse.ok) setReplayEvents(readEvents(await replayResponse.json()));
      setFeedView("replay");
      const firstRun = payload.scenarios.find((row) => row.runs.length > 0)?.runs[0];
      setSelectedRunId(typeof firstRun?.run_id === "string" ? firstRun.run_id : null);
    } catch (error) {
      setBenchmarkState("error");
      setBenchmarkError(error instanceof Error ? error.message : "The replay runner failed.");
    }
  }

  useEffect(() => {
    void loadExperimentData();
    return () => {
      generation.current += 1;
      releaseMicrophone();
      releaseLiveSignal();
      roomRef.current?.disconnect();
      roomRef.current = null;
    };
  }, [loadExperimentData, releaseLiveSignal, releaseMicrophone]);

  useEffect(() => {
    if (liveState !== "connected") return;
    const interval = window.setInterval(() => { void refreshEvents(); }, 1200);
    return () => window.clearInterval(interval);
  }, [liveState, refreshEvents]);

  useEffect(() => {
    if (!selectedRunId) {
      setFetchedRunEvents([]);
      return;
    }
    let cancelled = false;
    void fetchRunEvents(selectedRunId)
      .then((loaded) => {
        if (!cancelled) setFetchedRunEvents(loaded);
      })
      .catch(() => {
        if (!cancelled) setFetchedRunEvents([]);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedRunId]);

  useEffect(() => {
    if (!compareRunId) {
      setFetchedCompareEvents([]);
      return;
    }
    let cancelled = false;
    void fetchRunEvents(compareRunId)
      .then((loaded) => {
        if (!cancelled) setFetchedCompareEvents(loaded);
      })
      .catch(() => {
        if (!cancelled) setFetchedCompareEvents([]);
      });
    return () => {
      cancelled = true;
    };
  }, [compareRunId]);

  const selectedReport = report?.scenarios.find((row) => row.scenario_id === scenario.id);
  const sessionEvents = useMemo(() => {
    const filtered = liveRunId ? events.filter((event) => event.run_id === liveRunId) : events.filter((event) => !event.run_id || event.scenario_id === scenario.id);
    return filtered.sort((left, right) => eventSortValue(left) - eventSortValue(right));
  }, [events, liveRunId, scenario.id]);
  useEffect(() => {
    const runs = selectedReport?.runs ?? [];
    if (runs.length === 0) return;
    const ids = runs.map((run) => String(run.run_id));
    // Keep the trace pointed at a run of the scenario on screen, and prefer the
    // most recent one: an old, quieter run should not be the first thing a
    // reader sees.
    if (selectedRunId && ids.includes(selectedRunId)) return;
    setSelectedRunId(ids[ids.length - 1]);
  }, [selectedReport, selectedRunId]);

  const timelineEvents = useMemo(() => {
    const local = selectedRunId
      ? [...events, ...replayEvents].filter((event) => event.run_id === selectedRunId)
      : liveRunId
        ? events.filter((event) => event.run_id === liveRunId)
        : replayEvents.filter((event) => event.scenario_id === scenario.id);
    const source = local.length > 0 ? local : selectedRunId ? fetchedRunEvents : [];
    return source.filter((event) => event.elapsed_ms !== undefined).sort((left, right) => (left.elapsed_ms || 0) - (right.elapsed_ms || 0));
  }, [events, fetchedRunEvents, liveRunId, replayEvents, scenario.id, selectedRunId]);
  const liveFeedEvents = useMemo(() => [...events].sort((left, right) => eventSortValue(right) - eventSortValue(left)).slice(0, 8), [events]);
  const replayFeedEvents = useMemo(() => [...replayEvents].sort((left, right) => eventSortValue(right) - eventSortValue(left)).slice(0, 8), [replayEvents]);
  const feedEvents = feedView === "live" ? liveFeedEvents : replayFeedEvents;
  const feedLabel = feedView === "live" ? (liveRunId ? "ACTIVE ROOM" : "RECORDED LIVE") : "DETERMINISTIC REPLAY";
  const latestEvent = sessionEvents.at(-1);
  const latestTranscript = [...sessionEvents].reverse().find((event) => event.name === "stt_transcript");
  const latestMetric = [...sessionEvents].reverse().find((event) => event.name === "pipeline_metric");
  const benchmarkComparison = selectedReport?.comparison;
  const latestVoiceEvent = [...sessionEvents].reverse().find((event) => event.name === "user_speech_started" || event.name === "user_speech_ended");
  // A speech provider that rejects audio leaves the agent silent with a perfectly
  // good answer in hand, which reads as "the agent is broken". Say what happened.
  const speechFailure = sessionEvents.some((event) => event.name === "session_error" && event.data.error_type === "TTSError");
  const timelineMax = Math.max(1000, ...timelineEvents.map((event) => event.elapsed_ms || 0));
  const compareEvents = useMemo(() => {
    if (!compareRunId) return [];
    const local = [...events, ...replayEvents].filter((event) => event.run_id === compareRunId);
    const source = local.length > 0 ? local : fetchedCompareEvents;
    return source
      .filter((event) => event.elapsed_ms !== undefined)
      .sort((left, right) => (left.elapsed_ms || 0) - (right.elapsed_ms || 0));
  }, [compareRunId, events, fetchedCompareEvents, replayEvents]);
  const traceMax = Math.max(timelineMax, ...compareEvents.map((event) => event.elapsed_ms || 0));
  const primaryTrace = useMemo(() => buildTrace(timelineEvents, traceMax), [timelineEvents, traceMax]);
  const compareTrace = useMemo(() => (compareEvents.length === 0 ? null : buildTrace(compareEvents, traceMax)), [compareEvents, traceMax]);
  const reportMode = report?.provenance?.label || report?.source || "NO REPORT";
  const voiceActivity = liveState === "connected" ? agentSpeaking ? "AGENT SPEAKING" : userSpeaking ? "USER SPEAKING" : "ROOM LISTENING" : latestVoiceEvent?.name === "user_speech_started" ? "USER SPEAKING (RECORDED)" : latestVoiceEvent?.name === "user_speech_ended" ? "USER FLOOR OPEN" : "NO VOICE ACTIVITY";
  const providerState = liveState === "connected" ? latestMetric ? metricLabel(latestMetric) : "WAITING FOR PROVIDER METRIC" : latestMetric ? metricLabel(latestMetric) : "NO PROVIDER STATE";
  const connectionState = liveState === "connected" ? "CONNECTED" : liveState === "connecting" ? "CONNECTING" : liveState === "error" ? "ERROR" : "IDLE";
  const signalText = signalLevel === null ? "SIGNAL —" : `SIGNAL ${(signalLevel * 100).toFixed(1)}%`;

  return (
    <div className="console-shell">
      <a className="skip-link" href="#workspace">SKIP TO CONSOLE</a>
      <header className="console-header">
        <div className="brand-lockup"><span className="brand-mark" aria-hidden="true">///</span><span>BLUE MACHINES<span className="brand-period">.</span></span><span className="brand-slash">/ VOICE OPS</span></div>
        <div className="header-readout"><span className={`status-square ${liveState === "connected" ? "status-on" : ""}`} /> LINK {connectionState} <span className="header-divider">|</span> API {eventState === "ready" ? "ONLINE" : eventState === "loading" ? "POLLING" : "OFFLINE"}</div>
      </header>

      <main id="workspace" className="console-main">
        <section className="console-toolbar" aria-label="Console controls">
          <div><p className="kicker">LIVEKIT / MEASURED VOICE RUNTIME</p><h1>VOICE OPERATIONS CONSOLE</h1></div>
          <div className="toolbar-actions"><span className="toolbar-clock">{updatedAt ? `LAST POLL ${updatedAt}` : "WAITING FOR POLL"}</span><button className="square-button" onClick={() => void loadExperimentData()} disabled={eventState === "loading"} aria-label="Refresh all data"><Icon name="refresh" /></button></div>
        </section>

        <div className="console-grid">
          <section className="panel session-panel" aria-labelledby="session-heading">
            <div className="panel-head"><h2 id="session-heading">01 / SESSION</h2><span className={`panel-state ${liveState === "connected" ? "panel-state-on" : ""}`}>{connectionState}</span></div>
            <div className="session-body">
              <div className="session-title"><span className={`status-square ${liveState === "connected" ? "status-on" : ""}`} /><div><strong>{liveState === "connected" ? roomName || "LIVEKIT ROOM" : "NO ACTIVE ROOM"}</strong><small>{liveMessage}</small></div></div>
              <div className="signal-frame"><div className="signal-topline"><span>{liveState === "connected" ? "INPUT SIGNAL / LIVEKIT MIC" : "INPUT SIGNAL / LOCAL ANALYSER"}</span><span>{signalText}</span></div><div className={`signal-plate ${signalLevel !== null ? "signal-live" : ""}`}><div className="signal-grid" /><div className="signal-bars" aria-hidden="true"><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /></div><canvas className="waveform" width="640" height="110" ref={waveform} aria-label="Microphone signal waveform" /></div><div className="signal-foot"><span>MIC {liveState === "connected" ? "PUBLISHED" : micState === "active" ? "CAPTURING" : "NOT CAPTURING"}</span><span>{liveState === "connected" ? "LIVEKIT AUDIO TRACK" : "LOCAL CHECK ONLY"}</span></div></div>
              <div className="state-readouts"><div><span className="readout-label">VOICE ACTIVITY</span><strong>{voiceActivity}</strong></div><div><span className="readout-label">AGENT AUDIO</span><strong>{audioBlocked ? "BLOCKED" : liveState === "connected" ? agentSpeaking ? "PLAYING" : "READY" : "OFFLINE"}</strong></div><div><span className="readout-label">LAST EVENT</span><strong>{latestEvent ? eventLabel(latestEvent.name) : "—"}</strong></div></div>
              <div className="session-controls"><button className="primary-control" onClick={liveState === "connected" ? stopConversation : startConversation} disabled={liveState === "connecting"}>{liveState === "connected" ? "END CONVERSATION" : liveState === "connecting" ? "CONNECTING" : "START CONVERSATION"}<Icon name={liveState === "connected" ? "stop" : "play"} /></button><button className="secondary-control" onClick={micState === "idle" ? startMicrophone : stopMicrophone} disabled={liveState === "connected"}>{micState === "active" ? "STOP MIC CHECK" : micState === "pending" ? "CANCEL REQUEST" : "CHECK MICROPHONE"}<Icon name={micState === "idle" ? "mic" : "stop"} /></button>{audioBlocked && <button className="audio-unblock" onClick={() => void roomRef.current?.startAudio()}>ENABLE AUDIO</button>}</div>
              {(micError || liveError) && <p className="error-line" role="alert">ERR / {micError || liveError}</p>}
              {speechFailure && !liveError && <p className="error-line" role="alert">SPEECH PROVIDER REJECTED THE AUDIO / the agent generated a reply but could not speak it - check the speech quota (Groq allows 3600 speech tokens per day). The cue audio is pre-generated, so backchannels still play.</p>}
              <p className="privacy-line"><Icon name="lock" size={12} /> AUDIO SENT TO LIVEKIT ONLY DURING A CONVERSATION</p>
              <div className="remote-audio-host" ref={audioHost} aria-hidden="true" />
            </div>
          </section>

          <aside className="panel protocol-panel" aria-labelledby="protocol-heading">
            <div className="panel-head"><h2 id="protocol-heading">02 / PROTOCOL</h2><span>{String(selected + 1).padStart(2, "0")} / 08</span></div>
            <div className="protocol-selector"><span className="kicker">SCENARIO TARGET</span><strong>{scenario.title}</strong><small>{scenario.description}</small><p>{scenario.action}</p></div>
            <div className="scenario-list">{scenarios.map((item, index) => <button key={item.id} className={`scenario-row ${selected === index ? "scenario-selected" : ""}`} onClick={() => setSelected(index)} aria-pressed={selected === index}><span>{String(index + 1).padStart(2, "0")}</span><strong>{item.title}</strong><small>{item.tag}</small><Icon name="arrow" size={12} /></button>)}</div>
            <div className="mode-block"><span className="kicker">AGENT POLICY / {liveState === "connected" ? "RUN LOCKED" : "SELECT BEFORE CONNECT"}</span><div className="mode-grid">{(["baseline", "backchannel", "jev_backchannel"] as Mode[]).map((item) => <button key={item} className={mode === item ? "mode-selected" : ""} disabled={liveState === "connecting" || liveState === "connected"} title={liveState === "connected" ? "Mode is fixed for the active run" : undefined} onClick={() => void changeMode(item)}>{item === "baseline" ? "BASE" : item === "backchannel" ? "TIMER" : "JEV"}</button>)}</div></div>
          </aside>

          <section className="panel transcript-panel" aria-labelledby="transcript-heading">
            <div className="panel-head"><h2 id="transcript-heading">03 / TRANSCRIPT</h2><span>{latestTranscript?.data.is_final === true ? "FINAL" : latestTranscript ? "INTERIM" : "NO DATA"}</span></div>
            <div className="transcript-readout"><span className="prompt-symbol">&gt;_</span><p>{latestTranscript && typeof latestTranscript.data.transcript === "string" ? latestTranscript.data.transcript : "Transcript text is not emitted by the worker event contract."}</p></div>
            <div className="transcript-meta"><span>{transcriptLabel(latestTranscript)}</span><span>{latestTranscript?.elapsed_ms === undefined ? "OFFSET —" : `OFFSET ${Math.round(latestTranscript.elapsed_ms)} MS`}</span></div>
            <div className="event-rail"><span>STT EVENT RAIL</span>{sessionEvents.filter((event) => event.name === "stt_transcript").slice(-5).map((event, index) => <i key={`${event.name}-${event.elapsed_ms}-${index}`} className={event.data.is_final === true ? "rail-final" : ""} />)}</div>
          </section>

          <section className="panel provider-panel" aria-labelledby="provider-heading">
            <div className="panel-head"><h2 id="provider-heading">04 / PROVIDER + REPORT</h2><span className={latestMetric ? "panel-state-on" : ""}>{latestMetric ? "OBSERVED" : "WAITING"}</span></div>
            <div className="provider-scroll">
              <div className="provider-report-source">{liveState === "connected" ? `LIVE ROOM / ${sessionEvents.length} EVENTS · BENCHMARK BELOW IS RECORDED` : `BENCHMARK / ${reportMode}`}</div>
              <div className="provider-focus"><span className="kicker">{liveState === "connected" ? "LIVE ROOM TELEMETRY" : "LATEST TELEMETRY"}</span><strong>{providerState}</strong><small>{liveState === "connected" ? `${shortMode(mode)} / ${sessionEvents.length} LIVE EVENTS` : `${reportMode} / ${reportState === "ready" ? `${report?.run_count || 0} RUNS` : reportState.toUpperCase()}`}</small></div>
              <ComparisonTable scope="ALL SCENARIOS (AGGREGATE)" comparison={report?.overall} provenanceLabel={reportMode} />
              <ComparisonTable scope={`SCENARIO / ${scenario.title.toUpperCase()}`} comparison={benchmarkComparison} provenanceLabel={reportMode} />
              {report?.provenance && <p className="provider-note">BENCHMARK SOURCE / {report.provenance.provider_latency_available ? "PROVIDER LATENCY MEASURED" : "PROVIDER LATENCY NOT REPORTED"} / {report.provenance.note}</p>}
              {report?.replay_observation?.provenance && <p className="provider-note">REPLAY BASELINE ALSO AVAILABLE / {report.replay_observation.provenance.label} / {report.replay_observation.provenance.note}</p>}
              {reportError && <p className="error-line" role="alert">ERR / {reportError}</p>}
            </div>
          </section>

          <section className="panel feed-panel" aria-labelledby="feed-heading">
            <div className="panel-head"><h2 id="feed-heading">05 / FAST EVENT FEED</h2><div className="feed-switch" role="tablist" aria-label="Event source"><button className={feedView === "live" ? "feed-switch-selected" : ""} onClick={() => setFeedView("live")} role="tab" aria-selected={feedView === "live"}>LIVE</button><button className={feedView === "replay" ? "feed-switch-selected" : ""} onClick={() => setFeedView("replay")} role="tab" aria-selected={feedView === "replay"}>REPLAY</button></div></div>
            <div className="feed-source">{feedLabel} {feedView === "live" && eventState === "ready" ? "· POLL 1.2S" : ""}</div>
            <div className="feed-list" aria-busy={feedView === "live" && eventState === "loading"}>{feedView === "live" && eventState === "offline" && <p className="empty-line">{eventError}</p>}{feedView === "live" && eventState === "loading" && <p className="empty-line">FETCHING LIFECYCLE EVENTS...</p>}{feedEvents.length === 0 && !(feedView === "live" && eventState === "loading") && <p className="empty-line">NO {feedLabel} EVENTS</p>}{feedEvents.map((event, index) => <div className="feed-row" key={`${event.timestamp || event.run_id || "event"}-${event.name}-${event.elapsed_ms}-${index}`}><span className="feed-index">{String(index + 1).padStart(2, "0")}</span><span className="feed-marker" /><strong>{eventLabel(event.name)}</strong><span className="feed-mode">{shortMode(event.mode)}</span><span className="feed-offset">{event.elapsed_ms === undefined ? "—" : `${Math.round(event.elapsed_ms)}ms`}</span><time>{eventTime(event.timestamp)}</time></div>)}</div>
          </section>

          <section className="panel timeline-panel" aria-labelledby="timeline-heading">
            <div className="panel-head"><h2 id="timeline-heading">06 / RUN TRACE</h2><label>RUN <select value={selectedRunId || ""} onChange={(event) => setSelectedRunId(event.target.value || null)}><option value="">RECORDED SCENARIO</option>{liveRunId && <option value={liveRunId}>LIVE / {liveRunId.slice(-8)}</option>}{(selectedReport?.runs || []).map((run) => <option key={String(run.run_id)} value={String(run.run_id)}>{String(run.mode).toUpperCase()} / {String(run.run_id).slice(-8)}</option>)}</select></label></div>
            <div className="trace-controls">
              <label>COMPARE <select value={compareRunId || ""} onChange={(event) => setCompareRunId(event.target.value || null)}><option value="">OFF</option>{(selectedReport?.runs || []).filter((run) => run.run_id !== selectedRunId).map((run) => <option key={String(run.run_id)} value={String(run.run_id)}>{String(run.mode).toUpperCase()} / {String(run.run_id).slice(-8)}</option>)}</select></label>
              <span className="trace-legend"><i className="legend-primary" /> PRIMARY <i className="legend-compare" /> COMPARE</span>
            </div>
            {timelineEvents.length === 0 ? <p className="empty-line">NO RUN TRACE. START A CONVERSATION OR RUN REPLAY.</p> : <><div className="trace-axis"><span>0MS</span><span>{Math.round(timelineMax)}MS</span></div><div className="trace-lanes">{LANES.map((lane) => <div className="trace-lane" key={lane}><span>{lane.toUpperCase()}</span><div className="trace-track">{primaryTrace[lane].map((mark) => <i key={mark.key} className={`trace-${mark.kind}`} style={{ left: `${mark.left}%`, width: mark.width === undefined ? undefined : `${mark.width}%` }} title={mark.title} />)}{compareTrace?.[lane].map((mark) => <i key={`compare-${mark.key}`} className={`trace-${mark.kind} trace-compare`} style={{ left: `${mark.left}%`, width: mark.width === undefined ? undefined : `${mark.width}%` }} title={`COMPARE / ${mark.title}`} />)}</div></div>)}</div></>}
          </section>
        </div>
      </main>
      <footer className="console-footer"><span>BLUE MACHINES / LOCAL VOICE LAB</span><span>LIVE / REPLAY · SOURCE-LABELLED DATA</span><span>{benchmarkState === "error" ? `ERR / ${benchmarkError}` : benchmarkState === "running" ? "REPLAY RUNNING..." : <button onClick={() => void runReplay()}>RUN 3X DETERMINISTIC REPLAY <Icon name="play" size={11} /></button>}</span></footer>
    </div>
  );
}
