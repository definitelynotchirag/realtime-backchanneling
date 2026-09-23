# Voice-agent backchanneling: research brief for the Blue Machines SDE-1 assignment

## Bottom line

Backchanneling works when it is treated as a small, timed listener signal—not as an ordinary assistant turn. The strongest 10–14 hour submission would combine: (1) conservative acknowledgement gating, (2) immediate and correct barge-in cancellation, and (3) a small evaluation harness that measures timing and false stops. A compact bank of pre-recorded acknowledgement clips is a sensible latency optimization; it should be presented as an engineering choice, not as an OpenAI implementation detail.

## What is publicly documented about OpenAI Realtime

- **Turn detection is configurable at the API boundary.** OpenAI documents `server_vad`, which detects speech start/stop from audio volume, with `threshold`, `prefix_padding_ms`, and `silence_duration_ms`; shorter silence can reduce response delay but can fire during short pauses. It also documents `semantic_vad`, which uses a turn-detection model to estimate whether the user has finished and can wait longer when speech trails off or sounds incomplete. Its documented eagerness levels have maximum timeouts of 8 seconds (`low`), 4 seconds (`medium`), and 2 seconds (`high`). ([OpenAI Realtime API reference](https://platform.openai.com/docs/api-reference/realtime-client-events/session?lang=node.js), current API reference; see `turn_detection`.)

- **Response creation and interruption are separately controllable.** The same reference documents `create_response` and `interrupt_response`: VAD can still emit speech events while the application decides whether to create a response or cancel an in-progress one. `idle_timeout_ms` can trigger a response after an unexpected pause, measured after the previous response’s audio has finished playing. ([OpenAI Realtime API reference](https://platform.openai.com/docs/api-reference/realtime-client-events/session?lang=node.js).)

- **The public SDK describes a concrete cancellation path.** With VAD enabled, a user speaking over the agent can interrupt. The Agents SDK says the WebSocket path listens for `input_audio_buffer.speech_started`, truncates assistant audio to what the user actually heard, and emits `audio_interrupted`; WebRTC clears buffered output audio automatically, while WebSocket applications must stop local playback themselves. ([OpenAI Agents SDK, “Building Voice Agents”](https://openai.github.io/openai-agents-js/guides/voice-agents/build/), current guide.) This is the right public contract to build against: stop playback, cancel/interrupt the active response, and keep conversation state aligned with audio actually heard.

- **OpenAI publicly supports streaming speech and automatic interruptions, but does not publicly specify a separate backchannel policy.** The Realtime launch post describes direct streaming audio and automatic interruption handling. The public API reference allows instructions about audio behavior such as speaking speed and expressiveness, but says instructions are guidance, not guarantees. I found no public OpenAI specification that exposes ChatGPT’s private acknowledgement classifier, clip bank, or internal timing policy. Any claim about those internals would be out of scope. ([Introducing the Realtime API](https://openai.com/index/introducing-the-realtime-api/), OpenAI, 1 October 2024; [Realtime API reference](https://platform.openai.com/docs/api-reference/realtime?lang=javascript).)

## Findings from broader speech and dialogue research

### Timing and cues

- Human turn changes are fast enough that listeners must anticipate turn endings: average gaps are on the order of 200 ms, while language production takes substantially longer. Prosodic, syntactic, and pragmatic cues therefore matter before silence arrives. ([Levinson & Torreira, “Timing in turn-taking and its implications for processing models of language,” *Frontiers in Psychology*, 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4464110/).)

- Backchannels are often deliberately overlapping but brief. In the same study’s quantitative analysis, overlaps were short and largely involved backchannels rather than full turns; between-overlaps had a modal duration of 96 ms and occupied less than 5% of the speech signal. ([Levinson & Torreira, 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4464110/).) This supports a policy of allowing a tiny acknowledgement overlap while suppressing substantive speech.

- Prosody is useful but not sufficient by itself. Benus, Gravano, and Hirschberg found that affirmative backchannels differ prosodically from other affirmative words, including higher pitch/intensity and stronger pitch slope; phrase-final rising pitch was a salient trigger. ([Benus, Gravano & Hirschberg, “The Prosody of Backchannels in American English,” *Proceedings of the 16th ICPhS*, 2007](https://www.icphs2007.de/conference/Papers/1276/index.html).) More recent work likewise finds that backchannel timing depends jointly on prosody and turn-taking function, not one acoustic threshold. ([Kelterer & Schuppler, “Distribution and Timing of Verbal Backchannels in Conversational Speech: A Quantitative Study,” *Languages*, 2025](https://www.mdpi.com/2226-471X/10/8/194).)

- A small, feasible model can look ahead instead of waiting for end-of-turn. Lala et al. predicted a backchannel 500 ms into the future from pitch, intensity, silence, voice-activity, and overlap features, making predictions every 100 ms. Their time-based model reported AUC 0.851, precision 0.344, recall 0.889, and F1 0.496; in a 7-point listening study it outperformed a fixed-after-IPU baseline and was comparable to the counselor condition on the reported ratings. The authors explicitly conclude that continuous timing is preferable to waiting for utterance endpoints. ([Lala et al., “Attentive listening system with backchanneling, response generation and flexible turn-taking,” *SIGdial*, 2017, pp. 127–136](https://aclanthology.org/W17-5516/).)

- Recent work treats backchanneling as an imbalanced, frame-level prediction problem. Inoue et al. fine-tuned a Voice Activity Projection model on Japanese dialogue; their proposed multi-task/pre-trained model achieved frame-level F1 42.85, precision 32.52, and recall 62.80, and ran with real-time factor below 1.0. The study also found assessment backchannels harder than simple continuers and warns that its Japanese-only data limits generalization. ([Inoue, Lala, Skantze & Kawahara, “Yeah, Un, Oh: Continuous and Real-time Backchannel Prediction with Fine-tuning of Voice Activity Projection,” *NAACL-HLT*, 2025, pp. 7171–7181](https://aclanthology.org/2025.naacl-long.367/).)

### End-of-turn suppression and barge-in

- Acknowledgement timing must preserve the current speaker’s floor. The attentive-listening system generated a response in advance but only output a substantive response when its turn-taking model predicted that the user intended to yield; this avoids interrupting a user who wants to continue. ([Lala et al., 2017](https://aclanthology.org/anthology-files/anthology-files/pdf/W/W17/W17-5516.pdf), especially the system-flow and evaluation sections.)

- VAD alone is not enough for barge-in. Selfridge et al. describe false barge-ins from noise, background speech, and unstable partial recognition, and propose continuously deciding whether to pause, continue, or resume the system prompt. Their SIGdial 2013 system improved task success and efficiency over a standard single-stage barge-in baseline. ([Selfridge, Arizmendi, Heeman & Williams, “Continuously Predicting and Processing Barge-in During a Live Spoken Dialogue Task,” *SIGdial*, 2013, pp. 384–393](https://aclanthology.org/W13-4063.pdf).)

- Evaluation should separate interruption from listener backchannel, side conversation, and ambient speech. Full-Duplex-Bench v1.5 proposes exactly those four overlap scenarios and reports a metric suite including categorical behavior, stop latency, response latency, prosodic adaptation, and perceived speech quality. ([Full-Duplex-Bench v1.5, author preprint, 2025](https://arxiv.org/abs/2507.23159).)

- There is evidence that full-duplex handling can improve latency, but it is not evidence that every agent should speak more often. Lin et al.’s *Duplex Conversation* system separated user-state detection, backchannel selection, and barge-in detection and reported a 50% response-latency reduction in online A/B experiments. ([Lin et al., “Duplex Conversation: Towards Human-like Interaction in Spoken Dialogue Systems,” author preprint, 2022](https://arxiv.org/abs/2205.15060).) The result is directionally useful, but the system, domain, and production stack differ from a take-home Realtime prototype.

### Cached versus generated acknowledgements

- **Established practice:** research systems often use a constrained acknowledgement repertoire. *Duplex Conversation* describes mining/counting suitable backchannel responses and selecting from a limited set; Lala et al.’s listening evaluation used a recorded backchannel pattern. ([Lin et al., 2022](https://arxiv.org/abs/2205.15060); [Lala et al., 2017](https://aclanthology.org/anthology-files/anthology-files/pdf/W/W17/W17-5516.pdf).)

- **Inference for this assignment:** local pre-recorded clips should reduce acknowledgement start latency and make cancellation predictable, while generated speech can express more context but risks arriving after the relevant moment. A hybrid is therefore attractive: use a tiny local bank for non-content-bearing signals (`mm-hm`, `yeah`, `right`, `oh`) and reserve generated speech for actual answers or content-sensitive reactions. This is an engineering inference from the constrained-repertoire evidence, not a claim about OpenAI’s internal implementation.

## 10–14 hour differentiators

| Differentiator | Effort | Evidence / metric | Main risk |
|---|---:|---|---|
| **Floor-preserving acknowledgement gate.** Detect an opportunity from short rolling audio/prosody windows plus partial transcript cues; require a continuation/hold signal, impose a refractory window, and suppress on likely end-of-turn. | 2–3 h | Backchannel precision/recall or F1; acknowledgements per minute; user-rated “did not interrupt me.” Compare against fixed-after-silence. | Over-suppression makes the agent feel silent; prosody varies by speaker and language. |
| **Tiny pre-recorded acknowledgement bank.** Keep 3–5 short clips with a small amount of variation; choose only non-content-bearing forms and play them without starting a full model response. | 1.5–2 h | Time from gate decision to audible clip; repeated-clip rate; naturalness rating; no transcript pollution. | Repetition, speaker mismatch, or a clip colliding with user speech. |
| **Two-sided interruption transaction.** On confirmed barge-in, stop local playback immediately, cancel/interrupt the active response, and reconcile assistant history to the audio actually played. | 3–4 h | Stop latency p50/p95; false-stop rate on `mm-hm`, cough, noise, and side speech; recovery correctness after “No, I meant…”. | Race conditions between playback, cancellation, and conversation events; WebSocket buffering. |
| **Explicit overlap test matrix.** Test continuation, listener backchannel, real interruption, ambient/side speech, and repeated interruptions with timestamped logs. | 1.5–2.5 h | One table of stop latency, response latency, false stops, missed interruptions, and recovery outcome; short before/after audio examples. | Instrumentation can consume time if it becomes a dashboard project. |
| **Small adaptive policy, not a new model.** Expose “conservative / balanced / eager” modes mapped to documented VAD controls and local gating. | 1–2 h | Same scenario matrix across modes; plot latency versus false-stop rate; show a deliberate trade-off. | Tuning can be overfit to a few recordings; document that it is a policy layer. |

## Recommendation

Make the submission stand out with a narrow, evidence-led story: “the agent distinguishes a listener backchannel from a real floor grab, yields instantly when the user takes the floor, and proves both behaviors with measurements.” Use the documented Realtime VAD/interruption controls, add one small local acknowledgement path, and spend the final time on a four-scenario evaluation matrix and a clear latency/false-stop trade-off. Do not spend the take-home budget training a backchannel model or trying to reproduce undocumented OpenAI behavior; the public evidence supports a careful policy layer with strong instrumentation.

## Sources

1. OpenAI. *Realtime API reference* — turn detection, VAD, response creation, interruption, idle timeout, and audio behavior. https://platform.openai.com/docs/api-reference/realtime-client-events/session?lang=node.js
2. OpenAI Agents SDK. *Building Voice Agents* — VAD, interruption events, transport differences, truncation, and manual response control. https://openai.github.io/openai-agents-js/guides/voice-agents/build/
3. OpenAI. *Introducing the Realtime API*. 1 October 2024. https://openai.com/index/introducing-the-realtime-api/
4. Lala, D. et al. 2017. *Attentive listening system with backchanneling, response generation and flexible turn-taking*. SIGdial, 127–136. https://aclanthology.org/W17-5516/
5. Inoue, K. et al. 2025. *Yeah, Un, Oh: Continuous and Real-time Backchannel Prediction with Fine-tuning of Voice Activity Projection*. NAACL-HLT, 7171–7181. https://aclanthology.org/2025.naacl-long.367/
6. Selfridge, E. O. et al. 2013. *Continuously Predicting and Processing Barge-in During a Live Spoken Dialogue Task*. SIGdial, 384–393. https://aclanthology.org/W13-4063.pdf
7. Levinson, S. C. & Torreira, F. 2015. *Timing in turn-taking and its implications for processing models of language*. Frontiers in Psychology 6:731. https://pmc.ncbi.nlm.nih.gov/articles/PMC4464110/
8. Benus, S., Gravano, A. & Hirschberg, J. 2007. *The Prosody of Backchannels in American English*. ICPhS 2007. https://www.icphs2007.de/conference/Papers/1276/index.html
9. Kelterer, A. & Schuppler, B. 2025. *Distribution and Timing of Verbal Backchannels in Conversational Speech: A Quantitative Study*. Languages 10(8):194. https://www.mdpi.com/2226-471X/10/8/194
10. Lin, T.-E. et al. 2022. *Duplex Conversation: Towards Human-like Interaction in Spoken Dialogue Systems*. Author preprint. https://arxiv.org/abs/2205.15060
11. *Full-Duplex-Bench v1.5: Evaluating Overlap Handling for Full-Duplex Speech Models*. Author preprint, 2025. https://arxiv.org/abs/2507.23159
