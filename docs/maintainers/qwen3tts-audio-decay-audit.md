# Qwen3-TTS Audio Decay Audit

## Why this note exists

This repo currently has two separate Qwen3-TTS decode behaviors:

1. The tokenizer's ordinary decode surface uses bounded chunk replay with explicit left context.
2. The generation code path that most callers hit decodes through the incremental streaming decoder.

That split matters because the incremental decoder keeps causal state alive across the entire utterance, while the bounded decode path replays only a fixed amount of left context per chunk. If the incremental state drifts over long generations, we can get exactly the kind of symptom that prompted this audit: later audio loses energy even though earlier audio is healthy.

## Finding 1: ordinary Qwen3-TTS generation is not using the tokenizer's bounded decode path

### What the code is doing

`Qwen3TTSModel.generateVoiceDesign(...)` builds the full codec stream, then calls `decodeChunk(...)` for non-streaming output.

`decodeChunk(...)` does not call the tokenizer's regular `decode(...)` API. Instead, it loops over `speechTokenizer.streamingDecode(...)` and concatenates the streamed audio chunks back together.

### Why that is risky

The tokenizer already has a normal decode path:

- `Qwen3TTSSpeechTokenizer.decode(...)`
- `Qwen3TTSSpeechTokenizerDecoder.chunkedDecode(...)`

That path explicitly replays bounded left context for each decode chunk:

- default `chunkSize`: `300`
- default `leftContextSize`: `25`

The incremental path is materially different:

- `Qwen3TTSSpeechTokenizer.streamingDecode(...)`
- `Qwen3TTSSpeechTokenizerDecoder.streamingStep(...)`

That path keeps decoder state alive across the whole utterance.

So the current "offline" generation path is not actually exercising the tokenizer's bounded decode behavior. It is exercising the streaming decoder and then stitching those chunks together. If the streaming decoder drifts, ordinary full-utterance generation drifts too.

### Concrete code anchors

- `Sources/MLXAudioTTS/Models/Qwen3TTS/Qwen3TTS.swift`
- `Sources/MLXAudioTTS/Models/Qwen3TTS/Qwen3TTSSpeechTokenizer.swift`

## Finding 2: the incremental decoder ignores the configured sliding-window boundary

### What the code is doing

`Qwen3TTSTokenizerDecoderConfig` carries a `slidingWindow` field.

`DecoderTransformer.callAsFunction(...)` uses the cache offset to build ever-increasing position IDs and attention masks when a cache is present.

`Qwen3TTSSpeechTokenizerDecoder.streamingStep(...)` creates the cache once and keeps reusing it for the full decode.

There is no trim or windowing step that enforces `slidingWindow` inside the incremental decoder path.

### Why that is risky

The bounded decode path approximates a local-context decode by replaying only a small left context for each chunk. The streaming decoder does not do that. It carries the whole decoder history forward.

That means the repo currently has two different answers to the same question:

- "Decode this long code sequence with bounded local context."
- "Decode this long code sequence with unbounded carried state."

If loudness, timbre stability, or phase behavior changes as history grows, the incremental path can decay or skew later chunks even when the bounded path stays stable.

## Reference-audio mismatch worth tracking

There is a second inconsistency that is not the main decay repro but is still important:

- non-streaming generation prepends `refCodes` before decode and trims them back off afterward
- streaming generation decodes only newly generated codes and never warms decoder state with those reference codes

So reference-conditioned requests do not share the same decoder warm-up behavior between the streaming and non-streaming lanes.

## Current regression coverage

`Tests/MLXAudioTTSTests.swift` now includes two Qwen3-TTS regression probes:

- a synthetic-code decode comparison between `Qwen3TTSSpeechTokenizer.decode(...)` and `Qwen3TTSSpeechTokenizer.streamingDecode(...)`
- a cached-model decode comparison that encodes repeated real fixture audio into Qwen3 codec codes first, then compares the same bounded and incremental decode surfaces
- a conditioned generated-code comparison that captures an actual long-form Qwen generation, then runs that exact generated code sequence through:
  - the current `decodeChunk(...)` helper
  - bounded tokenizer decode
  - reference-warmed incremental decode

Both probes currently pass on this machine. That matters:

- the risky decode split is real and now documented
- the plain decoder mismatch has regression coverage
- but these tests do **not** yet reproduce the severe long-form RMS collapse that prompted the audit

So the remaining repro likely lives in a higher-level generation path, a model-specific fixture, or a conditioning-dependent case that these decoder-only probes do not capture yet.

## Additional result from the conditioned generated-code probe

The conditioned generated-code comparison is the closest local match so far to
the SpeakSwiftly investigation shape because it uses:

- real conditioned generation
- real generated Qwen codec tokens
- the repo's current `decodeChunk(...)` helper
- bounded decode on the exact same captured code sequence
- a manual reference-warmed incremental decode on the same captured code

On this machine, that probe still did **not** reproduce late-tail collapse.

What it did show instead:

- `decodeChunk(...)` and manual reference-warmed incremental decode matched each
  other closely
- both of those paths were noticeably hotter than bounded decode in the early
  quarter of the waveform
- the tail RMS stayed roughly aligned across all three paths

That pushes the current evidence one step forward:

- the decode-path mismatch is definitely real
- reference-warmed incremental decode behaves like the current helper on the
  same captured conditioned codes
- but the specific severe tail-decay symptom seen in SpeakSwiftly still is not
  reproduced by decode-only or captured-code decode comparison inside this repo

So the next likely hidden variable is no longer "plain decoder state drift by
itself." It is more likely one of:

- a profile/materialization-specific conditioning artifact
- a runtime-surface difference above the raw decode helpers
- a longer or more pathological generated token sequence than the current local
  cached probe produced

## Additional result from a real SpeakSwiftly conditioning artifact

The next pass used an actual persisted conditioning artifact from the local
SpeakSwiftly profile `probe-clear-masc-20260421` together with its matching
cached model materialization:

- profile artifact: `qwen-conditioning-qwen3.json`
- backend model: `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit`
- reference audio: `reference.wav`

The probe compared two end-to-end retained generations on the same long prompt:

- raw reference conditioning built from `reference.wav` plus the saved reference
  text at generation time
- persisted conditioning loaded directly from the saved SpeakSwiftly artifact

It also re-ran the artifact-generated code sequence through:

- the repo's current `decodeChunk(...)` helper
- bounded tokenizer decode

### What changed

The persisted artifact did make the retained output worse than rebuilding
conditioning from the raw reference assets:

- raw retained tail ratio: about `0.890`
- artifact retained tail ratio: about `0.776`

That means the saved conditioning artifact is not neutral. It pushes the same
model and long-form prompt toward more late-utterance energy loss than the raw
`refAudio` plus `refText` path on this machine.

### What did not change

Even on that artifact-conditioned generation, the decode helper still did not
show the kind of late-tail decode collapse that would explain the full
SpeakSwiftly field report by itself:

- artifact helper head gain vs bounded decode: about `0.874`
- artifact helper tail gain vs bounded decode: about `0.990`

So the helper decode path remained roughly aligned with bounded decode at the
tail on the exact same artifact-generated codec stream.

### Current significance

This narrows the picture further:

- persisted conditioning artifacts do appear to be a real amplifying factor
- they worsen long-form retained loudness relative to rebuilding conditioning
  from raw assets
- but they still do not make the decode helper reproduce the catastrophic
  profile collapse on their own

The strongest remaining explanation is now a combination effect rather than a
single isolated bug:

- profile-specific or artifact-specific conditioning changes the generated codec
  trajectory
- longer generations amplify the problem
- some additional runtime or sequence-shape factor beyond the decode helper
  comparison still appears necessary to reach the worst observed SpeakSwiftly
  failures

## Additional result from the profile-and-length matrix

The next pass compared two real SpeakSwiftly profiles across both a short and a
long prompt, always using the same local `0.6B Base 8bit` backend snapshot and
measuring retained-output tail-versus-head RMS for:

- raw rebuilt conditioning from `reference.wav` plus saved reference text
- persisted conditioning from `qwen-conditioning-qwen3.json`

Profiles:

- `probe-soft-femme-20260421`
- `probe-clear-masc-20260421`

### Measured retained tail ratios

`probe-soft-femme-20260421`

- short prompt, raw: about `0.742`
- short prompt, artifact: about `0.712`
- long prompt, raw: about `0.867`
- long prompt, artifact: about `0.0367`

`probe-clear-masc-20260421`

- short prompt, raw: about `0.689`
- short prompt, artifact: about `0.658`
- long prompt, raw: about `0.918`
- long prompt, artifact: about `0.894`

### Why this matters

This is the first local repro in `mlx-audio-swift` that crosses into the same
failure class as the SpeakSwiftly field report.

The important pattern is not just "artifacts are worse" or just "longer is
worse." It is the interaction:

- short prompts only showed modest artifact penalties on both profiles
- `probe-clear-masc-20260421` stayed relatively healthy even on the long prompt
- `probe-soft-femme-20260421` collapsed dramatically only when the long prompt
  was paired with the persisted conditioning artifact

So the severe decay is now locally reproducible, but only for a specific
profile-and-length combination. That strongly supports the idea that the core
bug is profile-sensitive and sequence-sensitive rather than a uniform decode
bug affecting all conditioned Qwen generations equally.

### Important caveat from reruns

This matrix result is not stable enough yet to treat as a deterministic
regression gate.

Later reruns of the same probe on the same machine did not always reproduce the
same catastrophic `probe-soft-femme-20260421` long-form artifact collapse.
One rerun instead showed a much healthier long-form artifact result for that
profile, with:

- raw head rms: about `0.0939`
- raw tail ratio: about `0.792`
- artifact head rms: about `0.0955`
- artifact tail rms: about `0.0803`
- artifact tail ratio: about `0.840`

Another rerun shifted the strong collapse to the other profile lane instead:

- `probe-clear-masc-20260421` long prompt, raw tail ratio: about `0.905`
- `probe-clear-masc-20260421` long prompt, artifact tail ratio: about `0.126`

After converting the matrix into a logging-style probe instead of a strict
pass/fail gate, another full rerun completed with no catastrophic collapse at
all. That pass measured:

`probe-soft-femme-20260421`

- short prompt, raw: about `0.553`
- short prompt, artifact: about `0.587`
- long prompt, raw: about `0.783`
- long prompt, artifact: about `0.790`

`probe-clear-masc-20260421`

- short prompt, raw: about `0.905`
- short prompt, artifact: about `0.756`
- long prompt, raw: about `0.968`
- long prompt, artifact: about `0.946`

So the strongest honest reading is now:

- the profile-and-length matrix did uncover the right failure family locally
- the severe retained-output collapse is stochastic enough that a one-shot
  numeric expectation is flaky
- the matrix is currently best treated as an investigation harness and logging
  surface, not a strict pass/fail regression test

### Follow-up on the downstream SpeakSwiftly harness

The local SpeakSwiftly checkout has since tightened the Qwen probe harness in a
few ways that make its future reports more trustworthy:

- `volume-probe` now validates explicit `--conditioning artifact` runs up front
  instead of silently falling back to rebuilt conditioning when no stored
  artifact is present
- the new quarter-style `head_rms`, `tail_rms`, and `tail_head_ratio` summary
  values are now computed with duration-weighted RMS instead of a plain average
  of per-window RMS
- the harness now writes structured JSON artifacts under
  `.local/volume-probes/` for both `compare-volume` and `matrix-volume`

That does not remove the stochastic behavior seen across reruns, but it does
mean newer SpeakSwiftly probe reports are less likely to be mislabeled and the
head/tail comparison metrics are now a better fit for upstream cross-checking.

### Updated significance

At this point the evidence stack looks like this:

- the decode-path mismatch is real
- persisted conditioning artifacts are a real amplifying factor
- profile choice matters materially
- longer retained generations matter materially
- the worst failures now appear when those factors line up, rather than from a
  simple decode-helper regression alone
- the exact profile that collapses hardest can still move across reruns, so the
  remaining bug likely includes a stochastic generation component rather than a
  fully deterministic decode-only failure
