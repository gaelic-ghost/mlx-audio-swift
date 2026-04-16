# Qwen3TTS upstream review notes

Date: 2026-04-16

These notes capture problems and oddities observed while comparing the reusable
reference-conditioning branch against upstream `main`.

## Main findings

### 1. Reference conditioning silently degrades into ordinary generation

In `Qwen3TTSModel.generateVoiceDesign`, the clone/reference path only activates
when all of these are true:

- `refAudio != nil`
- `refText != nil`
- `speechTokenizer.hasEncoder == true`

If any one of those conditions fails, the implementation quietly falls through
to `prepareGenerationInputs(...)` and behaves like ordinary synthesis instead of
throwing a validation error.

This is why a missing-`refText` call can sound like it "defaults" to a built-in
voice instead of failing loudly. The request is no longer treated as a
reference-conditioned request at all.

### 2. Upstream caches only the audio half of the reference pair

`ReferenceAudioContext` caches:

- `speakerEmbedding`
- `refCodes`
- `codecEmbedIcl`

That cache is keyed only by `ObjectIdentifier(refAudio)` on upstream `main`.
`refText` is retokenized on every request and is not part of the cached unit.

This means the implementation does not cache one coherent reference-conditioning
artifact. It caches the audio-derived half, then combines that cached audio-side
state with freshly rebuilt text-side state from whatever transcript is supplied
on the current call.

### 3. Transcript mismatch is still a real failure mode even without cache reuse

Harness results showed that broad transcript mismatch is not just a cache-key
problem. Even when the same `refAudio` clip is reloaded fresh per case so the
upstream cache cannot be hit, these behaviors still appear:

- punctuation-only transcript changes noticeably shift the voice
- paraphrased transcripts also shift the voice
- completely wrong transcripts break the output badly
- wrong-language transcripts also break badly

So the bad cache key is real, but the deeper issue is that the reference-audio
path is fragile when the transcript does not truly match the reference clip.

### 4. The shared public API overloads one parameter with different meanings

The `SpeechGenerationModel` protocol only exposes one generic `voice` slot.
Inside Qwen3TTS, that slot means different things depending on model type:

- VoiceDesign-style path: free-text instruction / voice description
- CustomVoice path: speaker name
- Base path: effectively unused in ordinary non-reference generation

The code "works" by repurposing one shared parameter rather than by modeling
those modes explicitly.

### 5. The internal naming is actively misleading now

`generateVoiceDesign(...)` is no longer only a VoiceDesign engine. It is the
shared internal generation engine for:

- Base
- CustomVoice
- reference-conditioned clone generation

The implementation already treats it as the common engine, but the name still
describes only one product mode.

### 6. There are still raw debug prints and crashy assumptions in library code

Current upstream `Qwen3TTS` still contains:

- unconditional `print(...)` statements in cache and CustomVoice paths
- `fatalError(...)` in internal request-assembly paths
- forced assumptions like `config.talkerConfig!`

That combination makes the library harder to reason about because some invalid
states quietly degrade, some produce noisy debug logs, and others crash.

### 7. Language handling around reference conditioning is loosely assembled

The reference transcript is retokenized each request, while codec-side language
conditioning is derived separately from the `language` parameter. Those are not
modeled as one validated, persisted unit.

That makes it easy to assemble reference requests that are formally accepted but
semantically inconsistent, such as:

- cached or recomputed English reference audio paired with unrelated text
- wrong-language transcript paired with an unchanged codec-side prefix
- incomplete reference pairs that silently fall back to ordinary generation

## Practical takeaway

The upstream implementation currently treats reference conditioning as a set of
pieces it can opportunistically combine at runtime. The reusable-conditioning
branch takes the opposite approach and treats the reference pair as a concrete
artifact that can be prepared once, validated, persisted, and reused.

That difference is bigger than just API taste. It affects correctness,
debuggability, and whether the caller can rely on a reference pair having one
stable meaning.
