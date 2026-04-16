# Marvis TTS

A fast conversational text-to-speech (TTS) model with built-in voices for English, French, and German.

[Hugging Face Model Repo](https://huggingface.co/Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit)

## Supported Voices

- `conversational_a` (English)
- `conversational_b` (English)
- `conversational_fr` (French)
- `conversational_de` (German)

## CLI Example

```bash
mlx-audio-swift-tts --model Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit --voice conversational_a --text "Hello world."
```

## Swift Example

```swift
import MLXAudioTTS

let model = try await MarvisTTSModel.fromPretrained("Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit")
let audio = try await model.generate(
    text: "Hello world.",
    voice: "conversational_a",
    refAudio: nil,
    refText: nil,
    language: nil,
    generationParameters: GenerateParameters()
)
```

## Maintainer Note

Marvis uses runtime-built RoPE caches inside `CSMLlama3ScaledRoPE`. Those caches are derived at model initialization time and are not checkpoint weights, so they must stay hidden from MLX module parameter reflection. Keep those cache fields underscore-prefixed so strict checkpoint verification with `verify: .all` continues to validate real weights without demanding runtime-only cache keys such as `rope.cosF32`.
