# Marvis Strict Verify Investigation

Date: 2026-04-16

## Symptom

`SpeakSwiftly` hit a resident Marvis preload failure after the fork moved `MarvisTTSModel.fromPretrained(...)` to `verify: .all`. The failing key was:

- `model.backbone.layers.0.self_attn.rope.cosF32`

The failure first showed up during a hosted dual-lane resident preload path, but the same code path succeeded again as soon as the Marvis loader temporarily dropped back to the older selective verification mode.

## What This Was Not

- Not a skipped `ropeInit()` step. `CSMLlama3ScaledRoPE` builds its caches during initialization.
- Not a lower-level API bypass in `SpeakSwiftly`. The package calls the normal `TTS.loadModel(modelRepo:)` surface, which dispatches to `MarvisTTSModel.fromPretrained(...)`.
- Not evidence that Marvis resident preload cannot run in parallel. The same dual-lane preload path worked once strict verification stopped demanding the runtime cache keys.

## Root Cause

`mlx-swift` module reflection treats stored fields as parameter keys unless they are hidden from parameter validation. The relevant behavior is:

- `Module.parameterIsValid(_:)` ignores keys that start with `_`
- reflection still sees ordinary stored `MLXArray`-shaped fields as module content

`CSMLlama3ScaledRoPE` stored these runtime-only caches without underscore prefixes:

- `cosF32`
- `sinF32`
- `cosByDType`
- `sinByDType`

Those fields are derived caches, not checkpoint-backed weights. Once Marvis adopted `verify: .all`, MLX started expecting the checkpoint to provide those keys, which caused the preload failure.

## Fix

Keep the runtime-only RoPE caches underscore-prefixed:

- `_cosF32`
- `_sinF32`
- `_cosByDType`
- `_sinByDType`

That preserves strict checkpoint verification for real model weights while keeping runtime caches out of the reflected parameter surface.

## Regression Coverage

Add a focused test that:

- instantiates `CSMLlama3ScaledRoPE`
- confirms the reflected parameter keys do not include the runtime cache names
- confirms `update(parameters: ModuleParameters.unflattened([:]), verify: .all)` succeeds

## Upstream Issue Check

As of 2026-04-16, no matching open issue was found on `Blaizzy/mlx-audio-swift` for this Marvis strict-verify cache mismatch. Search terms checked included:

- `Marvis verify`
- `cosF32`
- `rope`
