# KV-Base Status

Current repository status as of 2026-09-23.

## Current State

KV-Base contains the retained KVPress runtime and Press implementations plus
the common research infrastructure:

- `KVContext`, `KVPhase`, and `KVExecutionLifecycle`;
- read-only `KVObserver` hooks;
- file-backed `KVRecorder`;
- focused infrastructure tests for observation and intervention;
- compact project and extension documentation.

The first cleanup pass removed benchmark/evaluation code, notebooks, KVzap
training utilities, repository community files, and their dedicated tests.
The first Press pruning pass additionally removed five externally coupled
methods: KVzap, LU-KV, DuoAttention, FastKVzip, and KVCompose.
The second Press pruning pass removed seven specialized methods: QFilter, CAP,
CUR, Compactor, LeverageScore, NonCausalAttention, and EntropyGatedChunkKV.
The retained set still includes baseline scoring, prefill/decode wrappers,
chunk methods, decode merge, critical scoring, and channel compression.

## Repository Identity

```text
path:       /home/wz/AI/kvbase
branch:     main
HEAD:       7f359b84b22f591f21af7394046106ed9b18d916
version:    0.5.4
origin:     git@github.com:wz1114841863/KV-Base.git
upstream:   https://github.com/NVIDIA/kvpress.git
```

`HEAD` and `upstream/main` referred to the same upstream commit when this
status was recorded. The working tree also contains the KV-Base changes and
cleanup changes listed above.

## Implemented Infrastructure

### Context and lifecycle

`kvpress/context.py` provides per-layer context objects with:

```text
phase, layer_id, forward_id, layer_event_index
decode_step, prefill_length, total_length, stored_length
cache_position, hidden_states, keys, values, cache, recorder
```

The lifecycle is shared by Press and Observer paths. Layer 0 starts a logical
forward, layers in that forward share `forward_id`, and decode forwards use a
monotonic `decode_step`. Cache creation, growth, attention execution, and
write-back remain owned by Transformers/KVPress.

### Observer

`KVObserver` installs read-only post-attention hooks. It creates a context for
each supported layer, calls `on_layer`, then `on_prefill` or `on_decode`, and
returns the original attention output. It does not modify the cache.

### Recorder

`KVRecorder` writes explicit JSON, JSONL, and optional CPU tensor files:

```text
<root>/<run_name>/
  run_metadata.json
  samples/<sample_id>/
    sample_metadata.json
    scalars.jsonl
    stats.jsonl
    events.jsonl
    tensors.jsonl
    tensors/*.pt
```

Full tensors are recorded only through an explicit `record_tensor` call.

### Press compatibility

Existing `BasePress` and `compress(...)` behavior remains the intervention
path. The hook supplies `kwargs["kv_context"]` to custom Press code without
requiring existing Press subclasses to change their signatures. Decode-aware
wrappers retain their existing cache update behavior.

## Environment

The project environment is managed by `uv` and was verified with:

```text
Python 3.12.3
torch 2.11.0+cu128
Transformers 5.2.0
NumPy 2.5.3
```

Direct runtime dependencies are limited to the packages used by the retained
runtime and Presses: NumPy, PyTorch, Transformers, Datasets, Accelerate,
Hugging Face Hub, Fire, and tqdm. The former evaluation
extra and unused `peft` dependency were removed. Development dependencies now
contain only the test, formatting, lint, and type-check tools.

PyTorch is resolved from the CUDA 12.8 index configured in `pyproject.toml`.
The generated `uv.lock` remains ignored by the repository's existing policy.

The real user environment reports CUDA availability on an RTX 3060 Laptop
GPU. The Codex sandbox cannot access `/dev/nvidia*`, so sandbox GPU checks are
not representative.

## Verification

The focused infrastructure suite passes:

```text
11 passed
```

The suite covers context metadata, Observer output equivalence, Recorder
output, prefill intervention, decode intervention, and attention patching.

The full repository test command in the Codex sandbox reported:

```text
19 passed, 3 skipped, 6 errors
```

The six errors are fixture setup failures while loading remote Hugging Face
models (`MaxJeblick/llama2-0b-unit-test` and
`h2oai/h2o-danube3-500m-chat`). The sandbox cannot write the existing cache
path and does not resolve the cached repository's case variant. They are
environment/model-access failures, not failures in the retained infrastructure
tests. The user's real-GPU Qwen3-4B smoke test passed separately.

Also verified:

- `compileall -q kvpress tests` passes;
- `uv pip check` passes;
- cached local fixture generation and `KnormPress` execution pass;
- model-ID fixture loading and QuantizedCache execution were not fully
  validated in the sandbox.
- Real-GPU smoke testing passed on the user's RTX 3060 Laptop GPU with
  `Qwen/Qwen3-4B` loaded from the local Hugging Face cache:
  - ordinary `model.generate()`;
  - prefill `KnormPress(compression_ratio=0.25)`;
  - decode `DecodingPress(KnormPress(), compression_interval=2)`.
  All three generated new tokens successfully.

The smoke-test log identifies the tested model as `Qwen/Qwen3-4B`; a separate
`Qwen3-8B` result is not recorded here. Llama was not tested because the
requested local model was unavailable. OPT was not tested because it is not in
the current Press-supported model list and is outside the intended Press
smoke-test scope.

The retained Press tree no longer imports `requests` or `cachetools`; both
were removed from the direct runtime dependencies.

## Known Boundaries

- Generic Q is not exposed by the post-attention hook.
- Full attention weights are optional and implementation-dependent.
- GQA/MQA distinguishes query heads from stored KV heads.
- Cache gathering does not automatically preserve original token IDs.
- Quantized-cache support may depend on private Transformers cache fields.
- Logical KV reduction is not evidence of physical bandwidth or latency
  reduction.

## Reading Order

1. `README.md`
2. `docs/PROJECT_GOALS.md`
3. `docs/base/architecture.md`
4. `docs/base/extension_guide.md`
5. `docs/base/experiment_conventions.md`
