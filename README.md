# KV-Base

KV-Base is a research-oriented framework for experiments on Transformer KV
Cache behavior. It keeps the KVPress Press mechanism and adds a small common
infrastructure for lifecycle metadata, observation, intervention, and
experiment recording.

The repository is intended for paper experiments and rapid prototyping, not
production serving. A downstream project may modify attention, cache, or
model-specific code when an experiment requires it.

## Installation

The project uses `uv`:

```bash
uv sync
```

For development:

```bash
uv run pytest
```

## Minimal Usage

```python
from transformers import pipeline
from kvpress import KnormPress

pipe = pipeline(
    "kv-press-text-generation",
    model="Qwen/Qwen3-8B",
    device_map="auto",
    dtype="auto",
)

press = KnormPress(compression_ratio=0.5)
result = pipe(
    "A long context.",
    question="Summarize the context.",
    press=press,
)
print(result["answer"])
```

The same Press path works with `model.generate()` when the Press is installed
as a context manager:

```python
with press(model):
    output = model.generate(**inputs)
```

## Framework Components

- `kvpress/context.py`: `KVContext`, `KVPhase`, and
  `KVExecutionLifecycle`. Provides per-layer PREFILL/DECODE metadata.
- `kvpress/observer.py`: read-only `KVObserver` hooks for inspecting K/V
  behavior without changing the cache.
- `kvpress/recorder.py`: file-backed `KVRecorder` for run metadata, sample
  metadata, JSONL records, and explicitly requested tensor dumps.
- `kvpress/presses/base_press.py`: existing KVPress intervention interface.
- `kvpress/presses/decoding_press.py` and related wrappers: decode-time
  intervention paths.
- `kvpress/pipeline.py`: the custom text-generation pipeline.
- `kvpress/attention_patch.py`: attention-function patching used by supported
  head-wise Presses.

See:

- [BASE_STATUS.md](BASE_STATUS.md)
- [docs/base/architecture.md](docs/base/architecture.md)
- [docs/base/extension_guide.md](docs/base/extension_guide.md)
- [docs/base/experiment_conventions.md](docs/base/experiment_conventions.md)
- [docs/PROJECT_GOALS.md](docs/PROJECT_GOALS.md)

## Observation

Subclass `KVObserver` and implement `on_layer`, `on_prefill`, or `on_decode`.
The callback receives a `KVContext` containing available cache tensors and
lifecycle metadata. Observation is opt-in and does not materialize Q or full
attention weights automatically.

```python
from kvpress import KVObserver, KVRecorder

class NormObserver(KVObserver):
    def on_layer(self, context):
        key_norm = context.k.norm(dim=-1).mean().item()
        if context.recorder is not None:
            context.recorder.record_scalar(
                "key_norm", key_norm, step=context.decode_step
            )

recorder = KVRecorder("runs/example", run_name="norms")
observer = NormObserver(recorder=recorder)
with observer(model):
    output = model.generate(**inputs)
```

## Intervention

Existing Presses remain the intervention mechanism. New research code should
reuse `BasePress` where possible and use `KVContext` for phase, layer, step,
and cache metadata. Existing Press implementations are kept in
`kvpress/presses/`; they are not all guaranteed to support every model or
cache type.

## Cache and Model Scope

The framework consumes Hugging Face Cache objects through the current
Transformers API. Ordinary dynamic caches and supported quantized-cache paths
are handled by existing Press code, but individual Presses may have
model-specific requirements. Test a method with the target model family
before drawing research conclusions.

The current code includes support paths for common Llama, Mistral, Phi, Qwen,
and Gemma-family causal language models. Model support is determined by the
attention and cache interfaces used by the selected Press.

## Tests

The infrastructure tests are self-contained and use synthetic models:

```bash
uv run pytest \
  tests/test_context.py \
  tests/test_observer.py \
  tests/test_recorder.py \
  tests/test_infrastructure.py \
  tests/test_attention_patch.py -q
```

The tests cover context metadata, observe-only behavior, recording, prefill
intervention, decode intervention, and attention patching.
