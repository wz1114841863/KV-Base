# Extension Guide

Use the smallest extension that answers the research question.

## Choose a Path

### Observation or analysis

Use `KVObserver` and `KVRecorder` when the experiment does not need to change
the cache.

```text
model execution -> Observer -> statistics / traces -> offline analysis
```

Record phase, layer, head interpretation, token positions, and decode step
whenever relevant. Do not retain GPU tensors without an explicit memory plan.

### Prefill intervention

Reuse a `BasePress` or `ScorerPress` when the method selects or modifies
prefill K/V pairs.

```text
prefill K/V -> score/select -> cache update -> decode
```

Record original length, retained length, selection policy, and quality impact.

### Decode intervention

Reuse `DecodingPress` or another decode-aware Press when the method operates
during generation.

```text
decode step -> inspect/score -> update cache -> next step
```

Record `decode_step`, cache length before and after the operation, affected
layers/heads/tokens, and policy overhead.

### Quantization or sparse execution

First determine whether the current Transformers Cache path supports the
required representation. If it does, reuse it. Otherwise add the smallest
specialized representation needed by the experiment.

Always distinguish logical entries, represented bytes and metadata, physical
bytes read or written, and measured execution time.

## Shared Interfaces

Useful imports:

```python
from kvpress import KVContext, KVObserver, KVRecorder
from kvpress import BasePress, DecodingPress
```

Observer callbacks receive a `KVContext`. Existing Press implementations keep
their `compress(...)` signatures; custom Press code can inspect
`kwargs["kv_context"]` in the hook path.

`KVContext` fields may be `None`. Generic Q and full attention weights are not
guaranteed to exist.

## Recording Rules

Use run metadata for model, revision, device, dtype, seed, method, and commit.
Use sample metadata for prompt/sample identity and lengths. Use JSONL records
for scalar or structured events. Dump full tensors only when offline analysis
requires them.

Prefer logical dimensions:

```text
L x H x T
S x L x H x T       # decode traces
```

Declare whether `H` means query heads or KV heads.

## When to Modify Shared Code

Put a change in KV-Base when it is reusable across multiple research ideas,
such as lifecycle metadata, observation hooks, or recording formats.

Keep a change downstream when it is a single paper's scoring rule, a
model-specific workaround, a one-off visualization, or a specialized kernel.

If shared code must change:

1. keep the change localized;
2. add a focused regression test;
3. update `BASE_STATUS.md` and the affected base document;
4. verify observe-only behavior remains unchanged.
