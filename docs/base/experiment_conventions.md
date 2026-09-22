# Experiment Conventions

These conventions keep results from different research branches comparable.

## Lifecycle Fields

Every stage-sensitive record should include:

```text
phase = PREFILL | DECODE
layer_id
forward_id
layer_event_index
```

Decode records should also include `decode_step`.

## Logical Indices

Use these dimensions when applicable:

```text
L x H x T
S x L x H x T
```

`L` is layer, `H` is query or KV head as explicitly declared, `T` is KV
token, and `S` is decode step.

After selection or gathering, tensor position is not necessarily the original
logical token position. Record original positions when the experiment studies
reuse, locality, or irregular access.

## Minimum Metadata

Record enough information to reproduce or interpret a result:

```text
experiment / run
git commit
model and revision
prompt or sample identity
tokenizer
seed
dtype and device
observer or method configuration
```

## KV Metrics

Keep these quantities separate:

- logical KV entries;
- logical KV bytes;
- physical allocated bytes;
- estimated or measured bytes read/written;
- prefill latency;
- decode latency;
- Observer/Press overhead.

Logical compression alone is not evidence of physical acceleration.

## Tensor and Trace Policy

Prefer statistics before full tensors: norms, moments, quantiles, top-k
indices, histograms, sparsity, and correlations. Full tensors are opt-in.

Physical fields such as page IDs or addresses may be recorded only when the
implementation actually defines them. Do not infer hardware locality from a
gathered contiguous tensor.
