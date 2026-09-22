# KV-Base Project Instructions

## 1. Project purpose

This repository is a research-oriented KV Cache base framework derived from NVIDIA KVPress.

The primary goal is **rapid prototyping of new KV Cache research ideas**. A new idea should be able to clone or branch from this base repository and reuse the common infrastructure for model loading, generation, KV Cache access, prefill/decode execution, evaluation, tracing, and experiment recording.

Typical downstream research topics include:

- KV Cache distribution and redundancy analysis
- attention/head/token importance analysis
- prefill KV compression
- decode KV compression or eviction
- KV sparsity
- KV quantization
- KV merging or reconstruction
- head-wise or token-wise policies
- temporal stability analysis
- KV access-pattern analysis
- later-stage hardware/system-oriented studies

This is **not** intended to become a production serving framework or a universal plugin system that can express every possible KV method without modifying source code.

The design target is:

> Reuse roughly 80% of common infrastructure while allowing the remaining 20% to be modified freely for a specific research idea.

---

## 2. Core development principle

Do not over-engineer the framework.

Prefer a small number of stable, research-useful abstractions over a large generic architecture.

The base framework should make the following two operations easy:

1. **Observe KV-related behavior without changing model semantics.**
2. **Intervene in KV Cache behavior when an experiment requires it.**

A downstream project is allowed to modify attention implementations, cache internals, kernels, or model-specific code when necessary. The base framework exists to reduce repeated work, not to prohibit specialized modifications.

---

## 3. Before modifying code

When starting from a fresh clone or when the upstream KVPress version changes:

1. Read this file.
2. Read `BASE_STATUS.md`.
3. Read `docs/PROJECT_GOALS.md`.
4. Read the current documents under `docs/` that describe project goals and
   base architecture.
5. Inspect the actual checked-out KVPress source code.
6. Inspect the current source before changing architecture. Historical
   upstream audit documents are not required to remain in this personal base.
7. Verify the original upstream baseline before making architectural changes.

Do **not** assume that previously documented KVPress internals still match the current checkout.

Repository documentation must reflect the code that is actually present in the current branch.

If upstream KVPress contains additional `AGENTS.md` files or repository-specific instructions, preserve and follow them unless they conflict with the explicit goals of this fork. Do not silently discard upstream instructions.

---

## 4. Phase 0: understand upstream KVPress first

Before implementing KV-Base changes, document the current upstream execution path.

At minimum, determine:

- How models are loaded.
- How generation is invoked.
- How attention modules are reached.
- How Q/K/V are produced or accessed.
- How `past_key_values` / Hugging Face Cache objects flow through generation.
- How KVPress obtains and modifies KV Cache.
- When a Press executes relative to attention.
- How prefill and decode are distinguished.
- How decode-time cache growth is handled.
- How existing decoding-related Press implementations work.
- How quantized caches are handled, if supported in the current checkout.
- Which model families are supported.
- Which parts are model-specific.
- How evaluation scripts invoke generation.
- What outputs, metrics, and metadata are already recorded.

Record the findings in:

- `docs/base/architecture.md`
- `docs/base/extension_guide.md`
- `docs/base/experiment_conventions.md`

These documents should reference concrete files, classes, and functions from the checked-out repository.

Do not begin a large refactor until this upstream documentation is sufficiently complete to explain the existing execution path.

---

## 5. Target KV-Base architecture

The first version should remain close to KVPress and introduce only the minimum infrastructure needed for repeated research work.

The target conceptual architecture is:

```text
Model / Generation
       |
       v
Prefill / Decode Lifecycle
       |
       v
Attention / Cache Access
       |
       +----------------------+
       |                      |
       v                      v
   KV Observer            KV Intervention
       |                      |
       v                      v
 analysis / tracing        Press / policy
       |                      |
       +----------+-----------+
                  |
                  v
               Recorder
                  |
                  v
              Evaluation
```

The important additions are:

- a unified execution context
- observation hooks
- experiment recording
- explicit prefill/decode semantics

Do not replace all existing KVPress mechanisms merely for architectural cleanliness.

Where possible, keep existing KVPress Press behavior working. Evaluation
scripts and benchmark directories are intentionally outside this compact base;
research branches may add the evaluation code they need.

---

## 6. Stable concepts to introduce

### 6.1 KVContext

Introduce a lightweight context object or equivalent mechanism that makes important execution metadata available in a consistent form.

The exact implementation should be chosen only after inspecting the current KVPress code.

Conceptually it should expose fields such as:

```text
phase                 # PREFILL or DECODE
layer_id
decode_step
prefill_length
total_length

input_ids
position_ids
hidden_states

q                      # when available / explicitly requested
k
v
attention              # optional; avoid forcing expensive attention materialization

cache
recorder
```

Not every field must always be populated.

Avoid forcing expensive tensors such as full attention matrices to be materialized unless an experiment explicitly requests them.

---

### 6.2 Observer

Add an observation path that can inspect or record KV-related information **without modifying the cache**.

Typical uses:

- K/V norm statistics
- Q/K/V distribution analysis
- attention distribution analysis
- per-head behavior
- per-token importance
- temporal stability
- decode-step evolution
- reuse/access-pattern analysis

Observation must be separable from intervention.

A user should be able to run an analysis experiment without implementing a fake compression method.

---

### 6.3 Recorder

Provide one reusable experiment recorder rather than allowing every downstream idea to invent its own `torch.save`, NumPy, JSON, or directory conventions.

The recorder should support:

- run-level metadata
- sample-level metadata
- scalar metrics
- layer/head/token statistics
- optional tensor dumps
- trace-style records

The first implementation should remain simple and transparent.

Do not introduce a database or heavy experiment-management dependency unless later work clearly requires it.

---

### 6.4 KV intervention

Preserve KVPress's existing Press mechanism where practical.

Do not immediately replace `BasePress` or existing Press subclasses with a completely new policy hierarchy.

The first KV-Base version should support both:

```text
Observer path       -> inspect only
Press path          -> modify KV Cache
```

A more general `KVMethod` abstraction may be introduced later only if repeated experiments demonstrate that the existing separation is insufficient.

---

## 7. Prefill and decode are first-class concepts

The framework must explicitly distinguish:

```text
PREFILL
DECODE
```

Do not treat generation as one undifferentiated forward path.

Where possible, expose:

```text
phase
prefill_length
decode_step
total_length
```

This is a core requirement because many KV Cache research questions differ fundamentally between prefill and decode.

Examples include:

- prefill-only token selection
- decode-time KV growth
- online eviction
- temporal head stability
- reasoning-generated KV
- long-generation KV reuse

---

## 8. Standard research indexing convention

Where practical, use the following conceptual dimensions consistently:

```text
Layer x Head x Token
```

For decode-time traces:

```text
Step x Layer x Head x Token
```

Recommended notation:

```text
L = layer
H = head
T = KV token
S = decode step
```

This convention should be used in recorder metadata, analysis scripts, and saved traces whenever it is natural.

Do not force tensors into this shape when doing so creates unnecessary copies or memory overhead. Preserve logical indexing even if the physical layout differs.

---

## 9. Experiment modes

The framework should support two explicit research modes.

### Observe mode

Purpose:

- discover phenomena
- collect statistics
- generate heatmaps/traces
- test hypotheses without changing inference semantics

Expected behavior:

```text
model execution
    ->
observer
    ->
recorder
```

The observer should not modify KV state.

### Intervention mode

Purpose:

- compression
- eviction
- quantization
- sparsity
- merging
- reconstruction
- placement or other KV policy experiments

Expected behavior:

```text
model execution
    ->
policy / Press
    ->
modified KV state
    ->
evaluation
```

Where useful, an experiment may combine observation and intervention.

---

## 10. Evaluation philosophy

The base framework should preserve upstream evaluation functionality whenever possible.

A research change should ideally be testable at several levels:

1. **Functional smoke test**
   - generation still runs
   - cache length/state is sensible

2. **Semantic regression**
   - with all new features disabled, behavior should match upstream baseline as closely as expected

3. **Research metric**
   - accuracy / perplexity / benchmark metric / task-specific outcome

4. **System metric when relevant**
   - KV size
   - bytes read/written
   - decode latency
   - prefill latency
   - memory usage
   - metadata overhead

Do not claim hardware savings solely from logical KV reduction.

Logical compression and physical execution cost must be treated as different questions.

---

## 11. Scope boundary: algorithm base vs system/hardware validation

This repository is primarily the **algorithm and phenomenon exploration layer**.

It should answer questions such as:

- Does the KV pattern exist?
- Is it stable across layers/heads/steps?
- Can tokens be removed?
- Can KV be quantized?
- Does accuracy remain acceptable?
- What logical KV reduction is achieved?

For later system or accelerator validation, downstream projects may integrate with or port ideas to:

- vLLM
- FlashInfer
- LMCache
- custom CUDA
- custom simulator / hardware model

Do not prematurely reshape the entire base around paged serving or hardware-specific layouts.

---

## 12. Documentation requirements

Documentation is part of the framework.

After any meaningful architectural change, update the relevant documents.

Important files:

```text
AGENTS.md
BASE_STATUS.md

docs/PROJECT_GOALS.md
docs/base/architecture.md
docs/base/extension_guide.md
docs/base/experiment_conventions.md
```

When recording a change, document:

```text
What changed?
Why was it changed?
What was the upstream behavior?
What is the new behavior?
Which code path is affected?
Does it affect prefill, decode, or both?
How was it verified?
```

Do not maintain documentation as a file-by-file changelog only.

Explain design intent.

---

## 13. Change discipline

Prefer small, reviewable commits.

Recommended initial sequence:

```text
1. docs: document checked-out upstream KVPress architecture
2. test: establish reproducible upstream baseline
3. feat: introduce execution context
4. feat: add observer infrastructure
5. feat: add recorder infrastructure
6. test: validate observe mode does not change baseline behavior
7. docs: document KV-Base architecture and extension workflow
```

Do not mix a large architectural refactor with implementation of a new research algorithm in the same initial change.

---

## 14. Initial acceptance tests for KV-Base v0.1

Before considering the base framework usable, demonstrate at least these three cases.

### Case A: observation only

Run inference without modifying KV Cache and record at least one layer/head/token statistic.

Expected result:

- output semantics remain baseline-equivalent within the expected numerical behavior
- recorder output is produced
- prefill/decode context is correct

### Case B: prefill KV modification

Implement a simple reference intervention such as deterministic or random retention of a fraction of prefill KV tokens.

Purpose:

- validate that the base can alter prefill KV state
- not intended as a research contribution

### Case C: decode KV modification

Implement a simple reference decode-time intervention, e.g. every N decode steps perform a controlled cache operation.

Purpose:

- validate decode lifecycle access
- validate step tracking
- validate cache update correctness

These examples should remain simple and serve as infrastructure tests.

---

## 15. What not to do in v0.1

Do not make the first version depend on:

- a custom CUDA kernel
- vLLM internals
- a new database
- a complex physical KV memory manager
- a universal policy DSL
- a large plugin registry
- a full accelerator simulator

Do not require all downstream research ideas to fit one rigid API.

Do not optimize for production throughput before the research workflow is stable.

---

## 16. Expected downstream workflow

The intended workflow is:

```text
kv-base
   |
   +--> idea-catekv-style
   |
   +--> idea-decode-kv
   |
   +--> idea-head-stability
   |
   +--> idea-kv-quant
   |
   +--> idea-kv-sparsity
```

A downstream idea should primarily need to modify:

```text
method-specific code
analysis code
experiment configuration
```

Common infrastructure should remain reusable:

```text
model loading
generation
KV access
prefill/decode lifecycle
recording
evaluation
```

A good design target is that an early prototype of a new KV idea can often be expressed in a few hundred lines of method/analysis code without repeatedly rewriting model-generation infrastructure.
