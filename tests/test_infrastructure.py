# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The three minimal KV-Base v0.1 infrastructure acceptance tests."""

import json
from dataclasses import dataclass, field

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from kvpress import BasePress, KVObserver, KVPhase, KVRecorder


def make_model():
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(config).eval()


@dataclass
class StatsObserver(KVObserver):
    contexts: list = field(default_factory=list)

    def on_layer(self, context):
        self.contexts.append(context)
        context.recorder.record_stat(
            "key_norm_mean",
            context.keys.float().norm(dim=-1).mean(),
            phase=context.phase,
            forward_id=context.forward_id,
            layer=context.layer_id,
        )


@dataclass
class DeterministicPrefillPress(BasePress):
    """Reference intervention retaining the first half of prefill tokens."""

    calls: int = 0

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        self.calls += 1
        keep = max(1, keys.shape[2] // 2)
        return keys[:, :, :keep].contiguous(), values[:, :, :keep].contiguous()


@dataclass
class DecodeEveryNPress(BasePress):
    """Reference decode intervention dropping the oldest token periodically."""

    interval: int = 2
    calls: list = field(default_factory=list)
    cache_lengths: list = field(default_factory=list)

    def forward_hook(self, module, input, kwargs, output):
        output = super().forward_hook(module, input, kwargs, output)
        context = kwargs["kv_context"]
        if context.layer_id == 0 and context.phase is KVPhase.DECODE and context.decode_step is not None:
            if (context.decode_step + 1) % self.interval == 0:
                cache_layer = context.cache.layers[context.layer_id]
                if cache_layer.keys.shape[2] > 1:
                    cache_layer.keys = cache_layer.keys[:, :, 1:].contiguous()
                    cache_layer.values = cache_layer.values[:, :, 1:].contiguous()
                    self.calls.append(context.decode_step)
                    self.cache_lengths.append(int(cache_layer.keys.shape[2]))
        return output

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        return keys, values


def test_case_a_observation_only_records_statistics_and_preserves_output(tmp_path):
    torch.manual_seed(123)
    model = make_model()
    reference = make_model()
    reference.load_state_dict(model.state_dict())
    input_ids = torch.tensor([[1, 2, 3, 4]])
    recorder = KVRecorder(tmp_path, run_name="observe-only")
    observer = StatsObserver()
    observer.recorder = recorder

    with observer(model):
        observed = model.generate(input_ids, max_new_tokens=2, do_sample=False)
    baseline = reference.generate(input_ids, max_new_tokens=2, do_sample=False)

    assert torch.equal(observed, baseline)
    assert any(c.phase is KVPhase.PREFILL for c in observer.contexts)
    assert any(c.phase is KVPhase.DECODE for c in observer.contexts)
    stats = tmp_path / "observe-only" / "samples" / "default" / "stats.jsonl"
    assert stats.exists()
    assert len(stats.read_text().splitlines()) == len(observer.contexts)
    assert json.loads(stats.read_text().splitlines()[0])["name"] == "key_norm_mean"


def test_case_b_prefill_intervention_changes_cache_length():
    model = make_model()
    press = DeterministicPrefillPress()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])

    with press(model):
        outputs = model(input_ids, use_cache=True)

    assert press.calls == 2
    assert outputs.past_key_values.get_seq_length() == 3
    assert all(layer.keys.shape[2] == 3 for layer in outputs.past_key_values.layers)


def test_case_c_decode_intervention_uses_lifecycle_step_and_updates_cache():
    model = make_model()
    press = DecodeEveryNPress(interval=2)
    input_ids = torch.tensor([[1, 2, 3, 4]])

    with press(model):
        outputs = model.generate(input_ids, max_new_tokens=5, do_sample=False)

    assert outputs.shape == (1, 9)
    assert press.calls == [1, 3]
    assert press.cache_lengths == [5, 6]
