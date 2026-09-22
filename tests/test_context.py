# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from kvpress import BasePress, KVPhase


@dataclass
class ContextCapturePress(BasePress):
    contexts: list = field(default_factory=list)

    def forward_hook(self, module, input, kwargs, output):
        result = super().forward_hook(module, input, kwargs, output)
        self.contexts.append(kwargs["kv_context"])
        return result

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        return keys, values


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


def test_context_records_prefill_metadata():
    model = make_model()
    press = ContextCapturePress()
    input_ids = torch.tensor([[1, 2, 3, 4]])

    with press(model):
        model(input_ids, use_cache=True)

    assert len(press.contexts) == 2
    assert all(context.phase is KVPhase.PREFILL for context in press.contexts)
    assert all(context.layer_id in (0, 1) for context in press.contexts)
    assert all(context.prefill_length == 4 for context in press.contexts)
    assert all(context.total_length == 4 for context in press.contexts)
    assert all(context.stored_length == 4 for context in press.contexts)
    assert {context.forward_id for context in press.contexts} == {0}
    assert [context.layer_event_index for context in press.contexts] == [0, 1]
    assert all(context.keys is not None and context.values is not None for context in press.contexts)


def test_context_records_decode_steps_without_changing_generation():
    torch.manual_seed(0)
    model = make_model()
    press = ContextCapturePress()
    input_ids = torch.tensor([[1, 2, 3, 4]])

    with press(model):
        output = model.generate(input_ids, max_new_tokens=3, do_sample=False)

    assert output.shape == (1, 7)
    decode_contexts = [context for context in press.contexts if context.phase is KVPhase.DECODE]
    assert len(decode_contexts) >= 2
    assert {context.decode_step for context in decode_contexts} == {0, 1}
    assert all(context.total_length == 5 + context.decode_step for context in decode_contexts)
    assert all(context.stored_length == context.total_length for context in decode_contexts)
    assert {context.forward_id for context in decode_contexts} == {1, 2}
