# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from kvpress import KVObserver, KVPhase, KnormPress


@dataclass
class CaptureObserver(KVObserver):
    contexts: list = field(default_factory=list)

    def on_layer(self, context):
        self.contexts.append(context)


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


def test_observer_is_read_only_and_sees_prefill_and_decode():
    torch.manual_seed(7)
    model = make_model()
    reference = make_model()
    reference.load_state_dict(model.state_dict())
    input_ids = torch.tensor([[1, 2, 3, 4]])

    observer = CaptureObserver()
    with observer(model):
        observed_output = model.generate(input_ids, max_new_tokens=3, do_sample=False)
    reference_output = reference.generate(input_ids, max_new_tokens=3, do_sample=False)

    assert torch.equal(observed_output, reference_output)
    assert any(context.phase is KVPhase.PREFILL for context in observer.contexts)
    assert any(context.phase is KVPhase.DECODE for context in observer.contexts)
    assert all(context.keys is not None and context.values is not None for context in observer.contexts)
    assert observer.contexts[0].forward_id == 0
    assert observer.contexts[1].layer_event_index == 1
    assert observer.last_context is None


def test_observer_composes_with_press_without_changing_press_result():
    model = make_model()
    observer = CaptureObserver()
    press = KnormPress(compression_ratio=0.5)
    input_ids = torch.tensor([[1, 2, 3, 4]])

    with observer(model):
        with press(model):
            outputs = model(input_ids, use_cache=True)

    assert outputs.past_key_values.get_seq_length() == 2
    prefill = [context for context in observer.contexts if context.phase is KVPhase.PREFILL]
    assert len(prefill) == 2
    assert all(context.stored_length == 4 for context in prefill)
