# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only KV-cache observation hooks."""

import logging
from contextlib import contextmanager
from typing import Generator

import torch
from torch import nn
from transformers import Gemma3ForConditionalGeneration, PreTrainedModel

from kvpress.context import KVContext, KVExecutionLifecycle, KVPhase
from kvpress.presses.base_press import SUPPORTED_MODELS
from kvpress.utils import extract_keys_and_values

logger = logging.getLogger(__name__)


class KVObserver:
    """Base class for read-only attention-layer observation.

    Subclasses can override :meth:`on_layer`, :meth:`on_prefill`, or
    :meth:`on_decode`.  The observer never writes to the cache and does not
    alter the attention output.  Tensor references are passed directly; an
    observer that needs to retain them should explicitly clone or move them.
    """

    def __init__(self, recorder=None):
        self.recorder = recorder
        self._last_context: KVContext | None = None
        self.lifecycle = KVExecutionLifecycle()

    @property
    def last_context(self) -> KVContext | None:
        """Return the most recently observed context while attached."""

        return self._last_context

    def on_layer(self, context: KVContext) -> None:
        """Handle every attention-layer event."""

    def on_prefill(self, context: KVContext) -> None:
        """Handle a prefill attention-layer event."""

    def on_decode(self, context: KVContext) -> None:
        """Handle a decode attention-layer event."""

    def _reset(self) -> None:
        self._last_context = None
        if not hasattr(self, "lifecycle"):
            self.lifecycle = KVExecutionLifecycle()
        self.lifecycle.reset()

    def _make_context(self, module, kwargs: dict, output, hidden_states, keys, values) -> KVContext:
        q_len = hidden_states.shape[1]
        cache_position = kwargs.get("cache_position")
        if not hasattr(self, "lifecycle"):
            self.lifecycle = KVExecutionLifecycle()
        metadata = self.lifecycle.observe(kwargs.get("cache_position"), q_len, int(module.layer_idx))

        context = KVContext(
            phase=metadata["phase"],
            layer_id=int(module.layer_idx),
            forward_id=metadata["forward_id"],
            layer_event_index=metadata["layer_event_index"],
            decode_step=metadata["decode_step"],
            prefill_length=metadata["prefill_length"],
            total_length=metadata["total_length"],
            stored_length=int(keys.shape[2]),
            cache_position=cache_position,
            input_ids=kwargs.get("input_ids"),
            position_ids=kwargs.get("position_ids"),
            hidden_states=hidden_states,
            keys=keys,
            values=values,
            query=None,
            attention=output[1] if len(output) > 1 else None,
            cache=kwargs.get("past_key_values"),
            recorder=getattr(self, "recorder", None),
        )
        self._last_context = context
        kwargs["kv_context"] = context
        return context

    def _forward_hook(self, module: nn.Module, input, kwargs: dict, output):
        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        keys, values = extract_keys_and_values(cache, module.layer_idx)
        context = self._make_context(module, kwargs, output, hidden_states, keys, values)

        self.on_layer(context)
        if context.phase is KVPhase.PREFILL:
            self.on_prefill(context)
        else:
            self.on_decode(context)
        return output

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        """Attach read-only hooks to all supported attention layers."""

        if not isinstance(model, SUPPORTED_MODELS):
            logger.warning("Model %s not tested, supported models: %s", type(model), SUPPORTED_MODELS)
        if isinstance(model, Gemma3ForConditionalGeneration):
            logger.warning_once("Observation in Gemma3 skips sliding-window attention layers")

        self._reset()
        hooks = []
        try:
            language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
            for layer in language_model.layers:
                if isinstance(model, Gemma3ForConditionalGeneration) and layer.self_attn.is_sliding:
                    continue
                layer.self_attn.rotary_emb = language_model.rotary_emb
                hooks.append(layer.self_attn.register_forward_hook(self._forward_hook, with_kwargs=True))
            yield self
        finally:
            for hook in hooks:
                hook.remove()
            self._reset()
