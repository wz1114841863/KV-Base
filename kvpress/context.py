# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution context for KV-Cache research instrumentation.

The context is deliberately a data carrier.  It does not own cache mutation,
and it does not require Q or attention weights to be materialized.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import torch


class KVPhase(str, Enum):
    """Phase of the model forward that produced a KV event."""

    PREFILL = "prefill"
    DECODE = "decode"


def is_prefilling(cache_position: torch.Tensor | None, q_len: int) -> bool:
    """Preserve KVPress's initial-prefill predicate in one shared location."""

    if cache_position is None or cache_position.numel() == 0:
        return False
    prefilling = cache_position[-1] + 1 == q_len
    return bool(prefilling.item() if isinstance(prefilling, torch.Tensor) else prefilling)


@dataclass
class KVExecutionLifecycle:
    """Shared phase and forward-event state for one model execution scope.

    A lifecycle event is emitted once per attention-layer hook. ``layer_id ==
    0`` starts a new model forward, so all layers in that forward share the
    same ``forward_id`` and decode step. The lifecycle does not mutate cache
    state and can be used by both Press and Observer paths.
    """

    prefill_length: Optional[int] = None
    forward_id: int = -1
    layer_event_index: int = 0
    current_phase: Optional[KVPhase] = None
    _decode_step: Optional[int] = None

    def reset(self) -> None:
        self.prefill_length = None
        self.forward_id = -1
        self.layer_event_index = 0
        self.current_phase = None
        self._decode_step = None

    def begin_sample(self) -> None:
        """Start a fresh sample/generation execution scope."""

        self.reset()

    def end_sample(self) -> None:
        """End a sample/generation execution scope."""

        self.reset()

    def observe(self, cache_position: torch.Tensor | None, q_len: int, layer_id: int) -> dict[str, Any]:
        """Return normalized lifecycle metadata for one attention-layer event."""

        new_forward = layer_id == 0 or self.forward_id < 0
        if new_forward:
            self.forward_id += 1
            self.layer_event_index = 0
        else:
            self.layer_event_index += 1

        phase = KVPhase.PREFILL if is_prefilling(cache_position, q_len) else KVPhase.DECODE
        if phase is KVPhase.PREFILL:
            self.prefill_length = q_len

        total_length = None
        if isinstance(cache_position, torch.Tensor) and cache_position.numel() > 0:
            total_length = int(cache_position[-1].item()) + 1

        if new_forward and phase is KVPhase.DECODE:
            self._decode_step = 0 if self._decode_step is None else self._decode_step + 1
        decode_step = self._decode_step if phase is KVPhase.DECODE else None

        self.current_phase = phase
        return {
            "phase": phase,
            "decode_step": decode_step,
            "prefill_length": self.prefill_length,
            "total_length": total_length,
            "forward_id": self.forward_id,
            "layer_event_index": self.layer_event_index,
        }


@dataclass
class KVContext:
    """Metadata and tensors visible at an attention-layer hook.

    ``total_length`` is the logical position length from ``cache_position``;
    ``stored_length`` is the current physical sequence dimension of K/V.  They
    can differ after a Press has compressed the cache.
    """

    phase: KVPhase
    layer_id: int
    forward_id: int
    layer_event_index: int
    decode_step: Optional[int]
    prefill_length: Optional[int]
    total_length: Optional[int]
    stored_length: Optional[int]
    cache_position: Optional[torch.Tensor]
    input_ids: Optional[torch.Tensor]
    position_ids: Optional[torch.Tensor]
    hidden_states: Optional[torch.Tensor]
    keys: Optional[torch.Tensor]
    values: Optional[torch.Tensor]
    query: Optional[torch.Tensor]
    attention: Optional[torch.Tensor]
    cache: Any
    recorder: Any = None

    @property
    def q(self) -> Optional[torch.Tensor]:
        """Alias for the optional query tensor used by research code."""

        return self.query

    @property
    def k(self) -> Optional[torch.Tensor]:
        """Short alias for cached keys."""

        return self.keys

    @property
    def v(self) -> Optional[torch.Tensor]:
        """Short alias for cached values."""

        return self.values
