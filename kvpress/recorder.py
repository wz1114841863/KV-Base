# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple file-backed recorder for KV-cache research experiments."""

import json
import time
from pathlib import Path
from typing import Any

import torch

from kvpress.context import KVContext


def _jsonable(value: Any) -> Any:
    """Convert common research values to JSON-compatible data."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return {
            "type": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "value"):
        return value.value
    return str(value)


class KVRecorder:
    """Transparent on-disk recorder using JSON, JSONL, and optional tensor files.

    The recorder keeps writes explicit.  It never records full tensors unless
    :meth:`record_tensor` is called.
    """

    def __init__(self, root_dir: str | Path, run_name: str | None = None):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.root_dir = Path(root_dir)
        self.run_name = run_name or timestamp
        self.run_dir = self.root_dir / self.run_name
        self.samples_dir = self.run_dir / "samples"
        self.tensors_dir = self.run_dir / "tensors"
        self.current_sample_id = "default"
        self._closed = False

        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.tensors_dir.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def close(self) -> None:
        self._closed = True

    def set_run_metadata(self, metadata: dict[str, Any]) -> None:
        self._write_json(self.run_dir / "run_metadata.json", metadata)

    def set_sample_metadata(self, sample_id: str, metadata: dict[str, Any] | None = None, **kwargs) -> None:
        self.current_sample_id = str(sample_id)
        payload = dict(metadata or {})
        payload.update(kwargs)
        self._sample_dir(self.current_sample_id).mkdir(parents=True, exist_ok=True)
        self._write_json(self._sample_dir(self.current_sample_id) / "sample_metadata.json", payload)

    def record_scalar(self, name: str, value: Any, sample_id: str | None = None, **indices) -> None:
        self._append_sample_jsonl(
            "scalars.jsonl",
            {"name": name, "value": value, "indices": indices, "time": time.time()},
            sample_id=sample_id,
        )

    def record_stat(self, name: str, value: Any, sample_id: str | None = None, **indices) -> None:
        self._append_sample_jsonl(
            "stats.jsonl",
            {"name": name, "value": value, "indices": indices, "time": time.time()},
            sample_id=sample_id,
        )

    def record_trace(self, event: dict[str, Any], sample_id: str | None = None) -> None:
        payload = dict(event)
        payload.setdefault("time", time.time())
        self._append_sample_jsonl("events.jsonl", payload, sample_id=sample_id)

    def record_context(self, context: KVContext, sample_id: str | None = None, **extra) -> None:
        """Record lightweight context metadata without dumping tensor payloads."""

        event = {
            "event": "kv_context",
            "phase": context.phase,
            "layer_id": context.layer_id,
            "decode_step": context.decode_step,
            "prefill_length": context.prefill_length,
            "total_length": context.total_length,
            "stored_length": context.stored_length,
        }
        event.update(extra)
        self.record_trace(event, sample_id=sample_id)

    def record_tensor(self, name: str, tensor: torch.Tensor, sample_id: str | None = None, **indices) -> Path:
        sample = sample_id or self.current_sample_id
        sample_tensor_dir = self._sample_dir(sample) / "tensors"
        sample_tensor_dir.mkdir(parents=True, exist_ok=True)
        tensor_index = sum(1 for _ in sample_tensor_dir.glob(f"{name}-*.pt"))
        path = sample_tensor_dir / f"{name}-{tensor_index:06d}.pt"
        torch.save(tensor.detach().cpu(), path)
        self._append_sample_jsonl(
            "tensors.jsonl",
            {
                "name": name,
                "path": path.relative_to(self.run_dir),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "indices": indices,
                "time": time.time(),
            },
            sample_id=sample,
        )
        return path

    def _sample_dir(self, sample_id: str) -> Path:
        return self.samples_dir / str(sample_id)

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        self._ensure_open()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")

    def _append_sample_jsonl(self, filename: str, payload: dict[str, Any], sample_id: str | None = None) -> None:
        self._ensure_open()
        sample = sample_id or self.current_sample_id
        sample_dir = self._sample_dir(sample)
        sample_dir.mkdir(parents=True, exist_ok=True)
        with (sample_dir / filename).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("KVRecorder is closed")
