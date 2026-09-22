# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import torch

from kvpress import KVObserver, KVRecorder
from tests.test_observer import CaptureObserver, make_model


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_recorder_writes_metadata_records_and_tensors(tmp_path):
    recorder = KVRecorder(tmp_path, run_name="run-a")
    recorder.set_run_metadata({"model": "synthetic", "seed": 1})
    recorder.set_sample_metadata("sample-0", {"prompt_length": 4})
    recorder.record_scalar("accuracy", torch.tensor(1.0), split="smoke")
    recorder.record_stat("kv_norm_mean", 0.25, layer=1, head=2)
    recorder.record_trace({"event": "decode_step", "step": 0})
    tensor_path = recorder.record_tensor("k_norm", torch.arange(4).reshape(2, 2), layer=0)
    recorder.close()

    run_dir = tmp_path / "run-a"
    assert json.loads((run_dir / "run_metadata.json").read_text())["model"] == "synthetic"
    assert json.loads((run_dir / "samples" / "sample-0" / "sample_metadata.json").read_text())["prompt_length"] == 4
    assert read_jsonl(run_dir / "samples" / "sample-0" / "scalars.jsonl")[0]["value"] == 1.0
    assert read_jsonl(run_dir / "samples" / "sample-0" / "stats.jsonl")[0]["indices"] == {"head": 2, "layer": 1}
    assert read_jsonl(run_dir / "samples" / "sample-0" / "events.jsonl")[0]["event"] == "decode_step"
    assert tensor_path.exists()
    assert torch.equal(torch.load(tensor_path), torch.arange(4).reshape(2, 2))
    tensor_index = read_jsonl(run_dir / "samples" / "sample-0" / "tensors.jsonl")[0]
    assert tensor_index["shape"] == [2, 2]


def test_recorder_records_context_metadata(tmp_path):
    model = make_model()
    recorder = KVRecorder(tmp_path, run_name="ctx-run")

    class RecordingObserver(KVObserver):
        def on_layer(self, context):
            context.recorder.record_context(context, sample_id="s0")

    observer = RecordingObserver(recorder=recorder)
    with observer(model):
        model(torch.tensor([[1, 2, 3, 4]]), use_cache=True)

    events = read_jsonl(tmp_path / "ctx-run" / "samples" / "s0" / "events.jsonl")
    assert len(events) == 2
    assert events[0]["event"] == "kv_context"
    assert events[0]["phase"] == "prefill"
    assert events[0]["stored_length"] == 4


def test_observer_does_not_record_implicitly(tmp_path):
    model = make_model()
    recorder = KVRecorder(tmp_path, run_name="quiet-run")
    observer = CaptureObserver()
    observer.recorder = recorder

    with observer(model):
        model(torch.tensor([[1, 2, 3, 4]]), use_cache=True)

    assert observer.contexts[0].recorder is recorder
    assert not list((tmp_path / "quiet-run" / "samples").glob("*/events.jsonl"))
