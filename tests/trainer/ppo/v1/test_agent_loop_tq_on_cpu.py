# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio

import pytest
import torch
from omegaconf import OmegaConf

import verl.trainer.ppo.v1.agent_loop_tq as agent_loop_tq_module
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, AgentLoopWorker
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopWorkerTQ, _settle_session_tasks, apply_greedy_sampling_params
from verl.trainer.ppo.v1.replay_buffer import ReplayBufferAsync
from verl.workers.rollout.llm_server import RolloutAbortedError


def test_settle_session_tasks_waits_for_siblings_after_failure():
    async def run():
        settled = asyncio.Event()

        async def fail():
            raise RuntimeError("session failed")

        async def finish_later():
            await asyncio.sleep(0.01)
            settled.set()

        tasks = [asyncio.create_task(fail()), asyncio.create_task(finish_later())]
        errors = await _settle_session_tasks(tasks)

        assert settled.is_set()
        assert all(task.done() for task in tasks)
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)

    asyncio.run(run())


_WorkerTQ = AgentLoopWorkerTQ.__ray_metadata__.modified_class


class _DummyWorker:
    _compute_multi_modal_inputs = AgentLoopWorker._compute_multi_modal_inputs
    _compute_position_ids = AgentLoopWorker._compute_position_ids
    _compute_score = AgentLoopWorker._compute_score
    _compute_teacher_logprobs = AgentLoopWorker._compute_teacher_logprobs
    distillation_enabled = False

    def __init__(self, topk_log_probs: int):
        self.rollout_config = OmegaConf.create({"topk_log_probs": topk_log_probs})
        self.processor = None
        self.reward_loop_worker_handles = None


def _output(extra_fields: dict) -> AgentLoopOutput:
    return AgentLoopOutput(
        prompt_ids=[101, 102, 103],
        response_ids=[11, 12],
        response_mask=[1, 1],
        num_turns=2,
        metrics=AgentLoopMetrics(),
        extra_fields=extra_fields,
    )


async def _postprocess(monkeypatch, worker: _DummyWorker, output: AgentLoopOutput, validate: bool):
    captured = {}

    async def async_kv_batch_put(*, keys, fields, tags, partition_id):
        captured["fields"] = fields

    monkeypatch.setattr(agent_loop_tq_module.tq, "async_kv_batch_put", async_kv_batch_put)
    await _WorkerTQ._agent_loop_postprocess(
        worker, output, validate, uid="u0", session_id=0, global_steps=1, raw_prompt=[{"role": "user", "content": "hi"}]
    )
    return captured["fields"]


@pytest.mark.asyncio
async def test_agent_loop_tq_postprocess_stores_unpadded_rollout_topk(monkeypatch):
    output = _output(
        {"response_topk_ids": [[11, 13], [12, 14]], "response_topk_log_probs": [[-0.1, -2.0], [-0.2, -1.5]]}
    )

    fields = await _postprocess(monkeypatch, _DummyWorker(topk_log_probs=2), output, validate=False)

    ids, log_probs = fields["rollout_topk_ids"][0], fields["rollout_topk_log_probs"][0]
    assert ids.shape == (5, 2) and ids.dtype == torch.int32
    assert log_probs.shape == (5, 2) and log_probs.dtype == torch.float32
    assert ids[2].tolist() == [11, 13] and ids[3].tolist() == [12, 14]
    assert log_probs[2].tolist() == pytest.approx([-0.1, -2.0]) and log_probs[3].tolist() == pytest.approx([-0.2, -1.5])
    assert ids[0].tolist() == [0, 1] and ids[4].tolist() == [0, 1]
    assert "response_topk_ids" not in fields["extra_fields"][0]
    assert fields["rollout_support_complete"][0].tolist() == [True, True, False, False, True]


@pytest.mark.asyncio
async def test_agent_loop_tq_postprocess_stores_nested_provider_coverage(monkeypatch):
    output = _output(
        {
            "response_topk_ids": [[11, 13], [12, 14]],
            "response_topk_log_probs": [[0.0, float("-inf")], [-0.2, -1.5]],
            "response_support_complete": [True, False],
        }
    )
    short_output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[11],
        response_mask=[1],
        num_turns=2,
        metrics=AgentLoopMetrics(),
        extra_fields={
            "response_topk_ids": [[11, 13]],
            "response_topk_log_probs": [[-0.1, -2.0]],
            "response_support_complete": [False],
        },
    )
    fields = await _postprocess(monkeypatch, _DummyWorker(topk_log_probs=2), [output, short_output], validate=False)
    complete = fields["rollout_support_complete"]
    assert complete.is_nested and complete.dtype == torch.bool
    assert complete[0].tolist() == [True, True, True, False, True]
    assert complete[1].tolist() == [True, False, True]


def test_greedy_reward_baseline_does_not_request_score_centering_heads():
    params = {"top_p": 0.9, "top_k": 64, "temperature": 1.0, "topk_log_probs": 128, "logprobs": True}
    apply_greedy_sampling_params(params)
    assert params == {"top_p": 1.0, "top_k": -1, "temperature": 0, "logprobs": True}


@pytest.mark.asyncio
async def test_agent_loop_tq_postprocess_skips_rollout_topk_on_validate(monkeypatch):
    fields = await _postprocess(monkeypatch, _DummyWorker(topk_log_probs=2), _output({}), validate=True)

    assert "rollout_topk_ids" not in fields.keys()
    assert "rollout_topk_log_probs" not in fields.keys()


@pytest.mark.asyncio
async def test_abort_discards_whole_group_after_siblings_settle_and_refills(monkeypatch):
    # Exercise prompt lifecycle, postprocessing and replay sampling with mocked TQ transport.
    metadata = {"train": {"u0": {"is_prompt": True, "status": "pending", "global_steps": 7}}}
    scored_sessions, cleared, refills = [], [], []
    aborted, two_scored, release_last = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def put(*, key, partition_id, tag):
        metadata[partition_id].setdefault(key, {}).update(tag)

    async def put_batch(*, keys, fields, tags, partition_id):
        for key, tag in zip(keys, tags, strict=True):
            metadata[partition_id][key] = tag

    def clear(*, keys, partition_id):
        cleared.append(set(keys))
        for key in keys:
            metadata[partition_id].pop(key, None)

    monkeypatch.setattr(agent_loop_tq_module.tq, "async_kv_put", put)
    monkeypatch.setattr(agent_loop_tq_module.tq, "async_kv_batch_put", put_batch)
    monkeypatch.setattr(agent_loop_tq_module.tq, "kv_list", lambda: metadata)
    monkeypatch.setattr(agent_loop_tq_module.tq, "kv_clear", clear)

    class Worker(_DummyWorker):
        def __init__(self):
            super().__init__(topk_log_probs=0)
            self.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"n": 4}}})

        async def _compute_score(self, outputs, kwargs):
            scored_sessions.append(kwargs["session_id"])
            for output in outputs:
                output.reward_score = 1.0
            if len(scored_sessions) == 2:
                two_scored.set()

        async def _run_agent_loop(self, sampling_params, *, trajectory, trace, session_id, **prompt):
            if session_id == 0:
                aborted.set()
                raise RolloutAbortedError("rollout interrupted")
            if session_id == 3:
                await release_last.wait()
            await _WorkerTQ._agent_loop_postprocess(
                self, _output({"stop_reason": "completed"}), False, session_id=session_id, **prompt
            )

    task = asyncio.create_task(
        _WorkerTQ._run_prompt(Worker(), {"uid": "u0", "global_steps": 7}, {}, {"validate": False})
    )
    try:
        await asyncio.wait_for(aborted.wait(), timeout=2)
        await asyncio.wait_for(two_scored.wait(), timeout=2)
        assert metadata["train"]["u0"]["status"] == "running"
        assert "u0_0_0" not in metadata["train"]
        assert not task.done(), "The failed group must remain running while a sibling can still write"
    finally:
        release_last.set()
        await asyncio.wait_for(task, timeout=2)

    assert metadata["train"]["u0"]["status"] == "failure"
    assert set(scored_sessions) == {1, 2, 3}
    assert {key for key in metadata["train"] if key != "u0"} == {"u0_1_0", "u0_2_0", "u0_3_0"}

    def refill(count):
        refills.append(count)
        metadata["train"]["replacement"] = {"is_prompt": True, "status": "finished", "global_steps": 8}
        for session in range(4):
            metadata["train"][f"replacement_{session}_0"] = {
                "status": "success",
                "global_steps": 8,
                "min_global_steps": 8,
                "max_global_steps": 8,
            }

    replay = ReplayBufferAsync(
        trainer_mode="separate_async",
        trainer_config={},
        max_off_policy_threshold=8,
        max_off_policy_strategy="drop",
        sampler_kwargs={},
        refill_fn=refill,
    )
    batch, metrics = replay.sample(global_steps=8, partition_id="train", batch_size=1)

    assert refills == [1], "One failed GRPO group must produce exactly one replacement prompt"
    assert cleared[0] == {"u0", "u0_1_0", "u0_2_0", "u0_3_0"}
    assert set(batch.keys) == {f"replacement_{session}_0" for session in range(4)}
    assert metrics["training/rollout_failure/evicted_samples"] == 1
    assert not any(key.startswith("u0") for key in metadata["train"])
