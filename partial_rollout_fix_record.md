# 异步 partial_rollout 修改记录

对比基线：`5ac083c5`（`[ascend, rollout] chore: restore dedicated baseline launcher`）与当前工作区。当前修改未提交。

实现路径为 V1 异步 trainer、CheckpointEngineManager 和 vLLM rollout；复现配置使用 `separate_async + vllm + nccl`。finish RPC 的引擎实现位于 vLLM，SGLang 和 TRT-LLM 未扩展相同接口。

官方 [V1 Async Trainer 文档](docs/advance/v1_async_trainer.md#partial-rollout-and-staleness) 说明：中断后保留 token 和 log probability，客户端重试并使用新版本权重，KV cache 为保留前缀重新构建。`True` 沿用这一行为，`False` 按本次要求增加旧请求完成后再安装新权重的处理。

## 1. 原来的问题

权重同步前，CheckpointEngineManager 始终调用 `abort_all_requests()`，与 `async_training.partial_rollout` 无关。客户端收到中断结果后，若配置为 `False`，就退出续写循环并返回已经生成的 token。因此，非空的半截回答会进入后续评分。

问题同时涉及权重同步前的中断时机和客户端的结束判断。只修改客户端重试，无法让已经中断的回答在旧权重上完成；只修改服务器等待，客户端仍会把其他来源的中断结果直接返回。

## 2. 修改逻辑

| 异步配置 | 权重更新前 | 权重更新后 |
| --- | --- | --- |
| `partial_rollout=True` | 关闭入口，中断已接入请求，客户端保留已有 token | 安装 actor 权重，恢复入口，客户端以原 prompt 加已有 token 继续生成 |
| `partial_rollout=False` | 关闭入口，已接入请求继续使用旧权重，直到正常结束 | 安装 actor 权重，恢复入口，新请求使用新权重 |

正常结束包括 EOS、stop 和长度上限。等待期间仍然调度已接入的请求，尚未接入的请求停在入口处。

同步 trainer 创建 manager 时固定传入 `True`，保持基线行为；异步 trainer 从 `async_training.partial_rollout` 读取配置，未配置时保持原有的 `True` 默认值。共享／混合 rollout manager 和独立 rollout manager 都接入该开关。

## 3. 核心实现

`CheckpointEngineManager.abort_replicas()` 按开关调用实际的 abort 或 finish RPC：

```python
if self.partial_rollout:
    await asyncio.gather(*[r.abort_all_requests(reject_request=reject_request) for r in self.replicas])
else:
    await asyncio.gather(*[r.finish_all_requests(reject_request=reject_request) for r in self.replicas])
```

`RolloutReplica.finish_all_requests()` 将调用转发到各 server。vLLM 的 node-rank 0 执行完成等待；其他节点运行 headless 引擎，由主节点的调度控制覆盖。

`False` 的等待分两层：

1. 设置 `_submission_paused=True`，清除 `_resume_event`，等待 `_admitting==0`。该计数在请求完成 `engine.add_request()` 注册后释放，确保关门前已接入的请求都进入引擎。
2. 保持引擎调度，等待 `output_processor.request_states` 为空。这一集合包含已注册但尚未完成的请求，包括等待首个 token 和被抢占后重新排队的请求。

```python
self._submission_paused = True
self._rejecting = reject_request
self._resume_event.clear()

while self._admitting > 0:
    await asyncio.sleep(0.01)

while self.engine.output_processor.request_states:
    await asyncio.sleep(0.01)

await self.engine.pause_generation(wait_for_inflight_requests=True, clear_cache=clear_cache)
```

上面展示 finish 路径；完整实现复用原来的暂停函数，abort 路径继续使用原来的接入超时逻辑。finish 路径等待接入完成和生成完成后才返回。

**不能直接用 vLLM 原生 wait 代替第二层等待。** 当前安装的 vLLM 0.30 在 wait 模式下设置 `PAUSED_NEW`，只继续运行 running 请求，等待队列停止调度；其未完成计数也只统计 running。提前暂停会留下排队或被抢占的旧请求。先等前端所有请求完成，再调用原生 wait，才能在安装权重前完成旧回答。

vLLM 0.32 及之后调用 `pause_generation(mode="wait")`；此前调用 `wait_for_inflight_requests=True`。缓存清理沿用基线逻辑。

NCCL 耦合传输的顺序为：**finish／abort 返回 → 释放 KV cache → 建立通信组 → actor 发送／rollout 接收权重 → 恢复 KV cache → 恢复生成**。支持 publish 的传输和 vLLM 原生分离传输也复用同一个等待入口。

客户端保留原来的 token 合并和前缀续写逻辑，删除 `False → should_retry=False → break` 的结束分支：

```python
# 4. check stop reason
# Weight updates abort requests only when partial rollout is enabled;
# otherwise the server finishes them on the current weights first.
# Retry aborted requests instead of returning an unfinished response.
if output.stop_reason not in ("aborted", "abort"):
    break

await asyncio.sleep(1)
```

`partial_rollout` 在服务器控制权重更新顺序。客户端收到 `abort/aborted` 时继续请求，正常停止或达到总长度预算时返回。`True` 的中断结果保留前缀，重试请求等待新权重安装后续写；`False` 的已接入请求在旧权重上完整结束，入口拒绝返回的空 `abort` 则重新获取服务器，避免空结果进入评分。同步 trainer 使用普通 LLMServerClient，其结束判断不变。

另外，server 的 `generate()` 在接入时保存 `global_steps`，返回时使用这个版本号。即使旧回答在权重更新后才被读取，它仍记录旧权重版本。客户端只将实际生成 token 的请求计入 `min/max_global_steps`，排除入口拒绝返回的空 `abort`：

```python
global_steps = output.extra_fields.get("global_steps", None)
# Empty rejections did not generate tokens on this version.
if output.token_ids:
    if min_global_steps is None:
        min_global_steps = global_steps
    max_global_steps = global_steps
```

V1 仍保留 off-policy 控制，但当前 replay buffer 根据 prompt 入队时的 `global_steps` 判断 drop／wait；`min/max_global_steps` 用于轨迹版本跨度和 staleness 的日志统计。排除空 abort 纠正这些统计，不改变当前 drop 判定。例如，版本 6 的空拒绝后在版本 7 完成生成，轨迹应记录版本 7、跨度 1。

## 4. 相对基线的代码 diff

以下是七个功能文件的完整 diff（同卡异步 trainer 只同步更新注释）：

```diff
diff --git a/verl/checkpoint_engine/base.py b/verl/checkpoint_engine/base.py
index cd989e8f..66269bee 100644
--- a/verl/checkpoint_engine/base.py
+++ b/verl/checkpoint_engine/base.py
@@ -432,14 +432,14 @@ class CheckpointEngineManager:
 
     1. Coupled strong synchronization (``support_publish_weights=False``), such as NCCL/NIXL:
        the trainer sender and rollout receiver must participate in the same transfer. The manager waits for the
-       previous update to commit, aborts in-flight generation, releases the rollout KV cache, builds the communication
+       previous update to commit, pauses in-flight generation, releases the rollout KV cache, builds the communication
        topology, and starts trainer send plus rollout receive. ``update_weights`` returns only after rollout installs
        the version, both sides finalize the transfer, and rollout generation resumes.
 
     2. Decoupled publish/commit (``support_publish_weights=True``), such as a trainer D2H snapshot followed by an
        asynchronous rollout RDMA read: after synchronously building the control-plane topology, the manager starts
        trainer publish and schedules rollout commit on its background event loop. ``update_weights`` returns once the
-       immutable trainer snapshot is published; in parallel, the background task aborts generation, releases the KV
+       immutable trainer snapshot is published; in parallel, the background task pauses generation, releases the KV
        cache, fetches and installs that version, finalizes the transfer, and resumes rollout.
 
     The manager allows at most one uncommitted version. Every new update first waits for the previous rollout commit,
@@ -449,6 +449,7 @@ class CheckpointEngineManager:
         config: The checkpoint engine config.
         actor_wg: The actor worker group (the training side that produces weights).
         replicas: The list of rollout replicas.
+        partial_rollout: Abort unfinished generation when True, otherwise finish it before updating weights.
     """
 
     def __init__(
@@ -456,6 +457,7 @@ class CheckpointEngineManager:
         config: CheckpointEngineConfig,
         actor_wg: RayWorkerGroup,
         replicas: list[RolloutReplica],
+        partial_rollout: bool = True,
     ) -> None:
         self.config = config
         self.backend = config.backend
@@ -463,6 +465,7 @@ class CheckpointEngineManager:
         self.backend_cls = CheckpointEngineRegistry.get(config.backend)
         self.actor_wg = actor_wg
         self.replicas = replicas
+        self.partial_rollout = partial_rollout
 
         self._update_ticket = None
         self._background_loop = asyncio.new_event_loop()
@@ -551,17 +554,20 @@ class CheckpointEngineManager:
 
     @auto_await
     async def abort_replicas(self, reject_request: bool = False):
-        """Abort all in-flight requests on every replica.
+        """Pause replicas by aborting or finishing in-flight generation.
 
         Args:
             reject_request: Fail requests arriving behind the closed gate instead of
                 parking them, for replicas that will not resume generation soon.
         """
-        await asyncio.gather(*[r.abort_all_requests(reject_request=reject_request) for r in self.replicas])
+        if self.partial_rollout:
+            await asyncio.gather(*[r.abort_all_requests(reject_request=reject_request) for r in self.replicas])
+        else:
+            await asyncio.gather(*[r.finish_all_requests(reject_request=reject_request) for r in self.replicas])
 
     @auto_await
     async def resume_generation_replicas(self):
-        """Resume generation on all replicas after abort_all_requests."""
+        """Resume generation on all replicas after the weight update."""
         await asyncio.gather(*[r.resume_generation() for r in self.replicas])
 
     @auto_await
diff --git a/verl/trainer/ppo/v1/trainer_base.py b/verl/trainer/ppo/v1/trainer_base.py
index 2c3cad84..60ab3187 100644
--- a/verl/trainer/ppo/v1/trainer_base.py
+++ b/verl/trainer/ppo/v1/trainer_base.py
@@ -406,6 +406,8 @@ class PPOTrainer(ABC):
             config=checkpoint_engine_config,
             actor_wg=self.actor_rollout_wg,
             replicas=self.llm_server_manager.get_replicas(),
+            partial_rollout=self.trainer_mode == "sync"
+            or self.config.get("async_training", {}).get("partial_rollout", True),
         )
         logger.info("checkpoint engine manager initialized")
 
diff --git a/verl/trainer/ppo/v1/trainer_colocate_async.py b/verl/trainer/ppo/v1/trainer_colocate_async.py
index 400de9f6..02c877af 100644
--- a/verl/trainer/ppo/v1/trainer_colocate_async.py
+++ b/verl/trainer/ppo/v1/trainer_colocate_async.py
@@ -20,7 +20,7 @@ from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient
 class PPOTrainerColocateAsync(PPOTrainer):
     """Asynchronous PPO trainer
     1. Trainer and rollout are colocated.
-    2. Partial rollout is enabled.
+    2. Generation resumes after weight synchronization.
     """
 
     def get_llm_client(self):
@@ -42,7 +42,7 @@ class PPOTrainerColocateAsync(PPOTrainer):
             self.checkpoint_manager.resume_generation_replicas()
 
     def on_sample_end(self):
-        # abort all unfinished requests and pause generation
+        # pause generation by aborting or finishing unfinished requests
         self.checkpoint_manager.abort_replicas()
         # sleep all replicas to discard weights and kv cache
         self.checkpoint_manager.sleep_replicas()
diff --git a/verl/trainer/ppo/v1/trainer_separate_async.py b/verl/trainer/ppo/v1/trainer_separate_async.py
index 9d62b890..1a2fa83d 100644
--- a/verl/trainer/ppo/v1/trainer_separate_async.py
+++ b/verl/trainer/ppo/v1/trainer_separate_async.py
@@ -181,6 +181,7 @@ class PPOTrainerSeparateAsync(PPOTrainer):
             config=checkpoint_engine_config,
             actor_wg=self.actor_rollout_wg,
             replicas=self.standalone_server_manager.get_replicas(),
+            partial_rollout=self.config.get("async_training", {}).get("partial_rollout", True),
         )
 
         # hybrid engine is in rollout mode after initialization
@@ -275,7 +276,7 @@ class PPOTrainerSeparateAsync(PPOTrainer):
             self._timed_switch_to_trainer()
 
     def _timed_switch_to_trainer(self) -> None:
-        """Switch Hybrid back to training and record the full remove/abort/sleep cost."""
+        """Switch Hybrid back to training and record the full remove/pause/sleep cost."""
         switch_start = time.perf_counter()
         with marked_timer("switch_to_trainer", self.timing_raw, color="cyan"):
             self.switch_to_trainer()
@@ -433,7 +434,7 @@ class PPOTrainerSeparateAsync(PPOTrainer):
         self.current_mode = HybridEngineMode.ROLLOUT
 
     def switch_to_trainer(self):
-        """Stop routing to Hybrid, abort partial requests, and return its GPU memory to training."""
+        """Stop routing to Hybrid, pause generation, and return its GPU memory to training."""
         if not self._enable_hybrid_replicas:
             return
         self.remove_replicas_from_balancer()
diff --git a/verl/workers/rollout/llm_server.py b/verl/workers/rollout/llm_server.py
index 931c06d4..05598f6a 100644
--- a/verl/workers/rollout/llm_server.py
+++ b/verl/workers/rollout/llm_server.py
@@ -342,9 +342,11 @@ class FullyAsyncLLMServerClient(LLMServerClient):
 
             # update model weights version
             global_steps = output.extra_fields.get("global_steps", None)
-            if min_global_steps is None:
-                min_global_steps = global_steps
-            max_global_steps = global_steps
+            # Empty rejections did not generate tokens on this version.
+            if output.token_ids:
+                if min_global_steps is None:
+                    min_global_steps = global_steps
+                max_global_steps = global_steps
 
             # 3. update max_new_tokens
             if original_max_tokens is not None:
@@ -354,12 +356,10 @@ class FullyAsyncLLMServerClient(LLMServerClient):
                     break
 
             # 4. check stop reason
-            # If partial rollout not enable, aborted samples will be dropped.
-            # For v1 trainer, should_retry is always True. Since self.config.async_training is not exist.
-            should_retry = True
-            if hasattr(self.config, "async_training") and not self.config.async_training.partial_rollout:
-                should_retry = False
-            if output.stop_reason not in ("aborted", "abort") or not should_retry:
+            # Weight updates abort requests only when partial rollout is enabled;
+            # otherwise the server finishes them on the current weights first.
+            # Retry aborted requests instead of returning an unfinished response.
+            if output.stop_reason not in ("aborted", "abort"):
                 break
 
             await asyncio.sleep(1)
diff --git a/verl/workers/rollout/replica.py b/verl/workers/rollout/replica.py
index 958e0ad0..1feccffa 100644
--- a/verl/workers/rollout/replica.py
+++ b/verl/workers/rollout/replica.py
@@ -305,9 +305,15 @@ class RolloutReplica(ABC):
         )
 
     async def resume_generation(self):
-        """Resume generation on all servers after abort_all_requests."""
+        """Resume generation on all servers after aborting or finishing requests."""
         await asyncio.gather(*[server.resume_generation.remote() for server in self.servers])
 
+    async def finish_all_requests(self, reject_request: bool = False):
+        """Finish admitted requests on every server and keep new requests paused."""
+        return await asyncio.gather(
+            *[server.finish_all_requests.remote(reject_request=reject_request) for server in self.servers]
+        )
+
     async def clear_kv_cache(self):
         """reset kv cache in each rollout server."""
         await asyncio.gather(*[server.clear_kv_cache.remote() for server in self.servers])
diff --git a/verl/workers/rollout/vllm_rollout/vllm_async_server.py b/verl/workers/rollout/vllm_rollout/vllm_async_server.py
index 2ff5b9fd..528d4356 100644
--- a/verl/workers/rollout/vllm_rollout/vllm_async_server.py
+++ b/verl/workers/rollout/vllm_rollout/vllm_async_server.py
@@ -759,6 +759,7 @@ class vLLMHttpServer:
         if rejected is not None:
             return rejected
 
+        global_steps = self.global_steps
         cache_salt = prefix_cache_salt(self.config, self.global_steps)
         if cache_salt is not None:
             prompt["cache_salt"] = cache_salt
@@ -777,7 +778,7 @@ class vLLMHttpServer:
             )
             assert final_res is not None
 
-        extra_fields = {"global_steps": self.global_steps}
+        extra_fields = {"global_steps": global_steps}
         # Handle abort case: when the request is aborted by pause_generation(abort),
         # outputs may be empty. Return empty results with stop_reason="aborted"
         # instead of crashing with "IndexError: list index out of range".
@@ -1173,6 +1174,12 @@ class vLLMHttpServer:
     async def wait_for_requests_to_drain(self):
         await self.engine.wait_for_requests_to_drain()
 
+    async def finish_all_requests(self, reject_request: bool = False) -> dict[str, Any]:
+        """Finish all admitted requests on the current weights, then pause the engine."""
+        if self.node_rank != 0:
+            return {"finished_count": 0, "request_ids": []}
+        return await self._pause_requests(reject_request=reject_request, wait_for_inflight_requests=True)
+
     async def abort_all_requests(
         self,
         reset_prefix_cache: bool = True,
@@ -1230,6 +1237,15 @@ class vLLMHttpServer:
             await self.engine.abort([*request_ids, *parent_ids], internal=True)
             return {"aborted_count": len(request_ids), "request_ids": request_ids}
 
+        return await self._pause_requests(reset_prefix_cache=reset_prefix_cache, reject_request=reject_request)
+
+    async def _pause_requests(
+        self,
+        reset_prefix_cache: bool = True,
+        reject_request: bool = False,
+        wait_for_inflight_requests: bool = False,
+    ) -> dict[str, Any]:
+        """Close admission, abort or finish existing requests, and pause scheduling."""
         try:
             # Close the gate first, then let admissions already past it land, so the pause
             # below actually covers them. Closing the gate and declaring how late arrivals
@@ -1239,7 +1255,7 @@ class vLLMHttpServer:
             self._resume_event.clear()
             deadline = time.monotonic() + _GATE_BARRIER_TIMEOUT_S
             while self._admitting > 0:
-                if time.monotonic() > deadline:
+                if not wait_for_inflight_requests and time.monotonic() > deadline:
                     if self.config.external_kv_cache_invalidation == "salt":
                         # A straggler would run on the new weights under its old salt and write
                         # into the kept KV store, so do not proceed with the weight update.
@@ -1256,9 +1272,15 @@ class vLLMHttpServer:
             # Snapshot request IDs before pausing for reporting
             request_ids = list(self.engine.output_processor.request_states.keys())
 
-            # pause_generation in abort mode will:
+            if wait_for_inflight_requests:
+                # Keep scheduling queued and preempted requests on the old weights.
+                # Native wait pauses that queue, so drain it before pausing the engine.
+                while self.engine.output_processor.request_states:
+                    await asyncio.sleep(0.01)
+
+            # pause_generation will:
             # 1. Set engine to paused state (new requests are accepted but not scheduled)
-            # 2. Abort all in-flight requests
+            # 2. Abort in-flight requests, or wait for the completed generation to drain
             # 3. Wait for requests to drain
             # 4. Clear prefix and mm caches if clear_cache=True.
             #    EngineCore._reset_caches defaults reset_connector=True
@@ -1271,22 +1293,27 @@ class vLLMHttpServer:
             reset_connector = self.config.external_kv_cache_invalidation == "reset"
             clear_cache = reset_prefix_cache and reset_connector
             if _VLLM_VERSION >= version.parse("0.32.0"):
-                await self.engine.pause_generation(mode="abort", clear_cache=clear_cache)
+                await self.engine.pause_generation(
+                    mode="wait" if wait_for_inflight_requests else "abort", clear_cache=clear_cache
+                )
             else:
                 await self.engine.pause_generation(
-                    wait_for_inflight_requests=False,
+                    wait_for_inflight_requests=wait_for_inflight_requests,
                     clear_cache=clear_cache,
                 )
             if reset_prefix_cache and not reset_connector:
                 await self.clear_kv_cache()
 
+            if wait_for_inflight_requests:
+                logger.info(f"Finished {len(request_ids)} requests on the old weights: {request_ids}")
+                return {"finished_count": len(request_ids), "request_ids": request_ids}
             logger.info(f"Aborted {len(request_ids)} requests: {request_ids}")
             return {"aborted_count": len(request_ids), "request_ids": request_ids}
 
         except Exception:
             # Weight updates must not proceed unless every in-flight request was
-            # actually aborted and the old-weight caches were cleared.
-            logger.exception("Error aborting requests")
+            # actually aborted or finished and the old-weight caches were cleared.
+            logger.exception("Error pausing requests")
             raise
 
     async def resume_engine_generation(self):
```

## 5. 验证

新增覆盖复用三个已有测试文件：

| 测试文件 | 覆盖内容 |
| --- | --- |
| `tests/checkpoint_engine/test_d2h_checkpoint_engine.py` | 参数化 True／False，调用实际 manager 分支；旧请求未完成时，不释放 KV cache、不建立通信组、不发布权重 |
| `tests/workers/rollout/rollout_vllm/test_submitter_gate_on_cpu.py` | True／False 等待接入注册；旧／新 vLLM 暂停接口；排队和被抢占请求完成；新请求等待；延迟读取保留旧版本；实际客户端、replica、server 和 NCCL checkpoint manager 的权重同步联动顺序 |
| `tests/workers/rollout/test_llm_server_routed_experts_on_cpu.py` | 显式 True／False 配置下，空 abort 重新请求且不计入生成版本；True 保留前缀续写；False 返回旧版本的完整回答 |

七个文件的回归测试：**190 项通过，1 项原有 Ray 快照集成测试未选入**。相对基线新增 8 个用例，集中在上述三个已有测试文件；响应长度测试文件只同步更新了旧注释。pre-commit 全部通过。

验证使用 conda `verl` Python 和启动脚本所用的 CANN、ATB、driver 库；测试为 CPU 模拟，尚未运行四卡训练、四卡 rollout 的 NPU 实验。

原 Omega 配置、数据、训练／生成超参数和 CANN／ATB／driver 激活参数与 `5ac083c5` 一致。另提供 True／False 两份配置，完整 Hydra 配置只在 `async_training.partial_rollout` 上有区别。现有启动脚本接受 `true` 或 `false`，自动选择对应配置和独立 RUN_ID；不传选择参数时保持原来的配置和 RUN_ID 行为。脚本语法及选择逻辑已验证。


## 6. 行为审核

| 要求 | 实现和验证 |
| --- | --- |
| 异步 True 先安装新权重，再沿旧回答续写 | manager 等待接收和安装结束后恢复 replica；客户端用原 prompt 加旧 token 请求。联动测试在安装完成前保持权重传输阻塞，检查重试已发起但新请求尚未进入引擎；安装后续写使用版本 8、前缀为版本 7 的 token |
| 异步 False 先完成旧回答，再安装新权重 | server 关闭入口，等待接入注册和所有 request states 完成后暂停；manager 此后才传输权重。联动测试检查旧回答完成前传输未启动，完整回答的版本为 7，随后才安装版本 8 |
| 同步训练保持原行为 | 同步 manager 固定使用原来的 abort 路径，同步 trainer 使用普通 LLMServerClient；本次改动的结束判断位于异步客户端 |
| 版本记录对应实际生成 | server 在接入时保存版本；客户端排除空 abort 的版本。该修正影响日志中的轨迹版本统计，当前 replay buffer 的 prompt 年龄筛选保持原逻辑 |

联动测试调用实际的客户端、replica、server 和 manager 方法，Ray RPC、权重传输和引擎 token 输出使用 CPU 模拟。测试验证调用顺序和数据归因，四卡训练／四卡 rollout 的 NPU 实验仍需运行。

