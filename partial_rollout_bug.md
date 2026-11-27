# Verl源代码 Partial Rollout Bug

## verl 文档中的参数语义

verl 的历史 Fully Async 文档用 `async_training.partial_rollout` 描述权重同步时如何处理在途生成。当 `staleness_threshold > 0` 时，文档写明，设为 `True` 会在同步前打断在途回复，换权重后从已有 token 继续生成。设为 `False` 会停止提交新任务，等在途任务生成完毕，再同步权重。 [verl 原始文档，模式 3 和模式 4](https://github.com/verl-project/verl/blob/5371b9989c64077917ac5a352ffdaa79b5898e72/docs/advance/fully_async.md#L210-L229)

**“**`**False**` **会等在途回复生成完毕再同步”对本文的 V1** `**separate_async**` **+ vLLM 路径是错误的。** 这条路径在安装权重前主动中断在途请求，`False` 只让客户端停止续写。历史文档描述的是旧版 Fully Async 流程，当前仓库的文档也明确将该流程归为 V0，并另行指向 V1。 [当前 verl 文档](docs/advance/fully_async.md) · [V1 使用的权重同步实现](verl/checkpoint_engine/base.py) · [生成客户端](verl/workers/rollout/llm_server.py)

在 V1 `separate_async` + vLLM 的 3200 题实验中，`partial_rollout=False` 会停止续写被权重同步打断的回复。**如果 vLLM 交回了非空 token，这段回复仍会算 reward，并且可以进入训练 batch。** 原始提交说明要求丢弃被中断的样本，当前这条数据流缺少对应的淘汰步骤。

源码以实验提交 `dec97fcecb6970d6eec78a4259c916555e7f2f3a` 为准。各节标出该提交的行号，并链接到本地文件。原始重试分支来自 `5371b9989`（zhang shuai，#6882）。

## 1. 权重同步中断生成

实验配置为 V1 `separate_async`，每步同步一次权重，每题生成四条回复。Rollout 安装新权重前，checkpoint manager 调用 `abort_replicas()`。vLLM 以 abort 模式暂停调度。引擎交回的结果决定这一条回复还能返回多少 token。 [实验配置](jiutian/cross_dc/config/baseline_3200/Omega_3dc.yaml) · [权重同步](verl/checkpoint_engine/base.py) · [vLLM 服务](verl/workers/rollout/vllm_rollout/vllm_async_server.py)

`vllm_async_server.py:1202–1208` 的暂停代码。

```python
if _VLLM_VERSION >= version.parse("0.32.0"):
    await self.engine.pause_generation(mode="abort", clear_cache=reset_prefix_cache)
else:
    await self.engine.pause_generation(
        wait_for_inflight_requests=False,
        clear_cache=reset_prefix_cache,
    )
```

如果 `final_res.outputs` 为空，服务端构造 `TokenOutput(token_ids=[], stop_reason="aborted", ...)`。如果列表有元素，服务端从第一个元素读取 token，即使这条输出尚未生成完毕。结束原因和 token 一起装进 `TokenOutput`，通过 Ray RPC 返回。下面是原码的两个位置。

```python
token_ids = final_res.outputs[0].token_ids
```

```python
finish_reason = final_res.outputs[0].finish_reason
if finish_reason == "abort":
    stop_reason = "aborted"
elif finish_reason == "stop":
    stop_reason = "completed"
else:
    stop_reason = finish_reason  # for more stop reason in the future
```

## 2. 客户端先累积 token，再根据配置决定是否续写

`FullyAsyncLLMServerClient.generate()` 通过 `await super().generate(...)` 接住 RPC 的结果，在循环内先追加 token（实验提交 `llm_server.py:305`）。

```python
final_output.token_ids.extend(output.token_ids)
```

最关键的重试判断如下。保留 `5371b9989` 原始注释和代码（`llm_server.py:344–358`）。 [生成客户端](verl/workers/rollout/llm_server.py)

```python
while True:
    ...
  
    # 4. check stop reason
    # If partial rollout not enable, aborted samples will be dropped.
    # For v1 trainer, should_retry is always True. Since self.config.async_training is not exist.
    should_retry = True
    if hasattr(self.config, "async_training") and not self.config.async_training.partial_rollout:
        should_retry = False
    if output.stop_reason not in ("aborted", "abort") or not should_retry:
        break

    await asyncio.sleep(1)

final_output.extra_fields["global_steps"] = global_steps
final_output.extra_fields["min_global_steps"] = min_global_steps
final_output.extra_fields["max_global_steps"] = max_global_steps
return final_output
```

* **历史默认行为。** 原注释假定 V1 的配置没有顶层 `async_training`。按 Python 的短路规则，`hasattr(...)` 为假时，`should_retry` 保持初始值 `True`，收到 `abort` 时会续写。这里的前提是缺少配置字段，V1 `separate_async` 仍然是异步训练。
* **本次实验行为。** YAML 显式配置 `async_training.partial_rollout: false`。客户端收到 `abort` 后，`should_retry` 设为 `False`。循环退出，已经追加的 token 随 `final_output` 返回。
* **实现差异。** 原提交正文写的是 “the aborted (e.g. staleness samples) should be dropped”。这里的 `break` 只终止续写，`return final_output` 仍将这些 token 交给调用方。后续是否丢弃，需要下游明确执行。

## 3. 返回值如何进入 reward

```text
vLLM RequestOutput
  → TokenOutput(token_ids, stop_reason)
  → ### FullyAsyncLLMServerClient 返回 final_output ###
  → SingleTurnAgentLoop 用 output 接住返回值
  → 新建 AgentLoopOutput(response_ids, response_mask, ...)
  → V1 将单个对象包装成 [output]
  → _compute_score() 调用 reward worker
  → output.as_dict() 写入 rm_scores
  → TransferQueue → ReplayBuffer → 训练 batch
```

单轮 Agent Loop 的 `output = await self.server_manager.generate(...)` 接住客户端的返回值，再用这些 token 构造 `AgentLoopOutput`。这个转换没有复制独立的 `TokenOutput.stop_reason` 字段。因此 V1 评分入口会把中断后的非空回复当作普通回复处理。 [单轮 Agent Loop](verl/experimental/agent_loop/single_turn_agent_loop.py) · [V1 入队前处理](verl/trainer/ppo/v1/agent_loop_tq.py)

`agent_loop_tq.py:160–165` 的原码。这里检查的是输出对象列表。只要有一个 `AgentLoopOutput`，即使它的 `response_ids=[]`，包装后列表也有一个元素，仍会进入评分。

```python
outputs = output if isinstance(output, list) else [output]
if not outputs:
    logger.warning(f"Empty output for prompt {uid}_{session_id}")
    return

await self._compute_score(outputs, kwargs=kwargs)
```

在默认 GSM8K 评分路径中，reward worker 解码回复并提取 `#### 数字`。半截回复缺少可解析答案时得 0。若半截已经包含正确答案，也能得 1。评分返回后，`AgentLoopOutput.as_dict()` 把标量 reward 写到最后一个回复 token（`agent_loop.py:144–148`）。 [GSM8K 评分](verl/utils/reward_score/gsm8k.py) · [序列化训练轨迹](verl/experimental/agent_loop/agent_loop.py)

```python
reward_score = output.pop("reward_score", None)
if reward_score is not None:
    rm_scores = torch.zeros_like(output["response_mask"], dtype=torch.float32)
    rm_scores[-1] = reward_score
    output["rm_scores"] = rm_scores
```

| vLLM 交回的中断结果 | 评分之后的数据流                                                                                                                         |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| 非空 token          | `rm_scores[-1]` 可写，轨迹可以入队。整题 session 成功且通过 ReplayBuffer 过滤后，可以进入训练 batch。                            |
| 零 token            | 如果评分成功返回标量，`rm_scores[-1]` 对空张量抛 `IndexError`。V1 将整道题标为 `failure`，异步 ReplayBuffer 随后淘汰该题组。 |

## 本地 Git 复核

```bash
git show -s --format='%an%n%B' 5371b9989
git show 5371b9989 -- verl/workers/rollout/llm_server.py
git show dec97fce:verl/workers/rollout/llm_server.py | nl -ba | sed -n '291,370p'
git show dec97fce:verl/trainer/ppo/v1/agent_loop_tq.py | nl -ba | sed -n '136,232p'
```

## 实验探究: git仓库 Use Qwen2.5-7B for the linxid-parameter experiment

**以下实验是在一个基于三数据中心的verl框架下做的，会和verl原生的代码有一定的区别，你仅作参考，不能参考他们的代码。**

### 实验基础

### 每一条prompt+rollout回复一览

```json
{
  "input": "system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\nuser\nAnna goes trick-or-treating in a subdivision where she gets 14 pieces of candy per house. Her brother Billy goes trick-or-tricking in a neighboring subdivision where he gets 11 pieces of candy per house. If the first subdivision has 60 houses and the second subdivision has 75 houses, how many more pieces of candy does Anna get? Let's think step by step and output the final answer after \"####\".\nassistant\n", 
  "output": "To solve this problem step by step, we need to calculate the total number of pieces of candy Anna and Billy each receive, and then find the difference.\n\n### Step 1: Calculate the total candy Anna receives.\n- Anna gets 14 pieces of candy per house.\n- The first subdivision has 60 houses.\n\n\[\n\text{Total candy for Anna} = 14 \times 60\n\]\n\n\[\n\text{Total candy for Anna} = 840\n\]\n\n### Step 2: Calculate the total candy Billy receives.\n- Billy gets 11 pieces of candy per house.\n- The second subdivision has 75 houses.\n\n\[\n\text{Total candy for Billy} = 11 \times 75\n\]\n\n\[\n\text{Total candy for Billy} = 825\n\]\n\n### Step 3: Find the difference.\nTo find out how many more pieces of candy Anna gets compared to Billy, we subtract the total candy Billy receives from the total candy Anna receives.\n\n\[\n\text{Difference} = 840 - 825\n\]\n\n\[\n\text{Difference} = 15\n\]\n\n#### Anna gets 15 more pieces of candy than Billy.", 
  "gts": "15", 
  "score": 0.0, 
  "step": 1, 
  "uid": "02da2784-9b0d-4bae-bca4-b1b2f91b1024_1_0"
}
```

主要出错的有以下四种。

### 1. 格式或表达问题

标准答案是 `500`，模型也算出了 `500`，但写成 `#### $500`。评分器要求 `####`  后直接跟数字或负号，所以得了 `0`。 \[原始日志第 9 行 (line 9)\](/Users/shtsky/VScodeProjects/verl/outputs/cross\_dc\_core/qwen25\_7b\_megatron\_linxid\_epoch1\_1t1r\_r02/rollout/1.jsonl:9)

```plaintext
Therefore, Rachel should budget $500.

#### $500
```

### 2. 答案不一致

`#### 24` 的格式正确，评分器能读出 `24`，但标准答案是 `18`。 \[原始日志第 17 行 (line 17)\](/Users/shtsky/VScodeProjects/verl/outputs/cross\_dc\_core/qwen25\_7b\_megatron\_linxid\_epoch1\_1t1r\_r02/rollout/1.jsonl:17)

```plaintext
5. **State the answer:**
   - Bella and Bob celebrated their 2nd anniversary 24 months ago.

#### 24
```

### 3. 裸 `####`

正文已经算出正确答案 `41`，回复却在最后一个 `####` 处结束，没有填数字。这条在表里同时计入「格式或表达问题」和「裸 `####`」。日志没有记录结束原因，无法确认它是否被中断。 \[原始日志第 120 行 (line 120)\](/Users/shtsky/VScodeProjects/verl/outputs/cross\_dc\_core/qwen25\_7b\_megatron\_linxid\_epoch1\_1t1r\_r02/rollout/1.jsonl:120)

```plaintext
Therefore, Alexis paid $41 for the shoes.

The final answer is:

\[
\boxed{41}
\]

####
```

### 4. 推理未完成（集中出现在实验四）

```json
{
  "input": "system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\nuser\nKen created a care package to send to his brother, who was away at boarding school.  Ken placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds.  Then, he added enough brownies to cause the weight to triple.  Next, he added another 2 pounds of jelly beans.  And finally, he added enough gummy worms to double the weight once again.  What was the final weight of the box of goodies, in pounds? Let's think step by step and output the final answer after \"####\".\nassistant\n", 
  "output": "Let's break down the problem step by step:\n\n1. **Initial Weight with Jelly Beans:**\n   - Ken started with an empty box and added jelly beans to bring the total weight to 2 pounds.\n   - So, the box initially weighed 0 pounds, and after adding jelly beans, it weighed 2 pounds.\n\n2. **Triple the Weight with Brownies:**\n   - Next, he added brownies, which caused the weight to triple.\n   - The current weight is", 
  "gts": "16", 
  "score": 0.0, 
  "step": 2, 
  "uid": "0514676e-29a1-4caa-a553-44f956e594e6_0_0", 
  "finish_reason": "abort", 
  "stop_reason": "aborted", 
  "stop_reason_history": ["aborted"], 
  "abort_count": 1, 
  "response_truncated": false
}
```

这个 The current weight is 就是明显话没说完，被aborted了。

### 实验一

用moe\_smoke的配置文件跑一轮，记录rollout的输出

```shell
ASCEND_GLOBAL_LOG_LEVEL=4 DENSE_ASCEND_LOG_LEVEL=4 CROSS_DC_INIT_ONLY=0 CROSS_DC_DELTA_ENABLED=1 RUN_ID=partial_rollout_qwen25_7b_megatron_1t1r_r02 bash /root/work/tiankuan/shtskysmile/final_code/verl/jiutian/cross_dc/platform_run.sh --config /root/work/tiankuan/shtskysmile/final_code/verl/jiutian/cross_dc/config/partial_rollout_qwen25_7b_megatron.yaml
```

/root/work/tiankuan/shtskysmile/final\_code/verl/outputs/cross\_dc\_core/partial\_rollout\_qwen25\_7b\_megatron\_1t1r\_r02

| 实验 | Step一个step512个问题 | 零分总数 | 格式或表达问题 | 答案不一致 | 疑似推理未写完 | 裸 `####` |
| ---- | --------------------- | -------- | -------------- | ---------- | -------------- | ------------ |

| smoke                 | 1 | 44            | 39            | 5             | 0           | 2            |
| --------------------- | - | ------------- | ------------- | ------------- | ----------- | ------------ |
| smoke                 | 2 | 63            | 44            | 19            | 0           | 2            |
| smoke                 | 3 | 78            | 59            | 19            | 0           | 3            |
| smoke                 | 4 | 53            | 33            | 20            | 0           | 2            |
| smoke                 | 5 | 49            | 43            | 6             | 0           | 4            |
| smoke                 | 6 | 55            | 34            | 21            | 0           | 2            |
| smoke                 | 7 | 36            | 14            | 22            | 0           | 1            |
| smoke                 | 8 | 28            | 11            | 17            | 0           | 0            |
| **smoke 小计** |   | **406** | **277** | **129** | **0** | **16** |

### 实验二

用linxid megatron的配置文件跑一轮

/root/work/tiankuan/shtskysmile/final\_code/verl/outputs/cross\_dc\_core/qwen25\_7b\_megatron\_linxid\_epoch1\_1t1r\_r02

```shell
ASCEND_GLOBAL_LOG_LEVEL=4 DENSE_ASCEND_LOG_LEVEL=4 \
CROSS_DC_INIT_ONLY=0 \
MODEL_PATH=/root/work/models/Qwen2.5-7B-Instruct \
RUN_ID=qwen25_7b_megatron_linxid_epoch1_1t1r_r02 \
bash /root/work/tiankuan/shtskysmile/final_code/verl/jiutian/cross_dc/platform_run.sh \
--config /root/work/tiankuan/shtskysmile/final_code/verl/jiutian/cross_dc/config/qwen25_7b_megatron_linxid_epoch1_1t1r.yaml
```

| 实验                   | Step一个step512个问题 | 零分总数      | 格式或表达问题 | 答案不一致    | 疑似推理未写完 | 裸 `####` |
| ---------------------- | --------------------- | ------------- | -------------- | ------------- | -------------- | ------------ |
| linxid                 | 1                     | 126           | 93             | 33            | 0              | 12           |
| linxid                 | 2                     | 119           | 86             | 33            | 0              | 8            |
| linxid                 | 3                     | 111           | 87             | 24            | 0              | 5            |
| linxid                 | 4                     | 90            | 45             | 45            | 0              | 4            |
| linxid                 | 5                     | 41            | 13             | 28            | 0              | 0            |
| **linxid 小计** |                       | **487** | **324**  | **163** | **0**    | **29** |

可能是去掉日志+enforce\_eager=False，本身算的就很快，确实没有遇到疑似推理未写完的。后续可能必须暴力减速来做分析，或者先进行代码模拟。

### 实验三：trainer rollout降成4卡，然后打开日志和enforce\_eager，把速度降下来，尽可能模拟之前partial\_rollout出问题的状况

bash -ec 'source /root/work/tiankuan/shtskysmile/final\_code/verl/jiutian/cross\_dc/env.sh; unset RAY\_ADDRESS; export ASCEND\_GLOBAL\_LOG\_LEVEL=1 DENSE\_ASCEND\_LOG\_LEVEL=1 RUN\_ID=qwen25\_7b\_linxid\_single\_instance\_4t4r\_r01; exec "$CROSS\_DC\_PYTHON" -m verl.trainer.main\_ppo --config-path "$CROSS\_DC\_LOCAL\_DIR/config" --config-name qwen25\_7b\_megatron\_linxid\_single\_instance\_4t4r'

[https://aihuanxin.cn/kunlun/kl-web?poolId=2&amp;projectId=caed3808e9df473bb632d9470246d44e#/train-task/detail/dt-592b9e78397447dc8d7656d1e128cdda](https://aihuanxin.cn/kunlun/kl-web?poolId=2&projectId=caed3808e9df473bb632d9470246d44e#/train-task/detail/dt-592b9e78397447dc8d7656d1e128cdda)

/root/work/tiankuan/shtskysmile/final\_code/verl/outputs/cross\_dc\_core/qwen25\_7b\_linxid\_single\_instance\_4t4r\_r01

| 实验 | Step | 零分总数 | 格式或表达问题 | 答案不一致 | 疑似推理未写完 | 无法归类 | 裸 `####` |
| ---- | ---- | -------- | -------------- | ---------- | -------------- | -------- | ------------ |

| linxid                 | 1  | 101           | 71            | 30            | 0           | 0            | 3            |
| ---------------------- | -- | ------------- | ------------- | ------------- | ----------- | ------------ | ------------ |
| linxid                 | 2  | 142           | 112           | 23            | 0           | 7            | 11           |
| linxid                 | 3  | 118           | 85            | 24            | 0           | 9            | 7            |
| linxid                 | 4  | 100           | 53            | 44            | 0           | 3            | 0            |
| linxid                 | 5  | 50            | 19            | 27            | 0           | 4            | 1            |
| linxid                 | 6  | 37            | 18            | 19            | 0           | 0            | 0            |
| linxid                 | 7  | 46            | 18            | 28            | 0           | 0            | 0            |
| linxid                 | 8  | 150           | 112           | 28            | 0           | 10           | 10           |
| linxid                 | 9  | 90            | 23            | 32            | 0           | 35           | 35           |
| linxid                 | 10 | 42            | 5             | 22            | 0           | 15           | 14           |
| **linxid 小计** |    | **876** | **516** | **277** | **0** | **83** | **81** |

### 实验四：回归到之前Omega，出现严重partial\_rollout问题的参数组合，并且完成对所有rollout停止原因的记录

bash -ec 'source /root/work/tiankuan/shtskysmile/final\_code/verl/jiutian/cross\_dc/env.sh; unset RAY\_ADDRESS; export ASCEND\_GLOBAL\_LOG\_LEVEL=1 DENSE\_ASCEND\_LOG\_LEVEL=1 RUN\_ID=omega\_qwen25\_7b\_single\_instance\_4t4r\_r02; exec "$CROSS\_DC\_PYTHON" -m verl.trainer.main\_ppo --config-path "$CROSS\_DC\_LOCAL\_DIR/config/baseline\_3200" --config-name Omega\_qwen25\_7b\_single\_instance\_4t4r'

/root/work/tiankuan/shtskysmile/final\_code/verl/outputs/cross\_dc\_core/omega\_qwen25\_7b\_single\_instance\_4t4r\_r02

| 实验                  | Step | 零分总数        | 格式或表达问题 | 答案不一致   | 疑似推理未写完 | 无法归类      | 裸 `####` |
| --------------------- | ---- | --------------- | -------------- | ------------ | -------------- | ------------- | ------------ |
| Omega                 | 1    | 48              | 39             | 9            | 0              | 0             | 2            |
| Omega                 | 2    | 149             | 23             | 4            | 31             | 91            | 3            |
| Omega                 | 3    | 146             | 27             | 2            | 24             | 93            | 1            |
| Omega                 | 4    | 138             | 17             | 3            | 25             | 93            | 2            |
| Omega                 | 5    | 147             | 9              | 17           | 30             | 91            | 1            |
| Omega                 | 6    | 137             | 12             | 5            | 38             | 82            | 1            |
| Omega                 | 7    | 141             | 9              | 11           | 33             | 88            | 1            |
| Omega                 | 8    | 125             | 2              | 2            | 35             | 86            | 2            |
| Omega                 | 9    | 131             | 1              | 7            | 24             | 99            | 0            |
| Omega                 | 10   | 127             | 1              | 3            | 39             | 84            | 2            |
| **Omega 小计** |      | **1,289** | **140**  | **63** | **279**  | **807** | **15** |

这说明了在某种参数配置下，确实容易出现rollout被actor权重更新打断，然后根本不重新生成新的回复的现象。这证明了verl官方实现的partial\_rollout确实存在bug。但是有关参数配置的具体的原因（比如什么参数导致了rollout算的慢，从而被打断）还有待进一步的探究。目前值得一提的是，实验四比实验三出现了更多的疑似推理未写完，但是其运行时间反而更短一些。

### 备忘录（不必在意）

| 配置文件                                       | 模型       | trainer＋rollout 节点 | 训练 TP/PP | rollout TP | 训练 batch / 生成 batch | 同步间隔          | 保存实际回答             |
| ---------------------------------------------- | ---------- | ---------------------- | ----------- | ----------- | --------------------------- | ----------------- | ------------------------ |
| `partial_rollout_qwen25_7b_megatron.yaml`    | Qwen2.5-7B | 1＋1                   | TP2 / PP1 | 2           | 64 / 1                    | 每次 PPO 更新   | 是                       |
| `qwen3_32b_megatron_v17.yaml`                | Qwen3-32B  | 2＋2                   | TP4 / PP2 | 4           | 128 / 16                  | 每两次 PPO 更新 | 是                       |
| `qwen25_7b_megatron_v17.yaml`                | Qwen2.5-7B | 1＋1                   | TP2 / PP1 | 2           | 128 / 16                  | 每两次 PPO 更新 | **未配置落盘目录** |
| `qwen25_7b_megatron_linxid_epoch1_1t1r.yaml` | Qwen2.5-7B | 1＋1                   | TP4 / PP2 | 4           | 128 / 16                  | 每两次 PPO 更新 | **是，当前使用**   |
