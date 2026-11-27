# False：abort 后丢弃方案

等待完成方案及其 r01/r02 实验已保存于 commit `31a7b03a`。本次按 PR [#6882](https://github.com/verl-project/verl/pull/6882) 所述的“关闭 partial rollout 时，被中断样本应丢弃”意图，补齐 V1 的实际丢弃数据流。

| 配置 | 权重同步 | 收到 abort 后 |
| --- | --- | --- |
| `async_training.partial_rollout=True`，或缺少该字段 | 关闭入口并 abort 在途请求，再安装权重 | 保留 token/logprob，等待恢复后续写 |
| `async_training.partial_rollout=False` | 关闭入口并 abort 在途请求，再安装权重 | 停止续写，淘汰该 prompt 的整个题组 |

同步端恢复基线的 abort 路径，移除等待模式新增的 finish RPC 和 manager 配置参数。保留 server 接入时捕获权重版本，以及客户端不把空 abort 计入生成版本的修正。

客户端在合并中断结果、检查长度预算之前抛出 `RolloutAbortedError`。因此，空 abort、非空 abort 和 token 数恰好达到长度预算的 abort 都不会作为完成回答返回。True 保持原来的前缀续写和累计长度预算逻辑。

V1 AgentLoopWorkerTQ 对预期的中断淘汰记录 info 日志，避免将其当成普通异常打印堆栈。只有全部 sibling sessions 完成后，才将 prompt 状态设为 failure，防止 ReplayBuffer 清理题组后仍有 sibling 写入。ReplayBufferAsync 沿用已有的失败题组淘汰和补样本机制：一个题组被丢弃，恰好补一个新 prompt。正常完成的 sibling 可能已经评分，但其轨迹与整个失败题组一起被清理，不进入训练。

评分入口另有防御检查：如果自定义 AgentLoop 返回的任一输出保留了 `stop_reason=abort/aborted`，评分前直接拒绝该输出，不写入轨迹。验证阶段收到这样的失败也不会给半截回答评分；沿用原来的失败计数规则，不进行训练式补样本。

实验 YAML 未更改：仍使用相同模型、数据、训练和采样参数、10 步训练、第 5 和第 10 步完整验证、相同 CANN/ATB/driver 激活设置。启动脚本接受同样的 `false` 参数，但默认使用新 RUN_ID `omega_qwen25_7b_single_instance_4t4r_partial_rollout_false_drop_r01`，并尊重调用方显式提供的 RUN_ID，避免覆盖等待模式日志。

运行命令：

```bash
RUN_ID=omega_qwen25_7b_single_instance_4t4r_partial_rollout_false_drop_r01 \
/bin/bash /root/work/tiankuan/shtskysmile/fix_partial_rollout/verl/scripts/run_partial_rollout_baseline.sh false
```

日志目录为 `outputs/partial_rollout_repro/omega_qwen25_7b_single_instance_4t4r_partial_rollout_false_drop_r01/`。指标 JSONL 位于其同级的同名 `.jsonl` 文件。

观察丢弃情况：训练日志中的 `Dropping prompt group uid=... after ... aborted sessions`，以及每步指标 `training/rollout_failure/evicted_samples`。后者包含中断及其他原因导致的失败题组，单位是 prompt 组，不是回复条数。

CPU 测试覆盖：True 续写；False 在 Namespace 和实际 OmegaConf 配置下拒绝空/非空/满预算 abort；正常 stop/length 保留；客户端、replica、vLLM server 和 NCCL manager 的实际方法联动；异步 sibling 全部完成后才发布 failure；失败题组连同已写入的 sibling 被淘汰且只补一次；评分前拒绝中断输出；旧回答延迟读取时仍保留接入版本。Ray RPC、引擎输出和 TransferQueue 传输使用模拟，ReplayBuffer 采样/淘汰方法使用实际实现。

验证结果：五个 CPU 测试文件共 **88 项通过、1 项真实 Ray 快照集成测试未选入**。使用 uv 创建的 Python 3.12 环境，复用原 conda 环境依赖并激活相同 CANN/ATB/driver 库。命令为：

```bash
.venv/bin/python -m pytest -q --disable-warnings \
  tests/checkpoint_engine/test_d2h_checkpoint_engine.py \
  tests/workers/rollout/rollout_vllm/test_submitter_gate_on_cpu.py \
  tests/workers/rollout/test_llm_server_response_length_cap_on_cpu.py \
  tests/workers/rollout/test_llm_server_routed_experts_on_cpu.py \
  tests/trainer/ppo/v1/test_agent_loop_tq_on_cpu.py \
  -k 'not d2h_snapshot_is_published_before_rollout_consumes_it'
```

另验证了 shell 语法、新日志目录选择和显式 RUN_ID 覆盖逻辑。

本次未启动 NPU 训练，不对准确率改善作承诺。丢弃会影响生成效率和训练数据选择，即使 YAML 参数相同，也不保证与等待模式消费完全相同的题目。

## 用户完成的 drop r01 实验结果

读取 `omega_qwen25_7b_single_instance_4t4r_partial_rollout_false_drop_r01` 的实际 Hydra 配置、10 步指标、全部 rollout 和第 5/10 步 validation。用户运行实验；本次仅分析日志，未提交代码。

| 实验 | 第 5 步验证 | 第 10 步验证 | 第 5 → 10 步变化 | 第 10 步无严格答案匹配 |
| --- | --- | --- | --- | --- |
| 原 bug r02 | 1154/1319，87.4905% | 1114/1319，84.4579% | −3.0326 个百分点 | 64 |
| True r01 | 1178/1319，89.3101% | 1186/1319，89.9166% | +0.6065 个百分点 | 8 |
| False 等待 r01 | 1179/1319，89.3859% | 1048/1319，79.4541% | −9.9318 个百分点 | 155 |
| False 等待 r02 | 1175/1319，89.0826% | 1159/1319，87.8696% | −1.2130 个百分点 | 39 |
| False 丢弃 r01 | 1170/1319，88.7036% | 1183/1319，89.6892% | +0.9856 个百分点 | 5 |

这次在两个验证点之间未出现整体掉点，最终比 True 少答对 3 题，即相差 0.2274 个百分点；比等待 r02 多答对 24 题，即提高 1.8196 个百分点。按 input 配对，第 5 步正确、第 10 步错误有 32 题，反向改善有 45 题，净增加 13 题。不能将总体上升解释为每道题都未退化，也没有每一步验证的证据。

第 5 步所有回答都有严格答案匹配；第 10 步 5 条无匹配中，2 条以 `#####\n####` 结束，另外 3 条为 `#####\n25`、`#### #\n14`、`#####\n2`，没有缺空格直接接数字的失败。第 10 步另有 131 条能严格提取但答案字符串不匹配的失败。因此此前的大规模格式退化没有在本次重现，但异常结尾仍有少量残留。相较 True，drop 格式失败少 3 条、答案不匹配多 6 条，合计少答对 3 题。

训练数据方面，每步均为 64 个 prompt × 4 条回答；全部 2560 条训练记录为 `finish_reason=stop`、`stop_reason=completed`，没有最终 abort 记录，未发现残缺题组。使用仓库 GSM8K 严格规则复算全部 5198 条训练和验证记录，与日志 score 均一致。训练 reward 从 76.5625% 升至 97.2656%，中间有波动；第 10 步仍有 4 条正常结束但无严格答案匹配的训练回答，丢弃中断不等于消除模型自身的所有格式错误。

第 2–10 步 `training/rollout_failure/evicted_samples` 分别为 29、33、27、31、31、29、31、36、37，合计 284 个失败 prompt 组；这是组数，且包含所有失败原因，不能直接当作被 abort 的回答数。主 `train.log` 未包含预期的 `Dropping prompt group ...` 信息，因此本次无法从该文件逐组核实 abort 原因和中断 session 数。日志里两处 Traceback 均为启动阶段可选 MSC 配置源不可用，训练最终完成 10/10 步，未发现 `_run_prompt` 普通异常记录。

所有轨迹 spans 为 1，符合一条回答不跨权重版本的预期。第 2–10 步平均 staleness 为 0.421875–0.578125，第 10 步为 0.421875：256 条回答中 108 条落后当前更新前 actor 一版，148 条与其同版。等待模式此前全部落后一版；本次丢弃和补样本确实改变了版本组成，但仍有旧策略数据。结果支持继续采用丢弃方案，也支持调查版本滞后与格式变化的关系；单次实验尚不能证明之前退化的唯一原因，或保证长训练稳定。

实际 Hydra 配置与等待 False r02 仅 `trainer.default_local_dir` 不同；与 True r01 再多一个 `async_training.partial_rollout` 差异。模型、数据、学习率、采样参数、10 步及第 5/10 步完整 GSM8K 验证保持一致。数据丢弃和 refill 会改变实际训练题目与生成随机轨迹，配置一致不代表消费数据一致。

日志中平均 `timing_s/update_weights` 为 7.317 秒，等待 r02 为 158.026 秒，True 为 9.005 秒；平均 `timing_s/step` 分别为 241.496、280.343、361.247 秒。这是计时字段对比，不能直接用来宣称包含验证、初始化在内的端到端效率优势，也不能据此推算被丢弃的 token 成本。
