# ACT PHW 第一轮优化实验（2026-11-02）

以现有 PHW 40k 为固定起点，先验证执行环境，再比较重规划间隔。此次不改变 checkpoint、动作标签、归一化或训练损失；训练与纠错采集需根据本轮结果选择具体方案。

## 已完成的离线审计

严格使用 checkpoint 保存的 160/20/20 episode 划分。三个模型的既有离线预测与同一批验证帧按 episode/frame 对齐，首步动作标签逐项核对通过。

| 阶段 | 实际训练帧 | 验证帧 | PHW 铲斗示范幅度 rad/s | PHW 铲斗预测幅度 rad/s |
| --- | ---: | ---: | ---: | ---: |
| 离土 `pull_exit_cut` | 493 | 62 | 0.2342 | 0.0534 |
| 持料 `secure_load` | 215 | 22 | 0.4105 | 0.0117 |
| 卸料 `unload_to_bin` | 15020 | 1833 | 0.4084 | 0.4335 |

实际训练集共 32,027 帧，持料阶段约占 0.67%。这些值反映在专家记录状态上的预测，不证明闭环因果；持料验证帧尤其少。

已生成 3,908 个候选训练观测窗口：围绕离土、持料帧的前后各 10 帧，按 episode 去重，不跨 episode，动作 chunk 在 episode 末端截断。窗口只来自训练集，未创建新的专家动作标签，也尚未用于训练。

## 闭环实验顺序

所有实验固定 10 Hz、350 步、100 个入车源粒子及连续开斗条件。每个场景启动新的模拟器，并逐物理步渲染；不同设置的结果分别保存，不与 10 月 29 日的简化渲染结果合并计算。

1. 在同一个已有、可追溯的参考场景回放专家动作，分别检查 PhysX 提前移墙、PhysX 沉降后移墙、Newton。Newton 参数固定为已有成功专家回放的分辨率、粒子预算及 seed。
2. 选择第一个专家成功的配置，优先顺序为上述顺序。随后对同一参考场景比较 PHW 每 3 步与每 1 步重规划。
3. 相同配置下对调试 seeds 2–6 各运行两种重规划间隔，保存视频与完整观测轨迹，供失败状态诊断和后续专家纠错使用。
4. 对另 20 个选择场景 seeds 12–31 各运行两种间隔。此阶段用于选择方案，不作为独立最终测试；最终测试必须使用另一个 seed 范围。

参考设置来自已有 `reference_setup.json`，其中 183 个动作已与原始导出 episode 1 的动作前缀核对。该 episode 属于训练集，参考模型结果仅用于拟合与执行诊断，不代表泛化性能。专家回放保持 `execution_source=expert_replay`，模型结果保持 `execution_source=model`。

如果所有专家回放均失败，实验记录 `alignment_failed` 并停止模型选择；不会把执行环境失败当作模型失败，也不会据此启动训练。任一基础设施错误、文件哈希变化或轨迹审计错误均记录失败并保留日志供检查。

## 运行与隔离

入口：`scripts/run_act_phw_diagnostics.py`。

本轮目录：`outputs/experiments/act_phw_diagnostics_20261102_01/`，包含离线审计、训练窗口清单、参考设置副本、源码及场景资源副本。执行前后验证 snapshot、checkpoint、数据 parquet 和 LeRobot Python 源码的哈希。原始 checkpoint 与原数据不修改。

使用 GPU 1 的一个 worker，等待已有 PI05 控制进程及其训练/评估全部退出，再要求 GPU 连续三次采样低于 1500 MiB 且利用率低于 15%。运行中每 5 秒更新 `status.json`；每完成一个实验，更新 `REPORT.md` 和 `results.json`。只清理本控制器自身的子进程。

准备并启动：

```bash
PYTHON=/root/gpufree-data/conda/envs/ExcavatorVLA-current/bin/python
"$PYTHON" scripts/run_act_phw_diagnostics.py \
  --output outputs/experiments/act_phw_diagnostics_20261102_01 --prepare-only
"$PYTHON" scripts/run_act_phw_diagnostics.py \
  --output outputs/experiments/act_phw_diagnostics_20261102_01 --resume \
  --gpu 1 --wait-pid 927250
```

`--resume` 仅用于尚未启动评估的 prepared 实验；已运行或失败的实验不能被覆盖。实验输出是本次工作的主要证据；不生成临时训练 checkpoint。

## 已完成检查

- 入口 Python 编译及 Git diff 空白检查通过。
- 训练/验证/测试 episode 互斥；3,908 个窗口无验证/测试集样本，窗口边界有效。
- 三个模型的验证首步标签与数据逐项一致；参考专家动作与导出记录前缀一致。
- 源码快照、LeRobot 源码、数据及 PHW checkpoint 哈希预检通过。
- GPU 闭环结果在实验运行后写入本轮 `REPORT.md`，本文不预报实验结论。
