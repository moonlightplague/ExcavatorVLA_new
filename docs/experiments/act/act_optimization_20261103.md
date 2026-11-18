# ACT PHW 优化实验（2026-11-03）

用户已批准实施关键转换诊断、普通续训与窗口重采样对照，以及独立的时间集成实验。

实验目录：`outputs/experiments/act_phw_optimization_20261103_01/`。
入口：`scripts/run_act_optimization.py`。

## 数据与训练

固定现有 PHW 40k 权重、原 160/20/20 episode 划分和归一化。两组都严格加载原权重及保存的预处理器，重新初始化优化器；这是权重微调，不是包含优化器状态的精确断点续训。两组全局 batch 16、BF16、学习率 3e-6，各训练 5,000 步；分别使用一个 GPU。

- uniform：训练观测均匀、有放回采样。
- transition：70% 均匀观测 + 30% 转换窗口；后者先均匀选择 episode，再在该 episode 的候选观测中均匀选择。与对照均为有放回采样，PHW 原阶段损失权重保持一致。

7,268 个训练观测窗口来自离土/持料前后各 10 帧以及卸料阶段开始前后各 10 帧。不跨 episode，不使用验证/测试样本，不修改动作标签。`sampling_audit.json` 同时记录观测阶段概率及考虑未来 chunk、padding 和原 PHW 权重后的监督质量占比。持料观测概率预计从 0.67% 变为 1.35%；监督占比从 2.38% 变为 4.54%。重复窗口不代表新增状态多样性。

`failure_diagnosis.json` 描述上一轮 52 条模型轨迹中的拾料、保料下降、入车事件和跟踪误差，不把这些事件直接解释成阶段因果，也不生成专家纠错标签。

## 执行与选择

固定 PhysX、沉降前移墙、10 Hz、350 步、100 源粒子门槛及连续三步开斗条件。每场新模拟器，逐物理步渲染。先复核一个参考专家回放。

普通执行每 3 步重规划。时间集成单独使用原 PHW 40k 权重，每步预测，融合最近三次 chunk 中对应当前时间戳的动作，权重为 `exp(-0.5 * age)`，每场清空缓存。该路径与检查点原生 temporal ensemble 配置分开记录，不更改训练。

1. 四种策略在开发 seeds 2–6 上运行，保存视频与完整轨迹：原 PHW、普通续训、窗口续训、原 PHW 时间集成。
2. 三个候选中最多两个进入 seeds 12–31 的选择比较；原 PHW 同时重跑。排序首先按完整成功数，其次按未挖到料的场数、平均入车量。
3. 候选完整成功数超过原 PHW 时，训练候选用采样/dropout seed 1001 从同一原始权重重复微调，并在选择场景复核；不改变原 40k 初始化权重。时间集成不涉及新的训练随机种子。
4. 只有选择改善且复核通过才运行预先冻结的 100 个新 seed 的原 PHW/胜出方案配对比较。新 seed 在训练前写入 `heldout_protocol.json`，排除已查到的采集/评估 seed。seed ID 不重叠不等于物理场景一定不同。

开发及选择结果不作为独立最终测试成绩。没有改善或复核失败时保留最终场景，并明确报告下一阶段的专家接管/纠错采集需求。不能把模型失败动作或按时间粘贴的原专家动作当成纠错标签。

## 运行与验证

控制器和训练/评估源码、场景资产均使用独立快照；核验原模型、数据 parquet、元数据及 LeRobot Python 源码哈希。训练产物验证重新加载；闭环逐场审计物理时间、粒子计数、执行动作和成功条件。每组只保留最终一步检查点。

`status.json` 运行中每 5 秒更新，等待 GPU 时每 10 秒更新；`REPORT.md` / `results.json` 随阶段完成更新。只清理由本控制器启动的进程。实验级文件锁防止重复启动。

运行前验证：ACT 共 17 项检查通过；主仓库共 88 项检查，其中 87 项通过、1 项因归档参考数据未安装而跳过。涵盖实际数据续训首步的源权重逐项一致、训练更新生效、预处理器张量保持一致、训练窗口泄漏拒绝、时间戳对齐融合、跨场景重置与异常子进程清理。

当前控制器已后台启动，状态为 `waiting_gpu`；等待另外的 PI05 100 场双卡评估释放 GPU，不终止或修改该任务。两张 GPU 各连续三次满足显存低于 1500 MiB、利用率低于 15% 后自动开始两组训练。实时状态以实验目录的 `status.json` 为准。

```bash
PYTHON=/root/gpufree-data/conda/envs/ExcavatorVLA-current/bin/python
"$PYTHON" scripts/run_act_optimization.py \
  --output outputs/experiments/act_phw_optimization_20261103_01 --prepare-only
# 准备完成后，从快照内启动控制器；ACT_PROJECT_ROOT 指向原工作区。
ACT_PROJECT_ROOT="$PWD" "$PYTHON" \
  outputs/experiments/act_phw_optimization_20261103_01/frozen_sources/scripts/run_act_optimization.py \
  --output "$PWD/outputs/experiments/act_phw_optimization_20261103_01" --resume
```

`--resume` 只启动尚未开始训练的 prepared 实验，不覆盖已启动的实验。

## 用户要求缩减最终场数（2026-11-03 16:28）

用户要求“50场结束”，最终测试改为两种策略各使用原种子序列的前 50 场（1000000–1000049）。应用缩减时，原策略已完成 30 场、时间集成已完成 31 场；不是按成功与否筛选样本。两组已完成的数据全部保留。

`scripts/cap_act_evaluation.py` 接替原调度控制器，保持两条正在运行的评估进程；在各组完成第 50 场后停止该组自己的进程树。`episode_cap_amendment.json` 记录用户指令、调整时点、原配置哈希和来源身份；`before_episode_cap.json` 保留变更前的状态与报告。

原始评估配置仍记录启动时的 100 场请求，不重写原始逐场数据或执行来源身份。实际按用户修改后的 50 场范围生成独立派生汇总与轨迹审计，保存至 `amended_heldout_50/`；最终对比写入 `heldout_comparison.json`。若退出边界产生第 51 场的未完成目录，记录为范围外数据，不计入结果。

`status.json` 和 `REPORT.md` 已切换到 50 场目标，实时监控进程记录于 `cap_launch.json`。预计完成时间以实际进度更新。
