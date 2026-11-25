# ACT 持续记忆实验（2026-11-19）

上一轮三次预测历史的 ACT 在场景 16、35 均完成首次倒料，但两者都没有完成物理回程，十轮任务仍失败。离线误差下降不能作为连续作业修复的证据。

本轮只改 ACT 内部模型与训练：冻结上一轮完整视觉、Transformer、短历史模块；新增 256 维 GRU 和零初始化的动作残差头。GRU 输入为现有归一化 28 维状态、三路相机的冻结特征和 ACT 原始预测，输出十步动作块的学习残差。预测调用之间保留隐藏状态，episode reset 同时清空短历史与持续记忆。阶段、铲数、回程标记、成功信号不作为模型输入。

训练从 `act_temporal_repaired10k_20261118_01/train/checkpoints/010000/pretrained_model` 初始化。使用原修复数据和原 140/20/40 episode 划分，源数据保持只读。两张 GPU 分别缓存训练与验证 episode 的冻结特征，采样间隔三帧，与执行三步后重新预测一致。视频顺序解码的像素与 LeRobot PyAV 加载器精确一致；当前帧空间特征只计算一次，过去特征严格按 episode 的时间顺序组成三帧上下文。测试 episode 不生成训练特征。

每次参数更新采样八段连续 32 次预测。先使用当前 GRU 权重从该 episode 开头重建所采样片段之前的完整记忆，再截断梯度并训练片段。损失为有效动作的归一化 L1，另加系数 0.0001 的残差平方约束。AdamW 学习率 0.0001、weight decay 0.0001、梯度裁剪 1，训练 5,000 步。每 1,000 步使用完整验证 episode 顺序评估；仅保留最终训练 checkpoint。此实验没有更新原 ACT 核心，不宣称使用原生 CVAE 训练目标。

代码：`persistent.py`；私有运行时构建器 `scripts/prepare_act_temporal_runtime.py --persistent`；实验启动器 `scripts/run_act_persistent_experiment.py`；真实权重检查 `scripts/check_act_persistent.py`。cuDNN 使用当前 PyTorch 环境自带的版本；新模型关闭 TF32，缓存与在线均使用完整 FP32 精度。GPU GRU 参数的共享存储在保存时分别复制为标准 safetensors 张量，保持标准 ACT 的严格加载接口。

已通过五项记忆模块测试、六项既有短历史模型测试，以及八项真实权重/像素/批量一致性/重置/保存重载检查。新增分支零初始化时，连续六次原始图像推理与同精度源模型逐值一致，16 样本批量与在线单样本最大差异约 0.0000014；两次训练更新后，原 ACT 权重逐值不变。另使用两个训练 episode 的真实缓存完成了两步完整训练流程检查，覆盖全前缀记忆重建、GPU 优化、顺序验证、最终保存及严格重载；临时权重与误差数据已删除，仅保留检查结论 `trainer_preflight.json`。最终训练完成时再次检查冻结核心、严格重载和四个外部评估/仿真文件的哈希。

实验输出：`outputs/experiments/act_persistent_temporal5k_20261119_03`。运行时：`outputs/experiments/act_persistent_runtime_20261119_03/python`。`status.json` 记录缓存/训练阶段，`cache_worker_*.json` 记录缓存进度，`validation_*.json` 记录验证结果。

训练结束后自动启动 `evaluation_pair/`：仍使用场景 16、35、相同种子、每例 5,000 步、十轮任务及原成功标准，保留视频。既有 evaluator、控制器和仿真文件均不修改。是否改善闭环回程与多轮成功尚待该评估验证，不能由离线 MAE 或记忆结构本身推断。
