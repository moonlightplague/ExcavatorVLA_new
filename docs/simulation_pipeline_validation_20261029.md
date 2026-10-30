# 仿真采集与 SmolVLA 评估验证（2026-10-29）

`collect.sh`、`eval_smolvla_success.sh` 和跨场景重置均已在真实 Isaac Sim 中跑通，进程退出码均为 0。本次使用默认 Newton MPM 后端；评估保留 fresh simulator、逐物理步渲染和视频输出。单个 seed 的结果用于确认链路可用，不能代表总体成功率。

## 环境与范围

- Python：`/root/gpufree-data/conda/envs/ExcavatorVLA-current/bin/python`，Python 3.12、LeRobot 0.4.4、PyTorch 2.10.0+cu128。
- Isaac Sim：`6.1.0-rc.26+release.49347.2d230af4.gl`；GPU 为 RTX 4090。
- 采集在 GPU 0 上运行单 worker；评估在 GPU 1 上运行单 worker、8 个仿真 CPU 线程。
- 测试期间两张 GPU 上还有已有评估任务，下述耗时包含资源竞争。
- 数据集元数据：10 Hz、current-v4 的 28D 状态、4D 关节速度动作、三路 256×256 相机。

## 实机结果

| 检查 | 结果 |
| --- | --- |
| `collect.sh 1 1 0` | 一次尝试完成一条 trainable episode，并成功导出 LeRobot v3 数据 |
| 导出完整性 | 197 行有限状态/动作；动作限幅、帧序号、10 Hz 时间戳检查通过；三路视频各完整解码 197 帧 |
| LeRobot 加载 | 第 0、98、196 帧的状态、动作、任务文本和三路图像读取通过；另检查了三路首、中、尾画面 |
| `eval_smolvla_success.sh` | seed 2，最多 350 步，在第 347 步成功结束；入厢 230、散落 15、桶内剩余 0 个计数粒子 |
| 成功条件 | 连续 3 次观测达到 100 粒子门槛且桶已打开；`official_result_valid=true`，无基础设施错误 |
| 评估审计 | `audit_smolvla_success.py --verify-sources` 通过，包含源码哈希、动作边界、物理时序和成功条件复核 |
| 评估视频 | 完整解码 368 帧，1024×864、10 fps，与记录帧数一致 |
| Newton 场景重置 | `tests/run_success_eval_reset_live.py` 的 seed `2 → 12 → 2` 通过；三次使用同一 Isaac 进程，车厢边界、源粒子统计和桥接握手正确刷新 |

评估执行了 34.7 秒仿真动作，rollout 墙钟时间约 476 秒；含启动和进程收尾的父进程运行时间约 589 秒。诊断曾报告持料停滞，随后解除，最终完成运输和卸料；诊断记录没有观测到超过容差的关节限位越界。

## 回归测试维护

更新了三处过期测试夹具：

- `tests/test_success_eval_metrics.py`：同步轨迹开关、进度参数、仿真设置、关节限位以及拆分后的图像接口；补上 worker 目录，并校验诊断和最终进度实际落盘。
- `tests/test_success_eval_reset.py`：提取真实墙体设置辅助函数，并固定测试环境，恢复旧墙体行为用例的隔离性。
- `ExcavatorVLA-smolvla/tests/test_success_eval_observation.py`：使用 `decoded_frames`、`validate_state` 和 `policy_batch`，保留黑色有效画面与损坏图像的验证。

| 测试集 | 结果 |
| --- | --- |
| 主仓库 `tests` | 80 项：79 通过、1 项因缺少归档原始 reference episode 跳过 |
| `ExcavatorVLA-isaac/tests` | 118 项通过 |
| `ExcavatorVLA-smolvla/tests` | 72 项通过 |

共 269 项通过、1 项跳过。最后补充诊断落盘断言后，相关 15 项 metrics 测试再次通过。Shell 入口语法和 Git diff 空白检查通过；测试数据与两个仿真子仓库的 Warp/Python 缓存均已被现有 gitignore 覆盖。

## 建议优先处理的改进

1. **统一成功示范的筛选标准。** 本次采集被判为 trainable，但采集计数为抓取/提起 511、入厢 92、散落 419，质量分 49，散落比例约 82%。`ExcavatorVLA-isaac/scripts/excavator_app/excavator_runtime.py` 的 `compute_episode_quality_score` 将低分、高散落和低入厢计数作为 warning；评估则要求至少 100 个入厢粒子及连续开桶观测。建议增加可选的严格采集档，并统一源粒子与车厢计数定义，给成功示范训练使用。当前两种成功标签不能直接等同。
2. **把实际土壤配置写入采集和导出元数据。** `sand_config_snapshot` 在 run 创建时保存的状态仍可能是未初始化状态；本次文件记录 `real_sand_enabled=false`、粒子数 0。`dataset_sand_status` 又只保留少数旧字段，缺少 Newton 后端、网格、材料参数和有效计数粒子配置。建议在 episode 开始时记录实际配置，将其写入导出清单，并让 append 检查后端/配置兼容性。这样可以识别历史 PhysX 数据与新增 Newton 数据的物理差异。
3. **先细分物理解算耗时，再做性能优化。** 本回合桥接服务端平均约 1202 ms，其中物理推进约 931 ms（77%）、相机读回 155 ms、缩放 37 ms、编码 62 ms，桶/车厢计数合计约 16 ms。应先给 Newton 的 `solver.step`、`project_outside` 和渲染粒子更新分别计时，确认收益来源。切换 `--no-fresh-simulator` 或 `--no-render-every-physics-step` 会进入 diagnostic 配置，不应将该配置的结果当作标准评估。

## 复现入口

在仓库根目录执行，临时目录避免追加到现有训练数据：

~~~bash
POLICY_PYTHON=/root/gpufree-data/conda/envs/ExcavatorVLA-current/bin/python
TEST_ROOT=$(mktemp -d "$PWD/outputs/pipeline_check_XXXXXX")

POLICY_PYTHON="$POLICY_PYTHON" DATA_ROOT="$TEST_ROOT/collection" \
  bash collect.sh 1 1 0

POLICY_PYTHON="$POLICY_PYTHON" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  bash eval_smolvla_success.sh \
  outputs/train/excavator_smolvla_20261014_190059_474909_268032/checkpoints/040000/pretrained_model \
  1 350 100 --workers 1 --gpus 1 --cpu-threads 8 --output "$TEST_ROOT/eval"

"$POLICY_PYTHON" scripts/audit_smolvla_success.py \
  --batch-dir "$TEST_ROOT/eval" --verify-sources

"$POLICY_PYTHON" tests/run_success_eval_reset_live.py \
  --output "$TEST_ROOT/reset" --gpu 0
~~~

本次临时采集数据、评估明细、视频和测试日志按仓库 universal 技能要求清理；保留此验证记录。已有数据、checkpoint 和测试前正在运行的任务保留。
