# 上层 Pick-and-Place 远程训练交接

**整理日期**：2026-11-07

**本地分支**：`codex/low-shelf-pick-place`

**当前 HEAD**：`38951dd feat(tasks): add two-cylinder pick-and-place scene`

**远端**：`origin = https://github.com/Eurekahah/loco-manip-unified-rl-agent.git`

## 1. 当前交接状态

当前本地已完成三个提交：

```text
5634c59 feat(teleop): add EE-only debug visualization option
26ec282 feat(data): add low-shelf RGB-D HDF5 collection
38951dd feat(tasks): add two-cylinder pick-and-place scene
```

当前分支还没有 upstream，也没有执行 `git push`。如果远程训练主机使用另一个 checkout，必须先让这三个提交在远程主机可见。远程 session 开始前必须确认：

```powershell
git status --short --branch
git log -4 --oneline --decorate
git rev-parse --short HEAD
```

远程 HEAD 必须是 `38951dd` 或包含它的后代提交。若看不到该提交，不要继续训练，先报告缺少提交并等待同步。

## 2. 上层架构概览

当前任务沿用现有 M20 + Piper 上层 RL 架构，不重新训练低层 locomotion/WBC 策略。

```text
高层策略 RL 输出 13 维动作
  -> PreTrainedPickWBCAction / TeleopLLAction
  -> 现有低层 WBC policy checkpoint
  -> 腿、轮、机械臂 IK、夹爪执行
```

13 维动作布局：

```text
[vx, vy, wz,
 delta_ee_x, delta_ee_y, delta_ee_z,
 delta_ee_roll, delta_ee_pitch, delta_ee_yaw,
 delta_body_height, delta_body_pitch, delta_body_roll,
 gripper]
```

采集任务中键盘和 VR 都使用“重新标定锚点 + 绝对偏移”语义；`delta_*` 指相对锚点的偏移，不是逐帧滚动增量。`gripper` 使用 `+1 = 开`、`-1 = 闭`。

训练任务：

- 使用 `PreTrainedPickWBCAction`。
- 使用状态教师观测，不使用相机。
- `RecorderManager` 为空，不写 HDF5。
- 训练任务仍通过现有 WBC checkpoint 执行低层动作。

纯遥操任务：

- 使用 `TeleopLLAction`。
- 相机和 recorder 默认关闭。
- 用于键盘或 VR 交互测试。

采集任务：

- 使用 `TeleopLLAction`。
- 显式启用场景相机和腕部 RGB-D。
- 显式启用 `LowShelfHDF5Recorder`。
- 不属于本次远程训练范围。

## 3. 两个新建训练任务

### 3.1 低层货架 Pick-and-Place

训练任务：

```text
Isaac-M20-Piper-LowShelf-Place-v0
```

配置：

```text
source/rl_training/rl_training/tasks/manager_based/locomotion/highlevel/config/low_shelf_place/env_cfg.py
LowShelfPlaceEnvCfg
```

场景：

- 双层货架。
- 下层取料板高约 `0.30 m`。
- 上层放置板高约 `0.55 m`。
- YCB `004_sugar_box.usd`。
- 目标区在上层蓝色放置垫。

语言指令：

```text
把下层托盘里的糖盒放到上层桌面。
```

### 3.2 双圆柱台 Pick-and-Place

训练任务：

```text
Isaac-M20-Piper-TwoCylinder-Place-v0
```

配置：

```text
source/rl_training/rl_training/tasks/manager_based/locomotion/highlevel/config/two_cylinder_place/env_cfg.py
TwoCylinderPlaceEnvCfg
```

场景：

- 矮圆柱台：`(0.60, -0.22)`，顶面高 `0.30 m`。
- 高圆柱台：`(0.72, 0.38)`，顶面高 `0.52 m`。
- YCB `004_sugar_box.usd`，绕本地 X 轴旋转 90°。
- 单物体、单目标区。
- 两平台横向错开，搬运阶段需要 M20 底盘侧移。

语言指令：

```text
把糖盒从矮圆柱台放到高圆柱台顶面。
```

## 4. 训练任务配置分离

| 用途 | Low Shelf | Two Cylinder | 相机 | Recorder |
|---|---|---|---|---|
| RL 训练 | `Isaac-M20-Piper-LowShelf-Place-v0` | `Isaac-M20-Piper-TwoCylinder-Place-v0` | 关闭 | 关闭 |
| 纯遥操 | `Isaac-M20-Piper-LowShelf-Teleop-v0` | `Isaac-M20-Piper-TwoCylinder-Teleop-v0` | 关闭 | 关闭 |
| 数据采集 | `Isaac-M20-Piper-LowShelf-Collect-v0` | `Isaac-M20-Piper-TwoCylinder-Collect-v0` | 开启 | 开启 |

远程训练只使用两个 `Place-v0` 任务，不使用 `Teleop-v0` 或 `Collect-v0`。

## 5. 训练依赖

### 5.1 Conda 环境

远程主机需使用 Isaac Lab 对应环境，预期名称为：

```powershell
conda activate env_isaac_lab
```

远程主机需确认 Isaac Sim、Isaac Lab、PyTorch、CUDA 和 GPU 驱动可用。不要在当前 session 假定远程驱动版本与本地 `580.88` 相同。

### 5.2 低层 WBC checkpoint

两个训练任务都依赖现有低层 WBC policy。默认路径解析逻辑位于：

```text
source/rl_training/rl_training/tasks/manager_based/locomotion/highlevel/config/high_level/hl_flat_pick_env_cfg.py
```

默认候选：

```text
logs/rsl_rl/deeprobotics_m20_wbc_flat/2026-09-18_01-31-58/exported/policy.pt
```

也可以通过环境变量覆盖：

```powershell
$env:RL_TRAINING_LOW_LEVEL_POLICY_WBC="<absolute-path-to-policy.pt>"
```

Linux 远程主机使用：

```bash
export RL_TRAINING_LOW_LEVEL_POLICY_WBC="<absolute-path-to-policy.pt>"
```

远程 session 必须先确认该文件存在，且 checkpoint 的 `policy_layout.json` 与当前 replay 逻辑匹配。若文件缺失，应停止并报告，不要改任务去绕过低层策略。

### 5.3 机器人资产

机器人 USD 来自：

```text
rl_training.assets.deeprobotics.DEEPROBOTICS_M20_PIPER_CFG
```

远程主机需具备 `ISAACLAB_ASSETS_DATA_DIR` 下的 M20 Piper 资产和 Nucleus 访问能力。

## 6. 远程简单训练计划

本次只做“可运行性和短训练 smoke”，不要求策略收敛，也不要求抓取/放置成功率。

建议规模：

```text
--headless
--num_envs 4
--max_iterations 20
```

先做 1 环境、1 iteration，确认环境和低层 replay 正常；再做 4 环境、20 iteration，确认训练循环、显存和日志正常。

### 6.1 配置解析

```powershell
python scripts/reinforcement_learning/rsl_rl/probe_task_cfg.py --task Isaac-M20-Piper-LowShelf-Place-v0 --headless
python scripts/reinforcement_learning/rsl_rl/probe_task_cfg.py --task Isaac-M20-Piper-TwoCylinder-Place-v0 --headless
```

两个命令都应显示：

```text
cameras      : []
recorders    : []
```

### 6.2 Low Shelf 短训练

```powershell
python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-M20-Piper-LowShelf-Place-v0 --headless --num_envs 1 --max_iterations 1
python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-M20-Piper-LowShelf-Place-v0 --headless --num_envs 4 --max_iterations 20
```

日志预期位置：

```text
logs/rsl_rl/low_shelf_place/<timestamp>/
```

### 6.3 Two Cylinder 短训练

```powershell
python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-M20-Piper-TwoCylinder-Place-v0 --headless --num_envs 1 --max_iterations 1
python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-M20-Piper-TwoCylinder-Place-v0 --headless --num_envs 4 --max_iterations 20
```

日志预期位置：

```text
logs/rsl_rl/two_cylinder_place/<timestamp>/
```

## 7. 验收标准

每个任务至少记录：

- 环境配置名。
- 是否使用 `PreTrainedPickWBCAction`。
- `cameras=[]`、`recorders=[]`。
- 低层 WBC checkpoint 实际路径。
- 1 环境 smoke 的退出码。
- 4 环境短训练的退出码。
- 场景创建耗时。
- iteration 耗时。
- 实际日志目录。
- reward 项和 termination 项是否正常输出。
- 是否出现 CUDA、Vulkan、Nucleus、USD 或 checkpoint 错误。

训练结束不要求 `place_success` 非零。随机初始策略下成功率为零是正常现象。

## 8. 已知限制

- 自动抓取脚本还没有稳定抬升糖盒，不能把短训练 smoke 当成任务成功验证。
- Low Shelf 和 Two Cylinder 都只验证了场景、配置、相机或资产的一部分闭环，尚未完成真实 VR 成功 episode。
- Two Cylinder 的高台成功高度偏移 `0.088 m` 与当前糖盒旋转姿态绑定。
- 远程训练只验证上层 RL 能否运行，不验证完整 sim-to-real 部署。
- 当前本地提交尚未 push。远程主机看不到 `38951dd` 时，训练会落到旧代码。

## 9. 相关文件

- `docs/pick_place/low_shelf_place_zh.md`
- `docs/pick_place/two_cylinder_place_zh.md`
- `source/rl_training/rl_training/tasks/manager_based/locomotion/highlevel/config/low_shelf_place/`
- `source/rl_training/rl_training/tasks/manager_based/locomotion/highlevel/config/two_cylinder_place/`
- `scripts/reinforcement_learning/rsl_rl/probe_task_cfg.py`
- `scripts/reinforcement_learning/rsl_rl/train.py`

