# 下个 Session 开工 Prompt：Isaac M20-Piper 遥操与数采低延迟

> 直接把本文件整段作为下个 session 的起始 prompt。目标是继续优化 Isaac Sim 中的
> M20-Piper VR/键盘遥操和 HDF5 数据采集，不要重新设计任务专用采集循环。

## 1. 环境、仓库和当前状态

```text
仓库：D:\nvidia-isaac-sim\loco-manip-unified-rl-agent
Isaac Lab：D:\nvidia-isaac-sim\IsaacLab-5.1.0
Python：C:\Users\autolab\miniconda3\envs\env_isaac_lab\python.exe
Shell：PowerShell
分支：codex/low-shelf-pick-place
当前实现截至：f2ed46a feat: integrate fast stepping with data collection
工作区：干净
```

运行 Isaac 前设置：

```powershell
$env:PYTHONIOENCODING='utf-8'
```

Git 命令使用：

```powershell
git -c safe.directory=D:/nvidia-isaac-sim/loco-manip-unified-rl-agent status --short
```

不要使用 bash/WSL 运行本仓库命令。当前工作区可能再次出现用户改动；先读
`git status`，不要回滚不属于本 session 的修改。

## 2. 已经完成的架构拆分

### 训练和交互分离

- 训练仍使用完整 `ManagerBasedRLEnv.step()`，不要改训练语义。
- 遥操使用 opt-in 的 `--fast-teleop`。
- 数据采集使用 opt-in 的 `--fast-stepping`。
- Fast 路径跳过训练用的 reward、observation、termination、event、command manager，
  但保留 action manager、低层策略、history、IK 和 scene。

核心文件：

```text
scripts/teleoperation/fast_teleop.py
scripts/teleoperation/teleop_m20_piper_keyboard.py
scripts/teleoperation/teleop_m20_piper_vr.py
scripts/data/collect_demonstrations.py
scripts/data/collection/cli.py
scripts/data/collection/runner.py
scripts/data/collection/profiles.py
scripts/data/collection/builtin_profiles.py
scripts/data/replay_hdf5.py
source/rl_training/rl_training/recorders/hdf5.py
source/rl_training/rl_training/datasets/hdf5.py
```

### 数采和任务隔离

- 通用采集入口只有 `scripts/data/collect_demonstrations.py`。
- 具体任务只提供 collection profile，声明 task id、相机工厂、动作 term、被采实体、
  成功条件和输出目录。
- 新增任务时不要复制采集主循环。
- HDF5 使用通用 `teleop_manipulation/v1` schema。
- HDF5 回放不加载 Isaac Sim，不依赖 task id 或 collection profile。
- 旧的任务专用采集入口已删除，不要重新引入。

### Fast Teleop 当前循环

```text
读取最新遥操命令
  -> action_manager.process_action
  -> recorder.record_pre_step        # 数采时
  -> FastTeleopStepper.advance
       -> 4 个 5 ms 物理 tick
       -> action_manager.apply_action 每个物理 tick
       -> scene.write_data_to_sim 默认每 20 ms 控制周期
       -> robot.write_data_to_sim 默认每 10 ms hot loop
       -> sim.step(render=False)
       -> 墙钟到期时 sim.render()
       -> scene.update
       -> episode_length_buf/common_step_counter += 1
  -> recorder.record_post_step       # 数采时
```

关键约束：

- `action_manager.apply_action()` 必须每个 5 ms 物理 tick 调用，否则低层策略/history
  会从 50 Hz 掉到约 12.5 Hz，机械臂会明显抖动。
- 不要通过降低 decimation、提高手柄灵敏度或调整死区来掩盖主循环问题。
- 机械臂 target 默认每 10 ms 写入；5 ms 更平滑但更慢。
- 数据采集的 HDF5 writer 是异步的，控制线程只负责采样和入队。
- Fast 数采当前不自动运行 termination manager，结束时用键盘 `K/L` 或 VR
  success/failure 手动标记。

## 3. 当前实测结果

测试环境：单环境、Windows、RTX A4000、Isaac Lab 5.1、headless。

| 路径 | FPS | 控制周期 |
|---|---:|---:|
| 完整 `ManagerBasedRLEnv.step()` | 5.60 | 208.35 ms |
| Fast，20 ms target，无 robot hot loop | 16.33 | 68.52 ms，但严重抖动 |
| Fast，默认 10 ms robot hot loop | 10.65 | 101.78 ms |
| Fast，5 ms robot hot loop | 7.68 | 最平滑 |
| Fast 数采，腕部相机 10 Hz、224×224 | 10.43 | 112.74 ms |

固定命令轨迹的抖动指标：

| 模式 | q range | target 二阶差分 RMS |
|---|---:|---:|
| normal | 0.002141 | 1.48e-6 |
| Fast，20 ms target | 0.257056 | 8.16e-2 |
| Fast hot，10 ms robot | 0.001852 | 1.64e-5 |
| Fast hot，5 ms robot | 0.000323 | 1.36e-6 |

结论：20 ms target-only 虽然快，但不可用；10 ms hot loop 是当前速度和抖动的折中。

数采时间轴验证：

```text
50 个控制周期、record_hz=10
HDF5 写出 10 帧
sim_time = [0.0, 0.1, ..., 0.9]
step_index = [0, 5, 10, ..., 45]
wrist_camera RGB = (10, 224, 224, 3)
```

注意：`timestamp` 是墙钟时间，实测中位间隔约 0.507 s；`sim_time` 是仿真时间，
两者不能混用。当前数据是仿真时间对齐，不是实时 10 Hz 或 25 Hz。

已完成的 A/B：

- `--ik-update-period 0.01` 与默认 0 的轨迹指标相同。
- `--ik-update-period 0.02` 虽然少算 IK，但 target/q 轨迹幅度增大，台阶感更明显，
  不作为默认。
- `--fast-robot-only-write` 在 two_cylinder 场景没有 FPS 收益（10.22 vs 10.43），
  因为 robot actuator 写入仍是主要成本，默认关闭。
- 首个 episode step 有约 0.5 s 的策略/IK 缓存 spike；profiling 必须有 warmup。

## 4. 已修复的问题和测试基线

相关缺陷记录：

```text
DEF-071：Fast Teleop 机械臂持续抖动
DEF-072：Fast 数采 common_step_counter 未推进，HDF5 只记录首帧
```

已修复内容：

- Fast stepper 增加 `process_action()` / `advance()` 拆分，便于数采在推进前后采样。
- `apply_action()` 保持每个 5 ms 物理 tick 执行。
- 10 ms robot hot loop 降低机械臂目标跳变。
- 增加 `ik_update_period_s`、`--fast-robot-hot-period`、`--fast-render-hz`。
- Fast stepper 每个控制周期推进 `episode_length_buf` 和 `common_step_counter`。
- HDF5 的 `sim_time`、`step_index`、动作和图像恢复同一控制时间轴。
- 数采保持异步 HDF5 writer；replay 与具体任务解耦。

当前回归：

```text
tests/data/test_hdf5_async.py
tests/data/test_collection_profiles.py
tests/data/test_replay_hdf5.py
tests/fisheye/test_fisheye_gpu.py
tests/teleop/test_fast_teleop.py

结果：10 passed
```

## 5. 当前可用命令

键盘 Fast 遥操（带四鱼眼和全景）：

```powershell
$env:PYTHONIOENCODING='utf-8'
python scripts\teleoperation\teleop_m20_piper_keyboard.py `
  --task Isaac-M20-Piper-Teleop-History-v0 `
  --num_envs 1 `
  --fast-teleop `
  --fast-render-hz 15 `
  --fast-robot-hot-loop `
  --fast-robot-hot-period 0.01 `
  --ik-update-period 0.0 `
  --fisheye-rig four `
  --show-panorama `
  --fisheye-width 480 --fisheye-height 300 `
  --panorama-width 960 --panorama-height 480 `
  --rendering_mode balanced
```

two-cylinder 遥操把 task 换成：

```text
Isaac-M20-Piper-TwoCylinder-Teleop-v0
```

VR 遥操把入口换成 `scripts\teleoperation\teleop_m20_piper_vr.py`，Fast 参数相同。

Fast 数采（注意参数名是 `--fast-stepping`，不是 `--fast-teleop`）：

```powershell
python scripts\data\collect_demonstrations.py `
  --task Isaac-M20-Piper-TwoCylinder-Collect-v0 `
  --profile auto --input keyboard --headless `
  --no-top_camera --no-scene_camera --wrist_camera `
  --record-hz 10 --compression lzf `
  --fast-stepping --fast-render-hz 15 `
  --fast-robot-hot-period 0.01 --ik-update-period 0.0 `
  --output recordings\fast_teleop\two_cylinder_fast.hdf5 `
  --profile-json outputs\fast_teleop\fast_profile.json
```

检查回放：

```powershell
python scripts\data\replay_hdf5.py `
  --input recordings\fast_teleop\two_cylinder_fast.hdf5 `
  --check-only
```

## 6. 下一步性能路线（按风险从低到高）

### P0：先做精确 profiling，不改默认行为

使用更长 episode 和 warmup，分别测量：

```text
advance()
低层 policy forward
IK history / process / apply
action_manager.apply_action
robot.write_data_to_sim
scene.write_data_to_sim
sim.step
scene.update
sim.render
camera readback / GPU-to-CPU
HDF5 enqueue / writer
```

需要做两个快速实验：

1. `--device cpu` A/B。单环境 GPU 管线可能有额外同步开销，不能假设 GPU 一定更快。
2. 仅用于 profiling 的 PhysX 对照：临时关闭 armature self-collision 和 contact
   reporting，观察 `sim.step` 是否明显下降。正式采集是否关闭必须另做一致性验收。

不要把“提高灵敏度、缩小死区、降低 render_interval”当成主修法。

### P1：继续拆 Fast 控制热路径

当前仍是一个轻量 stepper，不是完整的专用 `TeleopFastEnv`。下一步可以：

- 明确缓存低层策略真正需要的 robot state、policy observation 和 history。
- 跳过不需要的训练 manager 状态，但保持 `apply_action()` 的 5 ms 语义。
- 尝试 target dirty detection：只有 target 变化时才重复写 actuator；必须验证
  PhysX target、external wrench 和 IK 语义，不能直接假设安全。
- 评估选择性写 robot actuator / 分 actuator 写入，但此前“只重算 piper_arm”已证明
  轨迹更差，不要重复同一实验。
- 把 `robot.write_data_to_sim`、action apply、scene update 的耗时分别固定下来，
  每次只改一个变量。

### P2：把相机和录制从控制热路径移走

当前 HDF5 磁盘写入已经异步，但相机 readback、RGB/depth `.cpu()`、帧转换仍可能在
主线程。下一步目标：

- VR/输入线程只更新 thread-safe 的 latest command；控制线程非阻塞读取最新值，
  永不等待 VR 线程。
- 控制线程只保存最新状态和相机 frame 编号的快照，不等待磁盘。
- 主线程负责 Isaac 必须的 sensor readback；CPU 转换、HDF5 写入、图像压缩放后台。
- 使用有界环形缓冲区，队列满时明确计数和丢帧，不允许阻塞控制。
- 验证每帧的动作、机器人状态、`sensor_frame`、`sim_time` 和图像严格对应。
- 渲染只消费最新状态，按墙钟 10–20 Hz 执行；控制线程不等待渲染。
- 如果 Python/GIL 或 Omni GUI 仍阻塞，再做进程拆分：Isaac 物理控制留在主进程，
  VR/渲染/录制放辅助进程，控制线程绝不等待相机或磁盘。

### P3：如果仍达不到目标，采用 MuJoCo 前面板方案

MuJoCo 对照实现给我们的重要经验不是继续微调 VR 滤波，而是：

- Isaac 完整 `ManagerBasedRLEnv` 主循环约 5 FPS 时，单靠灵敏度、死区或
  `render_interval` 无法解决交互延迟。
- 现实目标是 Fast path 稳定 15–25 Hz；当前默认约 10–11 Hz，仍未达到。
- 如果 Isaac 单进程优化后仍达不到目标，推荐：

```text
MuJoCo 前端负责实时遥操，记录高层命令和时间戳
Isaac 后端回放高层命令，生成物理状态和多相机数据
```

这条路径把“人手操作延迟”和“Isaac 高保真相机生成”分开，优先保证遥操手感。
如果采用它，回放输入必须是任务无关的命令 schema，不能把 MuJoCo 任务逻辑复制进
Isaac 采集器。

## 7. 数据质量和验收标准

在提速前先确认以下不变量：

- 训练路径仍走完整 `ManagerBasedRLEnv.step()`。
- Fast 路径必须显式 opt-in。
- `apply_action()` 不能降频。
- `record_hz`、相机分辨率、相机 update period 不能为了提高 FPS 被静默降低。
- HDF5 的 `sim_time` 单调递增，`step_index` 与动作/图像采样一致。
- 相机 frame 未更新时不能伪装成新帧。
- writer 丢帧必须有计数和日志。
- 每个新任务只新增 env cfg 和 collection profile，不复制采集循环。

下一阶段的验收至少包括：

1. 无相机、腕部相机、鱼眼全景三种配置的 FPS 和 p95。
2. 固定命令下 q、dq、target 的 range 和二阶差分 RMS。
3. 50–100 步 warmup 后的稳定控制周期。
4. HDF5 replay 能显示动作对应的连续图像。
5. 相关离线测试全通过，Isaac 冒烟退出码为 0。
6. 所有默认值和不采用某个实验的原因写回 `docs/teleop_fast_path_zh.md` 和
   `docs/review/DEFECT_LOG_zh.md`。

## 8. 推荐工作顺序

1. 先读 `git status`、`docs/teleop_fast_path_zh.md`、`DEF-071/072` 和相关测试。
2. 复现当前 10 ms hot loop 的 50 步腕部相机基线。
3. 加长 warmup，完成无相机/腕部/鱼眼和 CPU/GPU 的 profiling。
4. 只选一个瓶颈做修改：优先相机 snapshot/转换异步，其次 actuator 写入，最后才考虑
   进程拆分。
5. 每改一项都跑 `tests/data/*`、`tests/fisheye/test_fisheye_gpu.py`、
   `tests/teleop/test_fast_teleop.py`，并做一次短 Isaac 冒烟。
6. 把结果、被否掉的方案和下一阶段限制写回文档，再按逻辑拆分 commit。
