# Fast Teleop / Fast 数采与热路径 profiling

训练环境继续使用完整的 `ManagerBasedRLEnv.step()`。遥操和数采默认使用 Fast 路径；
需要对照 manager 路径时传 `--no-fast-teleop` 或 `--no-fast-stepping`。Fast 路径复用
现有 robot、scene、action manager、低层策略和 history。

## 默认配置与高级覆盖

```text
--fast-teleop / --no-fast-teleop    默认开启
--fast-render-hz                 墙钟渲染频率，默认 15 Hz
--fast-write-each-tick           默认关闭；关闭时每 20 ms 控制 tick 写一次目标
--fast-robot-hot-loop            默认开启；机械臂/robot target 以 hot period 更新
--fast-robot-hot-period          默认 10 ms
--fast-robot-only-write          实验开关；只写 robot，不写其它 scene 实体
--ik-update-period               IK 重算周期（仿真秒），默认 0.01，与 10 ms target 写入对齐
--profile-detailed               拆分 action/policy/IK/scene/sim/render 计时
--rendering_mode                 默认 balanced
```

单环境遥操和数采（Fast 或 manager 路径）在用户未显式传 `--device` 时默认使用 CPU 物理。
需要强制 GPU 时显式传 `--device cuda:0`。该默认值来自下面的同机 A/B，不是经验猜测。

## Fast path 结构

```text
读取最新遥操命令
  -> action_manager.process_action
  -> action_manager.apply_action          # 每 5 ms 物理 tick
  -> scene.write_data_to_sim              # 每 20 ms 控制 tick
  -> robot.write_data_to_sim              # 可选 hot loop，每 10 ms
  -> 4 x sim.step(render=False)           # 4 x 5 ms 物理 tick
  -> wall-clock 到期时 sim.render
  -> scene.update                         # 每 20 ms 控制 tick
  -> episode_length_buf/common_step_counter += 1
```

默认不再每 5 ms 重复写相同目标。`--fast-write-each-tick` 可恢复逐物理 tick
写入，用于 A/B 验证。

Fast 数采保持同一个控制时间轴：`record_pre_step()` 在推进前取动作、机器人状态和
相机；`advance()` 后由 `record_post_step()` 入异步 HDF5 队列。
`common_step_counter` 每个控制周期只推进一次，因此 `record_hz=10` 时 50 个控制周期
会得到 10 帧，`sim_time` 为 0、0.1、0.2 ...，不会出现所有帧都写在 `t=0` 的问题。

## Fast 数采命令

```powershell
$env:PYTHONIOENCODING='utf-8'
python scripts\data\collect_demonstrations.py `
  --task Isaac-M20-Piper-TwoCylinder-Collect-v0 `
  --headless `
  --no-top_camera --no-scene_camera --wrist_camera `
  --output recordings\fast_teleop\two_cylinder_fast.hdf5 `
  --profile-json outputs\fast_teleop\fast_profile.json
```

Fast 模式每个控制周期都会计算 termination term。`place_success` 触发后先记录
当前帧和 termination flags，再调用 `env.reset()` finalize 当前 episode 并切换下一集。
键盘 `K/L` 或 VR success/failure 仍作为手动覆盖。HDF5 仍通过异步 writer 写入，
控制线程只负责采样和入队。

## 详细 profiling

`DetailedStepProfiler` 会分别记录：

```text
action_process_ms
action_apply_ms
ll_history_ms
ll_obs_ms
ll_policy_ms
ll_ik_target_ms
ll_ik_process_ms
ll_ik_apply_ms
scene_write_ms
sim_step_ms
scene_update_ms
render_ms
obs_manager_ms
event_manager_ms
```

## 2026-11-17 官方实现对比

本地 Isaac Lab 5.1.0 的官方遥操入口是
`scripts/environments/teleoperation/teleop_se3_agent.py`。它的控制体仍然只有：

```text
teleop_interface.advance()
  -> env.step(actions)
```

官方没有独立的 Fast Teleop 循环。它把优化放在设备抽象、retargeter、场景配置和
CloudXR 管线中，并在文档里给出两项与本项目直接相关的建议：

- 单环境优先尝试 `--device cpu`，避免 GPU 管线的同步开销。
- CloudXR 以 90 Hz 显示、45 FPS 渲染为目标；120 Hz 物理需要高端工作站。

为避免拿不同机器人、不同 decimation 直接比较，本仓库新增
`scripts/tools/benchmark_teleop_control.py`，统一 warmup、固定动作、headless
和计时口径。测试机为 Ryzen 7 2700X + RTX A4000：

| 控制路径 | 设备 | decimation | FPS | p95 |
|---|---|---:|---:|---:|
| 官方 Lift/Franka，完整 `env.step()` | GPU | 2 | 41.72 | 25.90 ms |
| 官方 Lift/Franka，完整 `env.step()` | CPU | 2 | **159.17** | 6.66 ms |
| M20 Teleop，完整 `env.step()` | GPU | 4 | 5.94 | 180.35 ms |
| M20 Teleop，完整 `env.step()` | CPU | 4 | 22.05 | 47.24 ms |
| M20 Fast，10 ms hot，IK 200 Hz | GPU | 4 | 10.65 | 100.87 ms |
| M20 Fast，10 ms hot，IK 100 Hz | GPU | 4 | 12.94 | 83.17 ms |
| M20 Fast，10 ms hot，IK 100 Hz | CPU | 4 | **51.13** | 20.29 ms |

表中的“官方 Lift/Franka”是官方任务加官方 `ManagerBasedRLEnv.step()` 路径，
不包含网络、手柄轮询、GUI 和相机。它说明官方实现本身并不存在魔法：
任务简单、物理频率较低时可达很高 FPS；本项目 M20 的完整管理器链在 GPU 上仍只有
约 6 FPS。

最值得借鉴的官方结论是 CPU 单环境物理。M20 Fast 路径从 GPU 的 12.94 FPS
提升到 CPU 的 51.13 FPS，控制体周期从 77.29 ms 降到 19.56 ms。

固定动作轨迹验收（100 步，30 步 warmup）：

| 配置 | q range | q 二阶差分 RMS | target 二阶差分 RMS |
|---|---:|---:|---:|
| GPU Fast，10 ms hot，IK 100 Hz | 0.001852 | 8.08e-5 | 1.64e-5 |
| CPU Fast，10 ms hot，IK 100 Hz | **0.001275** | **1.70e-5** | **3.68e-6** |

CPU 路径不仅更快，固定命令下的轨迹抖动也更低。原 IK 200 Hz 并非必要：
`--ik-update-period 0.01` 与 10 ms target 写入同频，实测轨迹不差于原默认。

腕部相机 224x224、`record_hz=10`、`fast-render-hz=15` 的 CPU 采集冒烟：

| 配置 | 帧数 | 整体 FPS | 控制/渲染表现 |
|---|---:|---:|---|
| CPU Fast 数采，腕部相机 | 100 | 34.69 | 控制约 20 ms，15 Hz 渲染单次约 40 ms |
| CPU Fast 数采，腕部相机（默认参数冒烟） | 30 | 38.53 | 控制约 20 ms，渲染约 14 ms |

相机渲染仍会阻塞主线程，所以渲染发生时会出现约 40-60 ms 的尖峰；但控制空档已经
明显低于原 GPU 路径，满足 15-25 Hz 交互目标。

时间轴验收：

```text
sim_time   = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
step_index = [0, 5, 10, 15, 20, 25]
sensor_frame = [1, 2, 3, 4, 5, 6]
```

30 个控制周期、`record_hz=10` 写出 6 帧，RGB 与动作、机器人状态使用同一
`sim_time`；异步 writer 队列为 0，无丢帧。

复现命令：

```powershell
python scripts\tools\benchmark_teleop_control.py `
  --task Isaac-M20-Piper-Teleop-History-v0 `
  --mode fast_hot --warmup 30 --steps 100 `
  --fast-robot-hot-period 0.01 --ik-update-period 0.01 `
  --profile-detailed --device cpu
```

## A/B 结果

测试：`Isaac-M20-Piper-Teleop-History-v0`，单环境，headless，无相机，30 步。

| 路径 | FPS | loop mean |
|---|---:|---:|
| 完整 `env.step` | 5.60 | 208.35 ms |
| Fast，每 5 ms 写目标，IK 5 ms | 6.30 | 181.49 ms |
| Fast，每 20 ms 写目标，IK 20 ms | 17.57 | 87.52 ms |
| Fast，默认 10 ms robot hot loop，100 步稳定 | 10.65 | 101.78 ms |
| Fast collection，腕部相机，10 Hz，50 步 | 10.43 | 112.74 ms |

Fast 配置主要耗时：

```text
4 x sim.step              约 37 ms
scene.write_data_to_sim   约 13 ms
action apply              约 22 ms（含首次/偶发 IK spike）
action process            约 7 ms
scene.update              约 7 ms
```

100 步默认 Fast 样本：

```text
env_step mean    101.37 ms
env_step p95     102.25 ms
sim.step mean      8.54 ms（每个物理 tick）
scene.write       11.94 ms（每个控制 tick）
robot.write       11.91 ms（每 10 ms）
robot.update       0.27 ms（每个物理 tick）
scene.update       6.39 ms（每个控制 tick）
action apply       8.00 ms（每个物理 tick）
low-level policy   2.30 ms（每 4 个物理 tick）
IK apply           5.60 ms（每 4 个物理 tick）
```

关键修正：

- `action_manager.apply_action()` 必须每个 5 ms 物理 tick 调用，否则低层
  policy/history 计数器会降到约 12.5 Hz。
- 只写 20 ms 一次会让机械臂目标保持过久。固定命令轨迹 A/B：

```text
normal       q_range=0.002141  target_second_diff_rms=0.00000148
fast 20 ms   q_range=0.257056  target_second_diff_rms=0.08163100
fast_hot 10ms q_range=0.001852 target_second_diff_rms=0.00001641
fast_hot 5ms  q_range=0.000323 target_second_diff_rms=0.00000136
```

默认 Fast 使用 10 ms robot hot loop，兼顾抖动和速度。需要最平滑控制时可显式使用
`--fast-robot-hot-period 0.005`，代价约为 7.7 FPS。

首个 episode step 仍包含约 0.5 秒的首次策略/IK 缓存 spike，之后进入稳定区间。

IK 周期 A/B 的结论：`--ik-update-period 0.01` 与默认 0 的轨迹指标相同；
`0.02` 的 target 二阶差分 RMS 略低，但 target range 从 `0.00100` 升到
`0.00174`、q range 从 `0.00185` 升到 `0.00322`，台阶感更明显，因此不设成默认。

`--fast-robot-only-write` 只建议用于没有其它被控 articulation 的任务。two_cylinder
场景实测没有 FPS 收益（10.22 vs 10.43），因为耗时仍在 robot actuator 写入，
所以默认关闭。

当前结果说明：

- 完整 manager 链不是唯一瓶颈，但跳过它仍有收益。
- 逐 5 ms 重复写目标明显浪费，20 ms 写一次更适合 50 Hz 高层控制。
- Fast 默认使用 CPU 物理和 100 Hz IK 后，控制体已达到约 51 FPS；相机配置仍有渲染尖峰。
- 首次/偶发策略和 IK 编译 spike 仍会拉高 max，正式 profiling 必须保留 warmup。
- 关闭 armature self-collision/contact reporting 仍未做正式一致性验收。

## 当前限制

- Fast path 已接入键盘、VR 遥操和通用 HDF5 数采；相机采样仍在主线程，异步 writer
  只负责磁盘写入。
- Fast path 不执行训练 reward/event manager，但会执行 termination manager，
  以便自动判定成功/失败并切换 episode；手动 reset 仍走完整环境。
- 显式传 `--device cuda:0` 可复现旧 GPU 路径；多环境 Fast 路径不会自动切 CPU。
- 相机 RGB/depth `.cpu()`、帧转换和预览仍在主线程，下一步应把 CPU 转换和有界队列
  从控制热路径移走。
