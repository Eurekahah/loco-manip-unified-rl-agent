# Fast Teleop 原型与热路径 profiling

训练环境继续使用完整的 `ManagerBasedRLEnv.step()`。遥操新增 opt-in 的
`--fast-teleop`，复用现有 robot、scene、action manager、低层策略和 history，
但跳过完整 reward/observation/termination/event 管理链。

## 新增参数

```text
--fast-teleop / --no-fast-teleop
--fast-render-hz                 墙钟渲染频率，默认 15 Hz
--fast-write-each-tick           默认关闭；关闭时每 20 ms 控制 tick 写一次目标
--ik-update-period               IK 重算周期（仿真秒），默认 0 表示每个物理 tick
--profile-detailed               拆分 action/policy/IK/scene/sim/render 计时
```

## Fast path 结构

```text
读取最新遥操命令
  -> action_manager.process_action
  -> action_manager.apply_action          # 每 20 ms 控制 tick
  -> scene.write_data_to_sim              # 每 20 ms 控制 tick
  -> 4 x sim.step(render=False)           # 4 x 5 ms 物理 tick
  -> wall-clock 到期时 sim.render
  -> scene.update                         # 每 20 ms 控制 tick
```

默认不再每 5 ms 重复写相同目标。`--fast-write-each-tick` 可恢复逐物理 tick
写入，用于 A/B 验证。

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

## A/B 结果

测试：`Isaac-M20-Piper-Teleop-History-v0`，单环境，headless，无相机，30 步。

| 路径 | FPS | loop mean |
|---|---:|---:|
| 完整 `env.step` | 5.60 | 208.35 ms |
| Fast，每 5 ms 写目标，IK 5 ms | 6.30 | 181.49 ms |
| Fast，每 20 ms 写目标，IK 20 ms | 17.57 | 87.52 ms |
| Fast，每 20 ms 写目标，IK 20 ms，100 步稳定 | 16.33 | 68.52 ms |

Fast 配置主要耗时：

```text
4 x sim.step              约 37 ms
scene.write_data_to_sim   约 13 ms
action apply              约 22 ms（含首次/偶发 IK spike）
action process            约 7 ms
scene.update              约 7 ms
```

100 步稳定样本：

```text
env_step mean    68.08 ms
env_step p95     66.51 ms
sim.step mean     7.38 ms（每个物理 tick）
scene.write      11.55 ms（每个控制 tick）
scene.update      6.46 ms（每个控制 tick）
action apply      4.19 ms（每个物理 tick）
low-level policy  2.13 ms（每 4 个物理 tick）
IK apply          7.22 ms（每 4 个物理 tick）
```

关键修正：`action_manager.apply_action()` 必须每个 5 ms 物理 tick 调用，否则低层
策略/history 计数器会降到约 12.5 Hz，机械臂会明显抖动。Fast path 只降频
`scene.write_data_to_sim()`，不降频 action manager。

首个 episode step 仍包含约 0.5 秒的首次策略/IK 缓存 spike，之后进入稳定区间。

当前结果说明：

- 完整 manager 链不是唯一瓶颈，但跳过它仍有收益。
- 逐 5 ms 重复写目标明显浪费，20 ms 写一次更适合 50 Hz 高层控制。
- IK 是最值得继续做 A/B 的部分，首次/偶发 spike 仍会拉高 p95。
- 需要继续预热、增加样本数，并验证关闭 armature self-collision/contact reporting 的影响。

## 当前限制

- Fast path 当前只接入键盘和 VR 遥操，尚未接入 HDF5 数据采集。
- Fast path 不执行训练 reward/termination/event manager，手动 reset 仍走完整环境。
- IK 周期和每 tick/每控制 tick 写目标仍需做动作一致性验收。
- 30 步 profiling 仍受首次策略和 IK 编译/缓存影响，后续应使用更长 episode 和 warmup。
