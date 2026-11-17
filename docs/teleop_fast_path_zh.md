# Fast Teleop / Fast 数采与热路径 profiling

训练环境继续使用完整的 `ManagerBasedRLEnv.step()`。遥操新增 opt-in 的
`--fast-teleop`，数据采集新增 opt-in 的 `--fast-stepping`。两者复用现有 robot、
scene、action manager、低层策略和 history，但跳过完整
reward/observation/termination/event 管理链。

## 新增参数

```text
--fast-teleop / --no-fast-teleop
--fast-render-hz                 墙钟渲染频率，默认 15 Hz
--fast-write-each-tick           默认关闭；关闭时每 20 ms 控制 tick 写一次目标
--fast-robot-hot-loop            默认开启；机械臂/robot target 以 hot period 更新
--fast-robot-hot-period          默认 10 ms
--fast-robot-only-write          实验开关；只写 robot，不写其它 scene 实体
--ik-update-period               IK 重算周期（仿真秒），默认 0 表示每个物理 tick
--profile-detailed               拆分 action/policy/IK/scene/sim/render 计时
```

## Fast path 结构

```text
读取最新遥操命令
  -> action_manager.process_action
  -> action_manager.apply_action          # 每 20 ms 控制 tick
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
  --profile auto --input keyboard --headless `
  --no-top_camera --no-scene_camera --wrist_camera `
  --record-hz 10 --compression lzf `
  --fast-stepping --fast-render-hz 15 `
  --output recordings\fast_teleop\two_cylinder_fast.hdf5 `
  --profile-json outputs\fast_teleop\fast_profile.json
```

Fast 模式自动 termination 关闭，结束时使用键盘 `K/L` 或 VR success/failure
手动标记。HDF5 仍通过异步 writer 写入，控制线程只负责采样并入队。

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
- IK 是最值得继续做 A/B 的部分，首次/偶发 spike 仍会拉高 p95。
- 需要继续预热、增加样本数，并验证关闭 armature self-collision/contact reporting 的影响。

## 当前限制

- Fast path 已接入键盘、VR 遥操和通用 HDF5 数采；相机采样仍在主线程，异步 writer
  只负责磁盘写入。
- Fast path 不执行训练 reward/termination/event manager，手动 reset 仍走完整环境。
- IK 周期和每 tick/每控制 tick 写目标仍需做动作一致性验收。
- 30/50 步 profiling 仍受首次策略和 IK 编译/缓存影响，后续应使用更长 episode 和 warmup。
