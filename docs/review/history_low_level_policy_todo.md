# 待办：让高层 replay 支持「带 history encoder」的低层策略

状态：**✅ 已实现（2026-09-19，分支 `codex/hl-replay-history`，commit 见文末）**。
基分支：`codex/hl-replay-l2`（commit `9d52fac`）+ `codex/export-deploy-policy`（`e565065`）。

---

## 1. 背景：为什么现在用不了

目标模型：`logs/rsl_rl/history_adaptation/2026-09-18_19-33-47`（`model_7500.pt`，
训练环境 `History-Adaptation-Deeprobotics-M20-v0`，策略类 `ActorCriticHistory`）。

问题的根源不在 checkpoint，而在 **`play.py` 的导出方式**：
`export_policy_as_jit(policy_nn.actor, ...)` 只导出 actor，而

```python
# rsl_rl/rsl_rl/modules/actor_critic_history.py
self.actor = MLP(num_actor_obs + latent_dim, num_actions, ...)   # 注意 + latent_dim
```

latent 由**另一个模块** `history_encoder` 从 10 步本体感受历史算出来。
实测该 checkpoint：`actor.0.weight = (512, 108)`，而 policy obs 只有 76 维
（108 = 76 + 32 latent）；state_dict 里另有 `history_encoder.*`（CNN）与
`privileged_encoder.*`。所以 play.py 导出的 `policy.pt` 是个"要 108 维输入、
但那 32 维没人给"的模型——数值上还能跑，但**不是训练出来的那个策略**。

## 2. 已完成：部署态导出（`codex/export-deploy-policy`, `e565065`）

`scripts/reinforcement_learning/rsl_rl/export_deploy_policy.py`（纯 torch，不用起 Isaac Sim）
导出的包装等价于 `ActorCriticHistory.act_inference`：

```
forward(policy_obs, history_flat) -> action
    latent = history_encoder(history_flat)
    return actor(cat([policy_obs, latent], -1))
```

同时写 `policy_layout.json`，把"这个 checkpoint 要什么输入"固定下来：

```json
{"kind": "history", "policy_obs_dim": 76, "history_single_step_dim": 70,
 "history_length": 10, "latent_dim": 32, "action_dim": 16, ...}
```

已验证（`--run logs/rsl_rl/history_adaptation/2026-09-18_19-33-47 --checkpoint model_7500.pt`）：

| 检查 | 结果 |
|---|---|
| scripted vs eager | 0.000e+00 |
| 与 rsl_rl `ActorCriticHistory.act_inference` | **0.000e+00** |
| 产物 | `<run>/exported/{policy.pt, policy_layout.json}` |

关键数字：**history 单步 70 维** = `base_ang_vel(3) + projected_gravity(3) +
joint_pos(24) + joint_vel(24) + last_action(16)`，窗口 10 步 → 700 维。

## 3. 还差什么（下一步要做的）

> ### ✅ 三件事都已完成（`codex/hl-replay-history`）
>
> 代码位置：`highlevel/mdp/low_level_replay.py` 新增
> `history_single_step_ll()` / `run_low_level_policy()` / `LowLevelReplayState` /
> `build_history_window()`；三个 action term（`pre_trained_pick_action` /
> `pre_trained_pick_wbc_action` / `teleop_ll_action`）在 `__init__` 里各建一个
> `self._ll_replay_state`，在 `apply_actions()` 的低层 tick 里先
> `history_flat = self._ll_replay_state.on_tick()`，再
> `policy_output = run_low_level_policy(self.policy, low_level_obs, history_flat)`。
>
> 3.1 **环形缓冲**：直接复用 IsaacLab 的 `CircularBuffer`（就是训练侧
> `ObservationManager` 内部用的那个类），因此"首次 push 填满整窗 / reset 清零 /
> 展平成 `[t_oldest(D) ... t_newest(D)]`"这些语义与训练**同源**，不是重新实现一遍。
> `last_action` 取回放自己缓存的**低层 16 维动作**（`[leg|wheel|ee_ik]`）。
>
> 3.2 **按 layout json 决定单/双输入**：`expected_policy_obs_dim()` 里那个
> `NotImplementedError` 已删除；`run_low_level_policy()` 按 `history_flat is None`
> 分支调用 `policy(obs)` 或 `policy(obs, history_flat)`。
> `checkpoint_dims()` 仍然只用于**动作维度**校验，观测维度一律以 `policy_layout.json`
> 的 `policy_obs_dim` 为准（启动日志里会显式打印"actor 输入 115 = 83 + 32 latent，
> 不能用 checkpoint_dims 读观测维度"）。
>
> 3.3 **ee_goal 取舍**：不需要额外开关 —— 用"保留 ee_goal"的 history checkpoint
> （`policy_obs_dim=83`）时，`build_low_level_obs_manager()` 直接得到 83 维；
> 用旧的不含 ee_goal 的 checkpoint 时它会自动去掉那 7 维（仍保持严格校验）。
>
> ⚠️ 一个顺带修掉的坑（原实现的 `last_action()` 闭包）：闭包里用
> `episode_length_buf == 0` 来判断"复位"并清零低层动作缓存 —— 这个条件在复位后的
> **整个 env step** 里都成立（高层 env 一步有 10 个低层 tick），于是整步内每个 tick
> 都把上一帧动作清 0，与 history 帧里的 `last_action` 自相矛盾。现在改成
> "`episode_length_buf` 相比上一次低层 tick **变小**"的跳变检测，只在复位那一 tick
> 清一次（`LowLevelReplayState.on_tick()`）。

回放侧（`highlevel/mdp/low_level_replay.py` + 三个 action term）需要三件事：

### 3.1 维护 history 环形缓冲

- 缓冲：`(num_envs, history_length, 70)`，每个低层 step 推入一帧。
- 展平顺序必须与低层训练一致：`(T, D)` → `flatten` ——
  `HistoryEncoder.forward` 用的是
  `history_obs.view(B, history_length, num_single_step_obs).transpose(1, 2)`，
  即**先时间步、后通道**（`[t0 全部维度, t1 全部维度, ...]`）。
- **坑（最容易错）**：单步向量里的 `last_action` 在低层训练时是
  `env.action_manager.action`（**低层 16 维**动作）。回放里必须用自己缓存的
  低层动作（`[low_level_leg_actions | low_level_wheel_actions | low_level_ee_actions]`），
  不能用高层 env 的 `action_manager.action`（那是 12+2 或 11+2 维）。
  低层那个 `mdp.history_single_step_obs` 直接调 `base_mdp.last_action(env)`，
  在高层 env 里语义是错的，所以需要自己写一个等价的单步函数
  （照抄 `velocity/mdp/observations.py::history_single_step_obs` 前四项 +
  传入回放自己的 last_action）。
- **reset**：episode reset 时把该 env 的行清零（否则跨 episode 的历史会污染）。
  在 `apply_actions()` 里已有的 `reset_ids` 分支里做即可。

### 3.2 按 layout json 决定"单输入还是双输入"

- `low_level_replay.py` 已有 `read_policy_layout()` / `expected_policy_obs_dim()`，
  并且**已经会对 `kind == "history"` 抛 `NotImplementedError`**（带指向本文件的提示）。
  实现后把那个 `raise` 改成正常分支即可。
- 调用处（三个 action term 的 `apply_actions`）目前是
  `policy_output = self.policy(low_level_obs)`；history 要改成
  `self.policy(policy_obs, history_flat)`。
- **另一个坑**：`checkpoint_dims()` 用 `actor.0.weight` 读观测维度，对 history
  策略会得到 108（含 latent），所以**必须**以 `policy_layout.json` 的
  `policy_obs_dim` 为准（现在已经是这个优先级了，保留即可）。

### 3.3 ee_goal 是否参与观测

该模型的 `policy_obs_dim = 76`，即**不含 ee_goal**（训练时
`FlatEnvWBCConfig` 把 `policy/critic.ee_goal` 置了 `None`）。
回放侧已经支持按维度自动取舍：`build_low_level_obs_manager()` 会先按模板建观测组，
若比 checkpoint 多出正好一个 `ee_goal` 的宽度就去掉它重建，否则直接报错。
所以 history 支持落地后，ee_goal 的有无**不需要额外开关**。

> 注意：用户已决定**下一次训练要带 ee_goal**（分支 `codex/ll-keep-ee-goal`
> 已把 `FlatEnvWBCConfig` 改回保留）。届时新 checkpoint 的
> `policy_layout.json` 里 `policy_obs_dim` 会是 83，回放会自动把 ee_goal 加回来。

## 4. 建议的实现顺序与验收

> ### ✅ 验收实测（2026-09-19）
>
> 先导出部署态策略（**用 `codex/export-deploy-policy` 上的脚本**，本分支没有它，
> 所以那次是用 `git show 4276970:.../export_deploy_policy.py` 取出来跑的：
> `--run logs/rsl_rl/history_adaptation/2026-09-19_09-02-50 --checkpoint model_5500.pt
> --out_dir <run>/exported_deploy`）：
>
> ```json
> {"kind": "history", "policy_obs_dim": 83, "history_single_step_dim": 70,
>  "history_length": 10, "latent_dim": 32, "action_dim": 16}
> ```
> 自检：与 `ActorCriticHistory.act_inference` 的最大误差 `0.000e+00`。
>
> 新增探针 `scripts/reinforcement_learning/rsl_rl/probe_history_window.py`
> （`Isaac-M20-Piper-Teleop-v0`，8 envs，40 步，历史窗口逐 tick 复算）：
>
> | 检查 | 实测 |
> |---|---|
> | replay 建出的低层 obs / 窗口 | obs **83** ✓、history **10 × 70 = 700** ✓、action **16** ✓（全部与 layout json 一致） |
> | **单步内容**：窗口最后一帧 vs 用**低层训练函数** `mdp.history_single_step_obs` 独立复算（临时把 `action_manager._action` 换成低层 16 维动作） | `max|差| = 0.000e+00` |
> | **整窗顺序**：40 次 tick 的 700 维窗口 vs "最近 k 帧按 `[t_oldest..t_newest]` 拼接 + 复位填充" | `max|差| = 0.000e+00`（其中 72 个 env·tick 带复位填充） |
> | `CircularBuffer` 语义（顺序 + reset 后填满整窗） | 与训练侧同源，直测得 `[2,3,4]`、reset 后 `[9,9,9]` ✓ |
> | `last_action` 段宽度 | **16**（低层动作）；对照：高层 `action_manager.total_action_dim = 13` ⇒ 拿错宽度都不对 |
>
> 训练冒烟（都是 `--headless --num_envs 64 --max_iterations 2`）：
>
> | 任务 | 低层 checkpoint | 结果 |
> |---|---|---|
> | `Isaac-M20-Piper-Teleop-v0` | **history**（83/700/16） | **EXIT=0**，reward 0.11 → 0.15，启动打印 `history 窗口: 10 × 70 = 700` |
> | `...Pick-WBC-Flat-Teacher-v0` + `policy_path=<history 策略>` | **history** | **EXIT=0**，reward 0.92 → 1.28 |
> | `...Pick-WBC-Flat-Teacher-v0`（默认 76 维普通 ActorCritic） | actor | **EXIT=0**，obs 76 = 76，reward 0.88 → 1.26（回归） |
> | `...Pick-Flat-Teacher-v0`（L1，`ee_action_dim=7`） | actor（23 维动作） | **EXIT=0**，reward 0.81 → 1.11（回归） |

1. `history_single_step_yy(env, asset_cfg, last_action_fn)` 放进 `low_level_replay.py`
   （镜像低层函数，明确注释"差异仅在于 last_action 的来源"）。
2. 在 action term 基类/三个 term 里加：`self._history_buf`、`_push_history()`、
   `_reset_history(reset_ids)`，并按 `layout.kind == "history"` 分支调用策略。
3. 验收（都要贴实测数据）：
   - `policy_layout.json` 的 `policy_obs_dim` / `history_*` 与在建 env 时打印的一致；
   - 观测维度断言通过（`build_low_level_obs_manager` 不再 raise）；
   - `train.py --task Isaac-M20-Piper-Teleop-v0 --headless --num_envs 64 --max_iterations 2` **exit 0**；
   - 建议再加一条**数值回归**：把低层训练时同一时刻的 obs（用一个短脚本从
     `history_adaptation` 的 env 里 dump）和回放构造的 obs 逐项对比，误差应为 0
     —— 这是唯一能证明"10 步窗口真的对齐"的方法。
4. 参考命令（本次已验证可用）：

```
python scripts/reinforcement_learning/rsl_rl/export_deploy_policy.py \
    --run logs/rsl_rl/history_adaptation/2026-09-18_19-33-47 --checkpoint model_7500.pt
```

## 5. 临时替代（现在就能用）

用不带 history encoder 的普通 `ActorCritic` 低层策略，回放侧已完整支持：

```
logs/rsl_rl/deeprobotics_m20_wbc_flat/2026-09-18_01-31-58/exported/policy.pt
（model_6300.pt 导出，obs 76 / action 16，不含 ee_goal）
```

实测：`Isaac-M20-Piper-Teleop-v0` 与
`Isaac-Deeprobotics-High-Level-Pick-WBC-Flat-Teacher-v0`
都能 `--num_envs 64 --max_iterations 2` 跑通（exit 0）。
