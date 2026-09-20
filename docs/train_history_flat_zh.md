# 训练说明：平地 + 历史观测（History-Adaptation）低层模型

分支：`codex/ll-history-flat-eegoal`（基于 `main`，只改 3 个文件 + 1 个工具脚本）

## 这一版包含什么

| 改动 | 文件 | 说明 |
|---|---|---|
| `bad_orientation_2` 阈值放大 | `velocity/mdp/events.py`、`velocity_env_cfg.py` | 旧实现 `(g_z>0) \| (\|g_xy\|>0.5)` 等价于"绕单轴倾斜约 30°"，且边界是方形（沿 x/y 30°、沿对角 45°）。现改为旋转不变的总倾角 `acos(-g_z) > limit_angle`，默认 **0.8 rad ≈ 45.8°**（与仓库里另一处 `bad_orientation(limit_angle=0.8)` 一致），参数在 `DoneTerm` 里显式给出，便于再调 |
| 保留 `ee_goal` 观测 | `flat_env_wbc_cfg.py` | `FlatEnvWBCConfig` 不再把 policy/critic 的 `ee_goal` 置 None |
| 导出工具（训练后要用） | `scripts/reinforcement_learning/rsl_rl/export_deploy_policy.py` | 纯 torch，不起 Isaac Sim；普通 `ActorCritic` 与带 history encoder 的 `ActorCriticHistory` 都支持 |

### 为什么放大倾角阈值

对 `logs/rsl_rl/history_adaptation/2026-09-18_19-33-47`（7555 点）的实测：

- 终止构成：**`bad_orientation_2` 62.7%** + `time_out` 35.3% + `root_height_below_minimum` 2.0%。
  早期（500 iter）`bad_orientation_2` 高达 **96.6%**，episode 平均只有 184 步。
- 命令空间：`body_pose` 的 pitch ±0.35 rad(20°)、roll ±0.25 rad(14°)；
  两者叠加的名义倾角已达 **0.427 rad(24.5°)**，距旧阈值 30° 只剩 5.5°。
- 而实测跟踪误差 `Metrics/body_pose/pitch_error_mean / roll_error_mean` 本身就有
  **0.13~0.38 rad(7.5~21.6°)** —— 比这 5.5° 余量还大。
  ⇒ **正常跟踪误差会被判成"摔倒"**，这就是终止率居高不下的直接原因。
- 另外该 run 在 4000 iter 之后整体退步（reward 15.85→11.26、`error_vel_xy` 0.479→0.605、
  `mean_noise_std` 1.45→1.53），课程把 body pose 奖励权重在 2000~3000 iter 内
  从 0.001 拉到 0.8 —— 属于另一条独立线索，见 `docs/review/known_issues.md #19`
  （该文件在 `codex/hl-replay-l2` 分支上，不在本分支）。

0.8 rad 的取舍：比旧的 30° 明显放宽（轴向上 30°→45.8°），能容纳"名义倾角 + 典型误差"，
但仍然拦得住真摔（侧躺 90°、翻倒 180°），并且与仓库另一处终止项一致。
若仍偏紧可改 `velocity_env_cfg.py` 里的 `limit_angle`（1.0 rad ≈ 57.3° 更宽松）。

## 在新机器上训练

```bash
# 1) 取代码（**必须带 submodule**：机器人 USD 资产来自 deep_robotics_model）
git clone --recurse-submodules -b codex/ll-history-flat-eegoal \
    https://github.com/Eurekahah/loco-manip-unified-rl-agent.git
cd loco-manip-unified-rl-agent
# 若是已有仓库：git fetch && git checkout codex/ll-history-flat-eegoal && git submodule update --init --recursive

# 2) 装包（在装了 Isaac Lab 的 python 环境里）
python -m pip install -e source/rl_training

# 3) 训练（max_iterations 已在 runner cfg 里设为 20000；num_envs 默认 4096）
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task History-Adaptation-Deeprobotics-M20-v0 \
    --headless --num_envs 4096
```

### 启动自检（打印出来的维度必须与此一致）

```
policy      obs = 83   (base_ang_vel 3 + projected_gravity 3 + velocity_commands 3
                        + joint_pos 24 + joint_vel 24 + actions 16
                        + ee_goal 7 + body_pose_cmd 3)
critic      obs = 86
history     obs = 700  (10 步 × 70)
privileged  obs = 89
```

`actions` 是 16 维（12 腿 + 4 轮；IK 由 CommandManager 驱动，不占动作维度）。
checkpoint 里 `actor.0.weight` 应为 `(512, 115)`：115 = 83(policy obs) + 32(latent)。
history 的 70 维 = `base_ang_vel(3) + projected_gravity(3) + joint_pos(24) + joint_vel(24) + last_action(16)`。

> `env.yaml` 里应能看到 `terminations.bad_orientation_2.params.limit_angle: 0.8`，
> 以及 policy/critic 的 `ee_goal` 不为 null。这两条是最快的"配置是否生效"检查。

## 训练完成后

```bash
# 导出成高层 replay 能直接加载的 TorchScript（纯 torch，不用起仿真）
python scripts/reinforcement_learning/rsl_rl/export_deploy_policy.py \
    --run logs/rsl_rl/history_adaptation/<你的时间戳> --checkpoint model_19999.pt
```

会写出 `<run>/exported/policy.pt` 与 `policy_layout.json`
（`kind=history, policy_obs_dim=83, history_single_step_dim=70, history_length=10, action_dim=16`）。
导出时脚本会自检 scripted 与 `ActorCriticHistory.act_inference` 的数值一致性（应为 0）。

⚠️ 注意：**高层 replay 目前还不支持 history 策略**（缺 10 步窗口回放）。
要把它接到高层，需要先实现 `docs/review/history_low_level_policy_todo.md` 里
那三件事（该文件在 `codex/hl-replay-l2` 分支上）。在此之前，`policy_layout.json`
是给 replay 侧做维度校验用的，replay 遇到 `kind=history` 会明确报错并指向那份待办。

## 其它注意

- 本分支只动低层训练侧；`bad_orientation_2` 是所有低层 env（flat/rough/WBC/History）共用的终止项，
  阈值放大对它们都生效。
- 保留 `ee_goal` 使低层 policy 观测 **76 → 83**，之前不含 ee_goal 训出的 checkpoint
  （含 `logs/rsl_rl/deeprobotics_m20_wbc_flat/2026-09-18_01-31-58`、以及
  `history_adaptation/2026-09-18_19-33-47`）都不能再用，必须重训。
- 训练日志在 `logs/rsl_rl/history_adaptation/<timestamp>/`（`logs/` 已被 .gitignore 忽略，不会进 git）。
- 续训：`--resume --load_run <run> --checkpoint <absolute/path/model_xxxx.pt>`。
