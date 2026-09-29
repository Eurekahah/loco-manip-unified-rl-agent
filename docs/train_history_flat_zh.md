# 训练说明：平地 + 历史观测（History-Adaptation）低层模型

分支：`codex/ll-history-flat-eegoal`（基于 `main`，只改 3 个文件 + 1 个工具脚本）

> **2026-09-29 更新**：本文件现在也覆盖分支 `codex/ll-train-detail-fix` 带来的
> 「静止伫立 / 镜像符号 / 扰动加强 / 多地形 / 遥操 history」五项改动，
> 见下面新增的 **「2026-09-29 训练细节专项」** 一节；每条的来龙去脉在
> `docs/review/DEFECT_LOG_zh.md` DEF-026 ~ DEF-031。

---

## 2026-09-29 训练细节专项（分支 `codex/ll-train-detail-fix`）

### 1) 奖励项与课程项的变化（对比部署基线）

| 项 | 基线（`2026-09-20_00-50-31`） | 现在 | 作用 |
|---|---|---|---|
| `stand_still_vel` | 无 | **−0.8 → −8.0**（25k 步爬升） | 零线速命令时惩罚底盘残余 xy 线速度 + yaw 角速度 |
| `stand_still_wheel_vel` | 无 | **−0.001 → −0.01**（25k 步爬升） | 零线速命令时惩罚轮关节转速（"轮子空转"） |
| `joint_mirror` | −0.03，2 对（对角，**符号错误**） | **−0.06，4 对**（对角 + 左右，带符号约定） | 镜像对称惩罚，压"右后腿往右前方撇" |
| `commands.base_velocity.rel_standing_envs` | 0.02 | **0.02 → 0.15**（25k 步爬升） | 有多少比例的 env 拿到"零速命令"（站着不动的训练信号占比） |
| `events.randomize_push_robot` | 每 10~15 s，±0.5/±0.5 m/s | **每 5~10 s，x±2 / y±1 m/s / yaw±0.52 rad/s** | 扰动强度与频率 |
| `disturbance_ramp` | 0.3× → 1.0×，25k 步 | **0.2× → 1.0×，50k 步** | 扰动课程（新幅度更大，所以起点更低、拉得更长） |

两条新惩罚**只按命令门控**（不看实测速度）——旧配置里唯一沾边的
`joint_pos_penalty_wbc` 要求 `body_vel < 0.5` 才生效，也就是说"一旦真的漂起来这项就自己关了"。
权重按回报口径标定：IsaacLab 的 `Episode_Reward/*` 记的是**每秒速率**，基线在命令 (0,0,0) 时
约 1.75/s；`−8.0 × |v|²` 在 |v|=0.15 时给出约 −0.18/s（≈10%），有梯度但不喧宾夺主（推导见 DEF-026 §3）。

### 2) 任务清单（2026-09-29 新增两个）

```bash
# 平地 history（本轮主线，静止伫立/镜像/扰动的验收任务）
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task History-Adaptation-Deeprobotics-M20-v0 --headless --num_envs 4096

# 多地形：随机粗糙（噪声 0.01~0.05）+ 正/反斜坡 + 平地
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Rough-Slopes-History-Adaptation-Deeprobotics-M20-v0 --headless --num_envs 4096

# 高层：遥操 + 历史自适应低层（默认指向 history 部署态策略）
RL_TRAINING_LOW_LEVEL_POLICY_TELEOP_HISTORY=<run>/exported_deploy/policy.pt \
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-M20-Piper-Teleop-History-v0 --headless --num_envs 64
```

地形组成（`mdp.terrains.ROUGH_SLOPES_FLAT_TERRAINS_CFG`）：
`random_rough` 0.40（`noise_range=(0.01, 0.05)`、`noise_step=0.01`）、
`hf_pyramid_slope` 0.25、`hf_pyramid_slope_inv` 0.25、`flat` 0.10；**不含楼梯/boxes/rails/pit**。
与两个已有的多地形任务的关系：`Rough-History-*` 用官方 `ROUGH_TERRAINS_CFG`（含楼梯等，噪声 0.02~0.10）；
`Rough-WO-Stairs-History-*` 用 `NONE_STAIRS_TERRAINS_CFG`（无楼梯，噪声仍是 0.02~0.10）。

> ⚠️ **本机（Windows + RTX A4000）跑不了任何生成地形的任务** —— env 创建期死锁，
> 原始代码同样复现，见 `docs/review/DEFECT_LOG_zh.md` DEF-031。
> 上面第二条命令请到能跑生成地形的机器上执行。

### 3) 结论怎么验（静止伫立）

训练日志里的 `Train/mean_reward` 是**随机命令课程**上的均值，跨 run 不可比（命令范围随课程放宽）。
要比较"站得住不住"必须用固定命令探针：

```bash
python scripts/reinforcement_learning/rsl_rl/eval_fixed_command.py \
    --headless --num_envs 512 --steps 1100 --seed 42 \
    --commands "0,0,0;0.5,0,0;1.0,0,0" \
    --checkpoint <run>/model_2000.pt --out logs/smoke/eval_<run>.json
```

看 `(0,0,0)` 那一行的 `err_vel_xy`（= 命令为零时的实际速度，基线是 **0.148 m/s**）与摔倒率。

---

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
  从 0.001 拉到 0.8 —— 属于另一条独立线索，见 `docs/review/TODO_zh.md` P1-1（训练稳定性）。

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

**这一步是必做项**（部署 / sim2sim / 高层 replay 都只认部署态策略）。

```bash
# 导出成高层 replay / 部署能直接加载的 TorchScript + ONNX（纯 torch，不用起仿真）
# 训练结束时 train.py 会把这条命令连同 run 目录一起打印出来，可直接复制。
python scripts/reinforcement_learning/rsl_rl/export_deploy_policy.py \
    --run logs/rsl_rl/history_adaptation/<你的时间戳> --checkpoint model_19999.pt
```

默认写到 **`<run>/exported_deploy/{policy.pt,policy.onnx,policy_layout.json}`**
（`kind=history, policy_obs_dim=83, history_single_step_dim=70, history_length=10, action_dim=16`）。
导出时会自检：scripted ↔ eager、scripted ↔ `ActorCriticHistory.act_inference`（都应为 0），
以及 ONNX ↔ TorchScript 的相对误差（fp32 舍入量级，~1e-07）。

ONNX 的接口与 `policy.pt` 一致（`--no-onnx` 可跳过、`--opset` 改 opset，默认 17）：

```
输入: policy_obs   (batch, 83)      输入: history_flat (batch, 700)  = 10 步 x 70 维，最旧→最新
输出: action       (batch, 16)
```

batch 维是动态的（部署时通常是 1）。`history_flat` 需要**调用方自己维护环形缓冲**：
每步 70 维 = `base_ang_vel(3) + projected_gravity(3) + joint_pos(24) + joint_vel(24) + last_action(16)`，
reset 后第一次推进用整窗填满同一帧（详见 `policy_layout.json` 的 `history_note`）。

⚠️ 两个必须区分的目录：

- `<run>/exported_deploy/policy.pt` —— 本脚本出的**部署态**策略（含 history encoder，双输入）；
- `<run>/exported/policy.pt` —— `play.py` 出的 **actor-only** 策略（输入 115 = 83 + 32 latent，
  latent 没有来源）。**拿去 sim2sim 是错的**，脚本会在这种情况下打印警告。

高层 replay 侧现在**已支持** history 策略（10 步窗口回放已并入 `main`）：
把低层 checkpoint 指过去即可，replay 按 `policy_layout.json` 自动走双输入 forward，例如

```bash
RL_TRAINING_LOW_LEVEL_POLICY_WBC=<run>/exported_deploy/policy.pt   # 见 TODO_zh.md P0-1 的 ⑦
```

## 其它注意

- 本分支只动低层训练侧；`bad_orientation_2` 是所有低层 env（flat/rough/WBC/History）共用的终止项，
  阈值放大对它们都生效。
- 保留 `ee_goal` 使低层 policy 观测 **76 → 83**，之前不含 ee_goal 训出的 checkpoint
  （含 `logs/rsl_rl/deeprobotics_m20_wbc_flat/2026-09-18_01-31-58`、以及
  `history_adaptation/2026-09-18_19-33-47`）都不能再用，必须重训。
- 训练日志在 `logs/rsl_rl/history_adaptation/<timestamp>/`（`logs/` 已被 .gitignore 忽略，不会进 git）。
- 续训：`--resume --load_run <run> --checkpoint <absolute/path/model_xxxx.pt>`。
