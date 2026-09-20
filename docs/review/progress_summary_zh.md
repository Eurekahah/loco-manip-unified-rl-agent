# 工作进度汇总（截至 2026-09-19）

`main` 基线：`a855e8f`。**所有修复都在分支上，未合并 main。**
`docs/review/*` 只存在于这些分支上（main 上还没有）。

---

## 一、已提交的分支

| 分支 | 代码 commit | 内容 | 验证 |
|---|---|---|---|
| `codex/hl-replay-layout` | `dc45d0e`（文档 `0f96511`） | 新增 `low_level_replay.py`：关节分组/布局/观测组装/启动期校验的单一来源；修 ④⑤⑥⑨⑯ + 清单外 A/B | 3 个高层任务 train 2 iter exit 0；低层无回归 |
| `codex/hl-replay-l2` | `1f56b6e` | L2 布局推导（`ee_action_dim=-1` 默认从低层 cfg 推导）；按 `policy_layout.json` 匹配观测维度（`ee_goal` 自动取舍）；history 类 checkpoint 明确报错 | teleop obs 76=76、pick-WBC 83=83 等 |
| `codex/ll-keep-ee-goal` | `1253bba` | WBC 低层恢复保留 `ee_goal` 观测（76 → 83） | — |
| `codex/export-deploy-policy` | `4276970` | `export_deploy_policy.py`：把带 history encoder 的 ROA 策略导出成部署态 `forward(policy_obs, history_flat)`；纯 torch；与 rsl_rl `act_inference` 数值一致（0.000e+00） | 两个模型导出通过 |
| `codex/hl-replay-history` | （见文末 commit） | **任务 2**：高层 replay 支持带 history encoder 的低层策略 —— 10 步窗口（复用 IsaacLab `CircularBuffer`）+ 按 `policy_layout.json` 单/双输入调用 + history 里的 `last_action` 用低层 16 维动作 + 复位检测修成"跳变检测" | 见下 |
| `codex/hl-fix-ll-command` | `e064bc6`（文档 `b32c515`） | **①** `PreTrainedPickAction` 补 `ll_command`/`ll_command_w`；`ll_command_world()` helper；8 处世界系奖励项改用它 | 见下 |
| `codex/hl-fix-ee-command` | `7a22759`（文档 `749ff83`） | **②** IK 目标写 `pose_command_b` + 同步 `pose_start_b/pose_end_b`；**③** flat 的 `ee_goal` 改用 root 系 | 见下 |
| `codex/ll-history-flat-eegoal` | `45f9e74` **（已推 GitHub）** | 训练用配置：`bad_orientation_2` 改成旋转不变 0.8 rad(45.8°) + 保留 `ee_goal` + 训练说明 `docs/train_history_flat_zh.md` + 导出脚本 | History-Adaptation 2 iter exit 0，policy 83 / history 700 |

## 二、关键实测数据

### 低层观测/动作布局（清单外新增 A/B）

| 任务 | 修前 | 修后 | checkpoint 期望 |
|---|---|---|---|
| `...Pick-Flat-Teacher-v0` | 76（actions 16） | **83（actions 23）** | 83 ✓ |
| `...Pick-WBC-Flat-Teacher-v0` | 79 | **86** | 86 ✓ |
| `Isaac-M20-Piper-Teleop-v0` | 79 | **86** | 86 ✓ |

修前第一次 `env.step()` 就 `RuntimeError: mat1 and mat2 shapes cannot be multiplied (Nx76 and 83x512)`。

### 轮关节掩码（清单外新增 B）

```
wheel 原生 id = [15,16,17,18]；正确列下标 = [12,13,14,15]
旧写法被清零 = ['hr_wheel_joint','arm_joint1','arm_joint2','arm_joint3']  ← 错
新写法被清零 = ['fl_wheel_joint','fr_wheel_joint','hl_wheel_joint','hr_wheel_joint'] ← 对
```

### ① `ll_command`

* `...Pick-Flat-Teacher-v0` train 2 iter **exit 0**（修前第一次 step 抛
  `AttributeError: 'PreTrainedPickAction' object has no attribute 'll_command'`），reward 0.78→0.88。
* 坐标系与独立复算：`|root − 独立复算| pos=0.000e+00 quat=0.000e+00`；
  对照 `|world − root| = 8.90`（证明两套坐标系确实不同）。
* `ll_command_w` 与本 term 内部世界系目标 `0.000e+00`。

### ②③ IK 目标 + `ee_goal` 坐标系

| 任务 | `\|pose_command_b − 高层root目标\|` | 低层 obs 的 `ee_goal` 槽位 vs root / vs world | EE 距高层目标 |
|---|---|---|---|
| `...Pick-Flat-Teacher-v0` | 0.000e+00 / 0.000e+00 | **0.000e+00** / 8.9093 | 0.2575 m |
| `...Pick-WBC-Flat-Teacher-v0` | 0.000e+00 / 0.000e+00 | （该 checkpoint 不含 ee_goal） | 0.0429 m |
| `Isaac-M20-Piper-Teleop-v0` | 0.000e+00 / 0.000e+00 | — | 0.0057 m |

训练回归（三任务 `--num_envs 64 --max_iterations 2` 全 **EXIT=0**）：
flat pick 0.83→1.14、WBC pick 0.88→1.28、teleop 0.12→0.18。

### 低层训练用（`codex/ll-history-flat-eegoal`）

`History-Adaptation-Deeprobotics-M20-v0 --max_iterations 2` → exit 0；
`policy obs 83 / critic 86 / history 700 / privileged 89`；
checkpoint `actor.0.weight (512,115)=83+32latent`、`history_encoder.conv.0 (32,70,4)`；
`env.yaml` 里 `limit_angle: 0.8` 与 `ee_goal` 非 null 均生效。

### history 回放（`codex/hl-replay-history`）

`probe_history_window.py`（`Isaac-M20-Piper-Teleop-v0`，8 envs，40 步）：

```
policy_layout.json = {kind: history, policy_obs_dim: 83, history_single_step_dim: 70,
                      history_length: 10, latent_dim: 32, action_dim: 16}
低层布局: action_dim=16 leg=12 wheel=4 ee_ik=0 policy_joint_names=24
history 窗口: length=10 single_step=70 flat=700
高层 action_manager.total_action_dim = 13（不能拿来当 last_action）

(2) 窗口最后一帧 vs 用低层训练函数 mdp.history_single_step_obs 独立复算：max|差| = 0.000e+00
(3) 整窗顺序（最近 k 帧 + 复位填充，40 次 tick）：max|差| = 0.000e+00
(4) CircularBuffer 语义（顺序 / reset 填满整窗）：OK
```

训练冒烟（`--num_envs 64 --max_iterations 2`，全部 EXIT=0）：

| 任务 | 低层 checkpoint | 结果 |
|---|---|---|
| `Isaac-M20-Piper-Teleop-v0` | history（83/700/16） | reward 0.11 → 0.15 |
| `...Pick-WBC-Flat-Teacher-v0` + `policy_path=<history>` | history | reward 0.92 → 1.28 |
| `...Pick-WBC-Flat-Teacher-v0`（默认） | actor 76 维 | obs 76=76，reward 0.88 → 1.26（回归） |
| `...Pick-Flat-Teacher-v0`（L1） | actor 23 维动作 | reward 0.81 → 1.11（回归） |

## 三、还没做的（按建议优先级）

| 项 | 说明 | 预估 |
|---|---|---|
| **bad_orientation 课程**（详见 `bad_orientation_analysis_zh.md`） | EE 目标 s0→s3 课程 + 姿态 slerp +（可选）执行器刚度课程 | 先测默认半径，再改 `WBCCurriculumCfg`；验证 ~30 min/组 |
| **history 回放**（`history_low_level_policy_todo.md`） | ✅ 已完成（`codex/hl-replay-history`）：10 步窗口 + 单/双输入调用 + 低层 last_action；导出侧另在 `codex/export-deploy-policy` | 见上实测 |
| P1 ⑦ checkpoint 路径参数化 | 目前 `_LOW_LEVEL_WBC_POLICY` 是一处常量，改成环境变量/CLI | 小 |
| P1 ⑧ 模块级 `LOW_LEVEL_ENV_CFG` + `render_interval` 警告 | `high_level_env_cfg.py:30/456-458` | 小 |
| ⑤ 的 R1 步：抽 `LowLevelPolicyActionBase` | 把 nav/openvla/`pre_trained_policy` 也纳入；顺带修 nav 奖励项读不存在的 `ll_command` | 中 |
| 低层 known_issues ⑤⑥⑦ | `body_names=""` 占位符、`disable_zero_weight_rewards` 脆弱、`feet_distance_y_exp` 类型错误 | 小 |
| ⑯ 剩下的低层侧 | 低层 env 自己也加布局打印/断言；`joint_pos_rel_without_wheel` 补断言 | 小 |
| 清理 | `codex/docs-review`（两份清单的来源分支，未合并）；nav/openvla cfg 里指向不存在 checkpoint 的硬编码路径 | 小 |

## 四、环境与命令备忘

```bash
# 跑任何 Isaac 脚本都用这个 python；中文输出需要 UTF-8
C:\Users\autolab\miniconda3\envs\env_isaac_lab\python.exe
$env:PYTHONIOENCODING='utf-8'

# 训练 2 iter 冒烟（高层）
python scripts/reinforcement_learning/rsl_rl/train.py --task <task> --headless --num_envs 64 --max_iterations 2

# 导出部署态策略（纯 torch，不用起仿真）
python scripts/reinforcement_learning/rsl_rl/export_deploy_policy.py --run <run_dir> --checkpoint model_7500.pt
```

**踩坑备忘**（都踩过）：

* 沙箱里 `.git` 只读，`checkout/commit` 需要 escalate；`git` 要带 `-c safe.directory=...`。
* 同一进程里建第二个 Isaac env 会卡死；一个进程只建一个 env。
* `exit()`/`sys.exit()` 在 Isaac 脚本里会抛 SystemExit，probe 脚本里要用 `raise SystemExit(code)` 且注意清理。
* `ObsTerm.params` 会被原样透传给 `__call__`，自定义开关必须在 `__init__` 里 `pop` 掉。
* probe 里改过 `names` 这类局部变量会把前面的 `env.action_manager.active_terms` 覆盖掉，注意命名。
* 想"步进后"检查命令量，要用**步进前**的 root 位姿复算，否则会因为机器人在一帧内移动而出现 ~2e-2 的假差异。
