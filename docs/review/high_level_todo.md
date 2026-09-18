# high_level 侧待修改清单（只记录，未改代码）

基线：`main @ 905c2df`。行号会随代码漂移，请以符号名/函数名为准。
本文只覆盖 `source/rl_training/rl_training/tasks/manager_based/locomotion/highlevel/**`
（mdp 与 config），低层和工程性问题见同目录 `known_issues.md`。

---

## P0 — 会导致任务跑不起来，或训练信号本身就是错的

### 1. `PreTrainedPickAction` 没有 `ll_command`，普通 pick teacher 第一步就崩

- 读取处：`highlevel/mdp/rewards.py:1139` → `action_term.ll_command[:, 0]`
- 常开项：`HLFlatPickRewardsCfg.base_vel_cmd_action_l1_near_object`（`config/high_level/hl_flat_pick_env_cfg.py:260`，weight=1e-5，非 0 → 不会被 `disable_zero_weight_rewards` 清理）
- 只有 `PreTrainedPickWBCAction` / `TeleopLLAction` 定义了 `ll_command`
- 实测：`--task Isaac-Deeprobotics-High-Level-Pick-Flat-Teacher-v0` 在第一次 `env.step()`
  抛 `AttributeError: 'PreTrainedPickAction' object has no attribute 'll_command'`
- 可选修法：① 给 `PreTrainedPickAction` 补同义属性；② 该奖励内做 `hasattr` 判断后跳过；
  ③ 非 WBC 的 pick 配置里把这一项置 `None`

### 2. 高层给机械臂的目标写进了死字段，IK 根本收不到

- 写入处：`pre_trained_pick_action.py:327`、`pre_trained_pick_wbc_action.py:347`
  → `self._ee_command_term.pose_command_w[...] = ...`
- 但 IK 动作项读的是 `command_manager.get_command("ee_pose")` → `pose_command_b`
  （`velocity/mdp/actions.py::CommandDrivenIKAction.process_actions`）
- 并且 `HeightInvariantEECommand._update_command`（`velocity/mdp/commands.py:191`）
  覆盖父类时没有调用 `super()`，所以 `pose_command_w` **永远不会被更新**
  （只有父类的 `_debug_vis_callback` 会读它）
- 影响：高层的 EE 目标从未到达 IK；机械臂跟随的是 `ee_pose` 课程采样出来的随机位姿
- 正确写法参考：`teleop_ll_action.py:377` 写的是 `pose_command_b`
- 建议：统一写 `pose_command_b`（世界系 → root 系转换后写入）；或让 IK 改读
  `pose_command_w` 并补上它的更新逻辑

### 3. replay 喂给低层 policy 的 `ee_goal` 是**世界系**，训练时是 root 系

- 训练侧：`velocity/mdp/observations.py::ee_goal_local` 返回 `command_local` = `pose_command_b`（root 系）
- replay 侧把这些槽位覆盖成世界系量：
  `pre_trained_pick_action.py:131`、`pre_trained_pick_wbc_action.py:141/288`、
  `openvla_pick_action.py:129`、`teleop_ll_action.py:129`
  （`_raw_actions[:, 3:10]` / `_ll_command[:, 3:10]` 里存的是 `target_pos_w` + 世界系四元数）
- 影响：低层 policy 收到明显偏离训练分布的输入（世界系里含机器人的世界坐标），
  典型的"高层一开就抖/不收敛"
- 前置决策：你当前 WIP 已在 `flat_env_wbc_cfg.py:314-315` 把
  `policy.ee_goal / critic.ee_goal` 置 `None`；需要先定 WBC 版是否保留 `ee_goal`，
  再统一 replay 侧到底往哪个槽位喂什么坐标系的值

---

## P1 — 训练/回放一致性与可维护性（不改也能跑，但迟早出问题）

### 4. `[已修]` 低层动作 scale 在 replay 里被硬编码，且已经不一致

> 修复于 `codex/hl-replay-layout`（见文末"修复记录"）。
>
> 实测补充：这些硬编码赋值**本来就是死代码** —— `JointAction.__init__` 会把
> `cfg.scale` / `cfg.clip` / `cfg.joint_names` 编译成内部张量，之后
> `term.scale = 20.0` 只是新增了一个不起作用的实例属性。
> 实测（`Isaac-Deeprobotics-High-Level-Pick-Flat-Teacher-v0`）：
> `_wheel_vel_action_term._scale = 5.0`（来自低层 cfg），
> 而同名实例属性 `__dict__["scale"] = 20.0`（无人读取）。
> 因此"replay 20.0 vs 训练 5.0"不是行为不一致，而是**误导性的死代码**。
> 现修法：删掉全部事后赋值，改成从传入的低层 action cfg
> （`cfg.low_level_leg_actions` / `cfg.low_level_wheel_actions`）读取，
> 并在 `check_low_level_action_cfgs()` 里校验它与布局一致。

| 位置 | wheel 速度 scale |
|---|---|
| 低层训练 `rough_env_cfg.py:358` | `5.0` |
| `pre_trained_policy_action.py:109` | `20.0` |
| `pre_trained_pick_action.py:109` | `20.0` |
| `pre_trained_nav_action.py:99` | `20.0` |
| `pre_trained_pick_wbc_action.py:114` | `5.0` |

同理 `joint_pos` 的 `{".*_hipx_joint": 0.125, ...: 0.25}` 也复制了 6 份。
建议：replay 侧直接从"产生该 checkpoint 的低层 cfg"里取 scale/clip/joint_names，
不要手抄。

### 5. `[部分已修]` 6 份几乎相同的 "低层 replay action term" 实现

> 第一步（`codex/hl-replay-layout`）已完成 R3 级别抽取：新增
> `highlevel/mdp/low_level_replay.py`，把关节名单、低层观测组装、
> `last_action` 拼接口径、布局校验收敛成单一来源。
> `PreTrainedPickAction` / `PreTrainedPickWBCAction` / `TeleopLLAction`
> （三个已注册任务实际可达的 term）已改用它。
> 剩余：`pre_trained_policy_action.py` / `pre_trained_nav_action.py` /
> `openvla_pick_action.py` 的迁移，以及 R1 的基类合并 —— 见下一步计划。

`pre_trained_policy_action.py` / `pre_trained_nav_action.py` / `pre_trained_pick_action.py` /
`pre_trained_pick_wbc_action.py` / `openvla_pick_action.py` / `teleop_ll_action.py`
各自复制了：腿/轮/臂关节名单、`last_action` 闭包、低层 obs 覆写、marker 逻辑。
共同后果：上面第 2/3/4 条要在 6 个文件里各改一遍。
建议：抽一个 `LowLevelPolicyReplayMixin`（或把"低层观测构造"收拢成单一函数），
让"训练用什么、回放就用什么"成为结构性保证。

### 6. `[已修]` `__init__` 里就地修改传入的 cfg

> 修复于 `codex/hl-replay-layout`。`build_low_level_observation_group()`
> 对模板做 `deepcopy` 后再覆写，模板本身不再被改动；
> `PreTrainedPickWBCAction` 不再在运行时 `cfg.low_level_observations = WBCObservationsCfg().policy`，
> 改为由 `HLFlatPickWBCActionsCfg` / `TeleopActionsCfg` 在 cfg 层显式提供 WBC 观测模板。

- 例如 `pre_trained_pick_action.py:125-145`：`cfg.low_level_observations.actions.func = ...`、
  `.params = dict()`；`pre_trained_pick_wbc_action.py:130-166` 甚至直接
  `cfg.low_level_observations = wbc_obs_cfg.policy`（用一个新的 `WBCObservationsCfg()`）。
- 风险：cfg 是共享对象（配置类实例之间通过 deepcopy 隔离，但在同一 env 内多个 action term
  共享同一份 cfg 时会互相覆盖），且"回放观测"与"训练观测"的差异被藏在这些赋值里，
  很容易静默漂移。
- 建议：改成显式的 `low_level_obs_cfg` 字段 + 一个构造函数，不要就地改。

### 7. 硬编码的、带时间戳的 checkpoint 路径

- `hl_flat_pick_env_cfg.py:37 / 57 / 70`
- `hl_flat_nav_env_cfg.py:27`
- `hl_flat_openvla_env_cfg.py:32 / 42`

这些路径指向 `logs/`（`.gitignore` 里 `**/logs/*` 被忽略）→ 换机器/清理一次 logs 后，
所有高层任务都起不来。建议改成命令行参数或环境变量，并在加载失败时给出明确报错。

### 8. 模块级 `LOW_LEVEL_ENV_CFG = DeeproboticsM20RoughEnvCfg()`（`high_level_env_cfg.py:30`）

- 只为拿低层 cfg 就在 import 期实例化整个低层 env cfg（内部还会 deepcopy 所有嵌套配置）。
- 隐式耦合：低层 cfg 的任何改动都会静默改变高层行为（`sim.dt / render_interval / decimation`
  都是从它派生，见 `high_level_env_cfg.py:456-458`）。
- 顺带：`render_interval = LOW_LEVEL_ENV_CFG.decimation`(4) 小于 `decimation`(40)，
  每个 env step 会触发多次渲染（IsaacLab 已给出 WARNING）。

### 9. `[已修]` `pre_trained_policy_action.py:131` 引用不存在的观测项

> 修复于 `codex/hl-replay-layout`：`ee_pose_commands` → `ee_goal`，切片同步改为 `[:, 3:10]`。
> 该类仍未启用；完整迁移到 `low_level_replay` 属于第 5 条的后续工作。

`cfg.low_level_observations.ee_pose_commands.func = ...`，但低层 policy 观测组里只有
`ee_goal`（没有 `ee_pose_commands`）。该类目前未被启用（`high_level_env_cfg.py:265` 是注释掉的），
一旦启用会立刻 `AttributeError`。

### 10. 待确认的小问题（低置信度）

- `hl_flat_pick_env_cfg.py:423-431`：`HLFlatPickTerminationsCfg_PLAY` 里新定义了 `lift_object`
  DoneTerm，与 `pick_success` 语义重复、命名易混，建议确认是否冗余。
- `highlevel/mdp/encoder.py` 的视觉编码器注册表：`torch.hub.load` 需联网；
  `_frozen_encoders` 以 name 为唯一 key 全局缓存，换 device 会拿到旧设备上的模型。

---

## 建议的验证方式（改完 high_level 后）

1. **高层 EE 目标是否真的生效**：跑 `Isaac-M20-Piper-Teleop-v0`（走 `pose_command_b` 的正确写法）
   与 `Isaac-Deeprobotics-High-Level-Pick-WBC-Flat-Teacher-v0` 对比机械臂是否跟同一目标。
2. **加一个观测布局断言**：启动时打印 `action_manager.total_action_dim` 与各 obs group 的形状，
   并与 checkpoint 的 `actor.0.weight.shape` 对齐（不一致就直接报错，而不是等到训练跑偏）。
3. **坐标系回归**：让机器人初始 yaw 取非 0（例如 1.0 rad），比较"root 系目标 vs 世界系目标"
  两种算法下的跟踪奖励；二者应当明显不同。

---

## 清单外新增：本次实测发现并已修的两个 P0

### A. `[已修]` 高层喂给低层 policy 的 `actions` 观测少了 7 维 —— 比第 1 条更早触发

- 现象：`Isaac-Deeprobotics-High-Level-Pick-Flat-Teacher-v0` 在**第一次 `env.step()`**
  抛 `RuntimeError: mat1 and mat2 shapes cannot be multiplied (Nx76 and 83x512)`
  （`env.step()` 的顺序是 `apply_action()` → `reward_manager.compute()`，
  所以它挡在第 1 条的 `AttributeError` 前面）。
- 根因：replay 用 `_ee_ik_action_term.action_dim` 决定 `actions` 观测宽度。
  IK 改成 `CommandDrivenIKAction` 后该值为 **0**，于是 `actions` 只有 16 维
  （12 腿 + 4 轮），而旧 checkpoint 训练时是 **23 维**（12 + 4 + 7 IK），
  低层观测总维度 76 ≠ checkpoint 的 83。
- 修法：把"产生 checkpoint 的那次低层训练"的布局显式写进
  `low_level_replay.LowLevelActionLayout`（`ee_action_dim=7`，可由
  action term cfg 的 `ee_action_dim` 覆盖），`actions` 观测按 `[leg | wheel | ee_ik]`
  原样拼接；`ee_ik` 这 7 维取自低层 policy 上一帧输出，IK 仍由 CommandManager 驱动。
- 实测（`--num_envs 4`，直接建 env 不训练）：
  | 任务 | 修前 ll_policy obs | 修后 ll_policy obs | checkpoint 期望 | 结果 |
  |---|---|---|---|---|
  | `...Pick-Flat-Teacher-v0`（flat 低层） | 76（actions 16） | **83（actions 23）** | 83 | 维度一致 |
  | `...Pick-WBC-Flat-Teacher-v0`（WBC 低层） | 79 | **86** | 86 | 维度一致 |
  | `Isaac-M20-Piper-Teleop-v0` | 79 | **86** | 86 | 维度一致 |

### B. `[已修]` 低层 `joint_pos` 观测把"轮关节置零"作用到了错误的关节上

- 根因：replay 里 `joint_pos` 用的列顺序是 `policy_joint_names`
  （leg → wheel → arm，22 维），而 `wheel_asset_cfg.joint_ids` 是 articulation
  **原生 id**（实测 `[15,16,17,18]`）。用原生 id 索引重排后的列，
  清掉的是 `['hr_wheel_joint','arm_joint1','arm_joint2','arm_joint3']`，
  放行了 `fl/fr/hl_wheel` —— 与 known_issues #1 记录的原始 bug 完全同型。
- 实测：轮关节的正确列下标是 `[12,13,14,15]`。
- 修法：新增 `joint_pos_rel_without_wheel_columns()`，按**列**置零；
  并在 `verify_wheel_columns()` 里做启动期断言（列序、列下标都必须对得上），
  每次建 env 都会跑。

---

## 修复记录

| 条目 | 分支 | 提交 |
|---|---|---|
| ④ 低层动作 scale 硬编码（含"其实是死代码"的实测结论） | `codex/hl-replay-layout` | `dc45d0e` |
| ⑤ 6 份 replay 实现 → 第一步抽取（R3 级） | `codex/hl-replay-layout` | `dc45d0e` |
| ⑥ `__init__` 就地改 cfg | `codex/hl-replay-layout` | `dc45d0e` |
| ⑨ `ee_pose_commands` 引用不存在的观测项 | `codex/hl-replay-layout` | `dc45d0e` |
| 新增 A：`actions` 观测宽度少 7 维 | `codex/hl-replay-layout` | `dc45d0e` |
| 新增 B：`joint_pos` 轮关节掩码索引空间 | `codex/hl-replay-layout` | `dc45d0e` |

验证命令（均 `--headless`）：

```
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Deeprobotics-High-Level-Pick-WBC-Flat-Teacher-v0 \
    --headless --num_envs 64 --max_iterations 2      # exit 0
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Flat-Deeprobotics-M20-Piper-v0 \
    --headless --num_envs 64 --max_iterations 2      # exit 0（低层无回归）
```
