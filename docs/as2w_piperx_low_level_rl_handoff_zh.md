# AS2W-PiperX 低层 RL 配置 —— 实施交接单

> **状态**：计划已冻结，等待执行。本文档是唯一需求来源；执行时如与本文冲突，以本文为准。
> **基线提交**：`4d00769`（分支 `codex/as2w-piperx`，工作区干净）
> **执行环境**：conda `env_isaac_lab`，可实际启动 Isaac Sim / Isaac Lab
> **交付目标**：为 AS2W-PiperX 训练一个低层控制 RL 模型，结构干净、参数可追溯

---

## 0. 一分钟速读

执行前必须记住五条硬规则：

1. **不继承、不 import** `velocity/velocity_env_cfg.py` 的任何 config 类，也不 import `deeprobotics_m20/*` 的任何东西。
2. **不写** `if self.__class__.__name__ in (...)` 这种按类名分支的代码。
3. **机器人常量只能出现在** `as2w_piperx/profile.py`，不许散落在 MDP 或 cfg 里。
4. **每个 manager 组一个文件**，文件名即「参数在哪」的答案。
5. 本次**只做平地**；未来 rough 只允许「单层继承 flat、只改地形相关」。

---

## 1. 范围

### 在范围内

- 新建 AS2W-PiperX 平地低层速度/位姿跟踪 RL 配置（命令 + 动作 + 观测 + 奖励 + 事件 + 终止 + 课程 + runner）。
- 复用 `velocity/mdp` 的**算法函数**，对其做 4 处向后兼容参数化（见 §8）。
- 资产符号改名 `DEEPROBOTICS_AS2W_PIPERX_CFG` → `AS2W_PIPERX_CFG`（硬改名，不留 alias）。
- 目录改名 `deeprobotics_as2w_piperx` → `as2w_piperx`（`git mv`）。
- 注册 3 个 gym task、Phase 0 探针、smoke 验证、play / eval / 部署导出。

### 不在范围内

- 遥操（`highlevel/mdp/low_level_replay.py`）改造 —— 后续 phase。
- 任何 `highlevel/` 下的改动。
- 地形/stairs、rough 配置。
- 臂奖励（EE tracking / arm tracking）—— 明确不要，属 legacy。
- `arm_weight` 命令 —— 明确不要，属 legacy。
- 消融分支（V2/V3/V4/Expl0/PrivV4/NoHist 等）—— 只保留一套。

---

## 2. 已冻结决策

| ID | 决策 | 说明 |
|---|---|---|
| D-01 | 目录名 `as2w_piperx` | 去掉 `deeprobotics`（公司名） |
| D-02 | 资产符号 `AS2W_PIPERX_CFG` | 硬改名，不保留旧名 alias |
| D-03 | 任务 id `History-Adaptation-AS2W-PiperX-v0` / `...-play-v0` | 去掉 `Deeprobotics`；保留 `Visualize-AS2W-PiperX-v0` |
| D-04 | 基线 = **Anchor (h060)** | 见 §3，不是 V4、不是遥操 checkpoint |
| D-05 | 只保留 Anchor 一套配置 | 不实现任何消融开关 |
| D-06 | 命令项只有 `base_velocity` / `ee_pose` / `body_pose` | 无 `arm_weight` |
| D-07 | 奖励不包含任何臂项 | 无 `arm_joint_*`、无 EE 跟踪奖励 |
| D-08 | 镜像奖励第一版就开 | 只做 `FL↔FR`、`RL↔RR` 两对，权重 `-0.03` |
| D-09 | 不继承 M20 配置类、不 import `velocity_env_cfg` | 见 §4、§7 |
| D-10 | 资产模块不拆分 | 仍放 `assets/deeprobotics.py`，只改符号名 |
| D-11 | 第一版只做平地 | rough 将来单层继承 flat |
| D-12 | 共享 MDP 只允许 4 处参数化 | 全部默认值保持 M20 行为，见 §8 |
| D-13 | EE 碰撞盒按 DEF-066 修正 | 见 §3.1，这是**有意偏离**原 run 的缺陷行为 |
| D-14 | 遥操改动后置 | 本次只保证 env cfg 暴露关节名分组 |
| D-15 | `RobotProfile` 不放 `common/` | 只有一台机器人，不做提前抽象 |
---

## 3. 基线：为什么是 Anchor (h060)

用户已确认「用效果最好的那套」。证据链：

**文档证据** —— `docs/review/DONE_zh.md` 第六节（约 145–185 行）：

| 项 | 值 |
|---|---|
| 最佳 run | `history_anchor_flat_h060/2026-10-23_13-22-53_height_033_060` |
| 平均 `err_vel_xy` | **0.1093 m/s** |
| 平均摔倒率 | **1.56%** |
| 结论 | 综合排名第一；NoHist 摔倒率略低但速度误差更高 |

**权重证据** —— `logs/cloud_20k_checkpoints/logs/rsl_rl/history_anchor_flat_h060/2026-10-23_13-22-53_height_033_060/model_19999.pt` 的 state_dict（已读取）：

| 张量 | shape | 推论 |
|---|---|---|
| `history_encoder.conv.0.weight` | `(32, 70, 4)` | 历史单步 **70 维**，`history_length=10` ⇒ 700 |
| `privileged_encoder.encoder.0.weight` | `(128, 54)` | 特权 **54 维** |
| `actor.0.weight` | `(512, 115)` | policy obs **83** + latent 32 |
| `critic.0.weight` | `(512, 140)` | critic obs **86** + 特权 54 |
| `actor.6.weight` | `(16, 128)` | 动作 **16 维** |

⇒ Anchor = **V4 的 54 维特权向量 + 教师自编码锚（128,128）/ coef 1.0 + entropy 0 + max_noise_std 1.2 + 历史窗口用 legacy 70 维 `history_single_step_obs` + 非对称 critic**。

对应源码：

- task：`velocity/config/wheeled/deeprobotics_m20/__init__.py:454` → `flat_env_wbc_cfg:FlatEnvWBCConfigV4`
- runner：`agents/rsl_rl_ppo_cfg.py:326` `HistoryAnchorPPORunnerCfg`
- 解析后的完整 env：`.../params/env.yaml`（3470 行）
- 解析后的完整 agent：`.../params/agent.yaml`

**注意**：历史观测用的是 `history_single_step_obs`（70 维），**不是** `history_single_step_obs_full`（V2，79 维）。这是逐字节核对过的，不要改。

### 3.1 一个必须知道的历史缺陷（D-13）

Anchor run 的 `commands.ee_pose.collision_*_limits` 是 `[-0.3,-0.3,0.0]`–`[0.3,0.3,0.5]`，而 `f0831a5`（DEF-066，2026-10-08）已把这段比较从**世界坐标**改成**高度不变系局部坐标**，并把默认值改为 `[-0.6,-0.05]`。

含义：原 run 的 EE 碰撞盒**对 4096 个环境里的 4095 个是静默失效的**（只有 env 0 在原点才碰巧生效）。所以：

- 不要照抄 `[0.0, 0.5]` 这组数（在新坐标系下它含义完全不同，会变成「禁止抬手」）。
- AS2W 必须给一组**在新坐标系下正确**的盒子，并在 Phase 0 报告重采样拒绝率（建议 <30%）。
- 这是一次有意的行为修正，不是「复现原样」。

---

## 4. 不要照抄 M20 的四处结构

M20 主链路（**禁止**在 AS2W 复用）：

```
LocomotionVelocityRoughEnvCfg
  └─ DeeproboticsM20RoughEnvCfg      (rough_env_cfg.py:310)
       └─ DeeproboticsM20FlatEnvCfg  (flat_env_cfg.py:13)
            └─ FlatEnvWBCConfig      (flat_env_wbc_cfg.py:498)
                 └─ FlatEnvWBCConfig_PLAY (flat_env_wbc_cfg.py:619)
```

具体病症：

1. `flat_env_wbc_cfg.py` 有 **11 处** `if self.__class__.__name__ in (...)`；新增子类会静默跳过设置（例如 `FlatEnvWBCConfigV4.__name__` 不在元组里 ⇒ 分层速度采样、`disable_zero_weight_rewards()`、vx 课程关闭全部**没生效**）。
2. 奖励权重在 3 层 `__post_init__` 里反复改写，最终值不可追踪。
3. 机器人常量硬编码在 MDP 里：`mdp/utils.py:23` `radius_wheel = 0.09`、`mdp/rewards.py:975` 的镜像正则、`mdp/observations.py:452` 的 actuator 集合、散落的 `body_names=".*wheel"`。
4. 文件按「历史演进」而不是「manager 组」切分，`flat_env_wbc_cfg.py` 超 1200 行。
---

## 5. 目标目录结构

放在 `velocity/config/wheeled/` 下，理由见 §5.1。

```
source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/as2w_piperx/
  __init__.py              # 只做 gym.register（3 个 task）
  profile.py               # AS2W_PIPERX_PROFILE：唯一机器人常量来源
  scene_cfg.py             # 地面 + 机器人 + 灯光 + 接触传感器
  stand_env_cfg.py         # 由旧 deeprobotics_as2w_piperx/stand_env_cfg.py 迁入
  low_level/
    __init__.py
    env_cfg.py             # AS2WPiperXFlatEnvCfg / AS2WPiperXFlatEnvCfg_PLAY
    actions.py             # 12 腿位置 + 4 轮速度 + IK 臂（0 维）
    commands.py            # base_velocity / ee_pose / body_pose
    observations.py        # policy 83 / critic 86 / history 700 / privileged 54
    rewards.py             # 只列启用项，权重显式
    events.py
    terminations.py
    curriculum.py
  agents/
    __init__.py
    rsl_rl_ppo_cfg.py      # AS2WPiperXHistoryPPORunnerCfg
```

同时执行：

```
git mv source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_as2w_piperx \
       source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/as2w_piperx
```

### 5.1 为什么不放到 `locomotion/as2w_piperx/`

`locomotion/` 的第一层是**任务族**轴（`velocity` / `highlevel`），机器人是下一层。把机器人名放到第一层会：

- 把「任务族」和「机器人」两个正交维度压到同一层；
- 让 `velocity` 退化成「共享内核包」，包名与职责更不一致；
- 以后加 rough 只能写成 `locomotion/as2w_piperx/rough/`，把任务族嵌进机器人下面，与现有约定相反。

真正该隔离的是**配置层**，不是包层级。用 §7 的自动检查来硬保证，比挪目录可靠。

---

## 6. 模块设计

### 6.1 `profile.py`（唯一常量来源）

要求：`@dataclass(frozen=True)`，只存 `str / float / tuple / Mapping`，**不存 `SceneEntityCfg`**。

理由：`SceneEntityCfg` 会被 `resolve()` **原地改写**成 id 列表；如果放在模块级共享，第二次用就会被污染。所以 `profile.py` 只放数据，另提供**工厂函数**每次新建 cfg：

```python
def leg_joint_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=list(PROFILE.joint_groups.non_wheel_legs), preserve_order=True)

def wheel_joint_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=list(PROFILE.joint_groups.wheels), preserve_order=True)

def wheel_body_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", body_names=PROFILE.body_groups.wheel_regex)

def ee_body_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", body_names=PROFILE.body_groups.ee)
```

建议字段（按组拆小 dataclass，但不要拆文件）：

```python
class JointGroups:       legs(12 非轮) / wheels(4) / arm(6) / gripper(2) / hip / thigh / calf
class BodyGroups:        base / arm_base / ee / wheel_regex / non_wheel_regex
class MirrorPair:        a(3 个关节) / b(3 个关节) / signs{"hip","thigh","calf"}
class EEAnchor:          p_l / p_pitch / p_yaw
class WheeledArmProfile: name, joint_groups, body_groups, actuator_names, wheel_joint_regex,
                         wheel_radius, default_base_height, base_height_range,
                         action_scale, mirror_pairs, joint_kind_patterns, ee_anchor,
                         ee_workspace, ee_collision_box, termination_height
```

`AS2W_PIPERX_PROFILE = WheeledArmProfile(...)`。

### 6.2 `scene_cfg.py`

- 平地 `TerrainImporterCfg(terrain_type="plane")`。
- `robot = AS2W_PIPERX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")`。
- `contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", update_period=0.005)`。
- 灯光：dome + sun（沿用现有 stand cfg）。
- **不需要** `height_scanner` / `height_scanner_base`：平地 `slope_gate_rad=0` ⇒ `slope_gate()` 直接返回全 1（`mdp/rewards.py:930`），扫描器完全用不到。rough 阶段再加。
- stand 与 train 共用本文件。
### 6.3 `low_level/`

| 文件 | 内容 | 硬约束 |
|---|---|---|
| `actions.py` | `joint_pos`（12 腿位置）、`joint_vel`（4 轮速度）、`ee_ik`（`CommandDrivenIKAction`，0 维） | 总动作维度必须 = 16 |
| `commands.py` | `base_velocity` / `ee_pose` / `body_pose` | 不许出现 `arm_weight` |
| `observations.py` | 4 个组：policy / critic / history / privileged | 维度见 §9.3 |
| `rewards.py` | 只定义 §9.5 的 23 个启用项 | 不定义零权重项 |
| `events.py` | startup / reset / interval 三档 | 含 `check_policy_layout` |
| `terminations.py` | §9.7 的 4 项 | — |
| `curriculum.py` | §9.8 的 14 项 | — |
| `env_cfg.py` | 组合以上 + `__post_init__` 只设 sim/decimation/episode | **禁止按类名分支** |

`env_cfg.py` 的 `__post_init__` 只允许做这几件事：`sim.dt`、`decimation`、`episode_length_s`、`viewer`、`sim.physx.bounce_threshold_velocity`。所有奖励/命令/课程数值必须写在各自的 cfg 文件里。

### 6.4 `agents/rsl_rl_ppo_cfg.py`

只有一个类：`AS2WPiperXHistoryPPORunnerCfg`，字段逐项等于 §9.9。docstring 里注明「基线 = M20 History-Anchor (h060)，唯一保留的一套」。

### 6.5 命名

| 类别 | 命名 |
|---|---|
| 资产符号 | `AS2W_PIPERX_CFG` |
| env cfg | `AS2WPiperXFlatEnvCfg` / `AS2WPiperXFlatEnvCfg_PLAY` |
| runner cfg | `AS2WPiperXHistoryPPORunnerCfg` |
| task | `History-Adaptation-AS2W-PiperX-v0` / `-play-v0` / `Visualize-AS2W-PiperX-v0` |

---

## 7. 依赖规则与自动检查

| 档 | 内容 | 规则 |
|---|---|---|
| **禁止** | `velocity_env_cfg.py`、`velocity/config/**/deeprobotics_m20/**`、任何 `*_env_cfg` 里的 config 类 | 永不 import |
| **允许、不改** | `velocity/mdp` 里的算法 term（IK 动作、低速滤波、EE/body 命令类、history buffer…） | 通过参数注入 |
| **允许、加默认值参数** | 内部写死机器人常量的 4 个函数（§8） | kwarg 默认值保持 M20 现状 |

必须新增两个自动检查（放在 `scripts/reinforcement_learning/rsl_rl/`）：

1. `check_module_boundary.py`：AST 扫描 `as2w_piperx/**/*.py` 的 import 语句，断言不出现 `deeprobotics_m20`、`velocity_env_cfg`。
2. `check_no_class_name_branch.py`：对同一目录做文本扫描，断言没有 `__class__.__name__`。

两条都要秒级跑完，失败即拒绝合并。

---

## 8. 共享 MDP 的 4 处参数化（唯一允许改的 M20 文件）

所有改动必须**默认值保持 M20 行为**。改完跑 M20 2-iter smoke + golden YAML diff（§13 Phase 1）。

### 8.1 `velocity/mdp/utils.py` — `compute_base_height_rel_to_feet`

```python
def compute_base_height_rel_to_feet(
    env, asset_cfg, feet_cfg,
    wheel_radius: float = 0.09,   # 新增
) -> torch.Tensor:
    ...
    return base_h_w - feet_h_w.mean(dim=1) + float(wheel_radius)
```

### 8.2 `velocity/mdp/commands.py` — `BodyPoseCommandCfg` 增加 `feet_radius`

- `BodyPoseCommandCfg` 增加字段 `feet_radius: float = 0.09`（带注释：足端轮半径，传给 `compute_base_height_rel_to_feet`）。
- `BodyPoseCommand` 的两处调用（当前 `commands.py:1493`、`:1566`）改为：
  `compute_base_height_rel_to_feet(self._env, self.cfg.asset_cfg, self.cfg.feet_cfg, wheel_radius=self.cfg.feet_radius)`。

### 8.3 `velocity/mdp/rewards.py` — 两处

**(a) `body_height_tracking` 增加 `feet_radius: float = 0.09`**，透传给 `compute_base_height_rel_to_feet`。

**(b) `joint_mirror_signed` 增加 `joint_kind_patterns: dict[str, str] | None = None`**

- `None` ⇒ 完全保持现有行为（现正则 `_(hipx|hipy|knee)_joint$`，kind 取 group(1)）。
- 传入时（AS2W 用 `{"hip": r"_hip$", "thigh": r"_thigh$", "calf": r"_calf$"}`）⇒ 按给定正则逐个 `search` 匹配，匹配不到就抛错（**不要静默跳过**）。
- **同时把 `env.joint_mirror_signed_cache_key` 加上 patterns**，否则同进程里换机器人会命中错误缓存。

### 8.4 `velocity/mdp/observations.py` — 两处

**(a) `privileged_env_params_v4`**：新增 param `wheel_joint_regex: str = ".*_wheel_joint"`，把当前的

```python
self.wheel_joint_names = {jn for jn in self.asset.joint_names if "wheel" in jn}
```

改成用 `re.fullmatch(self.wheel_joint_regex, jn)` 判定。默认值对 M20 等价（M20 轮关节名都以 `_wheel_joint` 结尾）。同步更新那条「轮关节数 ≠ 轮 body 数」的错误信息。

> `foot_body_regex`、`included_actuators`、`base_body_name`、`gripper_body_name` **本来就有**，不用改。

**(b) `privileged_joint_gain_scale`**：`__init__` 里把硬编码的

```python
self.included_actuators = {"joint", "wheel", "piper_arm", "piper_gripper"}
```

改成

```python
self.included_actuators = set(cfg.params.pop("included_actuators", ("joint", "wheel", "piper_arm", "piper_gripper")))
```

**不需要改**：`stand_still_wheel_vel_l2` 已支持 `asset_cfg` / `feet_cfg` / `slope_gate_rad`。
---

## 9. Anchor 参数基准（照抄 + 替换）

以下数值除注明「待测」外，均来自 `params/env.yaml` / `params/agent.yaml`，可直接使用。

### 9.1 sim / scene / episode

| 项 | 值 |
|---|---|
| `sim.dt` | `0.005` |
| `decimation` | `4`（50 Hz 决策） |
| `episode_length_s` | `20.0` |
| `seed` | `42` |
| `scene.num_envs`（训练） | `4096` |
| `scene.env_spacing` | `2.5` |
| `terrain` | `terrain_type="plane"`, `collision_group=-1` |
| `replicate_physics` | `True` |
| `filter_collisions` | `True` |
| `contact_forces.update_period` | `0.005` |
| `robot.init_state.pos` | `(0, 0, 0.45)` |
| `soft_joint_pos_limit_factor` | AS2W 资产里是 `0.95`（M20 是 `0.9`） |

### 9.2 actions（总 16 维）

| term | 类型 | 关键参数 |
|---|---|---|
| `joint_pos` | `JointPositionAction` | `joint_names` = 12 个非轮腿关节；`scale = {hip: 0.125, thigh: 0.25, calf: 0.25}`；`use_default_offset=True`；`preserve_order=True`；`clip=(-100,100)` |
| `joint_vel` | `JointVelocityAction` | `joint_names` = 4 个 `*_foot`；`scale=5.0`；`use_default_offset=True`；`preserve_order=True` |
| `ee_ik` | `CommandDrivenIKAction` | `joint_names=["arm_joint[1-6]"]`；`body_name="gripper_base"`；`command_name="ee_pose"`；DLS `lambda_val=0.01`；`body_offset` 全零；动作维度 **0**（IK 由命令驱动，不占动作） |

> M20 的 `scale` 正则 `.*_hipx_joint: 0.125` 对应 AS2W 的 `hip`；`hipy/knee: 0.25` 对应 `thigh/calf`。
> `0.125/0.25/0.25` 是起点，Phase 4 用「关节目标速度 / 抖动」验证是否需要调。

### 9.3 observations（4 组，维度必须与 M20 完全一致）

| 组 | 维度 | 组成 |
|---|---|---|
| policy | **83** | `base_ang_vel`(3, scale 0.25) + `projected_gravity`(3) + `velocity_commands`(3) + `joint_pos`(24, 轮置零) + `joint_vel`(24, scale 0.05) + `actions`(16) + `ee_goal`(7) + `body_pose_cmd`(3) |
| critic | **86** | `base_lin_vel`(3) + policy 的全部 8 项（**不含** latent） |
| history | **70/步 × 10 = 700** | `mdp.history_single_step_obs`：`base_ang_vel`(3) + `projected_gravity`(3) + `joint_pos`(24) + `joint_vel`(24) + `last_action`(16) |
| privileged | **54** | `mdp.privileged_env_params_v4` |

要点：

- `joint_pos` 用 `mdp.joint_pos_rel_without_wheel`，**宽度仍是 24**（只把轮列置零，不删列），不是 20。
- `ee_goal` 用 `mdp.ee_goal_local`，宽度 7（pos 3 + quat 4）。
- 只有 `policy` 开 `enable_corruption=True`；其余三组都是 `False`。
- history 的 `history_length=10`、`flatten_history_dim=True`，`clip=(-100,100)`。
- privileged 的 params：
  ```python
  {
    "asset_cfg": SceneEntityCfg("robot"),
    "base_body_name": "base_link",
    "gripper_body_name": "gripper_base",
    "foot_body_regex": r".*_foot",
    "included_actuators": ("hip_thigh", "calf", "wheel", "piper_arm", "piper_gripper"),
    "wheel_joint_regex": r".*_foot",
  }
  ```

**AS2W 关节数也是 24**（12 非轮腿 + 4 轮 + 6 臂 + 2 夹爪），所以 83/86/70/54/16 与 M20 逐位相同。这是最强的验收信号。

### 9.4 commands

**`base_velocity`**（`UniformThresholdVelocityCommand`）

| 项 | 值 |
|---|---|
| `resampling_time_range` | `(10.0, 10.0)` |
| `heading_command` | `False` |
| `rel_standing_envs` | `0.02`（由课程爬到 0.15） |
| `rel_heading_envs` | `1.0` |
| `ranges.lin_vel_x` | `(-1.0, 1.0)`（初始，课程爬到 ±5） |
| `ranges.lin_vel_y` | `(-1.0, 1.0)` |
| `ranges.ang_vel_z` | `(-1.0, 1.0)` |
| `rel_zero_cmd_envs` / `rel_only_lin_x_envs` / `rel_only_lin_y_envs` / `rel_only_ang_z_envs` | **全部 0（默认）** |

> ⚠️ Anchor 用的是 `FlatEnvWBCConfigV4`，它的类名不在 `("FlatEnvWBCConfig", "FlatEnvWBCConfig_PLAY")` 里 ⇒ **没有启用分层采样**。不要照抄 `0.2/0.15/0.15/0.15`。

**`ee_pose`**（`HeightInvariantEECommand`）

| 项 | 值 |
|---|---|
| `resampling_time_range` | `(5.0, 5.0)` |
| `body_name` / `arm_base_link_name` | `gripper_base` / `arm_base_link` |
| `ranges.p_l`（Stage 0 锚点） | `(0.41, 0.41)` ← **待重定** |
| `ranges.p_pitch` | `(-0.08, -0.08)` ← **待重定** |
| `ranges.p_yaw` | `(0.0, 0.0)` |
| `ranges.o_roll / o_pitch / o_yaw` | 全 `(0, 0)` |
| `ranges.T_traj` | `(1.0, 3.0)` |
| `ranges.orn_cone_min_scale` | `0.1` |
| `sampled_height` | `0.6` ← **待重定**（height-invariant 帧原点世界 z） |
| `collision_lower_limits` | `[-0.3, -0.3, -0.60]` ← **待重定**（见 D-13） |
| `collision_upper_limits` | `[0.3, 0.3, -0.05]` ← **待重定** |
| `underground_limit` | `0.05`（世界 z） |
| `num_collision_check_samples` / `max_resample_attempts` | `10` / `10` |
| `target_blend_pos` / `target_blend_orn` | `1.0` / `1.0` |
| `urdf_path` / `ee_link_name` | 建议 **启用**：`AS2W_PIPERX_URDF_PATH` / `"gripper_base"` |

> `p_l / p_pitch / p_yaw` 是**球坐标**（以 `arm_base_link` 为原点的半径 / 仰角 / 方位角），不是笛卡尔 xyz。重定工作空间时不要把它当 x 用。
> `urdf_path` 启用后会用 FK 采样建可达栅格（`reach_samples=200_000`），DEF-049/054 记录过它对关节饱和的收益很大。AS2W 臂更小、更容易饱和，建议第一版就开；Phase 4 用「关节限位占比」验证。

**`body_pose`**（`BodyPoseCommand`）

| 项 | 值 |
|---|---|
| `resampling_time_range` | `(10.0, 10.0)` |
| `height_range` | `(0.33, 0.60)` ← **待重定** |
| `pitch_range` / `roll_range` | `(0.0, 0.0)` / `(0.0, 0.0)` |
| `steady_error_clip` | `0.15` |
| `asset_cfg` | `SceneEntityCfg("robot")` |
| `feet_cfg` | `SceneEntityCfg("robot", body_names=r".*_foot")` |
| `feet_radius` | `≈0.077` ← **§8.2 新增字段，待测** |
### 9.5 rewards（只定义这 23 项）

| term | weight | func | 关键 params |
|---|---|---|---|
| `is_terminated` | `-5.0` | `mdp.is_terminated` | — |
| `track_lin_vel_xy_exp` | `2.0` | `mdp.track_lin_vel_xy_exp` | `std=√2` |
| `track_ang_vel_z_exp` | `1.0` | `mdp.track_ang_vel_z_exp` | `std=√0.2` |
| `upward` | `0.08` | `mdp.upward` | — |
| `feet_contact_without_cmd` | `0.1` | `mdp.feet_contact_without_cmd` | `sensor_cfg` 轮 body |
| `body_height_tracking` | `0.001` | `mdp.body_height_tracking` | `std=0.04`，`feet_radius` |
| `body_pitch_tracking` | `0.001` | `mdp.body_pitch_tracking` | `std=0.05` |
| `body_roll_tracking` | `0.001` | `mdp.body_roll_tracking` | `std=0.04` |
| `stand_still_vel` | `-0.2` | `mdp.stand_still_vel_l2` | `yaw_weight=1.0` |
| `stand_still_wheel_vel` | `-5e-5` | `mdp.stand_still_wheel_vel_l2` | 轮关节 |
| `hip_joint_pos_penalty` | `-0.4` | `mdp.joint_pos_penalty_wbc` | `stand_still_scale=5.0`, `velocity_threshold=0.5`, `command_threshold=0.1`, `pose_command_name="body_pose"` |
| `thigh_joint_pos_penalty` | `-0.1` | 同上 | 同上 |
| `calf_joint_pos_penalty` | `-0.1` | 同上 | 同上 |
| `joint_mirror` | `-0.03` | `mdp.joint_mirror_signed` | 2 对 + signs（§10.2） |
| `action_rate_l2` | `-0.01` | `mdp.action_rate_l2` | — |
| `joint_torques_l2` | `-2.5e-5` | `mdp.joint_torques_l2` | 12 腿关节 |
| `joint_acc_l2` | `-2e-7` | `mdp.joint_acc_l2` | 12 腿关节 |
| `joint_acc_wheel_l2` | `-1e-7` | `mdp.joint_acc_l2` | 4 轮关节 |
| `joint_power` | `-2e-5` | `mdp.joint_power` | 12 腿关节 |
| `joint_pos_limits` | `-5.0` | `mdp.joint_pos_limits` | 12 腿关节 |
| `undesired_contacts` | `-1.0` | `mdp.undesired_contacts` | 非轮 body（见下），`threshold=1.0` |
| `undesired_base_link_contacts` | `-3.0` | `mdp.undesired_contacts` | `base_link`，`threshold=1.0` |
| `contact_forces` | `-1.5e-4` | `mdp.contact_forces` | 轮 body，`threshold=100.0` |

`undesired_contacts` 的非轮 body 正则：M20 是 `^(?!.*.*_wheel).*`，AS2W 改成 **`^(?!.*_foot).*`**。

> 不定义任何零权重项（等价于 Anchor 乘 0 的结果，但可审计）。
> **不要**写 `disable_zero_weight_rewards()`。
> **不要**任何 `arm_joint_*` / `ee_position_tracking` / `ee_orientation_tracking` / `ee_goal_reached`。

### 9.6 events

| term | mode | func | 关键 params |
|---|---|---|---|
| `randomize_rigid_body_material` | startup | `randomize_rigid_body_material` | `body_names=".*"`, static/dynamic `(0.35,1.5)`, restitution `(0,0.7)`, `num_buckets=1024` |
| `randomize_rigid_body_mass` | startup | `randomize_rigid_body_mass` | `body_names="^(?!.*base_link).*"`, `(0.85,1.15)` scale |
| `randomize_rigid_body_mass_end_effector` | startup | 同上 | `body_names="gripper_base"`, `(1.0,3.0)` |
| `randomize_rigid_body_mass_base` | startup | 同上 | `body_names="base_link"`, `(-1.0,3.0)` **add** |
| `randomize_rigid_body_inertia` | startup | `mdp.randomize_rigid_body_inertia` | `(0.85,1.15)` |
| `randomize_com_positions` | startup | `randomize_rigid_body_com` | `base_link`, `x/y ±0.03`, `z ±0.02` |
| `check_policy_layout` | startup | `mdp.check_policy_layout` | `group_name="policy"` |
| `randomize_apply_external_force_torque` | reset | `apply_external_force_torque` | `base_link`, force/torque `(-10,10)` |
| `randomize_reset_joints` | reset | `reset_joints_by_scale` | `(1.0,1.0)` / `(0.0,0.0)`（= 不随机化） |
| `randomize_actuator_gains` | reset | `randomize_actuator_gains` | `joint_names=".*"`, k/d `(0.85,1.15)` |
| `randomize_reset_base` | reset | `reset_root_state_uniform` | `pose` 的 `x/y ±1.0`、`roll/pitch ±0.3`、`yaw ±3.14`；`velocity` 的 `x/y/z ±0.2`、`roll/pitch ±0.05`、`yaw 0` |
| `randomize_push_robot` | interval | `push_by_setting_velocity` | `x ±2`, `y ±1`, `yaw ±0.52` |

### 9.7 terminations

| term | func | params |
|---|---|---|
| `time_out` | `mdp.time_out` | `time_out=True` |
| `terrain_out_of_bounds` | `mdp.terrain_out_of_bounds` | `distance_buffer=3.0` |
| `bad_orientation_2` | `mdp.bad_orientation_2` | `limit_angle=0.8` |
| `root_height_below_minimum` | `mdp.root_height_below_minimum` | `minimum_height=0.3` ← **待重定** |

> `bad_orientation_2` 定义在 `mdp/events.py`（不是 terminations.py），import 路径别写错。
> `minimum_height=0.3` 对 AS2W（站高 0.4466、蹲伏极限待测）大概率偏低，Phase 0 给出最低稳定高度后再定。
### 9.8 curriculum

| term | func | params |
|---|---|---|
| `body_pose_pitch_range_s3` | `override_value` | `commands.body_pose.pitch_range → (-0.35,0.35)`, `num_steps=50000` |
| `body_pose_roll_range_s3` | `override_value` | `commands.body_pose.roll_range → (-0.25,0.25)`, `num_steps=50000` |
| `body_height_rew_s2` | `ramp_reward_weight` | `body_height_tracking → 0.8`, `num_steps=25000` |
| `body_pitch_rew_s3` | `ramp_reward_weight` | `body_pitch_tracking → 0.8`, `num_steps=50000` |
| `body_roll_rew_s3` | `ramp_reward_weight` | `body_roll_tracking → 0.8`, `num_steps=50000` |
| `base_velocity_lin_vel_x_s4` | `override_value` | `→ (-2,2)`, `75000` |
| `base_velocity_lin_vel_x_s5` | `override_value` | `→ (-3,3)`, `100000` |
| `base_velocity_lin_vel_x_s6` | `override_value` | `→ (-4,4)`, `125000` |
| `base_velocity_lin_vel_x_s7` | `override_value` | `→ (-5,5)`, `150000` |
| `ee_goal_stages` | `apply_range_stages` | 3 个阶段，见下 |
| `disturbance_ramp` | `apply_event_scale` | `num_steps=50000`, `start_scale=0.2` |
| `standing_env_ratio_ramp` | `ramp_command_param` | `rel_standing_envs 0.02→0.15`, `25000` |
| `stand_still_vel_ramp` | `ramp_reward_weight` | `-0.2 → -2.0`, `25000` |
| `stand_still_wheel_vel_ramp` | `ramp_reward_weight` | `-5e-5 → -5e-4`, `25000` |

Anchor 里还有 3 个 **None** 的课程项，新配置不要定义：`terrain_levels`、`command_levels`、`body_pose_height_range_s2`。

`ee_goal_stages`（M20 值，AS2W **必须重算**）：

| stage | 步数 | `p_l` | `p_pitch` | `p_yaw` | `o_roll` | `o_pitch` | `o_yaw` |
|---|---|---|---|---|---|---|---|
| 1 | 25000 | (0.36,0.47) | (-0.35,0.2) | (±0.45) | ±0.09 | ±0.09 | 0 |
| 2 | 50000 | (0.33,0.5) | (-0.55,0.4) | (±0.85) | ±0.175 | ±0.175 | ±0.6 |
| 3 | 75000 | (0.3,0.52) | (-π/4, π/5) | (±2π/5) | ±π/8 | ±π/8 | ±π |

`disturbance_ramp` 的 spec：

```python
[
  {"term": "randomize_push_robot", "param": "velocity_range",
   "base": {"x": (-2.0, 2.0), "y": (-1.0, 1.0), "yaw": (-0.52, 0.52)}},
  {"term": "randomize_apply_external_force_torque", "param": "force_range", "base": (-10.0, 10.0)},
  {"term": "randomize_apply_external_force_torque", "param": "torque_range", "base": (-10.0, 10.0)},
]
```

### 9.9 runner（`agent.yaml` 逐字段）

```yaml
seed: 42
num_steps_per_env: 24
max_iterations: 20000
empirical_normalization: false
clip_actions: 100
save_interval: 500
class_name: OnPolicyRunnerHis
obs_groups: {policy: [policy], critic: [critic, privileged], history: [history], privileged: [privileged]}
policy:
  class_name: ActorCriticHistory
  init_noise_std: 1.0
  noise_std_type: log
  state_dependent_std: false
  actor_hidden_dims: [512, 256, 128]
  critic_hidden_dims: [512, 256, 128]
  activation: elu
  latent_dim: 32
  history_length: 10
  history_encoder_hidden_channels: (32, 32, 32)
  history_encoder_kernel_sizes: (4, 3, 2)
  history_encoder_strides: (2, 1, 1)
  history_encoder_projection_dim: 0
  history_encoder_type: tcn
  history_encoder_mlp_hidden_dims: (256, 128)
  history_encoder_gru_hidden_dim: 64
  history_encoder_gru_num_layers: 1
  privileged_encoder_hidden_dims: (128, 64)
  teacher_recon_hidden_dims: (128, 128)
  max_noise_std: 1.2
algorithm:
  class_name: rsl_rl.algorithms.ppo_roa.PPORoA
  num_learning_epochs: 5
  num_mini_batches: 4
  learning_rate: 1.0e-3
  schedule: adaptive
  gamma: 0.99
  lam: 0.95
  entropy_coef: 0.0
  desired_kl: 0.01
  max_grad_norm: 1.0
  value_loss_coef: 1.0
  use_clipped_value_loss: true
  clip_param: 0.2
  normalize_advantage_per_mini_batch: false
  rnd_cfg: null
  symmetry_cfg: null
  constraint_coef_schedule: (0.0, 0.1, 3000, 7000)
  teacher_recon_coef: 1.0
```

---

## 10. M20 → AS2W 映射

### 10.1 名字 / 正则替换

| 项 | M20 | AS2W |
|---|---|---|
| 资产符号 | `DEEPROBOTICS_M20_PIPER_CFG` | `AS2W_PIPERX_CFG` |
| 腿关节 | `fl_hipx_joint` / `fl_hipy_joint` / `fl_knee_joint` … | `FL_hip` / `FL_thigh` / `FL_calf` … |
| 轮关节 | `fl_wheel_joint` … | `FL_foot` / `FR_foot` / `RL_foot` / `RR_foot` |
| 轮 body | `.*wheel` | `.*_foot` |
| 非轮 body 正则 | `^(?!.*.*_wheel).*` | `^(?!.*_foot).*` |
| actuator | `joint` / `wheel` / `piper_arm` / `piper_gripper` | `hip_thigh` / `calf` / `wheel` / `piper_arm` / `piper_gripper` |
| 镜像关节类型键 | `hipx` / `hipy` / `knee` | `hip` / `thigh` / `calf` |
| 关节类型正则 | `_(hipx\|hipy\|knee)_joint$` | `_hip$` / `_thigh$` / `_calf$` |

### 10.2 镜像奖励（唯一一套）

只做左右两对：

```python
mirror_joints = [
    [r"FL_(hip|thigh|calf)", r"FR_(hip|thigh|calf)"],
    [r"RL_(hip|thigh|calf)", r"RR_(hip|thigh|calf)"],
]
mirror_signs = [
    {"hip": -1.0, "thigh": +1.0, "calf": +1.0},
    {"hip": -1.0, "thigh": +1.0, "calf": +1.0},
]
joint_kind_patterns = {"hip": r"_hip$", "thigh": r"_thigh$", "calf": r"_calf$"}
weight = -0.03
```

**符号依据（URDF 实测，需 Phase 0 复核）**：

- 四条腿的 `hip/thigh/calf` 关节 frame 的 `rpy` **全部是 (0,0,0)**，轴分别是 `(1,0,0)` 和 `(0,1,0)`，左右腿之间只有 `y` 方向平移差异（`±0.06`、`±0.1054`）。
- 所以左右腿不是「旋转副本」，而是「同向 frame 平移」：`hip` 绕 +x 旋转对左右腿产生相反方向的抬升 ⇒ `hip = -1`；`thigh/calf` 绕 +y 的矢状面运动左右一致 ⇒ `+1`。
- 默认站姿自证：`FL_hip=+0.10, FR_hip=-0.10`，而四条腿的 `thigh=+0.44, calf=-0.88` 完全相同。

**不做前后对、不做对角对**：后腿 `thigh` 限位是 `(-1.3, 4.19)`，前腿是 `(-2.0, 3.492)`，结构上不是前腿的副本，对称假设不成立。
---

## 11. 必须实测重定的参数（Phase 0 输出）

| 参数 | M20 值 | AS2W 处理 |
|---|---|---|
| 轮半径 | `0.09`（硬编码） | 实测轮心高，预计 `≈0.077` |
| 默认站高 | `0.513` | 实测 `base_link` 稳定高度，URDF 估算 `≈0.4466` |
| `body_pose.height_range` | `(0.33, 0.60)` | 实测蹲伏/抬升极限（不要照抄：M20 站高 0.513，AS2W 只有 0.4466） |
| 终止高度 | `0.30` | 实测「最低稳定高度 − 余量」 |
| EE `sampled_height` | `0.6` | 按 `arm_base_link` 世界 z（≈0.526）重定 |
| EE `p_l/p_pitch/p_yaw`（Stage 0 锚点） | `(0.41, -0.08, 0)` | 在 AS2W 球坐标工作空间里重找「低重心、前伸、臂不动」点 |
| EE 三阶段区间 | 见 §9.8 | 按实测可达区间重算 |
| EE 碰撞盒 | `[-0.3,-0.3,-0.6]`–`[0.3,0.3,-0.05]` | 按 AS2W 车体尺寸重定，并报告拒绝率 |
| 动作 scale | `hipx 0.125 / 其余 0.25` | `hip 0.125 / thigh 0.25 / calf 0.25`，smoke 验证 |
| `contact_forces.threshold` | `100.0` | 按轮半径/力矩重估（轮 effort 15 N·m ÷ 0.077 m ≈ 195 N） |
| self-collision | `True` | A/B 后决定（资产当前是 `False`） |

---

## 12. Phase 0：几何/工作空间探针（阻塞项）

**必须先完成，否则不要写 `profile.py` 的数值。** 用 conda `env_isaac_lab` 实跑。

新增脚本：`scripts/reinforcement_learning/rsl_rl/probe_as2w_geometry.py`
（可参考 `probe_ee_command_init.py`、`sweep_ee_anchor.py`、`probe_gait_symmetry.py` 的写法）

### 必须产出的 8 组数据

1. **名单与顺序**：完整 body / joint / actuator 名字与索引顺序；确认 `FL_foot` 同时是轮关节和轮 body。
2. **站高**：`base_link` 世界 z 的均值/标准差（200 步，零动作）。
3. **轮半径**：四个轮心世界 z 均值；等效半径 = 站高 − 轮心高（交叉验证 §11 的 0.077）。
4. **臂基座**：`arm_base_link` 世界 z；`gripper_base` 在默认站姿下的位姿。
5. **EE 工作空间**：以 `arm_base_link` 为原点的球坐标可达图（`p_l ∈ [0.2,0.8]`、`p_pitch ∈ [-1.2,1.2]`、`p_yaw ∈ [-π,π]`），输出可达栅格 + 一个低重心锚点候选。
6. **镜像符号 A/B**：对 `FL↔FR`、`RL↔RR`，扫 `θ_a ∈ [-0.3,0.3]`，求使两侧轮心位置镜像误差最小的 `s_b`，输出实际符号表。
7. **self-collision A/B**：开/关对比站高波动、关节力矩峰值、自接触次数；给出「开/关」建议 + 数值。
8. **最低稳定高度**：蹲伏扫描，给出 `height_range` 下界与 `root_height_below_minimum` 的建议值。

### 输出

```
logs/probe/as2w/<YYYYMMDD-HHMMSS>/
  summary.json          # 机器可读
  report.md             # 人读结论 + 建议值
  reachability.npz      # EE 可达栅格
```

`report.md` 必须逐条填写 §11 表格的「实测值」列。

---

## 13. Phase 1–5

### Phase 1：共享 MDP 参数化 + 配置导出工具

1. 完成 §8 的 4 处改动。
2. 扩展 `scripts/reinforcement_learning/rsl_rl/probe_task_cfg.py`：
   - 新增 `--dump-yaml PATH`；
   - 用 `env_cfg.to_dict()` + `yaml.dump` 导出解析后的 env cfg（对齐 `params/env.yaml` 的形态）。
3. **M20 黄金对照**：
   - 改前导出 `History-Anchor-Deeprobotics-M20-v0` → `m20_before.yaml`
   - 改后导出 → `m20_after.yaml`
   - `diff` 必须为空。
4. M20 2-iter smoke：

   ```
   conda run -n env_isaac_lab python scripts/reinforcement_learning/rsl_rl/train.py ^
     --task History-Anchor-Deeprobotics-M20-v0 --headless --num_envs 64 --max_iterations 2
   ```

   要求：正常启动、`check_policy_layout` 打印 83 / 86 / 700 / 54 / 16、无 NaN。

**门槛**：diff 为空 + smoke 通过，才允许进入 Phase 2。

### Phase 2：Profile + Scene + 骨架（只 import，不注册）

- `git mv` 目录改名；`assets/deeprobotics.py` 改符号名（同步 2 个 doc + 1 个 README）。
- 写 `profile.py`（数值来自 Phase 0）、`scene_cfg.py`、迁移 `stand_env_cfg.py`。
- 验证：

  ```
  conda run -n env_isaac_lab python scripts/tools/zero_agent.py --task Visualize-AS2W-PiperX-v0 --headless
  ```

  能起、能站 200 步不摔。

### Phase 3：完整低层配置 + 注册

- 写 `low_level/*` 与 `agents/*`，注册 3 个 task。
- `check_policy_layout` 必须打印：

  ```
  policy: 83   critic: 86   history: 10x70=700   privileged: 54   action: 16
  ```

### Phase 4：短程 smoke

- 64 envs × 2 iter：能起、无报错。
- 64–256 envs × 100–500 iter，观察：
  - 无 NaN / Inf
  - `base_height` 稳定（std < 2 cm）
  - 轮速不饱和
  - 臂关节力矩不长期顶满（joint4 饱和率 < 30%）
  - 终止率 < 20%
  - 开了 self-collision 的话无爆炸
- 若 `0.125/0.25/0.25` 导致腿抖，再调 scale；**不要**改奖励结构。

### Phase 5：Play / Eval / 导出

- `play.py --task History-Adaptation-AS2W-PiperX-play-v0`
- `eval_fixed_command.py`
- `export_deploy_policy.py`，产出 `policy_layout.json`，字段 `n=83 / history_length=10 / history_n=70 / action=16` 与训练一致。
- `probe_deploy_layout.py` 复核。
---

## 14. 验收标准

| 项 | 期望 |
|---|---|
| action terms | `joint_pos=12`, `joint_vel=4`, `ee_ik=0`, total **16** |
| policy obs | **83** |
| critic obs | **140**（86 + 54） |
| history | **10 × 70 = 700** |
| privileged | **54** |
| 维度与 M20 | 逐位相同（因为关节数同为 24） |
| M20 回归 | 共享 MDP 改动后 golden YAML diff 为空 + 2-iter smoke 通过 |
| 依赖边界 | `check_module_boundary.py` 通过 |
| 类名分支 | `check_no_class_name_branch.py` 通过 |
| 配置可审计 | AS2W 与 M20 的 YAML diff 只含 §10.1 / §11 列出的预期差异 |
| 部署导出 | `policy_layout.json` 与训练一致 |
| 文档 | 本文档 §11 表格填满实测值 |

---

## 15. 风险与坑

1. **关节数与 M20 相同是巧合，不是保证**。AS2W = 12 非轮腿 + 4 轮 + 6 臂 + 2 夹爪 = 24。若以后换臂，83/70/54 会变，必须重跑 layout 检查。
2. **`*_foot` 既是关节名也是 body 名**。`SceneEntityCfg(joint_names=".*_foot")` 和 `SceneEntityCfg(body_names=".*_foot")` 语义不同，别写混。
3. **`joint_pos_rel_without_wheel` 不删列**，只置零；所以宽度是 24 不是 20。
4. **`privileged_env_params_v4` 有硬性断言**：轮关节数必须等于轮 body 数，否则启动就报错。
5. **`joint_mirror_signed` 的缓存 key** 目前只有 `mirror_joints`，加 `joint_kind_patterns` 后必须一并写入，否则换机器人会命中错缓存。
6. **`HeightInvariantEECommand` 里 `find_bodies("gripper_base")` 是硬编码**；AS2W 恰好同名，若改夹爪名要同步。
7. **EE 碰撞盒坐标语义**：局部系，原点世界 z = `sampled_height`，机体在其下方 ⇒ z 为负。别照抄原 run 的 `[0.0, 0.5]`。
8. **`rel_standing_envs=0.02` 不是 `0.15`**：0.15 是课程终值，不是初值。
9. **不要启用分层速度采样**（`rel_zero_cmd_envs` 等）—— Anchor 没有用。
10. **`bad_orientation_2` 在 `mdp/events.py`**，不在 terminations。
11. **资产当前 `self_collision=False`、`enabled_self_collisions=False`**；M20 是 `True`。开启会显著影响计算量与稳定性，必须 A/B。
12. **轮半径 0.09 → 0.077 的差异不要用「整体下移高度区间」来抵消** —— 那是把一个单位换算藏进语义无关参数里，正是本文要消灭的问题。
13. **`UniformThresholdVelocityCommand` 的分层采样开关是「陷阱」**：它只在类名命中时生效，新配置里应当完全不写这些字段（保持默认 0），而不是写 0 显式覆盖。

---

## 16. 交付物

1. 新目录树（§5）+ `git mv` 后的旧目录消失。
2. `AS2W_PIPERX_CFG` 硬改名（源码 + 2 个 doc + 1 个 README）。
3. 4 处共享 MDP 参数化 + M20 golden diff 报告。
4. 2 个自动检查脚本。
5. `probe_as2w_geometry.py` + Phase 0 `report.md` / `summary.json`。
6. 3 个 task 注册 + `check_policy_layout` 输出日志。
7. smoke 日志（`logs/smoke/...`）。
8. `policy_layout.json`。
9. 一份不超过 30 行的说明，填在本文档末尾的「执行记录」小节。

---

## 附录 A：AS2W 资产事实（已实测，无需再查）

**关节（24，原生顺序：前→后、左→右）**

> 资产 docstring 写的是「24 DOF (16 leg + 4 wheel + 6 arm + 2 gripper)」，字面相加是 28，属于表述笔误：16 个 leg 关节里**已经包含** 4 个 `*_foot` 轮关节。真实总数是 24。Phase 0 第 1 项必须打印实际解析到的关节数来确认。

| 组 | 名字 |
|---|---|
| 腿（16，含轮） | `FL_hip, FL_thigh, FL_calf, FL_foot, FR_hip, …, RL_hip, …, RR_hip, …` |
| 臂（6） | `arm_joint1 … arm_joint6` |
| 夹爪（2） | `gripper_joint1, gripper_joint2` |

**关节限位 / 轴**

| 关节 | 限位 | 轴 |
|---|---|---|
| `*_hip` | `±0.785` | `(1,0,0)` |
| 前腿 `*_thigh` | `(-2.0, 3.492)` | `(0,1,0)` |
| 后腿 `*_thigh` | `(-1.3, 4.19)` | `(0,1,0)` |
| `*_calf` | `(-2.7677, -0.8133)` | `(0,1,0)` |
| `*_foot` | continuous | `(0,1,0)` |
| `arm_joint1..6` | `±2.618` / `[0,3.14]` / `[-2.9671,0]` / `±1.57` / `±1.57` / `±3.14` | `(0,0,1)` |
| `gripper_joint1/2` | `[0,0.05]` / `[-0.05,0]`（prismatic） | `(0,0,∓1)` |

**关节 frame**：所有腿关节 `origin.rpy = (0,0,0)`；左右腿只有 y 平移差（hip `±0.06`、thigh `±0.1054`）。

**默认站姿**：`hip = ±0.10`，`thigh = +0.44`，`calf = -0.88`，`foot = 0`；臂 `0 / 0.5 / -0.5 / 0 / 0 / 0`。

**几何**

| 项 | 值 |
|---|---|
| `base_link` 估算站高 | **≈0.4466 m**（URDF 注释 `≈0.449`，init 0.45） |
| 轮心高（相对 base） | **≈-0.379 m** ⇒ 等效轮半径 **≈0.077 m** |
| `arm_base_link` 相对 base | `z = +0.0798` |
| `base_link` 网格包围盒 | `x ∈ [-0.175, 0.335]`, `y ∈ ±0.113`, `z ∈ [-0.063, 0.135]` |
| `gripper_base` 网格 | `x ∈ [-0.105, 0.110]`, `y ∈ [-0.010, 0.085]`, `z ∈ [-0.033, 0.065]` |
| URDF 路径常量 | `AS2W_PIPERX_URDF_PATH` |

**actuator**

| 名 | 匹配 | effort / vel | k / d |
|---|---|---|---|
| `hip_thigh` | `.*_hip`, `.*_thigh` | 60 / 25 | 250 / 6 |
| `calf` | `.*_calf` | 90 / 16.67 | 250 / 6 |
| `wheel` | `.*_foot` | 15 / 98.47 | 0 / 0.6 |
| `piper_arm` | `arm_joint[1-6]` | 100 / 3 | 300 / 8 |
| `piper_gripper` | `gripper_joint[1-2]` | 10 / 1 | 286 / 5 |

---

## 附录 B：常用命令

```powershell
# 静态解析配置（不建 env，秒级）
conda run -n env_isaac_lab python scripts/reinforcement_learning/rsl_rl/probe_task_cfg.py `
  --task History-Adaptation-AS2W-PiperX-v0 --headless

# 视觉检查
conda run -n env_isaac_lab python scripts/tools/zero_agent.py --task Visualize-AS2W-PiperX-v0

# 短程 smoke
conda run -n env_isaac_lab python scripts/reinforcement_learning/rsl_rl/train.py `
  --task History-Adaptation-AS2W-PiperX-v0 --headless --num_envs 64 --max_iterations 2

# M20 回归 smoke
conda run -n env_isaac_lab python scripts/reinforcement_learning/rsl_rl/train.py `
  --task History-Anchor-Deeprobotics-M20-v0 --headless --num_envs 64 --max_iterations 2

# 整套 smoke 矩阵
conda run -n env_isaac_lab python scripts/reinforcement_learning/rsl_rl/smoke_regression.py --num_envs 64 --max_iterations 2

# 部署布局复核
conda run -n env_isaac_lab python scripts/reinforcement_learning/rsl_rl/probe_deploy_layout.py `
  --task History-Adaptation-AS2W-PiperX-play-v0
```

---

## 执行记录（由执行 session 填写）

- [ ] Phase 0 完成，`report.md` 路径：`______`
- [ ] Phase 1 完成，golden diff：`______`
- [ ] Phase 2 完成，stand 验证：`______`
- [ ] Phase 3 完成，layout 输出：`______`
- [ ] Phase 4 完成，smoke 日志：`______`
- [ ] Phase 5 完成，`policy_layout.json`：`______`