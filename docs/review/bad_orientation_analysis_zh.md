# `bad_orientation_2` 终止率暴涨的归因分析（含实测）

结论先行：**不是"IK 以前是错的"，而是"机械臂变得有能力把底盘顶翻"** ——
机器人资产、臂的执行器刚度、EE 参考系、以及一条新增的终止条件，都在这段时间里变了。

---

## 1. 现象：同一族任务，两代 run 差 20~100 倍

从各 run 的 tensorboard 里取末值（`Episode_Termination/bad_orientation_2`）：

| 实验 | run | 迭代 | `bad_orientation_2` | `time_out` | reward | 平均 episode 长度 |
|---|---|---|---|---|---|---|
| history_adaptation | 2026-06-26_16-53-49 | 19999 | **0.0060** | 0.9893 | 48.03 | 1000.0 |
| history_adaptation | 2026-06-29_17-18-16 | 19999 | **0.0074** | 0.9889 | 48.79 | 1000.0 |
| history_adaptation | 2026-06-30_22-18-34 | 19999 | **0.0149** | 0.9844 | 30.70 | 995.5 |
| history_adaptation | 2026-07-02_11-38-20 | 19999 | **0.0119** | 0.9881 | 40.24 | 993.9 |
| history_adaptation | 2026-07-09_09-54-35 | 19999 | **0.0327** | 0.9673 | 31.40 | 993.1 |
| history_adaptation | 2026-09-04_11-06-33 | 19999 | **0.7777** | 0.2146 | 5.92 | 429.1 |
| history_adaptation | 2026-09-18_19-33-47 | 7554 | **0.6244** | 0.3605 | 8.36 | 492.1 |
| deeprobotics_m20_wbc_flat | 2026-06-11_16-15-25 | 19999 | 0.0049 | 0.9903 | 48.03 | 990.2 |
| deeprobotics_m20_wbc_flat | 2026-09-18_01-31-58 | 6376 | 0.5889 | 0.3971 | 11.95 | 595.1 |

6~7 月那批：几乎不摔（0.6~3.3%），episode 顶满 1000 步，reward 30~49。
9 月起：62~78% 的 episode 以"倾角过大"结束，长度腰斩，reward 掉到 6~12。

**分界点在 2026-07-09 与 2026-09-04 之间。** 下面把这两代 run 的 `params/env.yaml`
展平成 `路径: 值` 做了完整 diff（脚本 `cfg_diff.py`）。

## 2. 配置差异（7 月 vs 9-04）

### 2.1 机械臂执行器：从"软的"变成"硬的"（最可疑）

| 项 | 7 月（好） | 9-04（差） |
|---|---|---|
| `piper_arm.class_type` | `ImplicitActuatorCfg` | **`DelayedPDActuatorCfg`** |
| `piper_arm.stiffness` | 40.0（但 Implicit 下**不生效**，用 USD 默认） | **300.0** |
| `piper_arm.damping` | 8.0（同上） | **20.0** |
| `piper_arm.friction / armature` | 0.01 / 0.01 | 0.01 / 0.01 |
| `piper_gripper.class_type` | Implicit | DelayedPD |
| `piper_gripper.armature / friction` | 0.0 / 0.0 | 0.01 / 0.01 |

当前 `assets/deeprobotics.py` 里 `piper_arm` 是 `DelayedPDActuatorCfg(stiffness=300, damping=20)`，
而**你自己在旁边注释里写的目标区间是"DelayedPD 在 60~100（stiffness）/ 0~20（damping）"**——
300 已经超出那个区间 3~5 倍。

### 2.2 机器人资产换了

| 项 | 7 月 | 9-04 |
|---|---|---|
| `scene.robot.spawn.usd_path` | `deep_robotics_model/M20/M20_usd/**M20_adjusted.usd**` | `deep_robotics_model/M20_Piper_own/usd/**M20_Piper_own.usd**` |
| `commands.ee_pose.body_name` / `actions.ee_ik.body_name` | `arm_link6` | `gripper_base` |
| `actions.ee_ik.body_offset.pos.z` | **0.135** | 0.0 |
| `commands.ee_pose.arm_base_link_name` | `arm_base`（该 body 在现 USD 里已不存在） | `arm_base_link` |
| 夹爪关节名 | `arm_joint[7-8]` | `gripper_joint[1-2]` |
| `scene.robot.init_state.joint_pos.arm_joint2` | **3.0** | **0.5** |

也就是说整条臂/夹爪的装配体、命名、默认姿态都换了（`M20_Piper_own` 是你自己装的），
质量和惯量分布随之改变。

### 2.3 新增了一条终止条件，压掉了稳定裕度

| 项 | 7 月 | 9-04 |
|---|---|---|
| `terminations.root_height_below_minimum` | **None（禁用）** | 启用，`minimum_height = 0.3` |
| `commands.body_pose.height_range` | (0.33, 0.6) | (0.33, 0.6)（不变） |

高度命令下界 0.33 m 与终止阈值 0.30 m **只差 3 cm** —— "认真跟踪高度命令"本身就在终止边界上。

### 2.4 EE 目标范围反而**变窄**了（所以不是"任务变难"）

| 项 | 7 月 | 9-04 |
|---|---|---|
| `commands.ee_pose.ranges.p_l` | (0.4, 0.7) | **(0.3, 0.52)** |
| `p_pitch` | (-1.0, 1.257) | (-0.785, 0.628) |
| `p_yaw` | ±1.885 | ±1.257 |
| `o_roll` / `o_pitch` | ±0.785 | **±0.393** |
| `commands.*.debug_vis` | True | False |

9 月的 EE 指令范围比 7 月**更小**，但目标半径更贴近机身（0.3~0.52 m），臂长期处于"折叠/近身"姿态。

## 3. 实测：臂的运动会怎样影响底盘（两个 rollout 实验）

用**已训好的 6300 iter 低层策略**（`deeprobotics_m20_wbc_flat/2026-09-18_01-31-58/exported/policy.pt`）
在 `Flat-Deeprobotics-M20-Piper-WBC-v0` 上做 rollout，直接自己算倾角
`tilt = acos(-g_z)` 与阈值 0.5 rad(28.6°) 比较（脚本 `rollout_tilt.py`）：

| 条件 | 倾角>阈值比例 | 平均倾角 | 峰值倾角 | **平均臂关节速度** | 底盘角速度 RMS |
|---|---|---|---|---|---|
| 硬臂 300/20（当前） | 0.0006 | 2.01° | 29.73° | **1.74 rad/s** | 0.644 |
| 软臂 40/8（=7 月那代） | **0.0002** | 1.89° | 29.52° | **0.95 rad/s** | 0.683 |

**结论**：

1. **任务本身是可学的** —— 已训策略下倾角越限比例只有 0.02~0.06%，平均倾角约 2°。
   所以问题出在**学习过程**（训练早期/中期一直处在"一倾斜就终止"的区间），
   不是"物理上做不到"。
2. 换成软臂后，同一个策略的臂关节速度**降低 45%**，倾角越限比例**降 3 倍**（0.0006 → 0.0002）。
   说明**臂的刚度直接决定它把多少扰动传给底盘**；软臂 ≈ 被动顺从/阻尼。
   （底盘角速度 RMS 反而略升：软臂"吸收"扰动但不"支撑"，所以底盘更晃但不容易翻。）

## 4. 对你原假设的判定

> "之前机械臂的 DifferentialIK 一直是错的，所以训出来的模型不会太受机械臂的影响"

* **不成立的部分**：那批"好"的 run（6~7 月）用的**就是** `CommandDrivenIKAction`
  （env.yaml 里 `class_type: ...mdp.actions:CommandDrivenIKAction`、`command_name: ee_pose`），
  而且 EE 目标课程是**开着的**、范围比现在还大。所以不是"IK 坏了所以臂不动"。
* **成立的部分（也就是真正的机制）**：**臂对底盘的"扰动能力"发生了量级变化**：
  ① 资产从 `M20_adjusted` 换成 `M20_Piper_own`（装配体/惯量不同、默认 `arm_joint2` 3.0→0.5）；
  ② 执行器从 Implicit(40/8) 换成 DelayedPD(**300/20**)，臂的跟踪速度实测 1.8 倍；
  ③ EE 参考系从 `arm_link6 + 0.135 m` 变成 `gripper_base + 0`，目标半径收到 0.3~0.52 m（更贴机身）；
  ④ 新增 `root_height_below_minimum = 0.3`（而高度命令下界 0.33）。
  这几项一起把"稳定裕度"吃掉了：`body_pose` 课程要求 pitch ±20°/roll ±14°，
  叠加起来的名义倾角已 24.5°，离 `bad_orientation_2` 的 30° 只剩 5.5° 余量（见 known_issues #19）。

## 5. 解决方案（按优先级）

### A. EE 目标课程（你的思路，推荐先做）

分阶段放开 `commands.ee_pose.ranges`，用仓库已有的
`mdp.modify_term_cfg` + `mdp.override_value`（`WBCCurriculumCfg` 里
`body_pose_height_range_s2` / `body_pose_pitch_range_s3` 就是这个写法）：

| 阶段 | 放开什么 | 目的 |
|---|---|---|
| **s0** | EE 目标固定在**默认位姿下的 EE**（`p_l`=默认半径、`p_pitch/p_yaw`=0、`o_*`=0） | 臂几乎不动，先让底盘学会站立/走动/姿态跟踪 |
| s1 | 位置半径（默认半径 → 目标区间），姿态仍固定 | 适应"臂在近身处移动" |
| s2 | 姿态 ±10° | 适应末端姿态变化 |
| s3 | 全范围（`p_l` 0.3~0.52 / `o_*` ±22.5°） | 完整任务 |

⚠️ 落地前要先测一个常数：**默认位姿在 height-invariant 坐标系下的半径**
（`HeightInvariantEECommandCfg.ranges.p_l` 的起点必须 ≤ 它，否则 s0 无法表达"默认位姿"）。
做法：在 env 里 reset 后打印 `HeightInvariantEECommand.pose_end_cart` / 或直接按
`sampled_height=0.6` + `arm_base_link` 的坐标系算一次。**这件事留到下次 session 第一步做。**

### B. 姿态插值（低成本、建议一起做）

`velocity/mdp/commands.py::HeightInvariantEECommand._update_command`：
位置按 `T_traj` 线性插值，但**姿态直接取终点四元数** —— 每次重采样（5 s）给机械臂一个
**瞬时**姿态参考跳变 → 关节速度/力矩尖峰 → 底盘冲击。
改法：对姿态做 slerp（`math_utils.quat_slerp`，或先只做"按 T_traj 的比例 slerp"）。

### C. 执行器刚度课程（可选，A 不够时再加）

把 `piper_arm` 的 stiffness/damping 作为**终值 300/20**，训练前段用 40/8，
后期线性升到终值；顺带检查 `piper_gripper` 的 4000/200（夹爪刚度极高，闭合冲击也不小）。
理由：7 月那代实测更"顺从"（臂速 0.95 vs 1.74 rad/s）；
这样底盘能先把"软臂"下的平衡学会，再逐步适应硬臂。
实现：新增一个 CurrTerm 直接改 `robot.actuators["piper_arm"].stiffness`
并调 `robot.write_joint_stiffness_to_sim(...)`（`assets/deeprobotics.py` 里那个注释
"DelayedPD 在 60~100" 也说明 300 偏高，可以顺便定标）。

### D. 两处稳定裕度小修（顺手）

1. `body_pose.height_range` 下界 0.33 与 `root_height_below_minimum=0.3` 只差 3 cm：
   要么把高度命令下界抬到 0.36，要么把终止阈值降到 0.26。
2. `bad_orientation_2` 阈值已在 `codex/ll-history-flat-eegoal`（`45f9e74`）改成
   旋转不変的 0.8 rad(45.8°)，与 A 的 s0 一起用效果最好。

### E. 验收指标（每阶段都要看）

| 指标 | 期望 |
|---|---|
| `Episode_Termination/bad_orientation_2` | s0 阶段应接近 0（7 月那代 0.6~3.3%）；每次放开后允许短时上升，但要能回落 |
| `Train/mean_episode_length` | 不超过 1000（=20 s 上限）时应逐步逼近上限 |
| `Metrics/base_velocity/error_vel_xy` | 应随迭代**下降**（7500 那个 run 是反向上升的，这是退步信号） |
| `Policy/mean_noise_std` | 不应无限上涨（7500 那个 run 涨到 1.53） |

> 还没做的对照实验（下次可选，约 30 min/组）：用同一份代码跑
> "EE 课程开 / EE 目标固定默认位姿" 两组各 300 iter 的训练，直接对比
> `bad_orientation_2` 曲线 —— 这是对 A 方案最直接的验证。
