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

> ## ✅ 已实现（2026-09-19，分支 `codex/ll-ee-goal-curriculum`，代码 commit `9ccb8ec`）
>
> 先用新探针 `scripts/reinforcement_learning/rsl_rl/probe_ee_default_pose.py`
> 测了"默认位姿在 height-invariant 坐标系下的球坐标"（`Flat-Deeprobotics-M20-Piper-WBC-v0`，
> 8 envs，reset 后未 step）：
>
> | 量 | 实测 | 备注 |
> |---|---|---|
> | 半径 `r0` | **0.4035 ± 0.0328**（0.361~0.461） | ✅ 落在当前 `p_l=(0.3,0.52)` 之内 —— 你不必担心"p_l 起点大于默认半径" |
> | 仰角 `pitch0` | **+1.2615 ± 0.1321 rad（+72.3°±7.6°）** | ⚠️ 当前 `p_pitch` 上界只有 **+0.628 rad（+36°）** |
> | 方位 `yaw0` | **+0.1739 ± 0.7335 rad（+10.0°±42.0°）** | 逐环境差异大 |
> | 目标坐标系原点 | z = 0.6000（= `sampled_height`），arm_base_link z = 0.6001，EE z = **0.9219** | 默认 EE 比采样平面**高 0.32 m** |
> | `o_*=(0,0)` 的姿态 vs 默认姿态 | 差 **68.5°±0.6°** | `o_*=(0,0)` 是"局部 +z 对齐位置方向"，不等于默认姿态 |
>
> 两条重要推论（都改变了原方案）：
>
> 1. **即使到了 s3，机械臂也必须比默认位姿低至少 36°**（默认仰角 +72.3° vs
>    `p_pitch` 上界 +36°）—— 这就是"臂被拽下来"的那段运动，也是 9 月与 7 月
>    行为差异的一部分（7 月 `p_l` 是 0.4~0.7、仰角范围也不同）。
> 2. **"把 `p_l/p_pitch/p_yaw` 收成常数点"表达不了"每个环境各自的默认位姿"**：
>    方位的逐环境标准差有 42°（主要来自复位时 root 的 roll/pitch 随机化 ——
>    这个坐标系只保留 yaw），而 `o_*=(0,0)` 的姿态与默认姿态差 68.5°。
>    所以 s0 若按"常数区间"写，臂在 s0 一开始仍会被要求转 ~68° 的手腕。
>
> **最终实现方式（等价目标、不需要任何常数）：给命令项加"混合比例"开关**
>
> `HeightInvariantEECommandCfg` 新增 `target_blend_pos` / `target_blend_orn`（默认 1.0）：
> 每次重采样时，把**采样出来的目标**与**这一刻真实的 EE 位姿**做混合
> （位置线性混合、姿态 slerp）：
>
> * `blend = 0` ⇒ 目标 = 当前位姿（reset 后即**默认位姿**）⇒ 机械臂不需要移动；
> * `blend = 1` ⇒ 完全采用采样目标 = **原来的行为**（s3 与原训练分布完全一致，无分布偏移）；
> * 中间值 ⇒ 目标沿直线/slerp 被拉回当前位姿，臂的移动幅度随之增长。
>
> 课程阶段（`WBCCurriculumCfg`，写法与 `body_pose_*_s2/s3` 完全一致）：
>
> | 阶段 | 触发步数 | `target_blend_pos` | `target_blend_orn` | 含义 |
> |---|---|---|---|---|
> | **s0** | 初始（`Flat/RoughEnvWBCConfig.__post_init__`） | **0.0** | **0.0** | 目标=默认位姿，臂不动 |
> | s1 | 25k | **0.35** | 0.0 | 臂在小范围内移动（位置放开 35%） |
> | s2 | 50k | 0.35 | **0.35** | 姿态再放开 35% ≈ 最终姿态范围 ±22.5° 的 1/3 ≈ **±8°**（接近原设想的 ±10°） |
> | s3 | 75k | **1.0** | **1.0** | 完整任务 |
>
> `FlatEnvWBCConfig_PLAY` / `RoughEnvWBCConfig_PLAY` 里关掉这 4 个课程项并把 blend 置 1.0。
>
> 实测（`probe_ee_curriculum.py`，`Flat-Deeprobotics-M20-Piper-WBC-v0`，16 envs）：
>
> * **课程阶段**：把 `common_step_counter` 设为 0 / 25001 / 50001 / 75001 后
>   `curriculum_manager.compute()` 得到 blend = (0,0) / (0.35,0) / (0.35,0.35) / (1,1)，全部符合预期。
> * **臂扰动机制 A/B**（同样零动作，只改 blend；已关掉外力/推力事件、并剔除复位瞬态）：
>
> | 条件 | 指标 | s0(blend=0) | s3(blend=1) |
> |---|---|---|---|
> | 站立（速度命令恒 0） | 倾角 p99 / 最大倾角 | **7.18° / 13.65°** | **35.01° / 45.40°** |
> | 站立 | 臂关节速度 RMS / 底盘角速度 RMS | 2.303 / 0.341 | 2.604 / 0.376 |
> | 速度命令 ±1 m/s | 倾角 p99 / 最大倾角 | **2.90° / 3.36°** | **6.52° / 36.77°** |
> | 速度命令 ±1 m/s | EE 位置跟踪误差 | 0.167 m | 0.091 m |
>
>   即"臂跟着 EE 目标动"这一个因素就能把最大倾角从 3.4° 拉到 36.8°（超过
>   `bad_orientation_2` 旧阈值 30°）；s0 把它按住后倾角回到个位数 —— 与 §3 的归因一致。
> * **训练冒烟**：`History-Adaptation-Deeprobotics-M20-v0 --num_envs 64 --max_iterations 2`
>   **EXIT=0**，`Episode_Termination/bad_orientation_2 = 0.0000`，
>   `env.yaml` 里 `target_blend_pos/target_blend_orn = 0.0`、`limit_angle = 0.8`、
>   `ee_goal` 非 null 全部生效。
>
> 未做（可选，见 §C）：执行器刚度课程；"EE 课程开 vs 目标固定"的 300 iter 对照实验。
> 需要的话可以按"`codex/ll-ee-goal-curriculum` 上把
> `--hydra` 覆盖 `env.commands.ee_pose.target_blend_pos=1.0`"来跑对照组（见 §E）。

### B. 姿态插值（低成本、建议一起做）

`velocity/mdp/commands.py::HeightInvariantEECommand._update_command`：
位置按 `T_traj` 线性插值，但**姿态直接取终点四元数** —— 每次重采样（5 s）给机械臂一个
**瞬时**姿态参考跳变 → 关节速度/力矩尖峰 → 底盘冲击。
改法：对姿态做 slerp（`math_utils.quat_slerp`，或先只做"按 T_traj 的比例 slerp"）。

> ## ✅ 已实现（同一分支 `codex/ll-ee-goal-curriculum`，commit `9ccb8ec`）
>
> `_update_command()` 的姿态改成 **slerp**（`pose_start_b → pose_end_b`，按 `T_traj` 比例），
> 位置仍是线性插值。
>
> ⚠️ 实现细节：**不能直接用** `isaaclab.utils.math.quat_slerp`：
>
> * 它用 `torch.dot` + `if tau == 0.0` 判断，**只支持单个四元数**（命令项是 `(N,4)`）；
> * 它内部 `q2 *= -1.0` 是**就地修改**（探针里已实测确认：传进去的 q2 确实被翻转了），
>   直接把 `pose_end_b` 传进去会被改符号。
>
> 因此新增批量版 `quat_slerp_batch()`（`velocity/mdp/commands.py`）：最短路径 + 无副作用 +
> 退化情形（近平行）走 lerp 再归一化。
>
> 实测（`probe_ee_curriculum.py`）：
>
> * 与官方单样本 `quat_slerp` 的**最大分量偏差 1.19e-07**（含 τ=0/1 端点与 q0==q1 退化）；
> * 受控测试（`T_traj=1 s`、`step_dt=0.02 s`、起止夹角 155.7°）：
>   **重采样瞬间的单步姿态跳变 155.74° → 3.12°**（≈ 155.7/50，正好是 `angle/T_traj*dt` 的量级）；
> * 新旧实现的差别在真实 rollout 里表现为"每次重采样给机械臂一个瞬时姿态参考跳变"，
>   现在变成沿最短路径平滑过渡。

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

> 现状（2026-09-19）：
>
> * 第 2 条已生效：`codex/ll-ee-goal-curriculum` cherry-pick 了 `d445007`，
>   `params/env.yaml` 里可见 `limit_angle: 0.8`。**实测：s0 阶段
>   `Episode_Termination/bad_orientation_2 = 0.0000`。**
> * 第 1 条**本次未改**（不在本次任务清单里，且它在实测中只占 ~2% 的终止；
>   s0 阶段高度命令被锁在 0.513，更几乎不触发）。若要改，推荐把终止阈值降到
>   **0.26**（比抬高命令下界更保守：不改任务分布）。

### E. 验收指标（每阶段都要看）

| 指标 | 期望 | 2026-09-19 实测 |
|---|---|---|
| `Episode_Termination/bad_orientation_2` | s0 阶段应接近 0（7 月那代 0.6~3.3%）；每次放开后允许短时上升，但要能回落 | **0.0000**（2 iter 冒烟）；机制探针里 s0 最大倾角 3.4° vs s3 36.8° |
| `Train/mean_episode_length` | 不超过 1000（=20 s 上限）时应逐步逼近上限 | 需长期 run |
| `Metrics/base_velocity/error_vel_xy` | 应随迭代**下降**（7500 那个 run 是反向上升的，这是退步信号） | 需长期 run |
| `Policy/mean_noise_std` | 不应无限上涨（7500 那个 run 涨到 1.53） | 需长期 run（2 iter 时 1.00） |

> 后三项只有在真实训练里才能下结论。可选的对照实验（~30 min/组）：
> "EE 课程开（默认）" vs "EE 目标固定默认位姿（关掉 s1~s3）"：
>
> ```bash
> # 课程开（本次实现，s0 全程有效）
> python scripts/reinforcement_learning/rsl_rl/train.py \
>     --task History-Adaptation-Deeprobotics-M20-v0 --headless --num_envs 64 --max_iterations 300
> # 对照组：把混合比例覆盖成 1.0（= 恢复"一上来就是完整任务"）
> python scripts/reinforcement_learning/rsl_rl/train.py \
>     --task History-Adaptation-Deeprobotics-M20-v0 --headless --num_envs 64 --max_iterations 300 \
>     env.commands.ee_pose.target_blend_pos=1.0 env.commands.ee_pose.target_blend_orn=1.0
> ```
>
> 300 iter = 7.2k 步 < 25k，所以整个 run 都停在 s0，正好对比
> "臂被按住" vs "臂一路跟着随机目标动"。

### ✅ 对照实验实测（2026-09-19，`codex/ll-ee-goal-curriculum`）

同一份代码、同一 seed、`--num_envs 64 --max_iterations 120`，只改 EE 目标混合比例：

| 指标（iter = 119） | 课程开（s0，blend=0） | 对照（blend=1，= 原行为） |
|---|---|---|
| `Episode_Termination/bad_orientation_2` | **0.594** | **0.883** |
| 同上（iter=100） | **0.521** | **0.936** |
| `Metrics/base_velocity/error_vel_xy` | 0.057 | 0.062 |
| `Policy/mean_noise_std` | 1.003 | 1.006 |
| `Train/mean_episode_length` | 30.7 | 33.0 |
| `Train/mean_reward` | −0.728 | −0.685 |

（两个 run：`logs/rsl_rl/history_adaptation/2026-09-19_14-16-49` 与 `2026-09-19_14-27-24`，两者都 EXIT=0。）

**怎么读这组数**：

1. EE 目标课程**确实把早期终止率压下来了**（119 轮 0.594 vs 0.883；100 轮 0.521 vs 0.936），
   方向与机制探针一致（"臂跟着随机目标动"能把最大倾角从 3.4° 拉到 36.8°）。
2. 但**没有**回到 7 月那代的 ~1% 量级：两个 run 的 episode 只有 ~30 步，
   说明"还没学会走"本身就在不停摔 —— s0 只拿掉了**臂**这一份扰动，
   拿不掉"腿还没学会"。7 月那代的 0.6~3.3% 是"软臂 + 旧资产"的整体结果。
3. 这组是 64 envs 的短跑（120×24 ≈ 2.9k 步/环境）；同代码在 4096 envs 下早期
   `bad_orientation_2` 只有 0.37~0.50（用户那份 `2026-09-19_09-02-50`），
   所以**绝对数值不能跨 env 数比较**，这里只看 A/B 的相对差。
4. 因此若要继续压这个终止项，建议先做 **⑤C 的执行器刚度课程**（40/8 → 300/20，
   这是 7 月那代最直接的差别），其次是把 `root_height_below_minimum` 从 0.30 降到 0.26。
