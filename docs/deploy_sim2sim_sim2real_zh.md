# 部署参考：sim2sim（MuJoCo）→ sim2real

**用途**：在**另一台电脑**上写/改部署脚本时的对照手册。策略本体已经训练好并导出，
但"策略能跑起来"取决于**接口复刻得对不对** —— 本文把接口逐项写死，并列出必须自己实现的部分
（最容易被漏掉的是**机械臂 IK**，见第 6 节）。

**别照抄本文的表**：在部署机上先跑一次探针，拿**实测输出**当权威（同一份代码在不同 commit 上
数值可能变过）：

```bash
python scripts/reinforcement_learning/rsl_rl/probe_deploy_layout.py \
    --task History-Adaptation-Deeprobotics-M20-v0 --headless --num_envs 2
```

本文里的数字都是这台机器上 `main`（2026-09-20）跑该探针的实测值。

---

## 0. 一眼速查

| 量 | 值 | 备注 |
|---|---|---|
| 策略频率 | **50 Hz**（`sim.dt=0.005` × `decimation=4`，周期 20 ms） | 每 4 个物理步跑一次策略 |
| 策略输入 | `policy_obs (batch,83)` + `history_flat (batch,700)` | ONNX / TorchScript 同接口 |
| 策略输出 | `action (batch,16)` | 12 腿（位置）+ 4 轮（速度）；**不含机械臂** |
| 腿目标 | `q_des = q_default + gain * a`，hipx **0.125**、hipy/knee **0.25** | 实测（探针 [2b]） |
| 轮目标 | `ω_des = 5.0 * a`（力矩 = `0.6*(ω_des-ω)`） | 轮子是速度伺服 |
| 机械臂 | 由 **IK** 从 `ee_pose` 命令驱动（不占动作维度） | 必须自己实现，见第 6 节 |
| history | 10 步 × 70 维，**最旧→最新** | 每步 70 维组成见第 4 节 |
| 训练终止阈值 | 倾角 > 0.8 rad(45.8°)、root 高度 < 0.30 m | 真机上当作**保护阈值**用 |
| 当前 run | `2026-09-20_00-50-31`，checkpoint `model_19999.pt` | 导出于 `<run>/exported_deploy/` |

---

## 1. 部署产物与"哪个文件能用"

`logs/rsl_rl/history_adaptation/2026-09-20_00-50-31/exported_deploy/`

| 文件 | 用途 |
|---|---|
| `policy.onnx` | **sim2sim / sim2real 用这个**。opset 17，batch 维动态，输入名 `policy_obs` / `history_flat`，输出名 `action` |
| `policy.pt` | TorchScript，接口同上（高层 replay 用；也可以拿它做数值对照） |
| `policy_layout.json` | 输入维度/顺序、history 语义、ONNX 校验结论 —— **部署脚本应该读它做断言**，别把维度写死 |

⚠️ **陷阱**：同 run 下的 `<run>/exported/policy.pt` 是 `play.py` 导出的 **actor-only** 版本
（输入 115 = 83 + 32 latent，那 32 维 latent 没有来源），拿去部署会静默算错。
导出脚本默认已经写到 `exported_deploy/`，并在遇到这种目录时打印警告。

数值一致性（导出时自检 + 独立复核）：TorchScript vs eager `0.000e+00`；
ONNX vs TorchScript 相对误差 `1.9e-07`（fp32 舍入量级）。

---

## 2. 时间与频率

- `sim.dt = 0.005 s`，`decimation = 4` → **策略 50 Hz**，每 4 个物理步算一次；
  中间 3 个物理步**重复上一次的关节目标**（IsaacLab 的 action manager 语义：
  策略输出在 decimation 步内保持不变，PD 仍然每个物理步执行）。
- IK 也在 50 Hz 更新一次（和策略同频）。
- 训练 episode = 20 s（部署时不要重置，只在人为干预时重置）。
- MuJoCo 侧注意：仓库里 MJCF 自带的 `timestep` 是 **0.002**，Isaac 是 **0.005**。
  要么把它改成 0.005，要么保持 0.002 但保证**策略周期仍是 20 ms**（= 10 个物理步）——
  两种都行，但别让"物理步变细"顺带把控制周期也改了。

---

## 3. 动作 → 关节（最需要照抄的一节）

16 维动作的**分块与顺序**由 action term 的 `joint_names` 决定（不是 articulation 原生序）：

| 动作槽 | 关节 | 目标 | 实测增益 |
|---|---|---|---|
| `a[0]` | `fl_hipx_joint` | 位置 `q_default + gain*a` | **0.125** |
| `a[1]` | `fl_hipy_joint` | 位置 | **0.25** |
| `a[2]` | `fl_knee_joint` | 位置 | **0.25** |
| `a[3..5]` | `fr_hipx / fr_hipy / fr_knee` | 位置 | 0.125 / 0.25 / 0.25 |
| `a[6..8]` | `hl_hipx / hl_hipy / hl_knee` | 位置 | 0.125 / 0.25 / 0.25 |
| `a[9..11]` | `hr_hipx / hr_hipy / hr_knee` | 位置 | 0.125 / 0.25 / 0.25 |
| `a[12..15]` | `fl_wheel / fr_wheel / hl_wheel / hr_wheel` | 速度 `ω = 5.0*a` | **5.0** |

即：**只有 hipx 是 0.125，其余腿关节都是 0.25**（`{".*_hipx_joint": 0.125, "(其它非臂关节)": 0.25}`）。
别从 IsaacLab 内部的 `_scale` 张量去推断（那张量的下标约定和动作槽位不是一回事，
本仓库的探针就是靠"把动作置 1.0 看关节目标"才测准的）。

**默认关节角**（`use_default_offset=True` ⇒ 动作 0 时回到这里；也是"零动作站立"的姿态）：

| 关节 | 默认角(rad) | | 关节 | 默认角(rad) |
|---|---|---|---|---|
| `f[l,r]_hipx_joint` | 0.0 | | `fl/fr_knee_joint` | +1.0 |
| `f[l,r]_hipy_joint` | **-0.6** | | `hl/hr_knee_joint` | **-1.0** |
| `h[l,r]_hipx_joint` | 0.0 | | `*_wheel_joint` | 0.0 |
| `h[l,r]_hipy_joint` | **+0.6** | | `arm_joint2 / arm_joint3` | **+0.5 / -0.5**（其余臂关节 0.0，夹爪 0.0） |

⚠️ **后腿的 hipy/knee 符号与前腿相反**（+0.6 / -1.0），这不是笔误 ——
URDF/`init_state` 就是这么定的，照抄即可。

**关节限位**（Isaac 侧软限位 = 以硬限位中点为心、**半宽 ×0.9**；例如 `hipx` 硬限位
[-0.436, 0.611] → 软限位 [-0.384, 0.559]。轮子无位置限位）：

| 关节组 | 硬限位(rad) | 力矩限幅 | 速度限幅 |
|---|---|---|---|
| `*_hipx_joint` | 前腿 [-0.436, 0.611]／后腿 [-0.611, 0.436] | 76.4 N·m | 22.4 rad/s |
| `*_hipy_joint` | 前腿 [-2.583, 2.286]／后腿 [-2.286, 2.583] | 76.4 N·m | 22.4 rad/s |
| `*_knee_joint` | 前腿 [-2.792, 2.809]／后腿 [-2.809, 2.792] | 76.4 N·m | 22.4 rad/s |
| `*_wheel_joint` | 连续（无位置限位） | 21.6 N·m | 79.3 rad/s |
| `arm_joint1..6` | 见第 6 节 | 100 N·m | 3 rad/s |
| `gripper_joint1/2` | [0, 0.035] / [-0.035, 0] | 10 N·m | 1 rad/s |

**动作里没有机械臂**：`ee_ik` 这个 action term 的 `action_dim = 0`
（IK 由 CommandManager 驱动）。所以 16 维动作里没有任何一个自由度对应 `arm_joint*` 和夹爪。

---

## 4. 观测：`policy_obs`（83 维）与 `history`（10×70）

**处理顺序**（IsaacLab `observation_manager`）：`compute → 加噪声 → clip → 乘 scale`。
部署时**不加噪声**，但 scale 和 clip 必须照做，顺序也要对（先 clip 后 scale）。

### 4.1 `policy_obs` 83 维

| 切片 | 项 | 维度 | 公式 | clip | scale |
|---|---|---|---|---|---|
| 0..2 | `base_ang_vel` | 3 | 机体角速度（**base 系**，IMU 直接给） | ±100 | **0.25** |
| 3..5 | `projected_gravity` | 3 | 重力方向在 base 系 = `R_b^T · [0,0,-1]` | ±100 | 1.0 |
| 6..8 | `velocity_commands` | 3 | 底盘命令 `[vx, vy, wz]`（base 系） | ±100 | 1.0 |
| 9..32 | `joint_pos` | 24 | `q − q_default`（**轮子列置零**，原生序） | ±100 | 1.0 |
| 33..56 | `joint_vel` | 24 | `q̇`（原生序，**不**置零） | ±100 | **0.05** |
| 57..72 | `actions` | 16 | **上一步的策略输出**（原始 16 维，不是换算后的关节目标） | ±100 | 1.0 |
| 73..79 | `ee_goal` | 7 | `[ee_pos_b(3), ee_quat_b(4) wxyz]`（root 系目标，见第 5 节） | **±3** | 1.0 |
| 80..82 | `body_pose_cmd` | 3 | `[height, pitch, roll]`（见第 5 节） | — | 1.0 |

训练时叠加的噪声量级（= 策略被训练成能容忍的传感器误差，**部署时不要加**，
但真机噪声不应明显超过它）：`base_ang_vel ±0.2`、`projected_gravity ±0.05`、
`joint_pos ±0.01`、`joint_vel ±1.5`、`ee_goal ±0.05`。

⚠️ **`joint_pos` / `joint_vel` 是 articulation 原生顺序，而且和 MuJoCo 的关节顺序不同**：

| 原生 idx | 关节 | 原生 idx | 关节 |
|---|---|---|---|
| 0 | `fl_hipx_joint` | 12 | `hl_knee_joint` |
| 1 | `fr_hipx_joint` | 13 | `hr_knee_joint` |
| 2 | `hl_hipx_joint` | 14 | `arm_joint3` |
| 3 | `hr_hipx_joint` | 15 | `fl_wheel_joint` |
| 4 | `arm_joint1` | 16 | `fr_wheel_joint` |
| 5 | `fl_hipy_joint` | 17 | `hl_wheel_joint` |
| 6 | `fr_hipy_joint` | 18 | `hr_wheel_joint` |
| 7 | `hl_hipy_joint` | 19 | `arm_joint4` |
| 8 | `hr_hipy_joint` | 20 | `arm_joint5` |
| 9 | `arm_joint2` | 21 | `arm_joint6` |
| 10 | `fl_knee_joint` | 22 | `gripper_joint1` |
| 11 | `fr_knee_joint` | 23 | `gripper_joint2` |

**这张表是本项目历史上最容易出错的地方**（`known_issues` #1 / `DEFECT_LOG_zh.md` DEF-016：
曾经把"列的下标"当"原生 id"用，静默清掉了 `hr_wheel + arm_joint1/2/3`）。
部署侧务必**按关节名建映射**，不要按位置硬编。

### 4.2 `history_flat` 700 维（10 步 × 70 维）

每步 70 维 = `[base_ang_vel(3), projected_gravity(3), joint_pos(24), joint_vel(24), last_action(16)]`

- **不加 scale、不加噪声**（只 clip ±100）——注意这与 `policy_obs` 里同名的项**不一样**：
  history 用的是**原始值**（`base_ang_vel` 没乘 0.25、`joint_vel` 没乘 0.05）。
- `joint_pos` 是**原始** `q − q_default` **包括轮子**（不置零），仍是原生序 24 维。
- 展平顺序是 **最旧 → 最新**（`history_length=10`）。
- 推窗节奏 = **策略周期**（每 20 ms 推一帧），不是每个物理步。
- **复位后第一帧**：用同一帧填满整窗（不要用 0 填或让窗口空着）。

---

## 5. 三个命令：部署时要自己生成

| 命令 | 维度 | 定义 | 训练终值（s3，20k iter 已跑满） |
|---|---|---|---|
| `base_velocity` | 3 | `[vx, vy, wz]`（base 系，m/s、rad/s） | vx **(-5, 5)**、vy **(-1, 1)**、wz **(-1, 1)** |
| `body_pose` | 3 | `[height, pitch, roll]` | height **(0.33, 0.55) m**、pitch **±0.35 rad**、roll **±0.25 rad** |
| `ee_pose` | 7 | `[pos_b(3), quat_b(4) wxyz]`（**root 系**）—— 同时喂给观测 `ee_goal` 和 IK | 见下 |

**`body_pose.height` 的度量**（别用 root 的绝对 z）：

```
height = root_z − mean(四个 wheel body 的 z) + wheel_radius(0.09 m)
```

（代码：`mdp/utils.py::compute_base_height_rel_to_feet`；`base_height_l2` 的目标 0.513 就是这个量。）
默认姿态下（MuJoCo 实测）足端 z 相同 → `0.55 − 0.1134 + 0.09 = 0.5266`，考虑接触压缩后实测 ~0.51。

**`ee_pose` 的训练采样方式是球坐标**（部署时你多半直接给"任务想要的 EE 位姿"，不一定要复刻采样器，
但要知道边界）：在"height-invariant 坐标系"（`arm_base_link` 的 xy + 固定 z = 0.6 平面）里

- 半径 `p_l ∈ (0.30, 0.52)` m、仰角 `p_pitch ∈ (-0.785, 0.628)` rad、方位 `p_yaw ∈ (±1.257)` rad；
- 姿态 `o_roll/o_pitch ∈ ±0.3927`、`o_yaw ∈ ±π`；
- 每条命令在 `T_traj ∈ (1, 3)` s 内从**重采样瞬间的真实 EE 位姿**插值（位置线性 + 姿态 slerp）到目标；
- 最后转到 **root 系**存进 `pose_command_b` —— 观测 `ee_goal` 和 IK 读的都是它。

⚠️ **两个坑**：

1. **不要用"举臂"姿态当默认目标**：实测把 EE 目标锁在默认（举起）位姿，比放开走完整分布更差
   （`root_z<0.30` 的 20 s 触发率 55.5% vs 25.8%）；锁在**低位锚点**
   （`p_l=0.41, p_pitch=-0.08, 姿态全 0`）只有 **1.0%**（`DEFECT_LOG_zh.md` DEF-006）。
   部署起步建议用这个低位锚点。
2. **复位后第一帧命令是 0**：`pose_command_b` 初值是全 0（= 位置 0 + 单位四元数），
   Isaac 侧这是已知问题（`TODO_zh.md` P1-3 ⑫）。部署侧**自己初始化命令**：
   第一帧就把 `ee_goal` 设成"当前 EE 位姿（root 系）"、`body_pose` 设成"当前实测
   height/pitch/roll"，不要发 0。

---

## 6. 机械臂 IK：**必须自己实现**（最容易漏）

Isaac 侧机械臂**不由策略动作驱动**，而是：`CommandManager` 的 `ee_pose` 目标
→ `CommandDrivenIKAction`（`velocity/mdp/actions.py`）→ 每 50 Hz 更新 `arm_joint1..6`
的**位置目标** → `DelayedPD(300/20)` 执行。

参数（照抄）：

| 项 | 值 |
|---|---|
| 控制器 | 微分 IK，`ik_method="dls"`，`lambda_val=0.01`，`use_relative_mode=False`（绝对位姿） |
| 关节 | `arm_joint1..6`（夹爪不在其中，`gripper_joint1/2` 保持在默认 0 = 闭合） |
| 末端 body | `gripper_base`（body 下标 24 / 共 27 个 body） |
| body offset | 位置 (0,0,0)、姿态单位四元数（即目标就是 `gripper_base` 的位姿） |
| 命令坐标系 | root 系（`pose_command_b`） |
| 频率 | 每个策略周期一次（50 Hz） |
| 关节限位 | j1 [-2.618, 2.168]、j2 [0, 3.14]、j3 [-2.967, 0]、j4 [-1.745, 1.745]、j5 [-1.22, 1.22]、j6 [-2.094, 2.094] |

**如果部署侧不实现 IK**：机械臂永远停在默认角，同时 policy 的 `joint_pos` 观测与 `ee_goal`
语义全部对不上 → 站立都困难。三种做法（任选）：

1. 用 MuJoCo/真机的 IK 复刻同样的 DLS（最贴近训练）；
2. 自己解到关节角再发位置目标（只要与上面 IK 的结果接近，效果一般可用）；
3. 先**冻结**：把 `ee_goal` 与 IK 目标都设成"当前 EE 位姿"，臂不动（用于分阶段调试）。

默认姿态下 `gripper_base` 在 root 系的位姿（对框架的有效性检查点）：
`pos = [0.3492, 0.0, 0.4327]`，`quat(wxyz) = [-0.7373, 0.0, -0.6756, 0.0]`。

---

## 7. MuJoCo 侧：模型已经有一份，先对齐再动策略

**可用模型**（submodule 里已有）：`deep_robotics_model/M20_Piper_own/mjcf/M20_Piper_own.xml`
（+ `meshes/`）。同一份 URDF 在 `deep_robotics_model/M20_Piper_own/urdf/M20_Piper_own.urdf`。

已核对（本机实测）：

- 关节轴与 URDF 一致（`hipx: (-1,0,0)`、`hipy/knee/wheel: (0,-1,0)`；`arm_joint*: (0,0,1)`），
  **符号没有反**；限位也与 Isaac 一致；
- 默认姿态下 `gripper_base` 相对 `base_link`：MuJoCo `pos=(0.3492, 0, 0.4326)` vs
  Isaac `pos=(0.3492, 0, 0.4327)` —— 差 1e-4，运动学链对齐 ✅
  （四元数两者差一个整体负号 = 同一个旋转，别逐元素比较，换算成角度差比）。

**必须调整/注意**：

1. **关节顺序不同**（下面的 MuJoCo 序 ≠ 第 4 节的 Isaac 原生序，也 ≠ 动作序）：
   `floating_base, fl_hipx, fl_hipy, fl_knee, fl_wheel, fr_hipx, …, hr_wheel, arm_joint1..6, gripper_joint1/2`
   → 一律**按关节名映射**。
2. `timestep`：XML 里是 0.002，Isaac 是 0.005；改 0.005 或按第 2 节说明处理控制周期。
3. **执行器**（`tau = kp·(q_des − q) + kd·(ω_des − ω)`，全部是 DelayedPD，延迟 0–5 个物理步）：

   | 组 | kp | kd | 额外 |
   |---|---|---|---|
   | 腿（hipx/hipy/knee） | 80 | 2 | friction 0、armature 0 |
   | 轮 | **0** | 0.6 | armature 0.00243216（轮子的惯性主要在这里） |
   | 臂 | 300 | 20 | friction 0.01、armature 0.01 |
   | 夹爪 | 4000 | 200 | friction 0.01、armature 0.01 |

   MuJoCo 里：腿/臂/夹爪用 `position`（k=…）或 `general` 手写 PD；
   轮子用速度伺服（`<velocity kv="0.6"/>` 等价于 kp=0、kd=0.6）。
4. **力矩/速度限幅**按第 3 节的表（腿 76.4 N·m/22.4 rad/s、轮 21.6/79.3、臂 100/3.0、夹爪 10/1.0）。
5. **时延**：Isaac 的 DelayedPD 每个执行器随机 0–5 个物理步（0–25 ms）。
   sim2sim 先按 0 做，再补 1–2 步看退化；真机务必实测通信+驱动延迟。
6. **摩擦/接触**：训练时静/动摩擦随机 0.35–1.5、restitution 0–0.7（MuJoCo 取中间值先跑通）；
   自碰撞开启、求解迭代 4（位置）/1（速度）。
7. **根初始高度** 0.55 m（spawn）；训练 reset 还带 roll/pitch ±0.3 rad 的随机 →
   站立姿态的初始条件不必完全复刻，但**第一帧观测要用真实状态**（见第 5 节的坑 2）。
8. 接触力归因陷阱（`DEFECT_LOG_zh.md`：sensor 的 body 顺序 ≠ articulation body 顺序）——
   轮子承重 85–115 N 是正常的，别把"臂/夹爪受力"归因错。

---

## 8. sim2sim 推荐上线顺序（每步都能单独判定成败）

1. **裸模型站立**：`qpos = 默认角`，PD 发力，看能不能稳住（对齐执行器/质量/接触）。
2. **零动作**：策略输出全 0 ⇒ 关节目标 = 默认角，轮速 0 ⇒ 应保持站立。
3. **接 ONNX，但命令全部"零位移"**：`base_velocity = 0`、
   `body_pose = 当前实测 height/pitch/roll`、`ee_goal = 当前 EE 位姿(root 系)`；
   history 用第一帧填满。此时策略应该**接近站立**。
4. **打开 IK**（目标仍是当前 EE 位姿）：确认机械臂不对底盘产生明显扰动。
5. **给 EE 目标小步进**（root 系 ±2 cm、姿态 ±5°）：确认臂能动、底盘不摔。
6. **加底盘速度**（先 vx=0.3 m/s 小幅）：确认能走、方向对。
7. 再逐步放开到训练区间（vx ±5 等），并复现 `body_pose` 的高度/俯仰变化。

**数值验收**（第 3 步之前就该做）：

- 取同一组 `(policy_obs, history_flat)`（可随机生成），分别喂给
  `policy.onnx`（onnxruntime）与 `policy.pt`（torch.jit）→ 动作**相对**误差应 ~1e-6；
  ⚠️ 判据要用相对误差：策略输出没归一化（实测 `|a|max ≈ 184`），fp32 舍入本身就有 ~3e-5 绝对误差。
- 对照量固定用同一套定义：root 高度（用第 5 节公式）、倾角
  （`acos(-g_z)`，直立 0 / 侧躺 90°）、EE 位置跟踪误差（root 系）。

---

## 9. sim2sim → sim2real 的差异清单

| 类别 | 注意什么 |
|---|---|
| 姿态/角速度 | `base_ang_vel`（base 系）与 `projected_gravity` 都来自 IMU；重力投影只需 roll/pitch（yaw 无关）。别用"积分角速度"当姿态。**单位要换**：`deg/s → rad/s`，且角速度要乘 0.25、关节速度乘 0.05 |
| 关节反馈 | 编码器给 `q`、`q̇`；减默认角时用**同一份默认角表**（第 3 节）；轮子关节位置不参与观测（`joint_pos` 置零），但 `joint_vel` 里有轮速 |
| EE 观测 | `ee_goal`/IK 目标都在 **root 系**；真机要用正运动学算当前 EE 位姿再套目标。四元数顺序是 **wxyz**（别和 xyzw 混） |
| 时延与同步 | 所有观测必须**同一时刻**采样并打时间戳；控制回路 20 ms 预算里不要有阻塞 IO；如果观测延迟 >1–2 个周期，先按第 7 节第 5 条在仿真里加同样延迟看退化 |
| 动作下发 | 保持"**喂给观测的 `actions` = 实际下发的动作**"。Isaac 里 `actions` 观测就是策略原始输出；如果你在部署侧加了限幅/斜率限制，就必须把**限幅后**的值写回下一步的 `actions`，否则观测与实际不一致 |
| 安全 | 训练终止阈值（倾角 >0.8 rad、root 高度 <0.30 m）当作**接管阈值**；另加力矩/速度限幅、通信超时→阻尼/站立、软启动、急停 |
| 分布外 | 命令别超第 5 节的训练区间；EE 目标别超球坐标范围；地形训练时是平地（flat）——真机起步也应在平地 |
| 复位语义 | 触发保护后要**重新初始化 history 整窗**与命令（第 4.2、第 5 节的坑），否则策略看到"上一段摔之前的历史" |

---

## 10. 常见失败模式对照表

| 症状 | 先查这个 |
|---|---|
| 一上电就抖 / 立刻趴下 | 关节映射错（Isaac 原生序 vs MuJoCo 序 vs 动作序三者混用） |
| 站得住但一走就歪/反向 | 轮速目标符号或 `wz`/`vx` 命令定义（base 系 vs 世界系）不一致 |
| 机械臂纹丝不动 | 没实现 IK（动作里没有臂的自由度） |
| 臂一动底盘就摔 | EE 目标位置过高（别用"举臂"锚点）；或 IK 输出被限幅后与策略预期不符 |
| 策略输出长期饱和、乱抖 | 观测 scale 漏乘（`base_ang_vel ×0.25`、`joint_vel ×0.05`）、或 clip/scale 顺序反了 |
| 一段时间后动作发散 | history 窗口顺序拼反（应为最旧→最新）或推窗节奏不对（应按 20 ms，而不是每个物理步） |
| 复位后第一步就崩 | 第一帧喂了全 0 的 `ee_goal` / `body_pose`（第 5 节坑 2） |
| 高度命令没反应 | height 度量定义不同（忘了 `+0.09` 轮半径 或 没用四轮 z 的均值） |
| ONNX 结果和 TorchScript 差很多 | 输入名/顺序搞混（先 `policy_obs` 后 `history_flat`），或把 700 维当成 10×70 的二维输入 |

---

## 11. 权威文件清单（改部署脚本时照着这些看）

### 本仓库（策略侧，权威定义）

| 文件 | 内容 |
|---|---|
| `scripts/reinforcement_learning/rsl_rl/probe_deploy_layout.py` | **先在部署机上跑它**：打印关节序/默认角/增益/观测布局 |
| `scripts/reinforcement_learning/rsl_rl/export_deploy_policy.py` | 导出（TorchScript + ONNX + layout），含自检 |
| `docs/deploy_sim2sim_sim2real_zh.md` | 本文件 |
| `docs/train_history_flat_zh.md` | 训练 + 导出流程、维度自检 |
| `docs/review/DEFECT_LOG_zh.md` | DEF-006（EE 锚点/高度）、DEF-008（actions 观测）、DEF-011（命令坐标系）、DEF-016（关节序陷阱）、DEF-020（ONNX 自检） |
| `source/.../velocity/velocity_env_cfg.py` | 观测/终止/动作默认值（`ObservationsCfg`、`TerminationsCfg`） |
| `source/.../wheeled/deeprobotics_m20/flat_env_wbc_cfg.py` | 本任务的命令与课程（`WBCCurriculumCfg` = 训练终值） |
| `source/.../wheeled/deeprobotics_m20/rough_env_cfg.py` | 动作 scale/clip、观测 scale、默认关节名单 |
| `source/.../velocity/mdp/actions.py` | `CommandDrivenIKAction`（IK 实现） |
| `source/.../velocity/mdp/commands.py` | `HeightInvariantEECommand`（EE 目标采样+插值）、`BodyPoseCommand` |
| `source/.../velocity/mdp/observations.py` | 观测函数（含 `joint_pos_rel_without_wheel` 的行/列陷阱） |
| `source/.../velocity/mdp/utils.py` | `compute_base_height_rel_to_feet`（height 命令的度量） |
| `rl_training/assets/deeprobotics.py` | 执行器增益/延迟、初始关节角、USD 路径 |

### 机器人模型

| 文件 | 用途 |
|---|---|
| `deep_robotics_model/M20_Piper_own/mjcf/M20_Piper_own.xml` | **MuJoCo sim2sim 直接用**（含 meshes/） |
| `deep_robotics_model/M20_Piper_own/urdf/M20_Piper_own.urdf` | 关节轴/限位/effort/velocity 的权威来源 |
| `deep_robotics_model/M20_Piper_own/usd/M20_Piper_own.usd` | Isaac 侧资产（不要改，改了策略就对不上） |

### 常用命令

```bash
# 1) 拿权威布局（在部署机上，需要 Isaac 环境）
python scripts/reinforcement_learning/rsl_rl/probe_deploy_layout.py \
    --task History-Adaptation-Deeprobotics-M20-v0 --headless --num_envs 2

# 2) 重新导出（换了 checkpoint 之后）
python scripts/reinforcement_learning/rsl_rl/export_deploy_policy.py \
    --run logs/rsl_rl/history_adaptation/2026-09-20_00-50-31 --checkpoint model_19999.pt

# 3) 只做数值对照（不启动 Isaac，纯 torch + onnxruntime）
#    见第 8 节：同一组输入分别喂 policy.onnx 与 policy.pt，比相对误差
```
