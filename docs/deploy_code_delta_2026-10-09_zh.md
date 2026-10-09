# 真机部署（sim2real）代码需要跟着改的地方 —— 相对部署基线

**基线**：`main @ 2d49f47`（tag `deploy-baseline-2026-09-20`，对应 run `2026-09-20_00-50-31`
的 `exported_deploy/*`）。
**目标**：`codex/ll-train-detail-fix @ f663ee1`（2026-10-09，比基线多 **37 个 source/scripts 提交**）。

> 如果上次上机用的**不是** `2d49f47`（例如从更早/更晚的 commit 拉的），告我一声，我按那个 commit 重算。

本文只写**会影响真机部署代码**的东西；纯训练侧（奖励/课程/地形/扰动）只在 §4 一句带过。
逐条结论都有实测数字，来源是 `docs/review/DEFECT_LOG_zh.md` 的对应条目（DEF-0xx）。

## 0. 一句话结论

**策略接口一个字都没变**（观测 83 / history 10×70 / 动作 16 / 增益与缩放全同）
⇒ **换 `policy.onnx` 不用改部署代码**；要改的是**执行器参数与命令层语义**（6 条，见 §2）。

## 1. 先跑这一条：拿实测值当权威

```bash
python scripts/reinforcement_learning/rsl_rl/probe_deploy_layout.py \
    --task History-Adaptation-Deeprobotics-M20-v0 --headless --num_envs 2
```

2026-10-09 在当前代码上实测，逐项与部署手册（§0 速查表）对比——**全部一致**，
旧部署脚本的接口层可以原样沿用：

| 项 | 实测值 | 与基线 |
|---|---|---|
| 策略输入 | `policy_obs 83` = base_ang_vel3 + projected_gravity3 + velocity_commands3 + joint_pos24 + joint_vel24 + actions16 + ee_goal7 + body_pose_cmd3 | 一致 |
| history | `700` = 10 步 × 70（最旧→最新；每步 [base_ang_vel3, projected_gravity3, joint_pos24, joint_vel24, last_action16]） | 一致 |
| 策略输出 | `action 16`（`joint_pos` 12 + `joint_vel` 4 + `ee_ik` **0**） | 一致 |
| 腿目标增益 | hipx **0.125**、hipy/knee **0.25**（`q_des = q_default + gain*a`） | 一致 |
| 轮目标 | `omega_des = 5.0*a`，力矩 `= 0.6*(omega_des - omega)` | 一致 |
| 默认关节角 | fl/fr: hipx 0、hipy -0.6、knee +1.0；hl/hr: hipx 0、hipy **+0.6**、knee **-1.0** | 一致 |
| 观测缩放 | base_ang_vel x0.25、joint_vel x0.05、ee_goal clip ±3 | 一致 |
| 保护阈值 | 倾角 > **0.8 rad**、root 高度 < **0.30 m** | 一致 |
| body_pose 区间 | height (0.33, 0.55) / pitch ±0.35 / roll ±0.25 | 一致 |
| ee_pose 区间（play） | p_l (0.41, 0.41)、p_pitch (-0.08, -0.08)、p_yaw 0、o_* 0 | 一致 |

## 2. 必须跟着改的 6 处（部署代码 / 真机控制器）

### ① 机械臂 PD：`damping 20 -> 8`（DEF-064）

| | 旧 | 新 |
|---|---|---|
| `piper_arm` | stiffness 300 / **damping 20** | stiffness 300 / **damping 8** |

* 为什么：`|qd|` 到 5 rad/s 时阻尼力矩 `20*5 = 100 N·m` 正好吃满限幅 ⇒ 关节长期饱和
  （实测逐档 `|tau|` 均 53.4 N·m、**饱和 23.5%**、原地抖）。改后 **22.5 N·m / 饱和 0%**，
  而且跟踪还更好（4.14 -> 2.19 cm）。
* 真机怎么改：若真机的臂控制器是自己写的 PD，**按同一比例降阻尼（刚度不动）**；
  只降刚度是错的（150/8 力矩只再省 1.5 N·m，跟踪退到 2.89 cm）。

### ② 臂关节速度上限 = 3 rad/s（DEF-065，新增约束）

* 旧：只写了 cfg `velocity_limit=3.0`，但仿真里真正生效的是 USD 的 `maxJointVelocity=5`
  ⇒ 实测 `|qd|` p99 **恒为 5.0**、超速时间占比 **33%**（真机按 5 放开会过热/失控）。
* 新：启动事件 `align_joint_velocity_limits` 把 PhysX 的 DOF 上限改成 3.0 ⇒
  `|qd|` p99 **3.00**、超速降到 5~18%、力矩再降到 14~17 N·m。
* 真机怎么改：**臂关节 1~6 的速度上限按 3 rad/s 设**（joint6 本来就是 3）。

### ③ IK 输出必须夹到关节限位（DEF-054，新增保护）

* 旧：`DifferentialIKController.compute()` 返回 `joint_pos + delta` 直接下发 ⇒
  目标不可达时关节目标停在限位**外**，PD 一路顶到 100 N·m（实测 joint5 顶限位 65%）。
* 新：`CommandDrivenIKAction.apply_actions()` 加了一层保护——**把目标夹进
  `joint_pos_limits`**（`protect_joint_pos=True` 默认开，内缩余量 `joint_limit_margin=0.0`）。
  实测 joint1/2/3/5/6 的饱和与顶限位全面下降（joint5 顶限位 65%->25.6%、joint6 30.2%->1.0%）。
* 真机怎么改：IK 解出的关节角在**下发前 clamp 到关节限位**。

### ④ 夹爪：PD `286/5` + 开合指令 ±0.035（DEF-058 / DEF-065）

| | 旧 | 新 |
|---|---|---|
| `piper_gripper` | stiffness **4000** / damping **200** | stiffness **286** / damping **5** |
| 开指令 | ±0.04 rad（**超出 ±0.035 行程**） | **±0.035** |
| 夹持力矩上限 | 10 N·m | 10 N·m（不变：满行程误差刚好给到限幅） |

* 实测（方波阶跃，`probe_gripper_response.py`）：旧参数**位置环实际失效**——
  饱和 88~100%、稳态误差最大 **0.0314 rad**（行程的 90%）、多数段到不了目标；
  新参数饱和 **0%**、上升时间 **40~80 ms**、稳态误差 <=0.0005、开到底误差 0.005 -> **0.0001**。
* 真机怎么改：夹爪位置环按"**满行程误差 ≈ 额定夹持力**"标定（≈286/5 这一档），
  并且**开指令不要写超出行程的值**。

### ⑤ EE 命令的复位锚定（known_issues ⑫，DEF-034）—— 自己实现 EE 命令采样的必看

* 旧：复位后第一步 `pose_command_b` 仍是父类初值 `(0,0,0,1,0,0,0)`（底盘原点 + 单位姿态），
  观测/奖励看到的是**零位姿目标**（实测与真实 EE 位姿差 **0.43 m**）。
* 新：`HeightInvariantEECommand.reset()` 覆写 —— 复位瞬间把命令锚到**那一刻的真实 EE 位姿**；
  修后与真实位姿差 **0.000e+00**。
* 真机怎么改：若部署脚本自己维护 EE 目标，**复位/上电后的第一拍必须把目标设成当前实测 EE 位姿**，
  别用 0 或固定常数。

### ⑥ EE 目标碰撞盒改成机体局部系（known_issues ⑭，DEF-066）

* 旧：盒子 `[-0.3,-0.3,0]...[0.3,0.3,0.5]` 语义是"机体周围"，代码却拿**世界坐标**去比
  ⇒ 只有 env 0（正好在原点）碰巧有效，其它环境**静默失效**。
* 新：改比**高度不变系局部坐标**（随机器人走），盒子重定义为机体区域
  `[-0.3,-0.3,-0.60]...[0.3,0.3,-0.05]`（原点在世界 z = 0.60 m）。
  实测拒绝率从 ≈0 变成 **24%**（204 个候选拒 49）。
* 真机怎么改：若部署侧也有"EE 目标不能穿身体"的检查，**把盒子定义在机体坐标系里**。
* 附带一条**默认关闭**的可选件：FK 可达性过滤（给 `urdf_path` 才开）。实测能把 joint4 饱和
  47.7%->10.7%，但 EE 跟踪误差 5.87->15.20 cm ⇒ **默认关**，要用先做受控验收。

## 3. 真机限幅对齐表（部署控制器按这个设）

| 部位 | 力矩上限 | 速度上限 | 备注 |
|---|---|---|---|
| 腿（hipx/hipy/knee） | 76.4 N·m | 22.4 rad/s | 本次未改 |
| 轮 | 21.6 N·m | 79.3 rad/s | 速度伺服：`tau = 0.6*(w_des - w)` |
| 臂 joint1~6 | 100 N·m | **3 rad/s**（旧 USD 是 5） | PD 300/**8**；IK 输出夹关节限位 |
| 夹爪 | 10 N·m | 1.0 rad/s | PD **286/5**；开合目标 ±0.035 |

## 4. 不用改的部分（只是训练侧，部署代码别跟着动）

* 扰动加强 + 扰动课程（`push` 间隔 5~10 s、幅度 ±2/±1/yaw±0.52、课程 0.2x→1.0x、可选 `peak_scale`）
  —— 只影响"策略见过多强的扰动"，部署侧不需要复现 push。
* 静止伫立惩罚与三条 25k 步课程、`rel_standing_envs 0.02->0.15`。
* 多地形任务（`Rough-Slopes-*`）与 SlowVx 课程配方（v_x 台阶 150k/200k/250k/300k）、坡度门控。
* `max_noise_std` 默认 0 -> **1.2**（探索噪声上界，训练侧）。
* `joint_mirror -> joint_mirror_signed`（奖励侧符号修正）、删除 weight=0 的
  `action_mirror`/`action_sync`（打开就炸的死代码）。
* 报告的各类口径修正（`--reset-grace`、A/B 机身姿态指令固定、逐地形 pitch/roll 均值）——
  只影响体检报告，不影响部署。

工具侧（也不影响部署代码，但知道一下省得困惑）：

* `export_deploy_policy.py` 与 `probe_deploy_layout.py` **自部署基线以来一行未改**
  （`git diff 2d49f47..HEAD -- <这两个文件>` 为空）⇒ 导出流程与 `policy_layout.json` 的
  格式同上次上机时完全一样。
* `play.py` 多了一小段：把本仓库改过名的推挤事件 `randomize_push_robot` 也关掉
  （官方旧名字关不掉它）—— 只影响"本地看策略跑"，不影响部署。
* 旧的 4 个可视化脚本（`gait_test` / `torque_test` / `tracking_test` / `test`）已删除，
  统一成 `policy_report.py`；新增 `eval_fixed_command.py`、`summarize_run.py`、
  `smoke_regression.py`、`sweep_ee_anchor.py`、`probe_arm_pd.py`、`probe_gripper_response.py`、
  `probe_gait_symmetry.py`、`probe_reset_anchor_timing.py` 等。

## 5. 上机前请记住的已知短板

1. **上坡 + 低/零速最弱**（DEF-061）：256 envs 逐地形实测，上坡 `(0,0,0)` 的速度误差是平地的
   **4.4x**、高度抖动 **24x**、终止 0.094 且全是翻倒。真机上"斜坡上原地站住"要格外小心。
2. **多地形模型的能力大致到 terrain level 4.7/9**（`cloud_slowvx20k`），更陡的坡是分布外。
3. **抓取成功率没在高层训练过的策略上复验过**（仓库里现成的高层 checkpoint 是 2026-05 的）；
   目前只能说"夹爪位置环从失效变正常"（DEF-058）。
4. 若部署的是**多地形**模型，用 `2026-10-02_00-44-42_cloud_slowvx20k/exported_deploy/`；
   **平地**用 `2026-09-30_19-36-00_cloud_cap12_20k/exported_deploy/`。两者接口逐字段相同。

## 6. 验收清单（部署机上照做）

1. `probe_deploy_layout.py`（§1）—— 接口逐项对上。
2. MuJoCo sim2sim 按 `docs/deploy_sim2sim_sim2real_zh.md` §8 的 7 步顺序（裸模型站立 →
   零动作 → 接 ONNX 零命令 → 开 IK → EE 小步进 → 加 vx → 放开到训练区间）。
3. 关节/夹爪限幅按 §3 设好，**特别是臂 3 rad/s**（仿真里已对齐，真机别放宽）。
4. 复位后第一拍的 EE 目标 = 当前实测位姿（§2 ⑤）。
5. 臂 IK 的输出 clamp 到关节限位（§2 ③）。
