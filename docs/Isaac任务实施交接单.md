# WBC-OmniBench Isaac 实施交接单（仓库对齐版）

版本：v0.2  
日期：2026-11-18  
用途：在现有 Isaac 仓库基础上继续实施 WBC-OmniBench 新任务

> 状态口径：
>
> - **已完成**：仓库中已有可运行代码或配置，并至少通过配置解析、smoke 或数据链路验证。
> - **进行中**：基础任务已可运行，但视觉策略、评测或消融尚未闭环。
> - **部分完成**：核心链路已存在，但缺少任务级观测、评测指标、消融或性能验收。
> - **未开始**：只有设计，仓库中没有对应任务实现。
>
> 本文只把“未开始”和“部分完成”的内容列入下一步。两个基础 pick-place 任务已归入已完成基线，不再作为新任务重复开发。

---

## 0. 仓库对齐结论

### 0.1 已完成，不再列入后续任务开发

| 类别 | 已完成内容 | 仓库依据 |
|---|---|---|
| 机器人资产 | M20 四轮足与 Piper 已合并为单一 articulation，含底盘、轮、6 自由度机械臂和夹爪 | `source/rl_training/rl_training/assets/deeprobotics.py` |
| 低层控制 | 已有 locomotion、arm、WBC 和 history-adaptive 低层策略入口 | `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/` |
| 高层动作 | 13 维动作：`vx/vy/wz`、EE 位置增量、EE 姿态增量、机身高度/pitch/roll、夹爪 | `highlevel/mdp/teleop_ll_action.py`、`pre_trained_pick_wbc_action.py` |
| 基础任务 1 | `low-shelf-pick-place` 的场景、训练/遥操/采集任务、奖励、终止和相机配置 | `highlevel/config/low_shelf_place/` |
| 基础任务 2 | `two-cylinder-pick-place` 的场景、训练/遥操/采集任务、奖励和终止 | `highlevel/config/two_cylinder_place/` |
| 任务注册 | 每个基础任务均有 `Place-v0`、`Teleop-v0`、`Collect-v0` 三个 Gym task | 两个任务目录下的 `__init__.py` |
| 数据采集 | 键盘、VR、任务相机、HDF5 记录、回放检查和 LeRobot v3 转换链路 | `scripts/data/`、`source/rl_training/rl_training/recorders/hdf5.py` |
| 标准相机 | `top_camera`、`wrist_camera`、`workcell_camera` 可按采集任务启用，支持 RGB/RGB-D | 两个任务的 `scene_cfg.py` |
| 鱼眼全景 | dual/four 鱼眼原始图、可选等距柱面拼接、全景预览和 HDF5 记录 | `scripts/tools/fisheye_*.py`、`docs/fisheye_teleop_collection_zh.md` |
| 采样同步 | 状态、动作和所有相机可按统一 `record_hz` 采样，记录 `sim_time` 和 `sensor_frame` | `HDF5Recorder` |
| 任务评测 | 两个基础任务已有物体抬起、物体到目标、稳定放置、掉落、摔倒和超时的任务级终止判定 | 两个任务的 `env_cfg.py` / `mdp.py` |

两个基础任务作为后续复杂任务的实现模板，覆盖以下结构：

```text
SceneCfg
  -> EnvCfg
  -> RewardsCfg / TerminationsCfg
  -> Place / Teleop / Collect 三类任务注册
  -> 高层动作
  -> 相机与 HDF5
```

### 0.2 已完成任务的验收尾项

以下内容属于两个基础任务的性能验收，不属于“再实现一个新任务”：

- 尚未完成稳定的真实物理抓取抬升和完整 pick-place 成功率统计。
- 尚未用真实 VR 头显完成并统计足够的成功 episode。
- `two_cylinder` 已有特权脚本示例 episode，但其中物体采用运动学跟随，不能替代真实抓取成功率。
- 后续只有在需要宣称基线性能时补做；在此之前应描述为“任务工程基线已完成，成功率待验收”。

### 0.3 尚未完成，进入下一步

1. 统一 evaluator 接口和标准 episode metrics。
2. 接触力、碰撞、轮速/轮滑移、关节物体状态、路径长度、能耗和失败类型记录。
3. 面向 RL/策略训练的相机 observation adapter，包括全景原始面和等距柱面输入。
4. 全景消融、固定机身消融和机体协作消融的统一开关。
5. Lab 场景资产：三层抽屉、门、隔断、托盘、窄通道和可移动障碍。
6. YCB 物体注册与训练/评测资产分层。
7. P1 RearPick、P3 PeekOver、W1 DrawerLadder 及其后续任务。

当前进展：P1 RearPick 的 Teacher、Teleop 和 Collect 任务已实现，状态教师 smoke 和四路鱼眼 HDF5 采集 smoke 已通过；视觉学生和正式评测尚未开始。详见 [P1-RearPick 任务设计](pick_place/P1-RearPick-任务设计.md)。

## 1. 当前平台能力契约

### 1.1 机器人

- 四轮足底盘；
- Piper 六自由度机械臂；
- Piper 夹爪；
- 腕部 RGB-D；
- 前视 RGB-D；
- 尾部高位全景相机。

仓库状态：上述机器人已通过 `M20_Piper_own.usd` 作为单一 articulation 接入；标准相机和双/四路鱼眼采集链路已存在。前视相机和全景相机尚未作为当前两个基础任务的策略 observation 正式接入。

### 1.2 底盘控制能力

当前只允许使用：

- 前进/后退；
- 左右平移；
- 左右转向；
- 机身高度；
- 机身 pitch；
- 机身 roll；
- 轮速差和轮制动。

### 1.3 明确不支持

当前不假设以下能力：

- 单腿末端 3D 位置控制；
- 单腿末端姿态控制；
- 单腿独立伸到门把手、按钮、线缆或柜门上；
- 用某条腿持续夹住/压住环境物体。

因此，不要在本轮实施中把“腿作为机械臂”作为成功必要条件。

---

## 2. 实施优先级

### P0：剩余公共基础设施（部分完成）

已完成部分不再重复：

- 单一 articulation；
- 13 维高层动作；
- 两个基础任务的场景、奖励和终止结构；
- 标准相机和鱼眼采集链路；
- HDF5、回放和 LeRobot 转换链路。

下一步只补以下公共能力：

1. 实现统一 evaluator 接口，并让任务成功、失败、进度和子目标状态可由外部评测代码读取；
2. 扩展 episode 记录：
   - 物体位姿；
   - 机器人关节状态；
   - 接触力；
   - 碰撞；
   - 轮速和轮滑移；
   - 关节物体角度/位移；
   - 路径长度和能耗；
3. 增加面向策略训练的前视和全景 observation adapter，明确原始鱼眼面片与等距柱面输入；
4. 增加统一时间戳与传感器延迟/遮挡配置；
5. 建立可复用的 Lab 场景和关节物体资产；
6. 增加标准指标汇总、失败类型分类和消融报告。

### 优先级 1：P1 RearPick（进行中）

实施规格：[P1-RearPick 任务设计](pick_place/P1-RearPick-任务设计.md)

目标：

- 目标起始只被全景相机看到；
- 前视和腕部看不到；
- 机器人完成发现、接近、局部重捕获和抓取。

最小场景：

- 一个后侧目标；
- 一个托盘；
- 一个矮隔断或货架；
- 一张桌子。

验收门槛：

- 起始帧前视/腕部中目标不可见；
- 全景中目标可见；
- 关闭全景后，任务无法从固定初始视角直接完成；
- 目标进入局部视场后可以被稳定重捕获。

### 优先级 2：P3 PeekOver（进行中）

目标：

- 利用高位全景相机越过 0.6–1.0 m 隔断；
- 前视被隔断完全遮挡；
- 机器人必须选择正确方向绕过隔断。

验收门槛：

- 隔断高度足以挡住前视和腕部；
- 全景能看到隔断后的目标或目标局部；
- 随机左右放置目标，机器人不能依赖固定路线。

### 优先级 3：W1 DrawerLadder（进行中）

目标：

- 三层抽屉；
- 目标在中间或底层；
- 机器人拉开抽屉、抓取目标、放入托盘、关闭抽屉。

验收门槛：

- 抽屉高度覆盖低位、中位和高位；
- 固定机身高度时至少一个抽屉无法舒适到达；
- 抽屉打开、目标取出和抽屉关闭分别可判定；
- 抽屉状态和接触力可记录。

### 优先级 4：W2 DoorBrace（进行中）

目标：

- 机械臂单独拉门不稳定；
- 机器人通过轮制动、轮速差和机身姿态建立有效支撑；
- 门打开后机器人通过门框。

验收门槛：

- 固定机身 + 仅机械臂的基线成功率应低于 30%；
- 允许机身高度/pitch/roll 和轮制动后，成功率应有明显提升；
- 记录轮滑移、底盘后退量和接触冲量；
- 不要求单腿独立接触门板。

### 优先级 5：W5 HighLowServe（进行中）

目标：

- 同一回合抓取低位和高位两个目标；
- 低位在桌下或低柜内；
- 高位在 1.2–1.5 m 货架上；
- 两个目标都放入托盘。

验收门槛：

- 固定机身姿态时，至少一个目标超出稳定可达空间；
- 改变机身高度/pitch/roll 后可达性明显提高；
- 高位和低位子目标分别计分。

### 优先级 6：P2 BlindIntercept（进行中）

目标：

- 目标沿轨道或传送带运动；
- 在 1–3 s 内离开腕部和前视视场；
- 全景持续可见；
- 机器人完成拦截和抓取。

验收门槛：

- 盲区事件必须可记录；
- 目标在盲区期间全景可见；
- 成功判定包括抓住、抬离和稳定保持。

### 优先级 7：L1 HomeMission（进行中）

目标：

- 组合搜索、开门/开抽屉、抓取、搬运、放置和复位；
- 使用语言或属性目标；
- 允许一次重抓取或一次路径重规划。

建议首发任务：

> 把右侧柜子里的蓝色盒子拿出来，放到门后的篮子里，然后关上柜门。

验收门槛：

- 至少 5 个子目标；
- 每个子目标有独立完成状态；
- 目标可能在起始时只被全景看到；
- 最终成功需要所有必要子目标完成。

---

## 3. 暂不实施的任务

以下任务移入备选池，等平台开放单腿末端控制后再做：

- B1 LegHoldPick：腿撑门取物；
- E2 CableHoldPlug：足固定线缆并插接；
- 任何要求足端按按钮、压把手、夹住柜门的任务；
- 任何要求某条腿独立保持门开度的任务。

当前可用替代：

- 用机身高度和 pitch/roll 扩展工作空间；
- 用轮制动和轮速差抵抗机械臂反作用；
- 用机身或轮组与环境的被动接触建立支撑；
- 用底盘重新站位代替腿部独立操作。

---

## 4. 公共场景与资产

仓库现状：两个基础任务已有各自的轻量场景配置；尚未建立统一的 Lab 场景，也没有门、抽屉、隔断等关节或交互资产。当前 YCB 实际使用 `004_sugar_box.usd`。

### 4.1 Lab 场景最小集合

- 地面；
- 一张桌子；
- 一个三层抽屉柜；
- 一扇可调阻力的门；
- 一个 0.6–1.0 m 隔断；
- 一个托盘；
- 一个窄通道或 L 形通道；
- 可移动障碍物。

### 4.2 YCB 首发物体

- mustard bottle；
- mug；
- cracker box；
- tomato soup can；
- bowl；
- sugar box；
- tennis ball；
- foam brick。

训练可使用简化碰撞体，评测使用高保真 YCB 资产。

### 4.3 关节物体

优先顺序：

1. Isaac 自带 cabinet/drawer；
2. ManiSkill 的 cabinet、drawer、fridge 资产；
3. AKB-48 / PartNet-Mobility 转换资产。

统一记录：

- 关节轴；
- 关节限位；
- 当前角度/位移；
- 目标开度；
- 接触力；
- 是否发生非物理破坏。

---

## 5. 统一 observation / action / evaluator

仓库现状：

- Action 已实现，当前实际接口与下方建议兼容。
- 状态、关节、物体位姿和相机已有记录能力。
- Observation 尚未形成跨任务统一 schema，全景尚未进入策略 observation。
- Evaluator 目前是任务级 termination 函数，尚无统一外部 evaluator 接口。

### 5.1 Observation 建议

```text
obs = {
  wrist_rgb,
  wrist_depth,
  front_rgb,
  front_depth,
  pano_raw,
  pano_equirect,
  camera_timestamps,
  base_pose,
  base_twist,
  base_height,
  base_rpy,
  joint_pos,
  joint_vel,
  wheel_velocity,
  wheel_contact,
  arm_joint_pos,
  arm_joint_vel,
  ee_pose,
  gripper_width,
  previous_action,
}
```

训练阶段可以额外使用特权信息，但评测阶段不得输入：

- 物体真值位姿；
- 门/抽屉真值关节状态；
- 接触力真值；
- 目标区域真值；
- 未来轨迹真值。

### 5.2 Action 建议

状态：**已完成**。当前实现为 13 维：

```text
action = {
  vx, vy, wz,
  delta_ee_position_x, delta_ee_position_y, delta_ee_position_z,
  delta_ee_roll, delta_ee_pitch, delta_ee_yaw,
  delta_body_height, delta_body_pitch, delta_body_roll,
  gripper,
}
```

如果低层 WBC 已经封装轮速差和轮制动，可以只在高层暴露底盘速度和机身姿态目标。

### 5.3 Evaluator

状态：**未完成**。两个基础任务有各自的 termination 判定，但还没有统一类接口、统一进度和指标输出。

每个任务至少提供：

```python
reset(scene_state)
step(scene_state, action, dt)
is_success() -> bool
is_failure() -> bool
progress() -> float
metrics() -> dict
```

必须支持：

- 稳定时间；
- 子目标进度；
- 超时；
- 摔倒；
- 严重碰撞；
- 越界；
- 物体损坏；
- 关节物体破坏。

---

## 6. 指标记录

状态：**部分完成**。当前 HDF5 已记录 success、动作、机器人状态、物体/目标位姿、相机帧和终止项；下列 benchmark 指标尚未统一实现。

每个 episode 至少记录：

- success；
- progress；
- completion_time；
- path_length；
- energy；
- collisions；
- base_roll_max；
- base_pitch_max；
- wheel_slip_distance；
- max_joint_torque_ratio；
- contact_force_peak；
- target_reacquisition_time；
- pano_visible_ratio；
- front_visible_ratio；
- wrist_visible_ratio；
- failure_type。

### 全景消融

- 无全景；
- 全景降采样；
- 全景延迟；
- 全景随机遮挡；
- 全景外参扰动。

### 机体协作消融

- 固定机身高度；
- 固定 pitch/roll；
- 禁用轮制动；
- 禁用轮速差；
- 禁用机身/轮组被动接触。

---

## 7. 任务验收顺序

已完成的两个基础任务不再列入开发顺序，只作为场景、动作、奖励、采集和评测结构的参考实现。

建议按以下顺序推进剩余工作：

0. 补齐统一 evaluator、metrics、全景 observation 和 Lab 资产；
1. P1 RearPick；
2. P3 PeekOver；
3. W1 DrawerLadder；
4. W2 DoorBrace；
5. W5 HighLowServe；
6. P2 BlindIntercept；
7. L1 HomeMission。

每个任务完成后，先提交：

- 场景截图；
- 相机视角截图；
- 成功/失败轨迹；
- evaluator 判定日志；
- 一次无全景消融；
- 一次固定机身消融；
- 失败类型统计。

---

## 8. 实现时的关键风险

### 8.1 全景相机成本

不要直接对每个并行环境输出高分辨率等距柱状图。优先：

- 训练使用 4 路低分辨率原始图；
- 或使用 cube map；
- 评测时再输出高分辨率全景。

### 8.2 抓取接触

YCB 小物体和夹爪接触是 sim-to-real gap 的主要来源。任务设计应避免把 P1/P3 的成功率完全绑定在透明、薄壁或极小物体上。

### 8.3 门轴和轮地接触

W2 必须做门轴阻力、门重、地面摩擦和轮附着随机化。否则任务很容易退化成“机械臂推一下就成功”。

### 8.4 任务作弊路径

必须防止：

- 使用物体真值位姿；
- 通过固定初始视角直接完成；
- 把目标放在前视/腕部初始可见位置；
- 用超长机械臂或非物理关节扩展；
- 用碰撞体穿透代替接触；
- 把抽屉/门直接设置为目标状态；
- 用外部传送带直接送入托盘。

---

## 9. 下一步交付物

已完成项不再重复交付。负责 Isaac 实施的 agent 应先交付：

1. 统一 `TaskEvaluator` 接口和 episode metrics schema；
2. 基础任务缺少的接触力、碰撞、轮滑移和失败类型记录；
3. 全景 observation adapter 及全景消融开关；
4. 可复用的 Lab 场景资产，包括门、三层抽屉、隔断和托盘；
5. P1 RearPick 完整可运行环境；
6. P1 的全景消融结果；
7. P3 和 W1 的场景占位与任务配置。

之后再依次推进 W2、W5、P2 和 L1。

